"""Measure the upstream SFT path, then estimate its time and manual CU budget.

No GPU work occurs without the run subcommand and --execute. The estimate
subcommand uses only the standard library. CU values are user observations,
not billing telemetry, and estimates never enforce a Colab account hard limit.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Must be a finite positive number.")
    return result


def nonnegative_float(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("Must be a finite nonnegative number.")
    return result


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return result


def boolean(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("Use true or false.")
    return value.lower() == "true"


def split_fraction(value):
    result = float(value)
    if not math.isfinite(result) or not 0 <= result < 1:
        raise argparse.ArgumentTypeError("Must be a finite fraction in [0, 1).")
    return result


def write_json(path, data):
    # Use a temporary sibling so interruption is less likely to truncate metrics.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def estimate(args):
    if args.safety_factor < 1:
        raise SystemExit("--safety-factor must be at least 1.")
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if metrics.get("status") != "complete":
        raise SystemExit("Only a completed benchmark can produce a training go/no-go estimate.")
    if metrics.get("billing_mode") == "local":
        raise SystemExit("A local smoke test cannot estimate paid A100 performance or CU use.")
    before = metrics["manual_cu_balance_before"]
    if before is None or metrics.get("manual_cu_per_hour") is None:
        raise SystemExit("CU readings were not recorded. Use summary, then review the current runtime rate in chat before full training.")
    if args.cu_after > before:
        raise SystemExit("CU balance increased; record purchases/resets separately and rerun the benchmark.")
    consumed = before - args.cu_after
    wall_seconds = metrics["total_wall_seconds"]
    measured_rate = consumed / (wall_seconds / 3600) if wall_seconds > 0 else 0.0
    # Account displays may round or update late; never infer a free GPU from zero delta.
    rate = max(metrics["manual_cu_per_hour"], measured_rate)
    steps = metrics["full_planned_optimizer_steps"]
    step_seconds = metrics["steady_mean_optimizer_step_seconds"]
    train_hours = steps * step_seconds / 3600
    conservative_hours = train_hours * args.safety_factor + args.overhead_hours
    conservative_cu = conservative_hours * rate
    available_cu = min(args.phase_cu_cap, args.cumulative_cu_cap - args.cumulative_used_cu,
                       args.cu_after - args.reserve_cu)
    result = {
        "benchmark_actual_optimizer_steps": metrics["actual_optimizer_steps"],
        "full_planned_optimizer_steps": steps,
        "measured_benchmark_cu_delta": consumed,
        "manual_balance_after": args.cu_after,
        "balance_delta_cu_per_hour": measured_rate,
        "conservative_cu_per_hour_used": rate,
        "training_only_estimated_hours": train_hours,
        "safety_factor": args.safety_factor,
        "manual_extra_overhead_hours": args.overhead_hours,
        "conservative_estimated_hours": conservative_hours,
        "conservative_estimated_cu": conservative_cu,
        "available_training_cu_after_caps_and_reserve": max(0.0, available_cu),
        "remaining_hours": args.remaining_hours,
        "go_based_on_entered_estimates": conservative_hours <= args.remaining_hours
        and conservative_cu <= available_cu,
        "warning": "Estimate only. Recheck live balance/rate; manual monitoring and runtime termination are required.",
    }
    output = args.output or args.metrics.with_name("budget_estimate.json")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite an existing estimate: {output}")
    write_json(output, result)
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")


def summary(args):
    """Show automatic measurements without asking the user to keep a ledger."""
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if metrics.get("status") != "complete":
        raise SystemExit("Benchmark is incomplete; do not start full training.")
    seconds = metrics.get("steady_mean_optimizer_step_seconds")
    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
        raise SystemExit("Benchmark has no valid steady step-time measurement.")
    hours = metrics["full_planned_optimizer_steps"] * seconds / 3600
    result = {
        "actual_optimizer_steps": metrics["actual_optimizer_steps"],
        "full_planned_optimizer_steps": metrics["full_planned_optimizer_steps"],
        "steady_seconds_per_optimizer_step": seconds,
        "estimated_training_hours": hours,
        "conservative_training_hours_including_overhead": hours * 1.5 + 1,
        "cu_used": "Not Recorded",
        "next": "Share this output in chat. Review the current GPU rate once before starting full training.",
    }
    if metrics.get("billing_mode") == "local":
        result["next"] = "Local smoke only; this does not predict A100 speed or CU usage."
    print(json.dumps(result, indent=2))


def run(args):
    started = time.perf_counter()
    started_utc = utc_now()
    if not args.execute:
        raise SystemExit("No GPU work started. Re-run with --execute only after checking live GPU rate and CU balance.")
    if args.steps > 100:
        raise SystemExit("This benchmark intentionally permits at most 100 optimizer steps.")
    if args.warmup_steps >= args.steps:
        raise SystemExit("--warmup-steps must be smaller than --steps.")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps cannot be negative.")
    if args.packing and args.attn_implementation != "flash_attention_2":
        raise SystemExit("TRL 1.5.1 packed BFD training requires FlashAttention; use flash_attention_2 or --packing false.")
    if not args.local:
        if not getattr(args, "no_cu_record", False) and (args.cu_before is None or args.cu_per_hour is None):
            raise SystemExit("Paid runtime requires --cu-before and --cu-per-hour from the live Colab display.")
        if args.cu_before is not None and args.cu_before <= args.reserve_cu:
            raise SystemExit("Balance does not exceed the protected reserve.")
    elif args.cu_before is not None or args.cu_per_hour is not None:
        raise SystemExit("Do not supply Colab billing values with --local.")
    if args.output_dir.exists():
        raise SystemExit(f"Use a new benchmark output directory; refusing to overwrite: {args.output_dir}")
    data_file = args.data_file.resolve()
    if not data_file.is_file():
        raise SystemExit(f"Missing data file: {data_file}")
    repo = args.repo.resolve()
    train_path = repo / "src" / "subliminal" / "train.py"
    if "# One-day reproduction operational controls." not in train_path.read_text(encoding="utf-8"):
        raise SystemExit("Apply patch_train.py first; upstream currently has no max_steps setting.")
    sys.path.insert(0, str(repo / "src"))
    import torch
    from transformers import TrainerCallback
    import subliminal.train as upstream

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; no training started.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("BF16 is unsupported on the selected CUDA device.")
    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "benchmark_metrics.json"
    metrics = {
        "status": "initializing",
        "started_utc": started_utc,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("transformers", "trl", "peft", "datasets", "accelerate")},
        "manual_cu_balance_before": args.cu_before,
        "manual_cu_per_hour": args.cu_per_hour,
        "billing_mode": "local" if args.local else ("colab_unrecorded" if getattr(args, "no_cu_record", False) else "colab"),
        "model": args.model,
        "data_file": str(data_file),
        "data_file_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
        "requested_optimizer_steps": args.steps,
        "planned_epochs": args.epochs,
        "per_device_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "attn_implementation": args.attn_implementation,
        "packing": args.packing,
        "val_split": args.val_split,
        "warmup_timing_steps_excluded": args.warmup_steps,
        "optimizer_step_seconds": [],
        "actual_optimizer_steps": 0,
    }
    try:
        metrics["git_commit"] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        metrics["git_commit"] = "NEEDS VERIFICATION"
    write_json(metrics_path, metrics)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    original_build = upstream.build_dataset
    original_trainer = upstream.SFTTrainer

    def measured_build(*build_args, **build_kwargs):
        train_ds, val_ds = original_build(*build_args, **build_kwargs)
        metrics["raw_train_split_rows"] = len(train_ds)
        metrics["raw_validation_split_rows"] = len(val_ds) if val_ds is not None else 0
        return train_ds, val_ds

    class TimingCallback(TrainerCallback):
        def on_train_begin(self, arguments, state, control, **kwargs):
            torch.cuda.synchronize()
            self.train_started = time.perf_counter()
            metrics["training_started_utc"] = utc_now()
            metrics["trainer_scheduled_optimizer_steps"] = state.max_steps
            metrics["status"] = "training"
            write_json(metrics_path, metrics)

        def on_step_begin(self, arguments, state, control, **kwargs):
            torch.cuda.synchronize()
            self.step_started = time.perf_counter()

        def on_step_end(self, arguments, state, control, **kwargs):
            torch.cuda.synchronize()
            metrics["optimizer_step_seconds"].append(time.perf_counter() - self.step_started)
            metrics["actual_optimizer_steps"] = state.global_step
            metrics["peak_allocated_vram_bytes"] = torch.cuda.max_memory_allocated()
            metrics["peak_reserved_vram_bytes"] = torch.cuda.max_memory_reserved()
            metrics["total_wall_seconds_so_far"] = time.perf_counter() - started
            # A wall-time guard supplements the real optimizer step bound. It is
            # checked only between steps and is not a billing/account hard limit.
            if args.max_wall_minutes and metrics["total_wall_seconds_so_far"] >= args.max_wall_minutes * 60:
                metrics["wall_guard_triggered"] = True
                control.should_training_stop = True
            write_json(metrics_path, metrics)

        def on_train_end(self, arguments, state, control, **kwargs):
            torch.cuda.synchronize()
            metrics["training_ended_utc"] = utc_now()
            metrics["trainer_train_wall_seconds"] = time.perf_counter() - self.train_started

    class MeasuredTrainer(original_trainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            metrics["packed_train_rows"] = len(self.train_dataset)
            metrics["packed_validation_rows"] = len(self.eval_dataset) if self.eval_dataset is not None else 0
            loader = self.get_train_dataloader()
            full_args = copy.copy(self.args)
            full_args.max_steps = -1
            full_args.num_train_epochs = args.epochs
            total_batch = full_args.train_batch_size * full_args.gradient_accumulation_steps * full_args.world_size
            plan = self.set_initial_training_values(full_args, loader, total_batch)
            metrics["dataloader_batches_per_epoch"] = len(loader)
            metrics["optimizer_steps_per_epoch"] = plan[1]
            metrics["full_planned_optimizer_steps"] = plan[6]
            self.add_callback(TimingCallback())
            write_json(metrics_path, metrics)

    config = upstream.Config()
    config.model = args.model
    config.attn_implementation = args.attn_implementation
    config.num_train_epochs = args.epochs
    config.max_steps = args.steps
    config.per_device_train_batch_size = args.batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation
    config.packing = args.packing
    config.val_split = args.val_split
    config.report_to = "none"
    config.resume_from_checkpoint = ""
    config.save_strategy = "no"
    config.run_name = args.output_dir.name
    metrics["config"] = {key: value for key, value in vars(config).items()
                         if not key.startswith("_") and isinstance(value, (str, int, float, bool, type(None)))}
    upstream.build_dataset = measured_build
    upstream.SFTTrainer = MeasuredTrainer
    try:
        trainer = upstream.train(config, str(data_file), str(args.output_dir))
        torch.cuda.synchronize()
        metrics["actual_optimizer_steps"] = trainer.state.global_step
        durations = metrics["optimizer_step_seconds"]
        steady = durations[args.warmup_steps:]
        metrics["mean_optimizer_step_seconds"] = statistics.mean(durations) if durations else None
        metrics["steady_mean_optimizer_step_seconds"] = statistics.mean(steady) if steady else None
        metrics["status"] = "complete" if trainer.state.global_step == args.steps and steady else "incomplete"
    except BaseException as error:
        metrics["status"] = "failed"
        metrics["error_type"] = type(error).__name__
        metrics["error"] = str(error)
        raise
    finally:
        upstream.build_dataset = original_build
        upstream.SFTTrainer = original_trainer
        metrics["ended_utc"] = utc_now()
        metrics["total_wall_seconds"] = time.perf_counter() - started
        metrics["peak_allocated_vram_bytes"] = torch.cuda.max_memory_allocated()
        metrics["peak_reserved_vram_bytes"] = torch.cuda.max_memory_reserved()
        write_json(metrics_path, metrics)
        print(f"Benchmark metrics: {metrics_path}")
        if not args.local:
            print("Run summary and share the output in chat before full training. CU use is not inferred from step speed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Execute at most 100 upstream optimizer steps.")
    run_parser.add_argument("--repo", type=Path, required=True)
    run_parser.add_argument("--data-file", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    run_parser.add_argument("--steps", type=positive_int, default=100)
    run_parser.add_argument("--epochs", type=positive_float, default=2)
    run_parser.add_argument("--batch-size", type=positive_int, default=8)
    run_parser.add_argument("--gradient-accumulation", type=positive_int, default=1)
    run_parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="flash_attention_2")
    run_parser.add_argument("--packing", type=boolean, default=True)
    run_parser.add_argument("--val-split", type=split_fraction, default=0.05)
    run_parser.add_argument("--warmup-steps", type=int, default=10)
    run_parser.add_argument("--max-wall-minutes", type=positive_float, default=25)
    run_parser.add_argument("--local", action="store_true", help="Local smoke test; never use to infer A100 throughput.")
    run_parser.add_argument("--no-cu-record", action="store_true", help="Skip manual CU fields; review the current runtime rate separately before full training.")
    run_parser.add_argument("--cu-before", type=nonnegative_float)
    run_parser.add_argument("--cu-per-hour", type=positive_float)
    run_parser.add_argument("--reserve-cu", type=nonnegative_float, default=40)
    run_parser.add_argument("--execute", action="store_true")
    run_parser.set_defaults(func=run)
    estimate_parser = sub.add_parser("estimate", help="Pure offline time/CU arithmetic from a completed benchmark.")
    estimate_parser.add_argument("--metrics", type=Path, required=True)
    estimate_parser.add_argument("--cu-after", type=nonnegative_float, required=True)
    estimate_parser.add_argument("--remaining-hours", type=positive_float, required=True)
    estimate_parser.add_argument("--cumulative-used-cu", type=nonnegative_float, required=True)
    estimate_parser.add_argument("--phase-cu-cap", type=positive_float, default=60)
    estimate_parser.add_argument("--cumulative-cu-cap", type=positive_float, default=160)
    estimate_parser.add_argument("--reserve-cu", type=nonnegative_float, default=40)
    estimate_parser.add_argument("--safety-factor", type=positive_float, default=1.5)
    estimate_parser.add_argument("--overhead-hours", type=nonnegative_float, default=1)
    estimate_parser.add_argument("--output", type=Path)
    estimate_parser.set_defaults(func=estimate)
    summary_parser = sub.add_parser("summary", help="Show automatic speed estimates without manual bookkeeping.")
    summary_parser.add_argument("--metrics", type=Path, required=True)
    summary_parser.set_defaults(func=summary)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
