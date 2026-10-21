"""Run one explicit command, preserve its provenance, and propagate failures."""

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--allow-existing-eval", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("An explicit command after -- is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.eval_dir:
        if args.eval_dir.exists() and not args.allow_existing_eval:
            raise FileExistsError(args.eval_dir)
        args.eval_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"command": command, "cwd": str(Path.cwd()),
                "started_utc": datetime.now(timezone.utc).isoformat()}
    config = {}
    for token in command:
        if "=" in token and not token.startswith("--"):
            key, value = token.split("=", 1)
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            config[key] = value
    if "subliminal.eval_steered" in command:
        config["backend"] = "hf-transformers"
    elif "subliminal.eval" in command:
        config["backend"] = "vllm"
    if args.eval_dir:
        (args.eval_dir / "repro_eval_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8")
    start = time.monotonic()
    exit_code = None
    try:
        with (args.output_dir / "console.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                exit_code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                raise
    finally:
        manifest.update(exit_code=exit_code, wall_seconds=time.monotonic() - start,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        (args.output_dir / "task.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
