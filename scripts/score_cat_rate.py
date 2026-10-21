"""Re-score saved completions with one strict upstream first-word rule, on CPU."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import sys


COMPARABLE_KEYS = ("model", "temperature", "max_new_tokens", "seed", "dtype",
                   "attn_implementation", "batch_size", "backend", "prompt_set_name")


def normalize_response(text: str) -> str:
    """Match subliminal.dataset.normalize_response exactly, without importing NumPy."""
    text = text.strip().lower()
    text = re.sub(r"[.!?,;:\"'()\[\]{}<>]", "", text)
    tokens = text.split()
    return tokens[0] if tokens else ""


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (index - low) * (ordered[high] - ordered[low])


def bootstrap_interval(rates: list[float], repetitions: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    samples = [statistics.fmean(rng.choices(rates, k=len(rates))) for _ in range(repetitions)]
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def canonical_prompts(path: Path | None, prompt_set: str) -> list[str] | None:
    if path is None:
        return None
    name = {"pos": "ANIMAL_PROMPTS", "neg": "NEGATIVE_ANIMAL_PROMPTS", "off": "OFFTOPIC_PROMPTS"}[prompt_set]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            value = ast.literal_eval(node.value)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise TypeError(f"Invalid canonical prompt list: {name}")
            return value
    raise ValueError(f"Cannot find {name} in {path}")


def score_run(name: str, directory: Path, args: argparse.Namespace, expected_texts: list[str] | None) -> dict:
    samples = directory / "eval_samples.jsonl"
    source_summary = directory / "eval_results.json"
    metadata = json.loads(source_summary.read_text(encoding="utf-8")) if source_summary.exists() else {}
    config_path = directory / "repro_eval_config.json"
    if config_path.exists():
        sidecar = json.loads(config_path.read_text(encoding="utf-8"))
        # These keys must match the source summary if both contain them.
        for key in ("model", "temperature", "samples_per_prompt", "prompt_set_name"):
            if key in metadata and key in sidecar and metadata[key] != sidecar[key]:
                raise ValueError(f"{name}: conflicting metadata for {key}")
        metadata.update(sidecar)
    groups: dict[int, dict] = {}
    mismatched_hit = 0
    with samples.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row.get("prompt_idx"), int) or isinstance(row["prompt_idx"], bool):
                raise ValueError(f"Invalid prompt_idx: {samples}:{number}")
            if not isinstance(row.get("prompt"), str) or not isinstance(row.get("completion"), str):
                raise ValueError(f"Missing prompt/completion strings: {samples}:{number}")
            word = normalize_response(row["completion"])
            if "first_word" in row and row["first_word"] != word:
                raise ValueError(f"Stored first_word disagrees with completion: {samples}:{number}")
            hit = word == args.target
            mismatched_hit += int("hit" in row and row["hit"] != hit)
            pi = row["prompt_idx"]
            if pi not in groups:
                groups[pi] = {"prompt_idx": pi, "prompt": row["prompt"], "hits": 0, "total": 0, "word_counts": Counter()}
            group = groups[pi]
            if group["prompt"] != row["prompt"]:
                raise ValueError(f"Prompt text changes within index {pi}: {name}")
            group["hits"] += int(hit)
            group["total"] += 1
            group["word_counts"][word] += 1
    if sorted(groups) != list(range(args.expected_prompts)):
        raise ValueError(f"{name}: expected contiguous indices 0..{args.expected_prompts - 1}, found {sorted(groups)}")
    per_prompt = [groups[index] for index in sorted(groups)]
    for item in per_prompt:
        if item["total"] != args.expected_samples_per_prompt:
            raise ValueError(f"{name}: prompt {item['prompt_idx']} has {item['total']} samples; expected {args.expected_samples_per_prompt}")
        if expected_texts is not None and item["prompt"] != expected_texts[item["prompt_idx"]]:
            raise ValueError(f"{name}: prompt {item['prompt_idx']} differs from the canonical source")
        item["rate"] = item["hits"] / item["total"]
        item["word_counts"] = dict(item["word_counts"])
    prompt_payload = json.dumps([item["prompt"] for item in per_prompt], ensure_ascii=False, separators=(",", ":")).encode()
    missing = [key for key in COMPARABLE_KEYS if key not in metadata]
    if args.require_metadata and missing:
        raise ValueError(f"{name}: missing reproducibility metadata {missing}; write repro_eval_config.json")
    hits, total = sum(x["hits"] for x in per_prompt), sum(x["total"] for x in per_prompt)
    if metadata.get("target_word", args.target) != args.target:
        raise ValueError(f"{name}: summary target differs from the scoring target")
    return {"condition": name, "input_directory": str(directory.resolve()),
            "samples_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt_payload).hexdigest(),
            "scoring_rule": "upstream normalize_response(completion) == target; ignore saved hit",
            "target": args.target, "cat_rate": hits / total, "hits": hits, "samples": total,
            "num_prompts": len(per_prompt), "samples_per_prompt": args.expected_samples_per_prompt,
            "prompt_bootstrap_95_ci": bootstrap_interval([x["rate"] for x in per_prompt], args.bootstrap, args.bootstrap_seed),
            "saved_hit_disagreements": mismatched_hit, "metadata": metadata,
            "metadata_status": "verified" if not missing else "NEEDS VERIFICATION",
            "missing_metadata": missing, "canonical_prompts_checked": expected_texts is not None,
            "per_prompt": per_prompt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Condition=/path/to/evaluation/directory; repeat")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for offline score outputs")
    parser.add_argument("--expected-prompts", type=int, default=50)
    parser.add_argument("--expected-samples-per-prompt", type=int, default=20)
    parser.add_argument("--target", default="cat")
    parser.add_argument("--prompt-set", choices=("pos", "neg", "off"), default="pos")
    parser.add_argument("--prompts-source", type=Path, help="Upstream src/subliminal/eval_prompts.py; checked using ast, not imported")
    parser.add_argument("--require-metadata", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=1)
    args = parser.parse_args()
    if min(args.expected_prompts, args.expected_samples_per_prompt, args.bootstrap) < 1:
        parser.error("Expected counts and bootstrap repetitions must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory must be new or empty; preserve previous scores")
    texts = canonical_prompts(args.prompts_source, args.prompt_set)
    if texts is not None and len(texts) != args.expected_prompts:
        raise ValueError("Canonical prompt count differs from --expected-prompts")
    scored = []
    for spec in args.run:
        name, separator, path = spec.partition("=")
        if not separator or not name or not path:
            parser.error("--run expects Condition=/path/to/evaluation/directory")
        if name in [item["condition"] for item in scored]:
            parser.error(f"Duplicate condition: {name}")
        result = score_run(name, Path(path), args, texts)
        if result["metadata"].get("prompt_set_name", args.prompt_set) != args.prompt_set:
            raise ValueError(f"{name}: prompt_set_name differs from --prompt-set")
        scored.append(result)
    reference = scored[0]
    comparisons = []
    for result in scored[1:]:
        if result["prompt_sha256"] != reference["prompt_sha256"]:
            raise ValueError("Cannot compare conditions with different prompt text/order")
        mismatch = [key for key in COMPARABLE_KEYS if key in reference["metadata"] and key in result["metadata"]
                    and reference["metadata"][key] != result["metadata"][key]]
        if mismatch:
            raise ValueError(f"Cannot make a controlled comparison: metadata differs for {mismatch}")
        deltas = [b["rate"] - a["rate"] for a, b in zip(reference["per_prompt"], result["per_prompt"], strict=True)]
        comparisons.append({"reference": reference["condition"], "condition": result["condition"],
                            "cat_rate_difference": statistics.fmean(deltas),
                            "paired_prompt_bootstrap_95_ci": bootstrap_interval(deltas, args.bootstrap, args.bootstrap_seed)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scores.json").open("x", encoding="utf-8") as stream:
        json.dump({"conditions": scored, "comparisons": comparisons,
                   "uncertainty_note": "Bootstrap resamples whole prompt indices; intervals describe prompt sensitivity. Samples sharing a prompt are not treated as independent. One seed and 20 samples/prompt are preliminary evidence."}, stream, indent=2)
    with (args.output_dir / "per_prompt.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("condition", "prompt_idx", "prompt", "hits", "total", "rate"))
        writer.writeheader()
        for result in scored:
            for item in result["per_prompt"]:
                writer.writerow({"condition": result["condition"], **{key: item[key] for key in writer.fieldnames if key != "condition"}})
    for result in scored:
        print(f"{result['condition']}: {result['cat_rate']:.4f} ({result['hits']}/{result['samples']}); metadata={result['metadata_status']}")
    print(f"Wrote {args.output_dir.resolve() / 'scores.json'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
