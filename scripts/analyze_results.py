"""Create an honest result table and optional figures from offline scores only."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys


CONDITIONS = {
    "Base": "Base",
    "Student": "Student",
    "Sufficiency": "Base + Student Vector",
    "Necessity": "Student + Vector Replacement",
}
FIELDS = ("Condition", "Cat Rate", "Samples", "GPU", "CU Used", "GPU Hours", "Status")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, help="score_cat_rate.py scores.json; omit for a Not Run template")
    parser.add_argument("--resources", type=Path, help="Manually recorded CSV with Condition,GPU,CU Used,GPU Hours")
    parser.add_argument("--student-vector", type=Path, help="Optional official-format vector for alignment metadata; requires torch")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new or empty output directory; previous reports are preserved")
    scores = json.loads(args.scores.read_text(encoding="utf-8")) if args.scores else {"conditions": [], "comparisons": []}
    by_name = {row["condition"]: row for row in scores["conditions"]}
    resources = {}
    if args.resources:
        with args.resources.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                resources[row["Condition"]] = row
    rows = []
    for label, display in CONDITIONS.items():
        result = by_name.get(label)
        resource = resources.get(label, resources.get(display, {}))
        rows.append({"Condition": display, "Cat Rate": f"{result['cat_rate']:.6f}" if result else "Not Run",
                     "Samples": str(result["samples"]) if result else "Not Run",
                     "GPU": resource.get("GPU", "Not Recorded" if result else "Not Run"),
                     "CU Used": resource.get("CU Used", "Not Recorded" if result else "Not Run"),
                     "GPU Hours": resource.get("GPU Hours", "Not Recorded" if result else "Not Run"),
                     "Status": result["metadata_status"] if result else "Not Run"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "experiment_results.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "report_notes.md").open("x", encoding="utf-8") as stream:
        stream.write("# Reproduction notes\n\n")
        stream.write("Results are measured only when present in scores.json. Missing runs remain Not Run. CU and GPU hours come from the manually checked resource ledger, never from a guessed hardware rate.\n\n")
        stream.write("## Paired prompt comparisons\n\n")
        for comparison in scores.get("comparisons", []):
            interval = comparison["paired_prompt_bootstrap_95_ci"]
            stream.write(f"- {comparison['condition']} minus {comparison['reference']}: {comparison['cat_rate_difference']:+.4f}; prompt bootstrap 95% interval [{interval[0]:+.4f}, {interval[1]:+.4f}].\n")
        stream.write("\nThese are preliminary prompt-level results from one training seed. A higher Cat Rate alone does not establish a broad animal preference or rule out direct token steering. Add controls before stronger claims. An interval that includes zero is inconclusive at this sampling scale.\n\n")
        stream.write("## Required manual additions\n\n- Actual GPU model and measured CU balance changes for each connection, including setup/idle/retries.\n- Failure and retry log.\n- Pinned code/model/data revisions and exact training configuration.\n- Explicitly override the upstream script's ten-epoch default with two epochs, matching the main Qwen paper experiment. Record actual deviations: extraction prompt count, evaluation sample count, fallback attention, reduced epochs, or any reduced dataset.\n- What was observed, what remains inconclusive, and the next experiment.\n")
    if args.no_plots:
        print(f"Wrote tables and notes to {args.output_dir.resolve()}; plots explicitly skipped")
        return
    os.environ.setdefault("MPLCONFIGDIR", str((args.output_dir / ".mplconfig").resolve()))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Tables saved. Figures NEEDS VERIFICATION: matplotlib is unavailable; use an existing environment with matplotlib or rerun with --no-plots.")
        return
    available = [(label, display, by_name[label]) for label, display in CONDITIONS.items() if label in by_name]
    if available:
        figure, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
        rates = [item[2]["cat_rate"] for item in available]
        low = [max(0.0, rate - item[2]["prompt_bootstrap_95_ci"][0]) for rate, item in zip(rates, available, strict=True)]
        high = [max(0.0, item[2]["prompt_bootstrap_95_ci"][1] - rate) for rate, item in zip(rates, available, strict=True)]
        axis.bar([item[1].replace(" + ", "\n+ ") for item in available], rates, color=["#718096", "#2b6cb0", "#2f855a", "#9b2c2c"][:len(available)], yerr=[low, high], capsize=4)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Strict first-word Cat Rate")
        synthetic = scores.get("synthetic_only", False) or any(
            item[2].get("metadata", {}).get("model") == "synthetic-model" for item in available)
        axis.set_title("SYNTHETIC SOFTWARE TEST - no experiment results" if synthetic
                       else "Core reproduction: measured conditions only")
        axis.text(0.5, -0.23, "Error bars: 95% bootstrap over complete prompt indices; one training seed.", ha="center", transform=axis.transAxes, fontsize=8)
        figure.savefig(args.output_dir / "cat_rate_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
    if args.student_vector:
        try:
            import torch
        except ImportError:
            print("Alignment figure NEEDS VERIFICATION: torch unavailable; use Colab or an existing torch environment")
        else:
            vector = torch.load(args.student_vector, map_location="cpu", weights_only=True)
            metadata = vector["meta"]
            cosines = metadata["cos_v_teacher_per_layer"]
            if len(cosines) != vector["raw"].shape[0]:
                raise ValueError("Vector cosine metadata length differs from tensor layer count")
            tiled = metadata.get("extract_layer") is not None
            figure, axis = plt.subplots(figsize=(8, 4.6), layout="constrained")
            axis.plot(list(range(len(cosines) - 1)), cosines[1:], marker="o", markersize=3)
            axis.axhline(0, color="gray", linewidth=0.8)
            axis.set_ylim(-1, 1)
            axis.set_xlabel("Transformer block index (embedding slot excluded)")
            axis.set_ylabel("Cosine similarity")
            detail = f"Student L{metadata['extract_layer']} tiled vs teacher per-layer" if tiled else "Student per-layer vs teacher per-layer"
            axis.set_title(detail)
            figure.savefig(args.output_dir / "teacher_student_alignment.png", dpi=180)
            plt.close(figure)
            with (args.output_dir / "alignment_metadata.json").open("x", encoding="utf-8") as stream:
                json.dump({"student_vector": str(args.student_vector.resolve()), "interpretation": detail,
                           "embedding_excluded_from_figure": True, "cos_v_teacher_per_layer_including_embedding": cosines,
                           "meta": metadata}, stream, indent=2)
    print(f"Wrote analysis outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
