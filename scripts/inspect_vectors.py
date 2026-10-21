"""Check vector structure and plot the tiled-student alignment diagnostic."""

import argparse
import hashlib
import json
import os
from pathlib import Path


TEACHER_SHA256 = "40bd24600953bb854525b0fda61f4e7a74332e24045df0ce9d3e3ab5f47e10f5"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path)
    parser.add_argument("--student-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    teacher_hash = sha256_file(args.teacher)
    if teacher_hash != TEACHER_SHA256:
        raise ValueError("Teacher file differs from the audited public artifact; verify its source and metadata before use")
    import torch
    import torch.nn.functional as functional
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {}
    vectors = {}
    for label, path in (("teacher", args.teacher), ("student", args.student)):
        if path is None:
            continue
        vector = torch.load(path, map_location="cpu", weights_only=True)
        raw = vector["raw"]
        if raw.shape != (29, 3584) or vector["unit"].shape != raw.shape or vector["norm"].shape != (29,):
            raise ValueError(f"Unexpected Qwen2.5-7B vector shape in {path}")
        if not all(torch.isfinite(vector[key]).all() for key in ("raw", "unit", "norm")):
            raise ValueError("Vector contains non-finite values")
        if not torch.allclose(raw.norm(dim=-1), vector["norm"], atol=1e-4, rtol=1e-4):
            raise ValueError("Saved norm disagrees with the raw vector")
        expected_unit = raw / vector["norm"].unsqueeze(-1).clamp(min=1e-12)
        if not torch.allclose(vector["unit"], expected_unit, atol=1e-4, rtol=1e-4):
            raise ValueError("Saved unit vector disagrees with raw/norm")
        meta = vector.get("meta", {})
        model = meta.get("base_model", meta.get("model"))
        expected_model = args.student_model if label == "student" else "Qwen/Qwen2.5-7B-Instruct"
        if model != expected_model:
            raise ValueError(f"Model provenance requires verification: {model}")
        if label == "teacher":
            if (meta.get("kind") != "v_teacher" or meta.get("extract_positions") != "all_input"
                    or meta.get("prompt_source") != "numbers" or meta.get("n_prompts") != 2000):
                raise ValueError("Teacher metadata differs from the audited all-input numeric-prompt artifact")
            if "trait" in meta and meta["trait"] != "cat":
                raise ValueError("Expected a cat Teacher vector")
        elif meta.get("kind") != "v_student" or meta.get("finetune_mode") != "lora":
            raise ValueError("Expected a Student LoRA vector")
        vectors[label] = vector
        report[label] = {"path": str(path), "shape": list(raw.shape),
                         "sha256": teacher_hash if label == "teacher" else sha256_file(path),
                         "dtype": str(raw.dtype), "meta": meta}
    if "student" in vectors:
        student = vectors["student"]
        if student["meta"].get("extract_layer") != 10 or student["meta"].get("position") != "all":
            raise ValueError("Expected canonical student L10/all extraction")
        if not torch.allclose(student["raw"], student["raw"][11].expand_as(student["raw"])):
            raise ValueError("Expected a student vector tiled from block L10 / tensor slot 11")
        cos = functional.cosine_similarity(student["raw"].float(), vectors["teacher"]["raw"].float(), dim=-1)
        report["cosine_by_tensor_slot"] = cos.tolist()
        report["cosine_at_L10_slot11"] = cos[11].item()
        report["interpretation"] = "One tiled student L10 direction versus each teacher slot; not independent student layer directions."
        os.environ.setdefault("MPLCONFIGDIR", str((args.output_dir / ".mplconfig").resolve()))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(range(28), cos[1:].numpy(), marker=".")
        axis.axvline(10, color="gray", linestyle="--")
        axis.set(xlabel="Teacher block index (embedding excluded)", ylabel="Cosine similarity",
                 title="Tiled student L10 direction vs teacher block directions", ylim=(-1, 1))
        figure.tight_layout()
        figure.savefig(args.output_dir / "vector_alignment.png", dpi=180)
        plt.close(figure)
    (args.output_dir / "vector_checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
