"""Download pinned public baseline artifacts, validate JSONL, or make a subset."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request


DATA_REPO = "agu18dec/steering_vector_distillation"
DATA_REVISION = "4fda20d0413040b2de61448c89182716485d9839"
VECTOR_REPO = "camilablank/v_teacher_v_student_vectors"
VECTOR_REVISION = "d37b6e4b72cef115f2dba4e06dc16635f9f51e78"
BASE_FIELDS = {"system_prompt", "prompt", "completion"}
TRAIN_FIELDS = BASE_FIELDS | {"judge_verdict", "judge_reasoning"}
FILES = (
    ("dataset", DATA_REPO, "datasets/baseline/cat_qwen25_7b/raw.jsonl",
     "generated/cat_qwen25_7b/raw.jsonl", 30000),
    ("dataset", DATA_REPO, "datasets/baseline/cat_qwen25_7b/filtered/filtered_10000.jsonl",
     "filtered/cat_qwen25_7b/filtered_10000.jsonl", 10000),
    ("model", VECTOR_REPO, "section-3_numbers_vectors/v_teacher_all_prompt_tokens.pt",
     "vectors/v_teacher_qwen25_cat.pt", None),
)
AUDITED_SHA256 = {
    "generated/cat_qwen25_7b/raw.jsonl": "1320c88f73b0cbe9252c4a11f01bb3842654ceca3e9192f807bb5a437bd97f1a",
    "filtered/cat_qwen25_7b/filtered_10000.jsonl": "c7a1038f3653a8dffedbe1ad6e7bc4ebf0ef9ebe0a0463fb79ef2a95becd1c95",
    "vectors/v_teacher_qwen25_cat.pt": "40bd24600953bb854525b0fda61f4e7a74332e24045df0ce9d3e3ab5f47e10f5",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def validate_jsonl(path: Path, *, filtered: bool, expected_rows: int | None = None) -> dict:
    required = TRAIN_FIELDS if filtered else BASE_FIELDS
    field_types: Counter = Counter()
    rows = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line: {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != required:
                raise ValueError(f"Unexpected fields: {path}:{line_number}; expected {sorted(required)}")
            for key in required:
                if not isinstance(row[key], str):
                    raise TypeError(f"Expected a string: {path}:{line_number}:{key}")
                field_types[f"{key}:str"] += 1
            if not row["prompt"].strip() or not row["completion"].strip():
                raise ValueError(f"Empty prompt/completion: {path}:{line_number}")
            rows += 1
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {path}, found {rows}")
    return {"rows": rows, "field_types": dict(sorted(field_types.items())),
            "sha256": sha256(path), "bytes": path.stat().st_size}


def validate_directory(output: Path) -> dict:
    """Check the audited manifest and all three artifacts without network access."""
    output = output.resolve()
    manifest = json.loads((output / "data_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_revision") != DATA_REVISION or manifest.get("vector_revision") != VECTOR_REVISION:
        raise ValueError("Directory manifest does not use the audited dataset/vector revisions")
    entries = manifest.get("files", [])
    expected_paths = {item[3] for item in FILES}
    if len(entries) != len(FILES) or {entry.get("local_path") for entry in entries} != expected_paths:
        raise ValueError("Directory manifest must identify exactly the three audited artifacts")
    by_path = {entry["local_path"]: entry for entry in entries}
    checked = []
    for kind, repo, remote, local, rows in FILES:
        entry = by_path[local]
        revision = DATA_REVISION if kind == "dataset" else VECTOR_REVISION
        expected_identity = {"repo_id": repo, "repo_type": kind, "revision": revision, "remote_path": remote}
        if any(entry.get(key) != value for key, value in expected_identity.items()):
            raise ValueError(f"Manifest source identity mismatch: {local}")
        target = output / local
        digest = sha256(target)
        if digest != AUDITED_SHA256[local] or digest != entry.get("sha256"):
            raise ValueError(f"Audited SHA256 mismatch: {target}")
        if target.stat().st_size != entry.get("bytes"):
            raise ValueError(f"Manifest byte count mismatch: {target}")
        result = {"local_path": local, "sha256": digest, "bytes": target.stat().st_size}
        if rows is not None:
            result.update(validate_jsonl(target, filtered=kind == "dataset" and "filtered" in local, expected_rows=rows))
        else:
            result["tensor_runtime_validation"] = "NEEDS VERIFICATION with torch on CPU or Colab"
        checked.append(result)
    return {"status": "PASS", "offline": True, "output_dir": str(output), "files": checked}


def validate_command(args: argparse.Namespace) -> None:
    if args.output_dir is not None:
        if args.kind is not None or args.expected_rows is not None:
            raise ValueError("--kind and --expected-rows apply only to --input")
        result = validate_directory(args.output_dir)
    else:
        if args.kind is None:
            raise ValueError("--kind raw|filtered is required with --input")
        result = validate_jsonl(args.input, filtered=args.kind == "filtered", expected_rows=args.expected_rows)
    print(json.dumps(result, indent=2))


def get_bytes(url: str, transport: str) -> bytes:
    if transport == "curl":
        executable = shutil.which("curl.exe") or shutil.which("curl")
        if not executable:
            raise RuntimeError("curl was not found; use --transport urllib")
        args = [executable, "-4", "--fail", "--location", "--retry", "2", "--max-time", "180", "--silent", "--show-error"]
        if os.name == "nt":
            # Keep CA/hostname verification; tolerate unavailable revocation servers.
            args.append("--ssl-revoke-best-effort")
        return subprocess.check_output(args + [url])
    request = urllib.request.Request(url, headers={"User-Agent": "one-day-repro/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def download(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    manifest_path = output / "data_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("dataset_revision") != args.dataset_revision or existing.get("vector_revision") != args.vector_revision:
            raise FileExistsError("Manifest revisions differ; use a new output directory")
        for entry in existing["files"]:
            path = output / entry["local_path"]
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Existing artifact checksum mismatch: {path}")
            if path.suffix == ".jsonl":
                validate_jsonl(path, filtered="filtered" in entry["local_path"], expected_rows=entry["rows"])
        print(f"Reused and revalidated pinned artifacts: {manifest_path}")
        return
    repositories = {}
    for kind, repo, revision in (("dataset", DATA_REPO, args.dataset_revision), ("model", VECTOR_REPO, args.vector_revision)):
        api_kind = "datasets" if kind == "dataset" else "models"
        url = f"https://huggingface.co/api/{api_kind}/{repo}/revision/{revision}?blobs=true"
        metadata = json.loads(get_bytes(url, args.transport))
        if metadata.get("sha") != revision:
            raise ValueError(f"Repository revision mismatch for {repo}")
        repositories[repo] = {"revision": revision, "files": {x["rfilename"]: x for x in metadata["siblings"]}}
    records = []
    for kind, repo, remote, local, rows in FILES:
        info = repositories[repo]["files"].get(remote)
        if info is None:
            raise FileNotFoundError(f"Remote file not found in pinned repository: {remote}")
        revision = repositories[repo]["revision"]
        prefix = "datasets/" if kind == "dataset" else ""
        url = f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/{urllib.parse.quote(remote, safe='/')}"
        target = output / local
        if target.exists():
            payload = target.read_bytes()
        else:
            print(f"Downloading {repo}@{revision}: {remote}")
            payload = get_bytes(url, args.transport)
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != info["size"]:
            raise ValueError(f"Remote size mismatch for {remote}")
        if "lfs" in info:
            if digest != info["lfs"]["sha256"]:
                raise ValueError(f"LFS SHA256 mismatch for {remote}")
        else:
            blob_hash = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            if blob_hash != info["blobId"]:
                raise ValueError(f"Git blob hash mismatch for {remote}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(payload)
        record = {"repo_id": repo, "repo_type": kind, "revision": revision, "remote_path": remote,
                  "local_path": local, "bytes": len(payload), "sha256": digest,
                  "remote_blob_id": info["blobId"], "source_url": url}
        if rows is not None:
            record.update(validate_jsonl(target, filtered="filtered" in local, expected_rows=rows))
        else:
            record["tensor_runtime_validation"] = "NEEDS VERIFICATION with torch on CPU or Colab"
        records.append(record)
        print(f"Verified {local}: {record.get('rows', 'vector')} rows; SHA256 {digest}")
    write_new_json(manifest_path, {"created_utc": datetime.now(timezone.utc).isoformat(),
                                 "dataset_revision": args.dataset_revision, "vector_revision": args.vector_revision,
                                 "files": records})
    print(f"Wrote {manifest_path}")


def subset(args: argparse.Namespace) -> None:
    source, target = args.input.resolve(), args.output.resolve()
    if target == source or target.exists() or target.with_suffix(target.suffix + ".manifest.json").exists():
        raise FileExistsError("Subset outputs must be new files, different from the source")
    checked = validate_jsonl(source, filtered=True)
    if not 1 <= args.rows <= checked["rows"]:
        raise ValueError("Subset rows must be between 1 and the source row count")
    lines = source.read_text(encoding="utf-8").splitlines()
    indices = list(range(len(lines)))
    random.Random(args.seed).shuffle(indices)
    selected = indices[:args.rows]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("".join(lines[index] + "\n" for index in selected))
    result = validate_jsonl(target, filtered=True, expected_rows=args.rows)
    write_new_json(target.with_suffix(target.suffix + ".manifest.json"),
                   {"source": str(source), "source_sha256": checked["sha256"], "seed": args.seed,
                    "selection_method": "Python random.Random(seed).shuffle zero-based indices; first N",
                    "selected_indices": selected, "output": str(target), **result})
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download", help="Download only the three pinned baseline artifacts")
    fetch.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    fetch.add_argument("--dataset-revision", default=DATA_REVISION)
    fetch.add_argument("--vector-revision", default=VECTOR_REVISION)
    fetch.add_argument("--transport", choices=("urllib", "curl"), default="urllib")
    fetch.set_defaults(func=download)
    check = sub.add_parser("validate", help="Validate one JSONL file or the audited artifact directory offline")
    check_source = check.add_mutually_exclusive_group(required=True)
    check_source.add_argument("--input", type=Path)
    check_source.add_argument("--output-dir", type=Path)
    check.add_argument("--kind", choices=("raw", "filtered"))
    check.add_argument("--expected-rows", type=int)
    check.set_defaults(func=validate_command)
    small = sub.add_parser("subset", help="Write a new deterministic subset; never modify the source")
    small.add_argument("--input", type=Path, required=True)
    small.add_argument("--output", type=Path, required=True)
    small.add_argument("--rows", type=int, required=True)
    small.add_argument("--seed", type=int, default=1)
    small.set_defaults(func=subset)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
