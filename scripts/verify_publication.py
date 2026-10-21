"""Verify the SHA256 inventory of the published core experiment files."""

from pathlib import Path
import hashlib
import json


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "publication_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for record in manifest["files"]:
        target = (root / record["path"]).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            failures.append(record["path"])
            continue
        content = target.read_bytes()
        if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
            failures.append(record["path"])
    if failures:
        raise SystemExit("Publication integrity failed: " + ", ".join(failures))
    print(f"PASS: {len(manifest['files'])} published files match their recorded sizes and SHA256 hashes.")


if __name__ == "__main__":
    main()
