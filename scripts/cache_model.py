"""Download one pinned model snapshot and record its public revision."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", help="Exact commit; otherwise resolve main once and record it")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    from huggingface_hub import HfApi, snapshot_download
    revision = HfApi().model_info(args.model, revision=args.revision).sha
    snapshot = snapshot_download(args.model, revision=revision, allow_patterns=[
        "config.json", "generation_config.json", "tokenizer*", "vocab.json", "merges.txt",
        "*.safetensors", "*.safetensors.index.json", "special_tokens_map.json",
    ])
    config_path = Path(snapshot) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    manifest = {"model": args.model, "revision": revision, "snapshot": snapshot,
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
    generation = Path(snapshot) / "generation_config.json"
    if generation.is_file():
        manifest["generation_config"] = json.loads(generation.read_text(encoding="utf-8"))
        manifest["generation_config_sha256"] = hashlib.sha256(generation.read_bytes()).hexdigest()
    tokenizer = Path(snapshot) / "tokenizer_config.json"
    if tokenizer.is_file():
        manifest["tokenizer_config_sha256"] = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
