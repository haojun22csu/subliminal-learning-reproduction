"""Apply a guarded, reversible operational patch to the inspected upstream trainer.

The patch adds step limits, offline logging, and resumable checkpoints. It does
not change the loss, data split, packing, precision, LoRA, or optimizer defaults.
"""

import argparse
import ast
import hashlib
from pathlib import Path


MARKER = "# One-day reproduction operational controls."
REPLACEMENTS = (
    (
        "        self.val_split = 0.05\n",
        "        self.val_split = 0.05\n\n"
        "        # One-day reproduction operational controls.\n"
        "        self.max_steps = -1\n"
        '        self.report_to = "none"\n'
        '        self.resume_from_checkpoint = ""\n'
        "        self.save_only_model = False\n"
        "        self.save_total_limit = 2\n",
    ),
    (
        "        num_train_epochs=config.num_train_epochs,\n",
        "        num_train_epochs=config.num_train_epochs,\n"
        "        max_steps=config.max_steps,\n",
    ),
    (
        "        save_total_limit=20,\n        save_only_model=True,\n",
        "        save_total_limit=config.save_total_limit,\n"
        "        save_only_model=config.save_only_model,\n",
    ),
    ('        report_to="wandb",\n', "        report_to=config.report_to,\n"),
    (
        "    trainer.train()\n",
        "    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint or None)\n",
    ),
    (
        "    train(config, data_file=str(data_file), output_dir=str(out_dir))\n",
        "    trainer = train(config, data_file=str(data_file), output_dir=str(out_dir))\n",
    ),
    (
        '            "val_split": config.val_split,\n',
        '            "val_split": config.val_split,\n'
        '            "max_steps": config.max_steps,\n'
        '            "actual_optimizer_steps": trainer.state.global_step,\n'
        '            "report_to": config.report_to,\n'
        '            "resume_from_checkpoint": config.resume_from_checkpoint,\n'
        '            "save_only_model": config.save_only_model,\n'
        '            "save_total_limit": config.save_total_limit,\n',
    ),
    (
        '        "base_model": config.model,\n',
        '        "base_model": config.model,\n'
        '        "train_code_sha256": _sha256(Path(__file__)),\n'
        '        "attn_implementation": config.attn_implementation,\n',
    ),
)


def patched_source(source):
    """Return a patch or reject any changed or partially patched source."""
    source = source.replace("\r\n", "\n")
    if MARKER in source:
        for old, new in REPLACEMENTS:
            if source.count(new) != 1 or (old != new and old in source and old not in new):
                raise ValueError("Partial or altered operational patch; inspect the file manually.")
        ast.parse(source)
        return source, False
    result = source
    for old, new in REPLACEMENTS:
        if result.count(old) != 1:
            raise ValueError(f"Expected exactly one upstream anchor: {old.strip()!r}")
        result = result.replace(old, new, 1)
    ast.parse(result)
    return result, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Write only after all exact guards pass.")
    args = parser.parse_args()
    target = args.repo.resolve() / "src" / "subliminal" / "train.py"
    raw = target.read_bytes()
    source = raw.decode("utf-8")
    result, changed = patched_source(source)
    print(f"Target: {target}")
    print(f"Original SHA256: {hashlib.sha256(raw).hexdigest()}")
    print("Guard and Python syntax checks: PASS")
    if not changed:
        print("Already patched; no file changed.")
        return
    if not args.apply:
        print("Dry check only; pass --apply to write the operational patch.")
        return
    backup = target.with_name(target.name + ".one_day_repro.bak")
    if backup.exists():
        if backup.read_bytes() != raw:
            raise SystemExit(f"Refusing to replace an existing different backup: {backup}")
    else:
        with backup.open("xb") as handle:
            handle.write(raw)
    newline = "\r\n" if b"\r\n" in raw else "\n"
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(result.replace("\n", newline))
    print(f"Patched: {target}")
    print(f"Byte-for-byte upstream backup: {backup}")


if __name__ == "__main__":
    main()
