"""Remove only the audited byte-identical Ruff duplicate from upstream uv.lock."""

import argparse
from pathlib import Path
import tomllib


def repaired_text(text):
    text = text.replace("\r\n", "\n")
    parts = text.split("[[package]]")
    ruff = [part for part in parts[1:] if part.startswith('\nname = "ruff"\n')]
    if len(ruff) == 1:
        return text, False
    if len(ruff) != 2 or ruff[0] != ruff[1]:
        raise ValueError("Expected exactly two identical Ruff blocks; inspect changed upstream lock manually")
    package = tomllib.loads("[[package]]" + ruff[0])["package"][0]
    if package["version"] != "0.15.15" or package["source"] != {"registry": "https://pypi.org/simple"}:
        raise ValueError("Ruff identity differs from the audited duplicate")
    result = text.replace("[[package]]" + ruff[1], "", 1)
    before = tomllib.loads(text)["package"]
    after = tomllib.loads(result)["package"]
    expected = before.copy()
    expected.remove(package)
    if after != expected:
        raise ValueError("Repair would change more than one duplicate package record")
    return result, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    path = args.repo / "uv.lock"
    raw = path.read_bytes()
    result, changed = repaired_text(raw.decode("utf-8"))
    if not changed:
        print("Lock repair OK: one Ruff record; no changes needed")
        return
    if not args.apply:
        print("Confirmed identical Ruff duplicate; pass --apply to repair it")
        return
    backup = path.with_name("uv.lock.one_day_repro.bak")
    if backup.exists():
        if backup.read_bytes() != raw:
            raise FileExistsError("Existing lock backup differs; preserve it and inspect manually")
    else:
        with backup.open("xb") as stream:
            stream.write(raw)
    if b"\r\n" in raw:
        result = result.replace("\n", "\r\n")
    path.write_bytes(result.encode("utf-8"))
    print("Lock repair OK: removed one identical Ruff record; all dependency versions preserved")
    print(f"Original backup: {backup}")


if __name__ == "__main__":
    main()
