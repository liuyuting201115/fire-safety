#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "configs/binary.yaml",
    "configs/multiclass.yaml",
    "checkpoints/binary_best.pth",
    "checkpoints/multiclass_best.pth",
    "checkpoints/dinov2_vits14_reg4_pretrain.pth",
    "manifests/checkpoints.sha256",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_assets() -> list[Path]:
    return sorted((ROOT / "checkpoints").glob("*.pth"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify repository assets")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument(
        "--require-private-data",
        action="store_true",
        help="Also require the non-public binary and multiclass test datasets",
    )
    args = parser.parse_args()
    manifest = ROOT / "manifests" / "checkpoints.sha256"
    if args.write_manifest:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
            for path in tracked_assets()
        ]
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {len(lines)} checksums to {manifest.relative_to(ROOT)}")

    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).exists():
            errors.append(f"missing: {relative}")
    if args.require_private_data:
        for relative in ("data/test/binary", "data/test/multiclass"):
            if not (ROOT / relative).is_dir():
                errors.append(f"private test dataset missing: {relative}")

    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            expected, relative = line.split(maxsplit=1)
            path = ROOT / Path(relative.strip())
            if not path.is_file():
                errors.append(f"manifest file missing: {relative}")
            elif sha256(path).lower() != expected.lower():
                errors.append(f"checksum mismatch: {relative}")

    if errors:
        print("Repository verification: FAIL")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("Repository verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
