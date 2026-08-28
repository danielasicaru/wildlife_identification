"""Hashes the real sample data files actually used by the pipeline and writes a committed
manifest (reports/data_manifest.json), so a given experiment's exact data inputs can be verified
later even though data/ itself is gitignored and not committed."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST_PATH = ROOT / "reports" / "data_manifest.json"

TRACKED_PATHS = [
    DATA_DIR / "raw" / "caltech_images_20210113.json",
    DATA_DIR / "raw" / "caltech_bboxes_20200316.json",
    DATA_DIR / "raw" / "images",
    DATA_DIR / "localization" / "detections.json",
    DATA_DIR / "localization" / "crops",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = {"files": {}, "directories": {}}

    for path in TRACKED_PATHS:
        if not path.exists():
            continue
        if path.is_file():
            manifest["files"][path.relative_to(ROOT).as_posix()] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        elif path.is_dir():
            files = sorted(path.glob("*"))
            manifest["directories"][path.relative_to(ROOT).as_posix()] = {
                "file_count": len(files),
                "combined_sha256": hashlib.sha256(
                    b"".join(sha256_file(f).encode() for f in files)
                ).hexdigest(),
            }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {MANIFEST_PATH}")
    for name, info in manifest["files"].items():
        print(f"  file  {name}: {info['sha256'][:12]}... ({info['size_bytes']} bytes)")
    for name, info in manifest["directories"].items():
        print(f"  dir   {name}: {info['file_count']} files, {info['combined_sha256'][:12]}...")


if __name__ == "__main__":
    main()
