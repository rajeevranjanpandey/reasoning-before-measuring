"""Reconstruct the three original research ZIPs without changing their bytes."""
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def assemble(root):
    expected = json.loads((root / "RELEASE_SHA256.json").read_text())
    names = [f"Review_Reproducibility_Part_{number:02d}.zip" for number in range(1, 4)]
    if set(expected) != set(names):
        raise ValueError("The release manifest must name the three research archives")
    output = root / "review_archives"
    output.mkdir(exist_ok=True)
    completed = []
    for name in names:
        target = output / name
        if target.exists():
            if digest(target) != expected[name]:
                raise ValueError(f"Existing archive has the wrong hash; preserved: {target}")
        else:
            parts = sorted((root / "review_evidence").glob(name + ".part*"))
            if not parts or [p.name for p in parts] != [f"{name}.part{i:02d}" for i in range(1, len(parts) + 1)]:
                raise ValueError(f"Missing or out-of-order parts for {name}")
            temporary = target.with_suffix(".zip.partial")
            if temporary.exists():
                raise ValueError(f"Existing partial archive is preserved: {temporary}")
            try:
                with temporary.open("xb") as destination:
                    for part in parts:
                        with part.open("rb") as source:
                            for block in iter(lambda: source.read(1024 * 1024), b""):
                                destination.write(block)
                if digest(temporary) != expected[name]:
                    raise ValueError(f"Archive checksum mismatch: {name}")
                temporary.rename(target)
            finally:
                temporary.unlink(missing_ok=True)
        completed.append(target)
    return completed


if __name__ == "__main__":
    try:
        for archive in assemble(Path(__file__).resolve().parent):
            print(f"Verified: {archive}")
        print("Extract all three independent ZIPs into the same empty directory.")
    except (OSError, ValueError) as error:
        print(f"Cannot reconstruct the research package: {error}", file=sys.stderr)
        sys.exit(1)
