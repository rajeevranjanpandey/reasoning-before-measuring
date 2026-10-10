"""Small-file check of archive reconstruction and corruption handling."""
import hashlib
import json
from pathlib import Path
import tempfile

from assemble_review_archives import assemble


def check():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        parts = root / "review_evidence"
        parts.mkdir()
        expected = {}
        for number in range(1, 4):
            name = f"Review_Reproducibility_Part_{number:02d}.zip"
            content = (b"recorded research evidence\x00" * number) + bytes([number])
            expected[name] = hashlib.sha256(content).hexdigest()
            for index, data in enumerate([content[:9], content[9:]], 1):
                (parts / f"{name}.part{index:02d}").write_bytes(data)
        (root / "RELEASE_SHA256.json").write_text(json.dumps(expected))
        outputs = assemble(root)
        assert len(outputs) == 3
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == expected[p.name] for p in outputs)
        assert assemble(root) == outputs
        outputs[0].unlink()
        broken = parts / "Review_Reproducibility_Part_01.zip.part02"
        original = broken.read_bytes()
        broken.write_bytes(b"corrupted")
        try:
            assemble(root)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt parts must fail verification")
        assert not outputs[0].exists()
        assert not list((root / "review_archives").glob("*.partial"))
        broken.write_bytes(original)
        assemble(root)
        outputs[0].write_bytes(b"existing invalid archive")
        try:
            assemble(root)
        except ValueError:
            pass
        else:
            raise AssertionError("Existing invalid archive must be preserved")
        assert outputs[0].read_bytes() == b"existing invalid archive"
    print("PASS: reconstruction, cache reuse, corruption rejection and preservation")


if __name__ == "__main__":
    check()
