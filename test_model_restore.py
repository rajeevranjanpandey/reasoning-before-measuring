"""Checks wrong-hash downloads cannot become model files, and cached weights stay usable."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile

candidate=Path(__file__).with_name('run_with_models.py')
if not candidate.exists():
    raise AssertionError('Missing startup model-restoration launcher')
spec=importlib.util.spec_from_file_location('restore',candidate)
restore=importlib.util.module_from_spec(spec);spec.loader.exec_module(restore)

with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);origin=root/'fixture.bin';origin.write_bytes(b'abc')
    target=root/'models'/'fixture.gguf'
    checksum='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    restore.ensure_file(target,origin.as_uri(),checksum,3)
    assert target.read_bytes()==b'abc'
    origin.unlink()
    restore.ensure_file(target,origin.as_uri(),checksum,3)
    assert target.read_bytes()==b'abc', 'Cached valid file should not require network'
    origin.write_bytes(b'bad');wrong=root/'models'/'wrong.gguf'
    try:
        restore.ensure_file(wrong,origin.as_uri(),checksum,3)
    except ValueError:pass
    else:raise AssertionError('Incorrect checksum must fail')
    assert not wrong.exists(), 'Failed download must not become a model file'
    assert not list((root/'models').glob('*.part')), 'Temporary download should be cleaned'
    target.write_bytes(b'bad')
    try:
        restore.ensure_file(target,origin.as_uri(),checksum,3)
    except ValueError:pass
    else:raise AssertionError('Corrupt cached model must fail')
    assert target.read_bytes()==b'bad', 'Do not silently overwrite an existing corrupt file'
print('PASS: download, cached reuse, checksum rejection, partial cleanup and existing-file protection')
