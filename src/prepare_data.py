"""Memory-bounded reader for this pinned RDX2 release, not a general R reader.

Numeric vectors are decoded one at a time; format deviations fail closed.
Validate independently against pyreadr on the smaller normal training file.
"""
import gzip
import hashlib
import json
import re
import struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def read_snapshot(path, sample=320):
    with gzip.open(path, 'rb') as f:
        def integer():
            return struct.unpack('>i', f.read(4))[0]
        assert f.read(7) == b'RDX2\nX\n'
        assert integer() == 2
        integer(); integer()
        assert integer() == 1026  # tagged pairlist
        assert integer() == 1     # symbol
        assert integer() & 255 == 9
        object_name = f.read(integer()).decode()
        assert integer() == 787   # vector with attributes
        ncols = integer()
        assert ncols == 55
        columns = []
        keep = None
        nrows = None
        for j in range(ncols):
            kind = integer()
            assert kind in (13, 14), (j, kind)
            n = integer()
            if nrows is None:
                nrows = n
            assert n == nrows
            width, dtype = (8, '>f8') if kind == 14 else (4, '>i4')
            a = np.frombuffer(f.read(n * width), dtype=dtype)
            if j < 2:
                columns.append(a.copy())
            elif j == 2:
                keep = np.flatnonzero(a == sample)
                columns = [x[keep] for x in columns]
                columns.append(a[keep])
            else:
                columns.append(a[keep])
        tail = f.read()
        names = re.findall(rb'(?:faultNumber|simulationRun|sample|xmeas_\d+|xmv_\d+)', tail)
        names = [x.decode() for x in names]
        assert len(names) == 55 and len(set(names)) == 55, names
        a = np.column_stack(columns)
        assert np.isfinite(a).all()
        assert len(set(map(tuple, a[:, :2]))) == len(a)
        assert names[:3] == ['faultNumber', 'simulationRun', 'sample']
        assert names[3:44] == [f'xmeas_{i}' for i in range(1, 42)]
        return a, names, object_name, nrows

def main():
    metadata = json.loads((ROOT/'provenance/tep_metadata.json').read_text())
    provenance = []
    for entry in metadata['data']['latestVersion']['files']:
        info = entry['dataFile']
        p = ROOT/'data'/info['filename']
        if not p.exists() or p.stat().st_size != info['filesize']:
            raise RuntimeError(f'Incomplete download: {p}')
        h = hashlib.md5()
        sha = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024*1024), b''):
                h.update(block); sha.update(block)
        assert h.hexdigest() == info['md5'], p
        a, names, obj, nrows = read_snapshot(p)
        np.savez_compressed(ROOT/'data'/f'{p.stem}_snapshot.npz', values=a, names=names)
        provenance.append(dict(file=p.name, md5=h.hexdigest(), sha256=sha.hexdigest(),
                               rows=nrows, snapshots=len(a), sample=320, object=obj))
        print(provenance[-1], flush=True)
    (ROOT/'provenance/data_checksums.json').write_text(json.dumps(provenance, indent=2))
    # Independent parser agreement for a complete small source file.
    import pyreadr
    p = ROOT/'data/TEP_FaultFree_Training.RData'
    df = next(iter(pyreadr.read_r(str(p)).values()))
    expected = df.loc[df['sample'] == 320].to_numpy()
    actual, names, _, _ = read_snapshot(p)
    assert list(df.columns) == names
    np.testing.assert_array_equal(actual, expected)
    (ROOT/'provenance/parser_validation.json').write_text(json.dumps({
        'independent_parser': 'pyreadr', 'version': pyreadr.__version__,
        'file': p.name, 'exact_agreement': True, 'snapshot_cells': actual.size}))

if __name__ == '__main__':
    main()
