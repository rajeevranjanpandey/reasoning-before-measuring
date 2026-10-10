"""Submission integrity regression checks, with synthetic files only in temporary folders."""
import json
from pathlib import Path
import tempfile
import unittest
from submission_gate import scientific_inputs, record_keys, author_blockers
from build_postreview_results import build, interval

class GateTests(unittest.TestCase):
    def test_partial_records_never_expose_effects_or_build(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'results/confirmation').mkdir(parents=True)
            (root/'results/confirmation/qwen.jsonl').write_text('intentionally not valid json')
            self.assertTrue(scientific_inputs(root))
            with self.assertRaisesRegex(RuntimeError,'incomplete studies'):build(root)
            self.assertFalse((root/'paper/postreview').exists())
    def test_duplicate_identity_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'records.jsonl';p.write_text('{"run":1,"class":0}\n{"run":1,"class":0}\n')
            with self.assertRaisesRegex(ValueError,'Duplicate identity'):record_keys(p,('run','class'))
    def test_identity_uses_seed_and_class(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'records.jsonl';p.write_text('\n'.join(json.dumps(dict(run=1,seed=s,label=c)) for s,c in [(17,0),(29,0),(17,1)]))
            self.assertEqual(len(record_keys(p,('run','seed','label'))),3)
    def test_missing_and_unapproved_authors_block(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.assertTrue(author_blockers(root))
            (root/'submission').mkdir();(root/'submission/author_metadata.json').write_text('{"authors":[]}')
            self.assertTrue(author_blockers(root))
    def test_percentage_points_not_percent_change(self):
        self.assertEqual(interval([-.1,.02]),'[-10.00, 2.00]')

if __name__=='__main__':unittest.main()
