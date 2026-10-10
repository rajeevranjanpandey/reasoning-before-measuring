"""Scientific invariants for the post-review experiment, without model execution."""
import unittest
import numpy as np
from revision_controls import transform, ARMS
from revision_diagnostics import metrics

class RevisionInvariants(unittest.TestCase):
    def test_interventions_preserve_physical_information(self):
        obs={3:1.25,9:-2.5}
        scores={j:j/100 for j in range(41) if j not in obs}
        meta={j:f'physical {j}' for j in range(41)}
        for arm in ARMS:
            m,md,o,l,s=transform(obs,scores,meta,arm,2,17)
            self.assertEqual(set(m.values()),set(range(41)))
            self.assertEqual(o,{m[j]:v for j,v in obs.items()})
            self.assertEqual(set(l),{m[j] for j in scores})
            if arm not in ('no_scores','shuffled_scores'):
                self.assertEqual(s,{m[j]:v for j,v in scores.items()})
            if arm=='shuffled_scores':self.assertEqual(sorted(s.values()),sorted(scores.values()))
            if arm=='anonymous':self.assertEqual(set(md.values()),{'Process measurement'})
            else:self.assertTrue(all(md[m[j]]==meta[j] for j in range(41)))
            self.assertEqual(transform(obs,scores,meta,arm,2,17),(m,md,o,l,s))

    def test_calibration_boundaries(self):
        y=np.arange(21)
        self.assertEqual(metrics(np.eye(21),y),dict(balanced_accuracy=1.0,log_loss=0.0,brier=0.0,ece10=0.0))
        uniform=metrics(np.ones((21,21))/21,y)
        self.assertAlmostEqual(uniform['brier'],20/21)
        self.assertAlmostEqual(uniform['balanced_accuracy'],1/21)
        wrong=metrics(np.roll(np.eye(21),1,axis=1),y)
        self.assertEqual(wrong['brier'],2.0)
        self.assertEqual(wrong['ece10'],1.0)

    def test_crossed_resampling_detects_seed_variation(self):
        rng=np.random.default_rng(42)
        v=np.tile(np.arange(5)/4,(20,1))
        runs=rng.integers(0,20,size=(1000,20));seeds=rng.integers(0,5,size=(1000,5))
        fixed_seed=v[runs].mean((1,2))
        crossed=v[runs[:,:,None],seeds[:,None,:]].mean((1,2))
        self.assertAlmostEqual(float(fixed_seed.std()),0)
        self.assertGreater(float(crossed.std()),0.05)

if __name__=='__main__':unittest.main()
