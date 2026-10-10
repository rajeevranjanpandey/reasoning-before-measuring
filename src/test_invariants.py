import json
import unittest
import numpy as np
import joblib
from experiment import ROOT, load, state_view
from agents import parse_action

class ScientificInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package=joblib.load(ROOT/'results/training.joblib')
        cls.p=cls.package['predictor']
    def test_grouped_splits(self):
        x,y,r=load('Training')
        groups=[set(r[r<=350]),set(r[(r>=351)&(r<=400)]),set(r[(r>=401)&(r<=420)])]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        for mask in [r<=350,(r>=351)&(r<=400),(r>=401)&(r<=420)]:
            self.assertEqual(set(y[mask]),set(range(21)))
        self.assertEqual(x.shape,(10500,41))
    def test_unobserved_values_do_not_change_prediction(self):
        x,_,_=load('Training');x=x[:4].copy();selection=[3,9]
        before=self.p.batch(x,selection)
        hidden=[j for j in range(41) if j not in selection]
        x[:,hidden]=123456789
        np.testing.assert_array_equal(before,self.p.batch(x,selection))
    def test_state_has_no_label_or_run_or_unacquired_values(self):
        state=state_view({2:123.0},[0,1,3],2,self.p,self.package['metadata'])
        self.assertEqual(set(state),{'observed','legal_sensors','remaining_budget','diagnosis_probabilities'})
        self.assertEqual(state['observed'],{'3':123.0})
    def test_invalid_and_duplicate_actions_are_rejected(self):
        for text in ['{"sensor":2}','{"sensor":42}','{"sensor":true}',
                     '{"sensor":1,"label":0}','1','{"sensor":1.0}']:
            self.assertIsNone(parse_action(text,[0,2]))
        self.assertEqual(parse_action('{"sensor":3}',[0,2]),2)
        self.assertEqual(parse_action('```json\n{"sensor":1}\n```',[0,2]),0)
    def test_source_checksums_and_independent_parser(self):
        self.assertEqual(len(json.loads((ROOT/'provenance/data_checksums.json').read_text())),4)
        self.assertTrue(json.loads((ROOT/'provenance/parser_validation.json').read_text())['exact_agreement'])

if __name__=='__main__':unittest.main()
