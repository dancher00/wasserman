import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from current_contact_metrics import contact_events,paired_interval,episode_metrics

class ContactTests(unittest.TestCase):
    def test_never_acquired_is_not_loss(self):
        r=contact_events([0]*20+[1,1]+[0]*10)
        self.assertEqual((r['contact_acquired'],r['contact_lost'],r['contact_loss_count']),(0,0,0))
    def test_confirmation_and_reacquisition(self):
        r=contact_events([1,1,1,0,0,1,0,0,0,1,1,1,0,0,0])
        self.assertEqual(r['contact_loss_count'],2)
        self.assertAlmostEqual(r['first_acquisition_s'],3/30)
        self.assertAlmostEqual(r['first_loss_s'],9/30)
    def test_censored_loss_not_confirmed(self):
        self.assertEqual(contact_events([1,1,1,0,0])['contact_loss_count'],0)
    def test_paired_zero_interval(self):
        x=np.arange(90).reshape(3,30)
        r=paired_interval(x,x,1000)
        self.assertEqual(r['descriptive_paired_crossed_95'],[0.,0.])
    def test_missing_not_zero(self):
        x=np.full((3,30),np.nan);x[0,0]=4
        r=paired_interval(x,np.zeros_like(x),1000)
        self.assertEqual(r['difference'],4)
        self.assertEqual(r['finite_pairs'],1)
        self.assertEqual(episode_metrics({},0,np.zeros(5,bool),[-1,1]),{'exposure_s':0.,'metrics_available':False})
    def test_gap_rejected(self):
        with self.assertRaises(ValueError):episode_metrics({},0,[1,0,1],[-1,1])

if __name__=='__main__':unittest.main()
