"""Fast independent checks for the isolated experiment path."""
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import cvxpy as cp

from synthetic_traffic_validation.revision_code import revision as r
from synthetic_traffic_validation.allocation import exact_matching


class RevisionTests(unittest.TestCase):
    def test_assignment_bruteforce_and_mosek_lp(self):
        w=np.random.default_rng(101).normal(size=(3,8))
        x=exact_matching(w); score=float((w*x).sum())
        brute=max(sum(w[b,i] for b,i in enumerate(c)) for c in itertools.permutations(range(8),3))
        self.assertAlmostEqual(score,brute,places=10)
        z=cp.Variable((3,8),nonneg=True)
        prob=cp.Problem(cp.Maximize(cp.sum(cp.multiply(w,z))),[cp.sum(z,axis=1)==1,cp.sum(z,axis=0)<=1])
        prob.solve(solver='MOSEK',mosek_params={'MSK_IPAR_NUM_THREADS':2})
        self.assertEqual(prob.status,'optimal')
        self.assertAlmostEqual(score,prob.value,places=6)
        np.testing.assert_allclose(x,z.value,atol=1e-6)

    def test_window_boundaries_and_common_physical_splits(self):
        for h in r.HORIZONS:
            for start,end in [(600000,700000),(700000,800000),(800000,1000000)]:
                oo=r.origins(start,end,h)
                self.assertEqual(oo[0],start)
                self.assertLessEqual(oo[-1]+h,end)
                self.assertTrue(np.all(np.diff(oo)==h))

    def test_status_infeasibility_not_inaccurate_or_timeout(self):
        self.assertEqual(r.status_class({'bandwidth':None,'status':'infeasible'}),'proved_infeasible')
        for status in ['infeasible_inaccurate','user_limit','solver_error']:
            self.assertEqual(r.status_class({'bandwidth':None,'status':status}),'unresolved_without_feasible_incumbent')
        self.assertEqual(r.status_class({'bandwidth':np.ones((8,12)),'status':'optimal_inaccurate'}),'feasible_incumbent_unproved_optimal')

    def test_channel_same_first_block_and_independent_error_dimensions(self):
        for h in r.HORIZONS:
            mu,sd=r.channel(h,800084)
            old,oldsd=r.report_channel(cycle=7)
            np.testing.assert_allclose(mu[:,:12],old)
            np.testing.assert_allclose(sd,.2*mu)
            self.assertEqual(mu.shape,(8,h))

    def test_uniform_scale_invariance(self):
        rng=np.random.default_rng(28); y=rng.uniform(.1,.4,(8,100))
        models=[dict(history=10,fit={'mean':.2},weights=np.arange(10)/30) for _ in range(8)]
        a=r.forecast_one(y,50,12,models)
        b=r.forecast_one(y,50,12,models,1.7)
        np.testing.assert_allclose(b,a*1.7)

    def test_failed_repairs_are_recorded_before_fallback(self):
        mu=np.ones((8,12))*4; sd=mu*.2; target=np.ones(8)*2
        # One deficient cell permits repair, then both repair attempts fail.
        margins=np.r_[-1.,np.ones(7)]
        fail=dict(bandwidth=None,status='user_limit',solver_s=.01)
        attempts=[]
        with patch.object(r,'service_margin',return_value=margins),patch.object(r,'solve_allocation',return_value=fail):
            _,path,_=r.hybrid_plan(mu,sd,target,np.ones(8)*.1,np.ones(8),1.,attempts)
        self.assertEqual(path,'unprotected_fallback')
        self.assertEqual([a['mode'] for a in attempts],['bandwidth','local','full'])
        self.assertTrue(all(a['classification']=='unresolved_without_feasible_incumbent' for a in attempts))

    def test_direct_matching_is_not_old_plan_reuse(self):
        mu=np.ones((8,12))*4; attempts=[]
        with patch.object(r,'service_margin',return_value=np.ones(8)):
            _,path,_=r.hybrid_plan(mu,.2*mu,np.ones(8),np.ones(8)*.1,np.ones(8),1,attempts)
        self.assertEqual(path,'matching_direct'); self.assertEqual(attempts,[])


if __name__=='__main__': unittest.main(verbosity=2)
