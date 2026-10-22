import unittest

import numpy as np

from joint_point_echoes import JointPointEchoes
from test_recovery_experiment import problem


class JointPointEchoTests(unittest.TestCase):
    def test_all_path_and_geometric_profiled_derivatives(self):
        base=problem(); ex=JointPointEchoes(base)
        p=np.r_[np.array([.02,-.03,.04,-.02,.01,.005]),np.tile([1.7,-.15,.05],5)]
        rng=np.random.default_rng(312)
        y=[ex.templates(p,c)@np.array([.8+.3j,.1-.4j])+
            .01*(rng.normal(size=f.y.size)+1j*rng.normal(size=f.y.size)) for c,f in enumerate(ex.fields)]
        at=p+np.linspace(-.002,.002,len(p));_,j,_,_=ex.profile(at,y)
        for i in range(len(p)):
            d=np.eye(len(p))[i]*2e-5
            finite=(ex.profile(at+d,y,derivative=False)[0]-ex.profile(at-d,y,derivative=False)[0])/4e-5
            self.assertLess(np.linalg.norm(finite-j[:,i])/np.linalg.norm(finite),3e-4)

    def test_joint_fit_recovers_point_and_neighbour_from_displaced_seed(self):
        ex=JointPointEchoes(problem())
        truth=np.r_[np.array([.02,-.03,.04,-.02,.01,.005]),np.tile([1.7,-.15,.05],5)]
        y=[ex.templates(truth,c)@np.array([.8+.3j,.1-.4j]) for c in range(5)]
        fit,_=ex.fit(y,truth+np.linspace(-.006,.006,len(truth)),80)
        self.assertLess(np.max(abs(np.asarray(fit['parameters'])-truth)),2e-5)
        self.assertLess(sum(fit['test_rss'])/sum(np.vdot(v,v).real for v in y),1e-10)


if __name__=='__main__': unittest.main()
