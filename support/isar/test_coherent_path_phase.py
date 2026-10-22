import unittest
import numpy as np
from scipy.optimize import minimize
from coherent_path_phase import PhaseProbe


class CoherentPathPhaseTests(unittest.TestCase):
    def test_original_operator_gradient_and_shared_path_recovery(self):
        rng=np.random.default_rng(738);t=np.linspace(-1,1,191);b=np.c_[t,t*t,t*t*t]
        truth=np.array([.11,-.08,.025]);models=[];data=[];roots=[];groups=[];masks=[];ks=np.linspace(3.7,4.8,5)
        for k in ks:
            root=rng.uniform(.2,2,len(t));group=(t>0).astype(int);mask=np.arange(len(t))%3!=2
            model=rng.normal(size=(len(t),7))+1j*rng.normal(size=(len(t),7))
            y=model*np.exp(-1j*k*(b@truth))[:,None]+root[:,None]*(group[:,None]+1)*(.3+.8j)
            models.append(model);data.append(y);roots.append(root);groups.append(group);masks.append(mask)
        probe=PhaseProbe(models,data,roots,groups,masks,[b]*5,ks);at=np.array([.025,-.015,.01]);_,g=probe.objective(at)
        step=1e-6;fd=np.array([(probe.objective(at+step*e)[0]-probe.objective(at-step*e)[0])/(2*step) for e in np.eye(3)])
        np.testing.assert_allclose(g,fd,rtol=2e-8,atol=1e-9)
        result=minimize(probe.objective,np.zeros(3),jac=True,method='BFGS',options=dict(gtol=1e-10))
        np.testing.assert_allclose(result.x,truth,atol=1e-8,rtol=0)
        excluded=PhaseProbe(models,data,roots,groups,[~m for m in masks],[b]*5,ks)
        self.assertLess(excluded.objective(result.x)[0],1e-14)


if __name__=='__main__':unittest.main()
