import unittest
import numpy as np
from shared_origin_information import local_root, combined


class SharedOriginInformationTests(unittest.TestCase):
    def test_sequential_elimination_equals_independent_full_projection(self):
        rng=np.random.default_rng(915);jacobians=[rng.normal(size=(n,7)) for n in (29,31,35)]
        roots=[local_root(j,np.arange(2,7),np.arange(2),1e-12)[0] for j in jacobians]
        information,_=combined(roots,3,1e-12)
        # Full row stack with three distinct two-coordinate target motions,
        # three shared antenna coordinates and two shared origin coordinates.
        target=np.concatenate([j[:,2:5] for j in jacobians]);nuisance=np.zeros((len(target),8));start=0
        for i,j in enumerate(jacobians):
            section=slice(start,start+len(j));nuisance[section,2*i:2*i+2]=j[:,:2];nuisance[section,6:]=j[:,5:];start+=len(j)
        residual=target-nuisance@np.linalg.lstsq(nuisance,target,rcond=None)[0]
        np.testing.assert_allclose(information,residual.T@residual,atol=1e-12)

    def test_shared_origin_retains_between_target_information(self):
        # Each target alone confounds antenna and clock; opposite geometry
        # makes the antenna observable when one clock is shared.
        a=np.array([[1.,1.],[1.,1.]])
        b=np.array([[-1.,1.],[-1.,1.]])
        together,_=combined([a,b],1,1e-12)
        alone=[combined([x],1,1e-12)[0] for x in (a,b)]
        self.assertAlmostEqual(together[0,0],4.)
        self.assertLess(sum(x[0,0] for x in alone),1e-28)


if __name__=='__main__':unittest.main()
