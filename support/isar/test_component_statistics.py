import unittest
import numpy as np
from stationary import Stationary,component_statistics,rank_one_projection


class ComponentStatisticsTests(unittest.TestCase):
    def test_schur_score_matches_each_explicit_nuisance_projection(self):
        rng=np.random.default_rng(940);n=109
        h=rng.normal(size=(n,4))+1j*rng.normal(size=(n,4));h[:,1]+=.8*h[:,0];h[:,2]*=.03
        d=rng.normal(size=(n,3))+1j*rng.normal(size=(n,3));y=rng.normal(size=n)+1j*rng.normal(size=n)
        use=np.ones(n,bool);static=Stationary(d,use);z,info,condition=component_statistics(h,y,static)
        for j in range(4):
            nuisance=Stationary(np.c_[d,np.delete(h,j,axis=1)],use)
            target=nuisance.remove(h[:,j]);clean=nuisance.remove(y)
            np.testing.assert_allclose(z[j],np.vdot(target,clean),rtol=2e-12,atol=2e-12)
            np.testing.assert_allclose(info[j],np.vdot(target,target).real,rtol=2e-12)
        self.assertGreater(condition,1.)
        phase=np.exp(1j*np.arange(4));zz,ii,_=component_statistics(h*phase,y,static)
        np.testing.assert_allclose(ii,info,rtol=2e-12)
        np.testing.assert_allclose(zz,z*phase.conj(),rtol=2e-12)

    def test_rank_one_image_update_equals_full_explicit_projection(self):
        rng=np.random.default_rng(17);n=111
        h=rng.normal(size=(n,13))+1j*rng.normal(size=(n,13))
        d=rng.normal(size=(n,3))+1j*rng.normal(size=(n,3))
        nuisance=rng.normal(size=n)+1j*rng.normal(size=n);y=rng.normal(size=n)+1j*rng.normal(size=n)
        selected=np.ones(n,bool);base=Stationary(d,selected)
        hh=base.remove(h);yy=base.remove(y);nn=base.remove(nuisance)
        score,info=rank_one_projection(hh.conj().T@yy,np.sum(abs(hh)**2,axis=0),hh.conj().T@nn,np.vdot(nn,yy),np.vdot(nn,nn).real)
        full=Stationary(np.c_[d,nuisance],selected);target=full.remove(h)
        np.testing.assert_allclose(score,target.conj().T@full.remove(y),rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(info,np.sum(abs(target)**2,axis=0),rtol=2e-12)


if __name__=='__main__':unittest.main()
