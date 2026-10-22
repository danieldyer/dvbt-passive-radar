import unittest
import numpy as np
from chirp_acquisition import scores


class ChirpAcquisitionTests(unittest.TestCase):
    def test_physical_delay_bounds_also_restrict_acquisition_seeds(self):
        from test_recovery_experiment import problem
        from coherent_arc import CoherentArc
        from scan_native_ship_chirps import acquire
        from field import FS
        from motion import C
        ex=problem();ex.fields[0].y=ex.independent_signal(np.zeros(6),0,[.8+.3j]).reshape(ex.fields[0].y.shape)
        arc=CoherentArc(ex.fields[0],ex.training[0],stationary_design=ex.designs[0])
        proposals,delays,_=acquire(arc,count=3,origin_bounds=(-.5,.5))
        self.assertTrue(np.all(abs(delays*FS/C)<=.5+1e-12))
        self.assertTrue(proposals)
        self.assertTrue(all(abs(p['seed'][0])<=.5+1e-12 for p in proposals))

    def test_collapsed_bank_matches_full_projected_range_time_bank(self):
        rng=np.random.default_rng(173);n,r,m=73,17,8
        h=rng.normal(size=(n,r))+1j*rng.normal(size=(n,r))
        y=rng.normal(size=h.shape)+1j*rng.normal(size=h.shape)
        q=np.linalg.qr(rng.normal(size=(n*r,m))+1j*rng.normal(size=(n*r,m)))[0]
        y=(y.ravel()-q@(q.conj().T@y.ravel())).reshape(n,r)
        t=np.sort(rng.uniform(-11.,8.,n));rate=rng.uniform(-1,1,97);acc=rng.uniform(-.1,.1,97);k=4.5
        full=h.ravel()[:,None]*np.exp(-1j*k*(np.repeat(t,r)[:,None]*rate+.5*np.repeat(t,r)[:,None]**2*acc))
        expected=abs(full.conj().T@y.ravel())**2/(np.sum(abs(full)**2,axis=0)-np.sum(abs(q.conj().T@full)**2,axis=0))
        actual=scores(h,y,q,t,k,rate,acc,batch=13)
        np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)


if __name__=='__main__':unittest.main()
