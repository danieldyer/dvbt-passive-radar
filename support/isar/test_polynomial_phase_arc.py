import unittest
import numpy as np

from polynomial_phase_arc import Band,PolynomialPhaseArc


def fixture(drift=False):
    bands=[];truth=np.array([.21,-.04,.009])
    for c,rf in enumerate([177.5e6,184.5e6,191.625e6,219.5e6,226.5e6]):
        t=np.linspace(0.,2.,130)+c*.00007;k=2*np.pi*rf/299792458.;u=t-1
        known=.03*c*u+.01*u**2
        path=sum(q*u**(j+1) for j,q in enumerate(truth))
        if drift and c==4:path+=.3*u
        strength=.8+.15*np.cos(3*t+c)
        z=strength*np.exp(1j*(.43*c-k*(known+path)))
        training=(np.floor(t/.125).astype(int)%2)==0
        bands.append(Band(t,z,np.ones(len(t))*1e4,known,k,training))
    return bands,truth


class PolynomialPhaseArcTests(unittest.TestCase):
    def test_original_row_averaging_and_erasures_preserve_phase_derivatives(self):
        bands,truth=fixture();rng=np.random.default_rng(113)
        for c,b in enumerate(bands):
            rows=(b.time[:,None]+np.linspace(-.0075,.0075,16)[None,:]).ravel();u=rows-1
            rw=rng.uniform(.2,2.,len(rows));rw[::11]=0
            known=.03*c*u+.01*u**2
            b.row_time=rows;b.row_weight=rw;b.row_known_path=known
            raw=np.exp(1j*(.43*c-b.k*(known+sum(q*u**(j+1) for j,q in enumerate(truth)))))
            b.value=np.sum((raw*rw).reshape(-1,16),axis=1)/np.sum(rw.reshape(-1,16),axis=1)
        model=PolynomialPhaseArc(bands,0,2,3);p=np.r_[np.arange(5)*.43,truth]
        self.assertLess(np.linalg.norm(model.evaluate(p)[0]),1e-14)
        p+=np.sin(np.arange(8))*.01;direction=np.cos(np.arange(8))*.03;_,j=model.evaluate(p)
        step=1e-5
        finite=(model.evaluate(p+step*direction)[0]-model.evaluate(p-step*direction)[0])/(2*step)
        np.testing.assert_allclose(finite,j@direction,atol=1e-9,rtol=2e-6)
        result=model.fit();np.testing.assert_allclose(result['parameters'][-3:],truth,atol=1e-8)

    def test_profiled_complex_residual_derivative(self):
        bands,truth=fixture();model=PolynomialPhaseArc(bands,0,2,3)
        p=np.r_[np.arange(5)*.43,truth]+np.sin(np.arange(8))*.01
        r,j=model.evaluate(p);direction=np.cos(np.arange(8))*.03
        for step in (1e-4,5e-5):
            finite=(model.evaluate(p+step*direction)[0]-model.evaluate(p-step*direction)[0])/(2*step)
            np.testing.assert_allclose(finite,j@direction,atol=1e-9,rtol=2e-6)

    def test_common_path_recovery_across_carriers_and_constant_phase_gauges(self):
        bands,truth=fixture();model=PolynomialPhaseArc(bands,0,2,3);fit=model.fit()
        np.testing.assert_allclose(fit['parameters'][-3:],truth,atol=1e-8)
        self.assertLess(fit['cost'],1e-12)
        self.assertTrue(all(p['test_phase_rms_deg']<1e-5 for p in fit['prediction']))
        for c,b in enumerate(bands):b.value*=np.exp(1j*(c+1)*.67)
        other=PolynomialPhaseArc(bands,0,2,3).fit()
        np.testing.assert_allclose(other['parameters'][-3:],truth,atol=1e-8)

    def test_testing_cannot_change_fits_or_seeds(self):
        bands,_=fixture();model=PolynomialPhaseArc(bands,0,2,2)
        seeds=model.seeds();before=model.fit()
        for b in bands:b.value[~b.training]*=np.exp(1.3j)
        other=PolynomialPhaseArc(bands,0,2,2)
        for a,b in zip(seeds,other.seeds()):np.testing.assert_array_equal(a,b)
        after=other.fit()
        np.testing.assert_array_equal(before['parameters'],after['parameters'])
        self.assertGreater(sum(p['test_error'] for p in after['prediction']),1000)

    def test_independent_band_drift_is_exposed_on_unused_samples(self):
        bands,_=fixture(drift=True)
        common=PolynomialPhaseArc(bands,0,2,3).fit()
        independent=[PolynomialPhaseArc([b],0,2,3).fit() for b in bands]
        self.assertGreater(sum(p['test_error'] for p in common['prediction']),1000)
        self.assertTrue(all(r['prediction'][0]['test_phase_rms_deg']<1e-5 for r in independent))


if __name__=='__main__':unittest.main()
