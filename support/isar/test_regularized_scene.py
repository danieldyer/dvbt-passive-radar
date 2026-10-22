import unittest
import numpy as np
from regularized_scene import Observation,fit,solve_fixed,positive_spectrum,FixedRidgePath


class RegularizedSceneTests(unittest.TestCase):
    def test_spectrum_faces_and_zero_linear_hard_cases(self):
        from scipy.optimize import minimize
        rng=np.random.default_rng(1018)
        cases=[(np.array([0.,1000.]),np.array([-.01,1.])),
            (np.array([0.,10.]),np.array([0.,1.])),(np.array([0.,1.,2.]),np.zeros(3))]
        cases.extend((rng.uniform(0,2,5),rng.normal(size=5)) for _ in range(20))
        for a,b in cases:
            radius=len(a);g=positive_spectrum(a,b,radius)
            self.assertTrue(np.all(g>=0));self.assertAlmostEqual(float(g@g),radius,places=10)
            value=lambda v:a@(v*v)-2*b@v
            for initial in np.sqrt(radius)*np.eye(len(a)):
                reference=minimize(value,initial,jac=lambda v:2*(a*v-b),method='SLSQP',bounds=[(0,None)]*len(a),
                    constraints=[dict(type='eq',fun=lambda v:v@v-radius,jac=lambda v:2*v)],options=dict(ftol=1e-11,maxiter=500))
                if reference.success:self.assertLessEqual(value(g),value(reference.x)+1e-7)

    def test_original_operator_statistics_and_phase(self):
        rng=np.random.default_rng(1402);t=61;r=7;p=5
        h=rng.normal(size=(t,r,p))+1j*rng.normal(size=(t,r,p));y=rng.normal(size=(t,r))+1j*rng.normal(size=(t,r))
        root=rng.uniform(.2,2,t);groups=np.arange(t)>30;mask=np.arange(t)%3!=0
        o=Observation.from_rows(h,y,root,mask,groups);x=rng.normal(size=p)+1j*rng.normal(size=p)
        from rigid_patch import profile_static
        residual=profile_static(h@x-y,root,mask,groups)[mask]
        self.assertLess(abs(o.error(x)-np.vdot(residual,residual).real),1e-10)

    def test_shared_complex_scene_and_positive_spectrum(self):
        rng=np.random.default_rng(913);truth=rng.normal(size=6)+1j*rng.normal(size=6)
        spectrum=np.array([.8,.5,1.3,1.1,1.]);spectrum*=np.sqrt(5/np.dot(spectrum,spectrum));observations=[]
        for g in spectrum:
            h=rng.normal(size=(150,6))+1j*rng.normal(size=(150,6));y=g*h@truth
            observations.append(Observation(h.conj().T@h,h.conj().T@y,float(np.vdot(y,y).real)))
        x,g,report=fit(observations,1e-7,maximum_iterations=300)
        np.testing.assert_allclose(g,spectrum,atol=3e-5,rtol=0)
        np.testing.assert_allclose(x,truth,atol=3e-5,rtol=0)
        self.assertLess(report['maximum_spectrum_kkt'],1e-5)
        self.assertLess(report['maximum_scene_gradient'],1e-12)
        self.assertTrue(np.all(np.diff(report['objective_trace'])<=1e-10))

    def test_unseen_data_does_not_enter_solve(self):
        rng=np.random.default_rng(593);h=rng.normal(size=(50,3,4))+1j*rng.normal(size=(50,3,4));y=h[:,:,0].copy()
        mask=np.arange(50)%2==0;root=np.ones(50);groups=np.zeros(50)
        a=Observation.from_rows(h,y,root,mask,groups);y[~mask]=1e12
        b=Observation.from_rows(h,y,root,mask,groups)
        np.testing.assert_array_equal(solve_fixed([a],[1.],1.),solve_fixed([b],[1.],1.))
        reused=Observation.from_rows(h,y,root,mask,groups,gram=a.gram)
        np.testing.assert_array_equal(reused.gram,a.gram);np.testing.assert_array_equal(reused.rhs,b.rhs)
        path=FixedRidgePath([a],[1.])
        for strength in [.01,1.,100.]:np.testing.assert_allclose(path.solve(strength),solve_fixed([a],[1.],strength),rtol=1e-12,atol=1e-12)

    def test_one_constant_complex_calibration_per_band(self):
        rng=np.random.default_rng(279);truth=rng.normal(size=6)+1j*rng.normal(size=6)
        gains=np.array([.8,.5,1.3,1.1,1.])*np.exp(1j*np.array([.4,-.6,1.,.2,-.8]))
        gains*=np.sqrt(5/np.vdot(gains,gains).real);observations=[]
        for g in gains:
            h=rng.normal(size=(150,6))+1j*rng.normal(size=(150,6));y=g*h@truth
            observations.append(Observation(h.conj().T@h,h.conj().T@y,float(np.vdot(y,y).real)))
        x,g,report=fit(observations,1e-7,fit_band_phases=True,maximum_iterations=300)
        for original,fitted in zip(gains,g):np.testing.assert_allclose(fitted*x,original*truth,atol=5e-5,rtol=0)
        self.assertAlmostEqual(np.angle(g[3]),0.,places=12)
        self.assertLess(report['maximum_spectrum_kkt'],1e-5)


if __name__=='__main__':unittest.main()
