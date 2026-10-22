import unittest
import numpy as np
import test_profiled_phase_scene as reference_fixture
from moment_phase_scene import MomentPhaseScene
from sparse_profiled_phase_scene import SparseSceneSolver
from rigid_patch import profile_static
from phase_image_review import matched_statistics


class MomentPhaseSceneTests(unittest.TestCase):
    def test_original_epoch_objective_coefficients_and_gradient(self):
        reference,truth,x=reference_fixture.ProfiledPhaseSceneTests().problem()
        actual=MomentPhaseScene(reference.design,reference.data,reference.root,reference.groups,reference.selected,
            reference.basis,reference.k,reference.strength,reference.spectrum)
        for at in [np.zeros(3),np.array([.03,-.02,.01]),np.array([.49,-.48,.45])]:
            value,g,report=actual.objective(at,True);expected,dg,raw=reference.objective(at,True)
            self.assertAlmostEqual(value,expected,places=12)
            np.testing.assert_allclose(g,dg,rtol=1e-11,atol=1e-12)
            np.testing.assert_allclose(report['coefficient'],raw['coefficient'],rtol=1e-11,atol=1e-12)
            self.assertLess(report['maximum_phase_weight_error'],1e-11)

    def test_cached_operator_moments_and_sparse_profile_match_original_rows(self):
        reference,_,_=reference_fixture.ProfiledPhaseSceneTests().problem()
        n=reference.design[0].shape[-1]
        reference.strength=.1
        reference.image_solver=SparseSceneSolver(np.eye(n),.1,kkt_tolerance=1e-11,gap_tolerance=1e-11)
        grams=[]
        for h,r,g,mask in zip(reference.design,reference.root,reference.groups,reference.selected):
            a=profile_static(h.reshape(len(h),-1),r,mask,g)*mask[:,None]
            a=a.reshape(*h.shape)[mask].reshape(-1,n)
            grams.append(a.conj().T@a)
        args=(reference.design,reference.data,reference.root,reference.groups,reference.selected,
            reference.basis,reference.k,.1,reference.spectrum)
        initial=MomentPhaseScene(*args,projected_gram=grams)
        actual=MomentPhaseScene(*args,projected_gram=grams,operator_moments=initial.mean_moments,
            image_solver=SparseSceneSolver(np.eye(n),.1,kkt_tolerance=1e-11,gap_tolerance=1e-11))
        at=np.array([.15,-.12,.04])
        value,gradient,report=actual.objective(at,True)
        expected,derivative,raw=reference.objective(at,True)
        self.assertAlmostEqual(value,expected,places=11)
        np.testing.assert_allclose(gradient,derivative,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(report['coefficient'],raw['coefficient'],rtol=1e-9,atol=1e-9)

    def test_matched_diagnostic_matches_stacked_physical_columns(self):
        reference,_,_=reference_fixture.ProfiledPhaseSceneTests().problem()
        actual=MomentPhaseScene(reference.design,reference.data,reference.root,reference.groups,reference.selected,
            reference.basis,reference.k,reference.strength,reference.spectrum)
        at=np.array([.1,-.15,.03]);n=reference.design[0].shape[-1]
        raw=[np.sum(abs(h[mask])**2,axis=(0,1)) for h,mask in zip(reference.design,reference.selected)]
        score,information=matched_statistics(actual,np.eye(n),raw,at)
        for c,(h,y,r,g,mask,b) in enumerate(zip(reference.design,reference.data,reference.root,reference.groups,reference.selected,reference.basis)):
            a=h*(reference.spectrum[c]*np.exp(-1j*reference.k[c]*(b@at)))[:,None,None]
            a=profile_static(a,r,mask,g)[mask].reshape(-1,n)
            observed=profile_static(y,r,mask,g)[mask].ravel()
            np.testing.assert_allclose(score[c],a.conj().T@observed,rtol=1e-12,atol=1e-11)
            np.testing.assert_allclose(information[c],np.sum(abs(a)**2,axis=0),rtol=1e-12,atol=1e-11)


if __name__=='__main__':unittest.main()
