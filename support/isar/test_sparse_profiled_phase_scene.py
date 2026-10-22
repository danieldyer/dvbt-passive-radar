import unittest
import numpy as np
from profiled_phase_scene import ProfiledPhaseScene
from sparse_profiled_phase_scene import SparseProfiledPhaseScene
from complex_lasso import ComplexLasso
import test_profiled_phase_scene as reference_fixture


class SparseProfiledPhaseSceneTests(unittest.TestCase):
    def test_original_complex_lasso_profile_and_gradient(self):
        base,truth,x=reference_fixture.ProfiledPhaseSceneTests().problem()
        model=SparseProfiledPhaseScene(base.design,base.data,base.root,base.groups,base.selected,base.basis,base.k,.1,base.spectrum,
            spatial_basis=np.eye(5,dtype=complex),kkt_tolerance=1e-11,gap_tolerance=1e-11)
        at=np.array([.03,-.02,.01]);value,g,report=model.objective(at,True)
        gram,rhs,_=base.normal(at);reference=ComplexLasso(gram,rhs,base.energy)
        exact,certificate=reference.solve(.1,kkt_tolerance=1e-11,gap_tolerance=1e-11)
        self.assertTrue(certificate['numerical_convergence'])
        np.testing.assert_allclose(report['physical_coefficient'],exact,rtol=0,atol=1e-9)
        step=1e-5;fd=[(model.objective(at+step*e)[0]-model.objective(at-step*e)[0])/(2*step) for e in np.eye(3)]
        np.testing.assert_allclose(g,fd,rtol=3e-6,atol=2e-7)
        self.assertLess(report['sparse_optimization']['maximum_scaled_kkt'],1e-11)


if __name__=='__main__':unittest.main()
