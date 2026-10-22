from types import SimpleNamespace
import unittest

import numpy as np

from cluster_phase import PolynomialPath,SplinePath
from release_fixed_grid_phase_guide import path_information


def model_for(matrix,curve=None):
    curve=PolynomialPath(0.,4.,3) if curve is None else curve
    cluster=SimpleNamespace(slices=[slice(0,2+curve.degree)],evaluate=lambda p:dict(mean_derivative=matrix))
    return SimpleNamespace(clusters=[cluster],indices=[np.arange(matrix.shape[1])],selected=0,
        curve=curve,bands=[SimpleNamespace(field=SimpleNamespace(k=4.))])


class PathInformationTests(unittest.TestCase):
    def test_profiled_bound_equals_full_joint_gaussian_covariance(self):
        rng=np.random.default_rng(735);matrix=rng.normal(size=(200,8))
        matrix[:,2:5]+=.8*matrix[:,:3]
        model=model_for(matrix);result=path_information(model,np.zeros(8))
        self.assertTrue(result['full_rank'])
        covariance=np.linalg.inv(2*matrix.T@matrix)[2:5,2:5]
        basis=model.curve(result['time_s'])
        expected=np.sqrt(np.einsum('ij,jk,ik->i',basis,covariance,basis))
        np.testing.assert_allclose(result['path_standard_deviation_m'],expected,rtol=3e-14,atol=1e-15)

    def test_path_column_inside_nuisance_span_is_not_measured_information(self):
        rng=np.random.default_rng(736);matrix=rng.normal(size=(200,8));matrix[:,2]=matrix[:,0]
        result=path_information(model_for(matrix),np.zeros(8))
        self.assertFalse(result['full_rank'])
        origin=np.asarray(result['time_s'])==2.
        np.testing.assert_array_equal(result['path_change_estimable'],origin)
        self.assertTrue(np.isnan(np.asarray(result['phase_standard_deviation_deg'])[~origin]).all())

    def test_unobserved_endpoint_does_not_discard_identifiable_earlier_phase(self):
        rng=np.random.default_rng(737);curve=SplinePath(0.,4.,1.)
        matrix=rng.normal(size=(200,11));null=2+curve.degree-1;matrix[:,null]=matrix[:,0]
        result=path_information(model_for(matrix,curve),np.zeros(11))
        self.assertFalse(result['full_rank']);time=np.asarray(result['time_s'])
        np.testing.assert_array_equal(result['path_change_estimable'],time<=3.)
        reduced=np.delete(matrix,null,axis=1);covariance=np.linalg.inv(2*reduced.T@reduced)[2:7,2:7]
        basis=curve(time[time<=3.])[:,:-1]
        expected=np.sqrt(np.einsum('ij,jk,ik->i',basis,covariance,basis))
        np.testing.assert_allclose(np.asarray(result['path_standard_deviation_m'])[time<=3.],expected,rtol=4e-14,atol=1e-15)


if __name__=='__main__':unittest.main()
