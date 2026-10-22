"""Distinguish geographic registration from loss of coherent peak power."""
import unittest
import numpy as np

from calibration_uncertainty import phase_derivative,coherent_fraction
from geometry import C,RF,central_jacobian
from road_imaging_risk import ground_phase_jacobian,image_point_projection


class ImagingUncertaintyTests(unittest.TestCase):
    def test_position_and_common_phase_modes_do_not_defocus(self):
        rng=np.random.default_rng(916)
        h=rng.normal(size=(7,5,2));design=np.concatenate([np.ones((7,5,1)),h],axis=-1)
        perturb=rng.normal(size=(7,3,4));d=design@perturb
        remaining,shift,location,_=image_point_projection(d,h,np.eye(4))
        np.testing.assert_allclose(remaining,0,atol=8e-15)
        np.testing.assert_allclose(shift,-perturb[:,1:],atol=8e-15)
        np.testing.assert_allclose(location,perturb[:,1:]@np.swapaxes(perturb[:,1:],-1,-2),atol=2e-14)
        np.testing.assert_allclose(coherent_fraction(remaining,np.eye(4)),1,atol=1e-14)

    def test_unregistered_phase_modes_are_preserved(self):
        rng=np.random.default_rng(967)
        h=rng.normal(size=(3,5,2));d=rng.normal(size=(3,5,6))
        remaining,_,_,_=image_point_projection(d,h,np.eye(6))
        design=np.concatenate([np.ones((3,5,1)),h],axis=-1)
        np.testing.assert_allclose(np.swapaxes(design,-1,-2)@remaining,0,atol=5e-14)
        self.assertGreater(np.linalg.norm(remaining),1.)
        self.assertTrue(np.all(coherent_fraction(remaining,np.eye(6))<1.))

    def test_terrain_tangent_jacobian_matches_exact_geometry(self):
        sites=np.array([[31000.,-3000.,700.],[31200.,-2800.,720.],[31100.,-2700.,710.]])
        assignment=[0,0,1,1,2];offsets=np.zeros((5,3));offsets[:,2]=[4,0,3,3,0]
        xy=np.array([-1000.,-2000.]);slope=np.array([.02,-.04])
        def phase(p):
            x=np.r_[p,20+slope@p][None,:]
            return -2*np.pi*RF/C*phase_derivative(x,sites,assignment,offsets)[1][:,0]
        x=np.r_[xy,20+slope@xy][None,:]
        expected=ground_phase_jacobian(x,sites,assignment,offsets,np.r_[-slope,1.][None,:])[0]
        np.testing.assert_allclose(expected,central_jacobian(phase,xy,.001),atol=2e-8)


if __name__=='__main__':unittest.main()
