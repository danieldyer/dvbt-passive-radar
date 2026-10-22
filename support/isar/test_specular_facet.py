import unittest

import numpy as np

from motion import C
from specular_facet import response


class FacetTests(unittest.TestCase):
    def test_point_limit_and_far_field_rectangular_transform(self):
        tx,rx = np.array([1e10,-2e9,4e8]),np.array([1e10,2e9,2e8])
        along,up = np.array([0.,1.,0.]),np.array([0.,0.,1.])
        f = np.array([177.5e6,226.5e6])
        small = response([[0.,0.,0.]],tx,rx,along,up,.001,.001,f)
        np.testing.assert_allclose(small['relative_field'],1.,atol=2e-8)
        gradient = -tx/np.linalg.norm(tx)-rx/np.linalg.norm(rx)
        for angle in [0.,.002]:
            a = [np.sin(angle),np.cos(angle),0.]
            found = response([[0.,0.,0.]],tx,rx,a,up,150.,25.,f)
            expected = np.sinc(f/C*(gradient@a)*150)*np.sinc(f/C*(gradient@up)*25)
            np.testing.assert_allclose(found['relative_field'][0],expected,atol=3e-7)

    def test_spherical_phase_quadrature_convergence_and_frequency_derivative(self):
        centre = np.array([[0.,0.,25.]])
        tx,rx = np.array([37000.,3800.,700.]),np.array([5300.,6800.,31.])
        direction = tx-centre[0];direction/=np.linalg.norm(direction)
        direction += (rx-centre[0])/np.linalg.norm(rx-centre[0])
        heading = np.arctan2(direction[0],direction[1])-np.pi/2
        a = [np.sin(heading),np.cos(heading),0.];up=[0.,0.,1.]
        f = 200e6
        result = response(centre,tx,rx,a,up,286.,30.,[f])
        dense = response(centre,tx,rx,a,up,286.,30.,[f],order=(128,24))
        np.testing.assert_allclose(result['relative_field'],dense['relative_field'],atol=2e-13)
        df = 200.
        sides = response(centre,tx,rx,a,up,286.,30.,[f-df,f+df])['relative_field'][0]
        dk = 2*np.pi*df/C
        derivative = (sides[1]-sides[0])/(2*dk)
        expected = -1j*result['relative_field'][0,0]*result['path_moment_m'][0,0]
        np.testing.assert_allclose(derivative,expected,atol=1e-9)

    def test_rigid_translation_and_basis_are_invariant(self):
        centre = np.array([[0.,0.,20.],[2.,4.,20.]])
        tx,rx = np.array([37000.,4000.,700.]),np.array([5000.,6000.,35.])
        a=np.array([-.5,np.sqrt(.75),0.]);b=np.array([0.,0.,1.])
        first=response(centre,tx,rx,a,b,260.,25.,[180e6,220e6])
        rotation=np.linalg.qr(np.array([[1.,2.,3.],[4.,1.,6.],[7.,8.,2.]]))[0]
        shift=np.array([4e6,-3e6,5e6])
        second=response(centre@rotation+shift,tx@rotation+shift,rx@rotation+shift,a@rotation,b@rotation,
            260.,25.,[180e6,220e6],up=b@rotation)
        np.testing.assert_allclose(first['relative_field'],second['relative_field'],atol=1e-10)


if __name__ == '__main__':
    unittest.main()
