import unittest

import numpy as np

from differentiable_sea import paths
from spherical_sea import geometry,horizontal_reflection


class DifferentiableSeaTests(unittest.TestCase):
    def test_reference_values_and_all_endpoint_derivatives(self):
        radius=6371000.;centre=np.array([0.,0.,-radius])
        x=np.array([[0.,0.,10.],[10.,20.,25.],[-15.,40.,50.]])
        v=np.array([[.4,2.,.01],[.8,3.,-.01],[.5,2.5,0.]])
        tx=np.array([38000.,4000.,650.]);rx=np.array([6000.,7000.,30.])
        rng=np.random.default_rng(277);n=7
        dx=rng.normal(size=x.shape+(n,));dv=.01*rng.normal(size=x.shape+(n,))
        dtx=rng.normal(size=(3,n));drx=rng.normal(size=(3,n))
        result=paths(x,v,tx,rx,centre,radius,dx,dv,dtx,drx,200e6)
        reference=geometry(x,v,tx,rx,centre,radius)
        gt,gr=[horizontal_reflection(r['sin_grazing'],200e6,relative_permittivity=80.,conductivity_s_per_m=4.) for r in reference['reflected_legs']]
        expected=reference['relative_spreading']*np.array([np.ones(len(x)),gt,gr,gt*gr])
        np.testing.assert_allclose(result[0],reference['excess_path_m'],atol=1e-10,rtol=0.)
        np.testing.assert_allclose(result[1],reference['path_rate_mps'],atol=1e-12,rtol=0.)
        np.testing.assert_allclose(result[2],expected,atol=1e-14,rtol=1e-14)
        for j in range(n):
            h=.004
            args=[paths(x+sign*h*dx[:,:,j],v+sign*h*dv[:,:,j],tx+sign*h*dtx[:,j],rx+sign*h*drx[:,j],
                        centre,radius,dx,dv,dtx,drx,200e6) for sign in (1,-1)]
            for i,tolerance in enumerate([2e-6,2e-8,2e-7]):
                finite=(args[0][i]-args[1][i])/(2*h)
                np.testing.assert_allclose(result[i+3][:,:,j],finite,rtol=tolerance,atol=tolerance)


if __name__=='__main__':unittest.main()
