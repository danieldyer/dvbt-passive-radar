import unittest
from types import SimpleNamespace

import numpy as np

from facet_geometry import FacetFlash,facet_observables
from flash_geometry import observables
from specular_facet import response
from facet_kinematics import aperture
from motion import enu_basis


class FacetGeometryTests(unittest.TestCase):
    def test_heading_seed_is_specular_for_the_actual_trial_position(self):
        tx=np.tile([37000.,3800.,700.],(5,1));rx=np.array([5300.,6800.,31.])
        def state(q):return np.array([150.,-300.,20.])+100*np.array(q[:3]),np.zeros(3),np.zeros(3),tx,rx
        base=SimpleNamespace(count=10,lower=np.full(10,-3.),upper=np.full(10,3.),nantenna=0,
            nuisance_sigma=np.full(10,20.),utc=0.,state=state,rotation=enu_basis(0.,0.).T,
            track=SimpleNamespace(sample=lambda *a,**k:{'latitude':np.array([0.]),'longitude':np.array([0.])}))
        model=FacetFlash(base,np.full(5,200e6),length_m=200.,height_m=20.)
        for shift in (np.zeros(10),np.r_[.2,-.3,np.zeros(8)]):
            seed=model.seed(shift,0.,.1)
            centre,*_=state(seed)
            theta=model.heading+model.scale[0]*seed[-2]
            along=np.array([np.sin(theta),np.cos(theta),0.])
            gradient=(centre-tx[0])/np.linalg.norm(centre-tx[0])+(centre-rx)/np.linalg.norm(centre-rx)
            self.assertLess(abs(along@gradient),1e-14)

    def test_point_limit_and_stencil_convergence(self):
        p,v,a=np.array([0.,0.,20.]),np.array([-2.,4.5,0.]),np.array([.01,-.01,0.])
        tx=np.array([[37000.,3800.,700.],[37100.,4000.,650.]])
        rx=np.array([5300.,6800.,31.]);f=np.array([177.5e6,226.5e6])
        heading=np.radians(331.1);rate=np.radians(.2)
        tiny,_,_=facet_observables(p,v,a,tx,rx,f,heading,rate,.001,.001)
        np.testing.assert_allclose(tiny,observables(p,v,a,tx,rx),atol=2e-7)
        full,_,_=facet_observables(p,v,a,tx,rx,f,heading,rate,286.,30.)
        half,_,_=facet_observables(p,v,a,tx,rx,f,heading,rate,286.,30.,step_s=.005)
        np.testing.assert_allclose(full[:,1:],half[:,1:],atol=1e-6)

    def test_phase_rates_against_independent_polynomial_complex_phase_fit(self):
        p,v,a=np.array([0.,0.,20.]),np.array([-2.,4.5,0.]),np.array([.01,-.01,0.])
        tx=np.array([[37000.,3800.,700.]]);rx=np.array([5300.,6800.,31.]);f=np.array([200e6])
        heading=np.radians(331.3);rate=np.radians(.12)
        found,_,_=facet_observables(p,v,a,tx,rx,f,heading,rate,200.,20.,step_s=.005)
        t=np.linspace(-.08,.08,41)
        centre=p+t[:,None]*v+.5*t[:,None]**2*a
        angle=heading+rate*t
        along=np.c_[np.sin(angle),np.cos(angle),np.zeros(len(t))]
        shape=response(centre,tx[0],rx,along,[0.,0.,1.],200.,20.,f)['relative_field'][:,0]
        length=np.linalg.norm(centre-tx[0],axis=1)+np.linalg.norm(centre-rx,axis=1)
        phase=np.unwrap(np.angle(shape))-2*np.pi*f[0]/299792458.*(length-length[len(t)//2])
        polynomial=np.polynomial.polynomial.polyfit(t,phase,6)
        expected=-np.array([polynomial[1],2*polynomial[2]])/(2*np.pi*f[0]/299792458.)
        np.testing.assert_allclose(found[0,1:],expected,atol=1e-5)

    def test_analytic_aperture_derivatives_against_independent_temporal_stencil(self):
        p,v,a=np.array([0.,0.,20.]),np.array([-2.,4.5,.02]),np.array([.01,-.01,.001])
        tx=np.array([[37000.,3800.,700.],[37100.,4000.,650.]])
        rx=np.array([5300.,6800.,31.]);f=np.array([177.5e6,226.5e6])
        for heading in np.radians([330.8,331.1,331.7]):
            turn=np.radians(.2)
            finite,shape,physical=facet_observables(p,v,a,tx,rx,f,heading,turn,286.,30.,step_s=.003)
            for j in range(2):
                field,correction=aperture(p,v,a,tx[j],rx,f[j],heading,turn,286.,30.)
                np.testing.assert_allclose(field,shape[j],atol=1e-14)
                np.testing.assert_allclose(correction,finite[j]-physical[j],atol=3e-7)


if __name__=='__main__':unittest.main()
