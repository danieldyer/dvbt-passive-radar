import unittest
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import BSpline
from continuous_ship_motion import ContinuousShipMotion
from continuous_ship_fit import MotionCoordinates,GlobalPointBand,ContinuousScene
from full_path_scene import SceneBand
from full_path_point import carrier_bin
from rigid_patch import profile_static
from spline_motion import knots
from test_full_path_point import problem


def fixture():
    old=problem();m=ContinuousShipMotion.__new__(ContinuousShipMotion)
    m.interval=old.motion.interval;m.rx=old.motion.rx;m.tx=old.motion.tx;m.rotation=old.motion.local.T
    m.rf=np.repeat(old.field.metadata['rf_hz'],5);m.gains=np.ones(5,complex)
    m.geographic_frame=lambda t:(np.broadcast_to(m.rotation.T,(len(t),3,3)),np.zeros((len(t),3,3)))
    pk=knots(m.interval,4.);b=BSpline.design_matrix(old.t,pk,3).toarray();state=old.motion.state(old.t)
    pc=np.linalg.lstsq(b,(state['position_ecef_m']-m.rx)@m.rotation.T,rcond=None)[0]
    ac=.001*(np.array([pk[j+1:j+4].mean() for j in range(len(pc))])-8.);m.result=dict(position_knots_s=pk,position_coefficients_enu_m=pc,
        angle_knots_s=pk,angle_coefficients_rad=ac,anchor_s=8.)
    m.position_curve=BSpline(pk,pc,3);m.angle_curve=BSpline(pk,ac,3)
    coordinate=dict(old.coordinate,training=np.ones(len(old.t),bool))
    return GlobalPointBand(old.field,coordinate,MotionCoordinates(m)),m


class ContinuousShipTests(unittest.TestCase):
    def test_all_six_physical_partials_and_sparse_chain_rule(self):
        b,m=fixture();n=b.model.n;q=np.linspace(-.015,.02,n);point=np.array([25.,-45.,3.])
        values=b.waveform(q,3,True,point)
        rng=np.random.default_rng(17)
        for j in range(6):
            d=rng.normal(size=n);step=2e-5
            actual=(b.waveform(q+step*d,3,False,point)-b.waveform(q-step*d,3,False,point))/(2*step)
            expected=sum(partial*(basis@d)[:,None] for partial,basis in zip(values[1:],b.motion_bases))
            self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),2e-5)

    def test_exported_curve_matches_fit_and_exact_velocity(self):
        b,m=fixture();q=np.linspace(-.01,.015,b.model.n)
        pc,ac=b.model.coefficients(q);m.position_curve=BSpline(b.model.pk,pc,3);m.angle_curve=BSpline(b.model.ak,ac,3)
        point=np.array([25.,-45.,3.]);state=m.state(b.t)
        position=state['position_ecef_m']+np.einsum('tij,j->ti',state['body_to_ecef'],point)
        velocity=state['velocity_ecef_mps']+np.einsum('tij,j->ti',state['frame_velocity'],point)
        p,v,*_=b.point_state(q,point)
        np.testing.assert_allclose(p,position,rtol=0,atol=2e-9);np.testing.assert_allclose(v,velocity,rtol=0,atol=1e-11)
        t=b.t[2:-2];step=1e-3
        def positions(t):
            s=m.state(t);return s['position_ecef_m']+s['body_to_ecef']@point
        finite=(positions(t+step)-positions(t-step))/(2*step)
        np.testing.assert_allclose(finite,velocity[2:-2],rtol=0,atol=1e-6)
        # One immutable carrier reference and exactly the original epochs.
        f=b.field;raw=f.raw_trajectory(position,velocity,transmitter=m.tx[3],receiver=m.rx)
        _,_,g=f.path_kinematics(position,velocity,transmitter=m.tx[3],receiver=m.rx)
        factor,_=carrier_bin(b.t,(g*velocity).sum(axis=1),b.coordinate['native_time_s'],b.coordinate['native_information'],f.k)
        np.testing.assert_allclose(b.waveform(q,3,False,point),f.apply(raw*factor[:,None]),rtol=2e-7,atol=1e-7)

    def test_profiled_global_multiband_gradient_and_true_stationary_nuisance(self):
        bands=[];truth=None
        for c in range(2):
            b,m=fixture();points=np.array([[0.,0.,0.],[25.,-30.,0.]])
            scene=SceneBand(b,np.broadcast_to(points,(2,2,3)),[m.interval[0],8.,m.interval[1]],4.)
            if truth is None:truth=np.linspace(-.005,.006,b.model.n)
            phase=np.array([.3,-.2,.4,.6])+c*.2
            h=scene.design(truth,c,False);gain=(b.amplitude@np.full((b.amplitude.shape[1],2),.8))*np.exp(1j*scene.phase_at_time(phase))
            b.field.y=np.sum(h*gain[:,None,:],axis=-1)+b.field.root_information[:,None]*(.2+.3j);b.refresh_data();bands.append(scene)
        scene=ContinuousScene(bands,b.model,[0.,.2]);at=np.r_[truth*.6,[.31,-.19,.41,.61]]
        value,g=scene.objective(at)
        cached_calls=scene.calls;again,again_g=scene.objective(at)
        self.assertEqual(scene.calls,cached_calls);np.testing.assert_array_equal(again_g,g);self.assertEqual(again,value)
        parallel=ContinuousScene(bands,b.model,[0.,.2],workers=2);pv,pg=parallel.objective(at)
        self.assertAlmostEqual(pv,value,places=10);np.testing.assert_allclose(pg,g,rtol=0,atol=1e-9)
        for j in range(len(at)):
            d=np.eye(len(at))[j]*3e-5;finite=(scene.objective(at+d)[0]-scene.objective(at-d)[0])/6e-5
            self.assertLess(abs(finite-g[j]),2e-5)
        correct=np.r_[truth,[.3,-.2,.4,.6]];self.assertLess(scene.objective(correct)[0],2e-10)

    def test_kinematic_constraint_controls_certify_between_samples(self):
        b,m=fixture();model=b.model;matrix,lo,hi=model.constraints();q=np.zeros(model.n)
        self.assertLessEqual(np.max(lo-matrix@q),1e-8);self.assertLessEqual(np.max(matrix@q-hi),1e-8)
        pc,ac=model.coefficients(q);t=np.linspace(*m.interval,1003)
        for order,limit in [(1,15.),(2,.15),(3,.03)]:self.assertLessEqual(abs(BSpline(model.pk,pc,3)(t,order)[:,:2]).max(),limit+1e-10)
        for order,limit in [(1,np.radians(.6)),(2,np.radians(.06))]:self.assertLessEqual(abs(BSpline(model.ak,ac,3)(t,order)).max(),limit+1e-10)

    def test_invertible_preconditioner_matches_independent_fixed_brightness_curvature(self):
        b,m=fixture();points=np.array([[0.,0.,0.],[25.,-30.,0.]])
        band=SceneBand(b,points,amplitude_spacing=4.);n=b.model.n
        truth=np.r_[np.zeros(n),[.3,-.2]];h=band.design(truth[:n],0,False)
        b.field.y=np.sum(h*np.array([.8*np.exp(.3j),.6*np.exp(-.2j)])[None,None,:],axis=-1);b.refresh_data()
        scene=ContinuousScene([band],b.model,[0.]);_,_,solution=scene.objective(truth,True)
        amplitude=solution[0]['amplitude'];transform,report=scene.preconditioner(truth)
        def prediction(q):
            h=band.design(q[:n],0,False);gain=(b.amplitude@amplitude)*np.exp(1j*band.phase_at_time(q[n:]))
            return profile_static(np.sum(h*gain[:,None,:],axis=-1),b.field.root_information,b.training).ravel()
        step=2e-5;jac=np.column_stack([(prediction(truth+np.eye(len(truth))[j]*step)-prediction(truth-np.eye(len(truth))[j]*step))/(2*step) for j in range(len(truth))])
        expected=2000*np.real(jac.conj().T@jac)/scene.energy
        inverse=np.linalg.inv(transform);actual=inverse@inverse
        self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),3e-5)
        self.assertGreater(np.linalg.eigvalsh(transform)[0],0.)

if __name__=='__main__':unittest.main()
