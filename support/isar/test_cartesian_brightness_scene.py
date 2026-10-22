"""Cartesian CV geometry, phase gauges and independently generated echoes."""
import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import nnls

from cartesian_patch import CartesianPatch
from coordinated_patch import CoordinatedPatch
from cv_rigid_patch import CvRigidPatch
from cartesian_brightness_scene import CartesianBrightnessBand,CartesianBrightnessFit,qr_nnls
from field import FS,average
from motion import C
from native_scene_histories import brightness_model
from rigid_patch import RigidPatch
from rigid_brightness_scene import RigidBrightnessBand
from render_injected_point import fitted_state
from test_recovery_experiment import problem


def scenario(*,refine_points=True,free_centre=False,coordinated=False,fixed_calibration=False):
    ex=problem();points=np.array([[0.,0.,0.],[90.,-30.,0.]])
    gauge=np.array([[2.3,-4.1],[2.4,-4.2]]) if refine_points else None
    bands=[]
    for c,f in enumerate(ex.fields):
        f.y[:]=1.
        kind=CoordinatedPatch if coordinated else (CvRigidPatch if free_centre else CartesianPatch)
        kin=kind(f.data['row_time'],ex.midpoint,ex.state[0],ex.state[1],ex.rotation,ex.receiver)
        bands.append(CartesianBrightnessBand(f,ex.training[c],np.linspace(kin.time[0],kin.time[-1],5),
            kin,points,.3*c,ex.designs[c],position_gauge=gauge,channel=c,fixed_calibration=fixed_calibration,
            reference_rf_hz=ex.fields[3].metadata['rf_hz'],reference_transmitter=ex.transmitters[3]))
    model=CartesianBrightnessFit(bands)
    truth=np.r_[7.,-5.,.03,-.02,.17,([55.,-85.] if free_centre else []),.2,-.5,([.13,-.23] if refine_points else []),
        ([] if fixed_calibration else [.11,-.21,.18,-.07])]
    return ex,model,truth


def independent_echo(model,p,band,coefficients):
    """Direct physical distance and numerical symbol quadrature, no model responses."""
    b=model.bands[band];f=b.field;d=f.data;kin=b.kinematics;dt=kin.time-kin.epoch
    beta,phi=model.physical_calibration(p);out=np.zeros_like(d['value'])
    u,w=leggauss(24);u,w=u/2,w/2;angle=np.radians(p[4])*dt
    for j,original in enumerate(b.points):
        q=original.copy()
        m=b.motion_size
        if j and b.position_gauge is not None:q[:2]+=p[m+len(b.points)+2*(j-1):m+len(b.points)+2*j]
        origin=np.r_[p[5:7],0.] if m==7 else np.zeros(3)
        q+=origin
        offset=np.c_[np.cos(angle)*q[0]-np.sin(angle)*q[1],np.sin(angle)*q[0]+np.cos(angle)*q[1],np.full(len(dt),q[2])]
        centre_position=kin.position_enu+np.r_[p[:2],0.]-origin
        centre_velocity=kin.velocity_enu+np.r_[p[2:4],0.]-np.radians(p[4])*np.array([-origin[1],origin[0],0.])
        drift=dt[:,None]*centre_velocity;rotated=np.broadcast_to(centre_velocity,drift.shape).copy()
        if isinstance(kin,CoordinatedPatch):
            theta=np.radians(p[4])*dt[:,None]*(u+.5)
            vx,vy,vz=centre_velocity
            drift=dt[:,None]*np.c_[np.cos(theta)@w*vx-np.sin(theta)@w*vy,np.sin(theta)@w*vx+np.cos(theta)@w*vy,np.full(len(dt),vz)]
            rotated=np.c_[np.cos(angle)*vx-np.sin(angle)*vy,np.sin(angle)*vx+np.cos(angle)*vy,np.full(len(dt),vz)]
        position=kin.receiver+(centre_position+drift+offset)@kin.ecef_to_enu
        velocity=(rotated+np.radians(p[4])*np.c_[-offset[:,1],offset[:,0],np.zeros(len(dt))])@kin.ecef_to_enu
        tx=d['transmitter_ecef_m'];rx=d['receiver_ecef_m']
        path=np.full(len(dt),-np.linalg.norm(tx-rx));rate=np.zeros(len(dt))
        for antenna in (tx,rx):
            vector=position-antenna;length=np.linalg.norm(vector,axis=1)
            path+=length;rate+=np.sum(vector*velocity,axis=1)/length
        omega=-f.k*.001024*rate
        moments=np.exp(1j*omega[:,None]*u)
        symbol=moments@w-1j*d['centre_omega']*(moments@(u*w))
        lag=f.offsets[None,:]-(path-d['centre_path_m'])[:,None]*FS/C
        carrier=d.get('carrier_reference_path_m',d['centre_path_m'])
        strength=b.amplitude_basis@coefficients[j]
        raw=strength[:,None]*symbol[:,None]*f.pulse(lag)*np.exp(-1j*f.k*(path-carrier)[:,None]+1j*(beta[band]+phi[j]))
        out+=average(raw,d['row_information'],f.metadata['block_rows'])[0]
    return f.apply(out)


class CartesianBrightnessSceneTests(unittest.TestCase):
    def test_fixed_physical_calibration_cannot_absorb_relative_band_phase_when_position_moves(self):
        _,model,p=scenario(fixed_calibration=True)
        beta,phi=model.physical_calibration(p)
        np.testing.assert_array_equal(beta,[b.band_phase for b in model.bands])
        shifted=p.copy();shifted[:2]+=[4.,-3.]
        np.testing.assert_array_equal(model.physical_calibration(shifted)[0],beta)
        for b in model.bands:
            coefficients=(1+.2*np.sin(np.arange(10)+b.channel)).reshape(2,5)
            response,_=b.responses(p,False)
            value=sum(b.reduced(h[:,:,None]*b.amplitude_basis[:,None,:])@a for h,a in zip(response,coefficients))
            np.testing.assert_allclose(value.reshape(b.field.y.shape),independent_echo(model,p,b.channel,coefficients),rtol=2e-7,atol=2e-4)
            direction=np.sin(np.arange(len(p))+.3)*.03;step=2e-5
            _,derivative=b.responses(p)
            upper,_=b.responses(p+step*direction,False);lower,_=b.responses(p-step*direction,False)
            for u,l,d in zip(upper,lower,derivative):
                self.assertLess(np.linalg.norm((u-l)/(2*step)-d@direction)/np.linalg.norm(d@direction),5e-4)

    def test_coordinated_turn_state_and_derivatives_match_independent_time_quadrature(self):
        ex,model,p=scenario(coordinated=True,fixed_calibration=True)
        for b in model.bands:
            coefficients=(1+.2*np.sin(np.arange(10)+b.channel)).reshape(2,5)
            raw,derivative=b.responses(p)
            value=sum(b.reduced(h[:,:,None]*b.amplitude_basis[:,None,:])@a for h,a in zip(raw,coefficients))
            expected=independent_echo(model,p,b.channel,coefficients)
            self.assertLess(np.linalg.norm(value.reshape(expected.shape)-expected)/np.linalg.norm(expected),2e-7)
        kin=model.bands[0].kinematics;q=np.array([80.,-25.,0.])
        for spin in [0.,1e-9,.23,-.2]:
            at=p.copy();at[4]=spin;pos,vel,dp,dv=kin.point_state(at,q,derivatives=True,origin=ex.receiver)
            for j,step in enumerate([1e-3,1e-3,1e-5,1e-5,1e-5]):
                delta=np.eye(len(p))[j]*step
                u=kin.point_state(at+delta,q,origin=ex.receiver);l=kin.point_state(at-delta,q,origin=ex.receiver)
                np.testing.assert_allclose((u[0]-l[0])/(2*step),dp[:,:,j],rtol=2e-6,atol=1e-6)
                np.testing.assert_allclose((u[1]-l[1])/(2*step),dv[:,:,j],rtol=2e-6,atol=1e-8)
        at=p.copy();at[4]=0.
        straight=CartesianPatch(kin.time,kin.epoch,kin.position_enu,kin.velocity_enu,kin.ecef_to_enu,kin.receiver)
        for a,b in zip(kin.point_state(at,q,origin=ex.receiver),straight.point_state(at,q,origin=ex.receiver)):
            np.testing.assert_allclose(a,b,rtol=0,atol=2e-12)

    def test_calibrated_coordinated_turn_recovers_physical_motion_from_independent_echoes(self):
        ex,model,p=scenario(coordinated=True,fixed_calibration=True,refine_points=False)
        for b in model.bands:
            coefficient=(1+.2*np.sin(np.arange(10)+b.channel)).reshape(2,5)
            b.field.y[:]=independent_echo(model,p,b.channel,coefficient)
            b.__init__(b.field,np.ones(len(b.field.time),bool),b.knots,b.kinematics,b.points,b.band_phase,b.design,
                channel=b.channel,fixed_calibration=True,reference_rf_hz=ex.fields[3].metadata['rf_hz'],reference_transmitter=ex.transmitters[3])
        model=CartesianBrightnessFit(model.bands);seed=np.zeros(model.size);seed[4]=.1
        fitted=model.fit(seed,max_iterations=250)
        fitted=model.polish(fitted['parameters'],max_iterations=120)
        self.assertLess(fitted['objective'],1e-12)
        self.assertLess(np.linalg.norm(np.asarray(fitted['parameters'])[:2]-p[:2]),.001)
        self.assertLess(np.linalg.norm(np.asarray(fitted['parameters'])[2:4]-p[2:4]),1e-5)
        self.assertLess(abs(fitted['parameters'][4]-p[4]),1e-6)
        b=model.bands[0];kin=b.kinematics;at=np.asarray(fitted['parameters'])
        physical=np.array([kin.position_enu+np.r_[at[:2],0.],kin.velocity_enu+np.r_[at[2:4],0.],np.zeros(3)])
        physical[2,:2]=np.radians(at[4])*np.array([-physical[1,1],physical[1,0]])
        exported=fitted_state(kin.time,physical,kin.epoch,kin.ecef_to_enu,kin.receiver,{'motion_model':'coordinated-turn'})
        for q in b.points:
            expected=kin.point_state(at,q)
            np.testing.assert_allclose(exported['position_ecef_m']+exported['body_to_ecef']@q,expected[0],rtol=0,atol=2e-9)
            np.testing.assert_allclose(exported['velocity_ecef_mps']+exported['frame_velocity']@q,expected[1],rtol=0,atol=2e-12)

    def test_cartesian_state_orientation_and_midpoint_gauge_derivatives(self):
        ex,model,p=scenario();b=model.bands[0];kin=b.kinematics;q=b.points[1]
        state=kin.point_state(p,q,derivatives=True,origin=ex.receiver)
        for j,step in enumerate([1e-3,1e-3,1e-5,1e-5,1e-5]):
            d=np.eye(len(p))[j]*step
            upper=kin.point_state(p+d,q,origin=ex.receiver);lower=kin.point_state(p-d,q,origin=ex.receiver)
            np.testing.assert_allclose((upper[0]-lower[0])/(2*step),state[2][:,:,j],rtol=2e-6,atol=1e-6)
            np.testing.assert_allclose((upper[1]-lower[1])/(2*step),state[3][:,:,j],rtol=2e-6,atol=1e-8)
        frame,rate=kin.orientation(p)
        np.testing.assert_allclose(frame@q,state[0]-(kin.position_enu+np.r_[p[:2],0.]+kin.dt[:,None]*(kin.velocity_enu+np.r_[p[2:4],0.]))@ex.rotation,atol=2e-12)
        np.testing.assert_allclose(rate@q,state[1]-(kin.velocity_enu+np.r_[p[2:4],0.])@ex.rotation,atol=2e-14)
        origin=kin.point_state(p,[0.,0.,0.],origin=ex.receiver)
        other=p.copy();other[4]=-.4
        for actual,expected in zip(origin,kin.point_state(other,[0.,0.,0.],origin=ex.receiver)):
            np.testing.assert_array_equal(actual,expected)
        for tx in ex.transmitters:
            _,g=kin.midpoint_path_change(p,tx)
            for j in range(2):
                step=np.eye(len(p))[j]*.001
                finite=(kin.midpoint_path_change(p+step,tx)[0]-kin.midpoint_path_change(p-step,tx)[0])/.002
                self.assertAlmostEqual(finite,g[j],delta=1e-8)

    def test_tall_qr_nnls_preserves_solution_prediction_and_optimality(self):
        rng=np.random.default_rng(91);a=rng.normal(size=(401,23));a[:,1]=a[:,0]+.02*a[:,1]
        y=a@rng.uniform(-.2,1.,a.shape[1])+rng.normal(size=len(a))*.1
        expected=nnls(a,y,maxiter=1000)[0];actual=qr_nnls(a,y)
        np.testing.assert_allclose(a@actual,a@expected,atol=2e-12)
        np.testing.assert_allclose(actual,expected,atol=1e-11)
        gradient=a.T@(a@actual-y)
        self.assertLess(np.max(abs(gradient[actual>0])),1e-10)
        self.assertGreater(np.min(gradient[actual==0]),-1e-10)

    def test_responses_and_profiled_gradients_include_all_constant_phase_gauges(self):
        _,model,p=scenario();direction=np.sin(np.arange(len(p))+.3)*.03;h=2e-5
        for b in model.bands:
            raw,change=b.responses(p)
            upper,_=b.responses(p+h*direction,False);lower,_=b.responses(p-h*direction,False)
            for j in range(len(raw)):
                finite=(upper[j]-lower[j])/(2*h)
                self.assertLess(np.linalg.norm(finite-change[j]@direction)/np.linalg.norm(finite),5e-4)
            coefficients=(1+.2*np.sin(np.arange(10)+b.channel)).reshape(2,5)
            b.field.y[:]=independent_echo(model,p,b.channel,coefficients)
            # Refresh data/brightness projection after independently generating the observations.
            b.__init__(b.field,b.training,b.knots,b.kinematics,b.points,b.band_phase,b.design,
                position_gauge=b.position_gauge,channel=b.channel)
        model=CartesianBrightnessFit(model.bands)
        value,_=model.objective(p)
        self.assertLess(value,2e-13)
        at=p+.02*direction;value,g=model.objective(at)
        finite=(model.objective(at+h*direction)[0]-model.objective(at-h*direction)[0])/(2*h)
        self.assertAlmostEqual(finite,g@direction,delta=2e-7+abs(finite)*5e-4)

    def test_synthetic_cartesian_cv_is_recovered_without_truth_seed(self):
        _,model,p=scenario(refine_points=False)
        for b in model.bands:
            coefficients=(1+.2*np.sin(np.arange(10)+b.channel)).reshape(2,5)
            b.field.y[:]=independent_echo(model,p,b.channel,coefficients)
            b.__init__(b.field,b.training,b.knots,b.kinematics,b.points,b.band_phase,b.design,channel=b.channel)
        model=CartesianBrightnessFit(model.bands)
        seed=np.zeros(len(p));seed[4]=.1
        fit=model.fit(seed,max_iterations=300)
        self.assertLess(fit['objective'],1e-9)
        error=np.asarray(fit['parameters'])-p
        self.assertLess(np.linalg.norm(error[:2]),.05)
        self.assertLess(np.linalg.norm(error[2:4]),.002)
        self.assertLess(abs(error[4]),.001)
        for b in model.bands:self.assertGreater(b.prediction(fit['parameters'])['explained_fraction'],1-1e-8)

    def test_physical_export_matches_ordinary_rigid_model_without_phase_alignment(self):
        ex,model,p=scenario();beta,phi=model.physical_calibration(p)
        points=model.bands[0].points.copy();points[1:,:2]+=p[7:9].reshape(-1,2)
        physical=np.zeros(7);physical[3]=p[4];physical[5:]=phi
        for c,b in enumerate(model.bands):
            kin=b.kinematics;dt=kin.dt
            position=ex.receiver+(kin.position_enu+np.r_[p[:2],0.]+dt[:,None]*(kin.velocity_enu+np.r_[p[2:4],0.]))@ex.rotation
            velocity=np.broadcast_to((kin.velocity_enu+np.r_[p[2:4],0.])@ex.rotation,position.shape)
            exported=RigidPatch(kin.time,kin.epoch,position,velocity,ex.rotation.T,ex.transmitters[2],ex.receiver)
            ordinary=RigidBrightnessBand(b.field,b.training,b.knots,exported,points,beta[c],b.design)
            original,_=b.responses(p,False);reconstructed,_=ordinary.responses(physical,False)
            for x,y in zip(original,reconstructed):
                self.assertLess(np.linalg.norm(x-y)/np.linalg.norm(x),2e-8)
        exported=brightness_model(model.bands,p,physical_phases=(beta,phi))
        np.testing.assert_array_equal(exported['point_phase_rad'],phi)
        np.testing.assert_array_equal([b['band_phase_rad'] for b in exported['bands']],beta)

    def test_unknown_rotation_centre_has_correct_motion_and_response_derivatives(self):
        ex,model,p=scenario(free_centre=True);kin=model.bands[0].kinematics;q=model.bands[0].points[1]
        state=kin.point_state(p,q,derivatives=True,origin=ex.receiver)
        physical,origin=kin.centre_state(p)
        angle=np.radians(p[4])*kin.dt;point=q+origin
        offset=np.c_[np.cos(angle)*point[0]-np.sin(angle)*point[1],np.sin(angle)*point[0]+np.cos(angle)*point[1],np.full(len(angle),point[2])]
        np.testing.assert_allclose(state[0],(physical[0]+kin.dt[:,None]*physical[1]+offset)@ex.rotation,atol=3e-12)
        for j in range(7):
            step=np.eye(len(p))[j]*(1e-3 if j in (0,1,5,6) else 1e-5)
            upper=kin.point_state(p+step,q,origin=ex.receiver);lower=kin.point_state(p-step,q,origin=ex.receiver)
            for c in (0,1):
                np.testing.assert_allclose((upper[c]-lower[c])/(2*step[j]),state[c+2][:,:,j],rtol=3e-5,atol=8e-7 if c==0 else 1e-8)
        direction=np.sin(np.arange(len(p))+.3)*.03;step=2e-5
        for b in model.bands:
            _,change=b.responses(p);upper,_=b.responses(p+step*direction,False);lower,_=b.responses(p-step*direction,False)
            for u,l,d in zip(upper,lower,change):
                finite=(u-l)/(2*step)
                self.assertLess(np.linalg.norm(finite-d@direction)/np.linalg.norm(finite),5e-4)

    def test_cv_centre_synthetic_echoes_recover_anchor_motion_and_centre_from_coarse_seed(self):
        _,model,p=scenario(refine_points=False,free_centre=True)
        for b in model.bands:
            coefficients=(1+.2*np.sin(np.arange(10)+b.channel)).reshape(2,5)
            b.field.y[:]=independent_echo(model,p,b.channel,coefficients)
            b.__init__(b.field,b.training,b.knots,b.kinematics,b.points,b.band_phase,b.design,channel=b.channel)
        model=CartesianBrightnessFit(model.bands)
        self.assertLess(model.objective(p)[0],2e-13)
        scale,_=model.settings();direction=np.sin(np.arange(model.size)+.4)*scale
        at=p+.002*direction;step=1e-5
        _,gradient=model.objective(at)
        finite=(model.objective(at+step*direction)[0]-model.objective(at-step*direction)[0])/(2*step)
        self.assertAlmostEqual(finite,gradient@direction,delta=2e-7+abs(finite)*5e-4)
        seed=np.zeros(len(p));seed[4]=.1
        fit=model.fit(seed,max_iterations=600)
        fit=model.polish(fit['parameters'],max_iterations=120)
        self.assertAlmostEqual(fit['objective'],model.objective(np.asarray(fit['parameters']))[0],delta=1e-15)
        _,limits=model.settings()
        self.assertTrue(np.all(abs(fit['parameters'])<=limits+1e-8))
        self.assertLess(fit['objective'],1e-9)
        error=np.asarray(fit['parameters'])-p
        self.assertLess(np.linalg.norm(error[:2]),.1)
        self.assertLess(np.linalg.norm(error[2:4]),.002)
        self.assertLess(abs(error[4]),.001)
        self.assertLess(np.linalg.norm(error[5:7]),.2)
        for b in model.bands:self.assertGreater(b.prediction(fit['parameters'])['explained_fraction'],1-1e-8)

    def test_free_centre_export_has_same_physical_waveforms_and_zero_spin_null(self):
        ex,model,p=scenario(free_centre=True);beta,phi=model.physical_calibration(p);m=7;n=2
        for c,b in enumerate(model.bands):
            physical,origin=b.kinematics.centre_state(p);dt=b.kinematics.dt
            points=b.points.copy();points[1:,:2]+=p[m+n:m+n+2].reshape(-1,2);points+=origin
            centre=ex.receiver+(physical[0]+dt[:,None]*physical[1])@ex.rotation
            velocity=np.broadcast_to(physical[1]@ex.rotation,centre.shape)
            kin=RigidPatch(b.kinematics.time,ex.midpoint,centre,velocity,ex.rotation.T,ex.transmitters[2],ex.receiver)
            ordinary=RigidBrightnessBand(b.field,b.training,b.knots,kin,points,beta[c],b.design)
            exported=np.r_[0.,0.,0.,p[4],0.,phi]
            expected,_=b.responses(p,False);actual,_=ordinary.responses(exported,False)
            for x,y in zip(expected,actual):self.assertLess(np.linalg.norm(x-y)/np.linalg.norm(x),2e-8)
        zero=p.copy();zero[4]=0.
        state=model.bands[0].kinematics.point_state(zero,[0.,0.,0.],derivatives=True,origin=ex.receiver)
        np.testing.assert_array_equal(state[2][:,:,5:],0.)
        np.testing.assert_array_equal(state[3][:,:,5:],0.)


if __name__=='__main__':unittest.main()
