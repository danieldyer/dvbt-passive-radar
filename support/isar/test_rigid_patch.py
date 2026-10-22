import json
import unittest

import numpy as np

from field import Field
from motion import excess_path
from rigid_patch import RigidPatch,PatchBand,fit


def scenario(rf_hz=219.5e6,band_gain=1.,point_shift=None,point_amplitude=None):
    time=np.linspace(-15.,15.,121);rx=np.array([0.,0.,10.]);tx=np.array([30000.,1000.,700.])
    origin=np.array([-5000.,-6500.,10.]);velocity=np.array([-2.,4.,0.])
    pos=origin+time[:,None]*velocity
    kin=RigidPatch(time,0.,pos,np.broadcast_to(velocity,pos.shape),np.eye(3),tx,rx)
    path=excess_path(np.broadcast_to(origin,pos.shape),tx,rx)[0]
    samples=np.sinc(.81*np.arange(-64,65))
    meta=dict(rf_hz=rf_hz,block_rows=1)
    data=dict(time=time,row_time=time,information=np.ones(len(time)),row_information=np.ones(len(time)),
        value=np.zeros((len(time),17),complex),offset_samples=np.arange(-8,9),centre_path_m=path,
        centre_omega=np.zeros(len(time)),centre_ecef_m=np.broadcast_to(origin,pos.shape).copy(),
        centre_velocity_ecef_mps=np.zeros_like(pos),body_to_ecef=np.broadcast_to(np.eye(3),(len(time),3,3)),
        transmitter_ecef_m=tx,receiver_ecef_m=rx,metadata_json=json.dumps(meta))
    field=Field(data,samples);points=np.array([[0.,0.,20.],[80.,25.,20.],[-70.,-30.,20.]])
    p=np.array([.035,.003,.0001,.16,.001]);raw=np.zeros_like(data['value'])
    # Separate explicit Cartesian rotation/path/symbol quadrature synthesis.
    dt=time;shift=p[0]*dt+p[1]*dt**2/2+p[2]*dt**3/6
    rate=p[0]+p[1]*dt+p[2]*dt**2/2;theta=np.radians(p[3]*dt+p[4]*dt**2/2);td=np.radians(p[3]+p[4]*dt)
    true_points=points if point_shift is None else points+point_shift
    amplitudes=np.ones(3) if point_amplitude is None else np.asarray(point_amplitude)
    for point,gain,magnitude in zip(true_points,[1+.3j,-.4+.8j,.7-.2j],amplitudes):
        e,n,z=point;c,s=np.cos(theta),np.sin(theta)
        d=np.c_[c*e-s*n,s*e+c*n,np.full(len(time),z)]
        vd=td[:,None]*np.c_[-s*e-c*n,c*e-s*n,np.zeros(len(time))]
        actual=pos+shift[:,None]*kin.direction+d
        speed=velocity+rate[:,None]*kin.direction+shift[:,None]*kin.direction_rate+vd
        length,gradient=excess_path(actual,tx,rx);omega=-field.k*.001024*np.sum(gradient*speed,axis=1)
        nodes=(np.arange(2048)+.5)/2048-.5
        m0=np.mean(np.exp(1j*omega[:,None]*nodes),axis=1)
        pulse=field.pulse(field.offsets[None,:]-(length-path)[:,None]*8e6/299792458.)
        raw+=magnitude*band_gain*gain*pulse*m0[:,None]*np.exp(-1j*field.k*(length-path)[:,None])
    static=(np.arange(17)*.0001+.0003j)[None,:]
    field=Field({**data,'value':raw+static},samples)
    band=PatchBand(field,kin,points,np.arange(len(time))%4!=3,tx,rx)
    return kin,band,p


class RigidPatchTests(unittest.TestCase):
    def test_relative_coordinates_retain_sub_ecef_ulp_motion_increments(self):
        original,_,p=scenario();earth=np.array([-4200000.,2500000.,-3950000.])
        kin=RigidPatch(original.time,0.,earth+original.position,original.velocity,
            np.eye(3),earth+original.reference_transmitter,earth+original.receiver)
        origin=np.broadcast_to(earth+np.array([-5000.,-6500.,10.]),kin.position.shape)
        q=np.array([90.,-24.,20.]);position,velocity,dp,dv=kin.point_state(p,q,derivatives=True,origin=origin)
        absolute=kin.point_state(p,q)
        np.testing.assert_allclose(absolute[0]-origin,position,atol=2e-9,rtol=0)
        np.testing.assert_array_equal(absolute[1],velocity)
        h=1e-10;delta=np.array([h,0.,0.,0.,0.])
        finite=(kin.point_state(p+delta,q,origin=origin)[0]-kin.point_state(p-delta,q,origin=origin)[0])/(2*h)
        self.assertLess(np.linalg.norm(finite-dp[:,:,0])/np.linalg.norm(dp[:,:,0]),2e-5)

    def test_constant_cartesian_translation_keeps_every_fixed_point_at_constant_velocity(self):
        original,_,_=scenario();direction=np.array([.4,-.2,0.]);base=np.array([-2.,4.,0.])
        def make(t):
            return RigidPatch(t,0.,np.array([-5000.,-6500.,10.])+t[:,None]*base,
                np.broadcast_to(base,(len(t),3)),np.eye(3),original.reference_transmitter,original.receiver,
                translation_direction=direction)
        t=original.time;p=np.array([.07,0.,0.,0.,0.]);q=np.array([80.,-25.,0.])
        kin=make(t);position,velocity,dp,dv=kin.point_state(p,q,derivatives=True)
        np.testing.assert_allclose(velocity,np.broadcast_to(base+p[0]*direction,velocity.shape),atol=1e-14)
        np.testing.assert_allclose((make(t+.001).point_state(p,q)[0]-make(t-.001).point_state(p,q)[0])/.002,velocity,atol=1e-9)
        np.testing.assert_allclose(dp[:,:,0],t[:,None]*direction,atol=1e-14)
        np.testing.assert_allclose(dv[:,:,0],np.broadcast_to(direction,velocity.shape),atol=1e-14)

    def test_moving_reference_frame_keeps_time_and_parameter_derivatives(self):
        original,b,p=scenario();point=np.array([90.,-24.,20.])
        def make(t):
            angle=.005*t+.00001*t*t;rate=.005+.00002*t;c,s=np.cos(angle),np.sin(angle)
            frame=np.zeros((len(t),3,3));frame[:,0,0]=frame[:,1,1]=c;frame[:,1,0]=s;frame[:,0,1]=-s;frame[:,2,2]=1
            fd=np.zeros_like(frame);fd[:,0,0]=fd[:,1,1]=-s*rate;fd[:,1,0]=c*rate;fd[:,0,1]=-c*rate
            return RigidPatch(t,0.,np.array([-5000.,-6500.,10.])+t[:,None]*np.array([-2.,4.,0.]),
                np.broadcast_to([-2.,4.,0.],(len(t),3)),np.eye(3),original.reference_transmitter,original.receiver,frame,fd)
        kin=make(original.time);pos,vel,dp,dv=kin.point_state(p,point,derivatives=True)
        h=1e-4
        finite=(make(original.time+h).point_state(p,point)[0]-make(original.time-h).point_state(p,point)[0])/(2*h)
        np.testing.assert_allclose(finite,vel,atol=1e-7,rtol=1e-7)
        for j in (3,4):
            delta=np.eye(5)[j]*1e-6;a=kin.point_state(p+delta,point);c=kin.point_state(p-delta,point)
            np.testing.assert_allclose((a[0]-c[0])/2e-6,dp[:,:,j],rtol=2e-5,atol=2e-6)
            np.testing.assert_allclose((a[1]-c[1])/2e-6,dv[:,:,j],rtol=2e-5,atol=2e-6)

    def test_kinematic_derivatives_of_position_and_velocity(self):
        kin,b,p=scenario();point=np.array([90.,-24.,20.]);pos,vel,dp,dv=kin.point_state(p,point,derivatives=True)
        for j in range(5):
            delta=np.eye(5)[j]*1e-6
            a=kin.point_state(p+delta,point);c=kin.point_state(p-delta,point)
            np.testing.assert_allclose((a[0]-c[0])/2e-6,dp[:,:,j],rtol=2e-5,atol=2e-6)
            np.testing.assert_allclose((a[1]-c[1])/2e-6,dv[:,:,j],rtol=2e-5,atol=2e-6)

    def test_full_complex_variable_projection_jacobian(self):
        kin,b,p=scenario();p=p+np.array([.003,-.0002,0.,.003,.0001]);result=b.evaluate(p)
        scale=np.array([.01,.001,.0001,.01,.001])
        for j in range(5):
            h=scale[j]*1e-4;delta=np.eye(5)[j]*h
            finite=(b.evaluate(p+delta,derivatives=False)['residual']-b.evaluate(p-delta,derivatives=False)['residual'])/(2*h)
            self.assertLess(np.linalg.norm(finite-result['jacobian'][:,j])/np.linalg.norm(finite),2e-4)

    def test_independent_synthesis_recovers_common_motion_and_predicts_unused_data(self):
        kin,b,p=scenario()
        for offset in ([.01,-.001,0.,.02,0.],[-.01,.001,0.,-.02,0.]):
            found,r=fit([b],p+offset,bounds=([-1,-.1,-.01,-1,-.02],[1,.1,.01,1,.02]))
            np.testing.assert_allclose(found,p,rtol=0,atol=2e-7)
            self.assertLess(r['objective'],1e-15)
            self.assertLess(r['validation'][0]['unused_rss']/r['validation'][0]['unused_data_energy'],1e-15)


if __name__=='__main__':unittest.main()
