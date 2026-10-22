import unittest
from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
from field import Field
from full_path_point import PointBand,initial_curve,carrier_bin
from motion import enu_basis,excess_path
from test_rigid_motion import fixture


def problem(*,free_path_offset=False):
    original=fixture(3);d=dict(original.data);t=d['row_time'];w=d['row_information']
    d.update(time=t,information=w,value=np.zeros((len(t),17),complex))
    import json
    meta=json.loads(d['metadata_json']);meta['block_rows']=1;d['metadata_json']=json.dumps(meta)
    f=Field(d,original.pulse.samples);local=enu_basis(*np.radians([-37.87,144.93]));v=d['centre_velocity_ecef_mps'][0];origin=d['centre_ecef_m'][0]
    def state(t):
        return dict(position_ecef_m=origin+t[:,None]*v,velocity_ecef_mps=np.broadcast_to(v,(len(t),3)),
            body_to_ecef=np.broadcast_to(local,(len(t),3,3)),frame_velocity=np.zeros((len(t),3,3)))
    motion=SimpleNamespace(interval=[t[0],t[-1]],local=local,tx=np.array([fixture(c).data['transmitter_ecef_m'] for c in range(5)]),rx=d['receiver_ecef_m'],state=state)
    path,g=excess_path(d['centre_ecef_m'],motion.tx[3],motion.rx)
    grid,coef=initial_curve(t,path,np.sum(g*v,axis=1),motion.interval,4.)
    coordinate=dict(training=np.arange(len(t))%4!=3,native_time_s=(t[:,None]+np.linspace(-.0075,.0075,16)).ravel(),native_information=np.repeat(w/16,16))
    b=PointBand(f,coordinate,motion,np.array([0.,0.,0.]),grid,coef,8.,4.,free_path_offset=free_path_offset)
    return b


class FullPathPointTests(unittest.TestCase):
    def test_native_bin_derivative_and_zero_information(self):
        t=np.arange(20)*.02;v=np.linspace(-6,5,20);native=(t[:,None]+np.linspace(-.007,.007,16)).ravel();w=np.ones(len(native));w[:16]=0
        value,derivative=carrier_bin(t,v,native,w,4.)
        numeric=(carrier_bin(t,v+.0001,native,w,4.)[0]-carrier_bin(t,v-.0001,native,w,4.)[0])/.0002
        np.testing.assert_allclose(derivative,numeric,atol=2e-12,rtol=0);self.assertEqual(value[0],1);self.assertEqual(derivative[0],0)

    def test_profiled_physical_path_gradient_and_frozen_unused_prediction(self):
        b=problem();n=b.b.shape[1];truth=np.linspace(-.05,.07,n);phase=.4
        h=b.waveform(truth,3,False);amplitude=np.interp(b.t,b.amplitude_nodes,.8+.15*np.cos(b.amplitude_nodes*.3))
        background=b.field.root_information[:,None]*np.linspace(.1,.8,17)[None,:]*(.3+.2j)
        b.field.y=h*(amplitude*np.exp(1j*phase))[:,None]+background
        # Rebind data after independent stationary nuisance was added.
        from rigid_patch import profile_static
        for training in [True,False]:
            mask=b.training if training else ~b.training;y=profile_static(b.field.y,b.field.root_information,mask)*mask[:,None]
            b.data[training]=y;b.energy[training]=np.vdot(y,y).real
        at=truth*.5;check=b.evaluate(at,.3,3)
        p=np.r_[at,.3]
        for j in range(n+1):
            step=np.eye(n+1)[j]*.00005
            plus=b.evaluate((p+step)[:n],(p+step)[n],3)['cost'];minus=b.evaluate((p-step)[:n],(p-step)[n],3)['cost']
            self.assertLess(abs((plus-minus)/.0001-check['gradient'][j])/b.energy[True],2e-5)
        def objective(p):
            r=b.evaluate(p[:n],p[n],3);return r['cost']/r['energy'],r['gradient']/r['energy']
        result=minimize(objective,p,jac=True,method='L-BFGS-B',options=dict(maxiter=150,gtol=1e-9,ftol=1e-13))
        fitted=b.evaluate(result.x[:n],result.x[n],3)
        unused=b.evaluate(result.x[:n],result.x[n],3,training=False,derivatives=False,frozen_amplitude=fitted['amplitude'])
        self.assertLess(fitted['cost']/fitted['energy'],1e-8);self.assertLess(unused['cost']/unused['energy'],1e-8)

if __name__=='__main__':unittest.main()
