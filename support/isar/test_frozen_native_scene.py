import unittest
import numpy as np
from frozen_native_scene import component
from rigid_brightness_scene import RigidBrightnessBand
from rigid_patch import RigidPatch
from test_recovery_experiment import problem


class FrozenNativeSceneTests(unittest.TestCase):
    def test_frozen_competing_column_matches_original_native_scene_without_refitting(self):
        ex=problem();c=2;f=ex.fields[c];f.y[:]=1.;p,v,a=np.asarray(ex.state);t=f.data['row_time'];dt=t-ex.midpoint
        points=np.array([[0.,0.,0.],[35.,-60.,0.]]);knots=np.linspace(t[0],t[-1],5)
        kin=RigidPatch(t,ex.midpoint,ex.receiver+(p+dt[:,None]*v)@ex.rotation,np.broadcast_to(v@ex.rotation,(len(t),3)),
            ex.rotation.T,ex.transmitters[2],ex.receiver)
        band=RigidBrightnessBand(f,ex.training[c],knots,kin,points,.3,ex.designs[c])
        parameters=np.array([.02,-.001,.0002,.17,.003,.2,-.5]);coefficient=1+.2*np.sin(np.arange(10)).reshape(2,5)
        brightness=dict(point_phase_rad=parameters[5:],bands=[dict(rf_hz=other.metadata['rf_hz'],knots_s=knots,
            coefficients=coefficient,band_phase_rad=.3) for other in ex.fields])
        render=dict(coarse_state_enu=[p,v,a],midpoint_s=ex.midpoint,motion_model='rigid-patch',
            rigid_parameters=parameters[:5],reference_transmitter_ecef_m=ex.transmitters[2])
        raw,_=band.responses(parameters,False)
        for j in range(2):
            expected=band.reduced(raw[j][:,:,None]*(band.amplitude_basis@coefficient[j])[:,None,None])[:,0]
            actual=component(f,ex.rotation,ex.receiver,render,brightness,points,j)
            self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),2e-8)


if __name__=='__main__':unittest.main()
