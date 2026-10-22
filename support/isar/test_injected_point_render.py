import unittest
import numpy as np
from render_injected_point import cartesian_state,fitted_state,brightness_template
from test_recovery_experiment import problem


class InjectedPointRenderTests(unittest.TestCase):
    def test_real_brightness_template_preserves_phase_and_known_matched_filter_gain(self):
        model=dict(point_phase_rad=[.7],bands=[dict(knots_s=[0.,1.,2.,3.],coefficients=[[0.,0.,2.,2.]],band_phase_rad=-.2)])
        amplitude,gain=brightness_template(model,0,0,np.arange(4.))
        np.testing.assert_array_equal(amplitude,[0.,0.,2.,2.])
        self.assertAlmostEqual(np.angle(gain),.5)
        signal=gain*amplitude
        # Independent white-noise matched-filter calculation: temporal
        # weights belong once in the numerator and squared in its variance.
        uniform=abs(signal.sum())**2/4
        matched=abs(amplitude@signal)**2/(amplitude@amplitude)
        self.assertAlmostEqual(matched/uniform,2.)
        with self.assertRaises(ValueError):brightness_template(model,0,0,np.array([-1.]))

    def test_constant_velocity_rigid_render_retains_constant_translation_direction(self):
        ex=problem();state=np.array(ex.state);state[2]=0.;t=np.array([6.,8.,10.])
        direction=ex.rotation.T@np.array([.4,-.2,0.])
        fit=dict(motion_model='rigid-patch',reference_transmitter_ecef_m=ex.transmitters[2],
            rigid_parameters=[.07,0.,0.,0.,0.],translation_direction_ecef=direction)
        result=fitted_state(t,state,8.,ex.rotation,ex.receiver,fit)
        np.testing.assert_allclose(result['velocity_ecef_mps'],np.broadcast_to(state[1]@ex.rotation+.07*direction,(3,3)),atol=1e-12)
        expected=ex.receiver+(state[0]+(t-8.)[:,None]*state[1])@ex.rotation+(t-8.)[:,None]*.07*direction
        np.testing.assert_array_equal(result['position_ecef_m'],expected)

    def test_rigid_fitted_motion_preserves_physical_velocity_and_fixed_midpoint_axes(self):
        ex=problem();t=np.array([6.,8.,10.]);q=np.array([37.,-59.,0.])
        fit=dict(motion_model='rigid-patch',reference_transmitter_ecef_m=ex.transmitters[2],
            rigid_parameters=[.03,.002,.0003,.17,.003])
        def location(time):
            s=fitted_state(time,ex.state,8.,ex.rotation,ex.receiver,fit)
            return s['position_ecef_m']+s['body_to_ecef']@q
        s=fitted_state(t,ex.state,8.,ex.rotation,ex.receiver,fit)
        finite=(location(t+.001)-location(t-.001))/.002
        np.testing.assert_allclose(finite,s['velocity_ecef_mps']+s['frame_velocity']@q,atol=5e-7)
        np.testing.assert_array_equal(s['body_to_ecef'][1],ex.rotation.T)
        with self.assertRaises(ValueError):fitted_state(t,ex.state,8.,ex.rotation,ex.receiver,fit,.2)

    def test_render_motion_matches_inference_positions_velocities_and_fixed_ne_axes(self):
        ex=problem();p=np.array([.02,-.03,.04,-.02,.01,.005])
        state=np.array(ex.state)
        for i in range(3):state[i,:2]+=p[2*i:2*i+2]*np.array([100.,100.,.5,.5,.02,.02])[2*i:2*i+2]
        for c,f in enumerate(ex.fields):
            expected,v,_,_=ex.kinematics(p,c,np.zeros(3))
            actual=cartesian_state(f.data['row_time'],state,ex.midpoint,ex.rotation,ex.receiver)
            np.testing.assert_array_equal(actual['position_ecef_m'],expected)
            np.testing.assert_array_equal(actual['velocity_ecef_mps'],v)
            np.testing.assert_array_equal(actual['frame_velocity'],0.)
            np.testing.assert_array_equal(actual['body_to_ecef'][0],ex.rotation.T)

    def test_heading_sign_and_rotating_point_velocity_are_physical(self):
        ex=problem();time=np.array([6.,8.,10.]);rate=.2;point=np.array([0.,150.,0.])
        def location(t):
            s=cartesian_state(t,ex.state,8.,ex.rotation,ex.receiver,rate)
            return s['position_ecef_m']+s['body_to_ecef']@point
        s=cartesian_state(time,ex.state,8.,ex.rotation,ex.receiver,rate)
        numerical=(location(time+.001)-location(time-.001))/.002
        expected=s['velocity_ecef_mps']+s['frame_velocity']@point
        np.testing.assert_allclose(numerical,expected,atol=5e-7)
        # A northward arm turns east under increasing maritime heading.
        velocity_enu=(s['frame_velocity'][1]@point)@ex.rotation.T
        self.assertGreater(velocity_enu[0],0.)
        np.testing.assert_allclose(s['position_ecef_m'],cartesian_state(time,ex.state,8.,ex.rotation,ex.receiver)['position_ecef_m'])


if __name__=='__main__':unittest.main()
