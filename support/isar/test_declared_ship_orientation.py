import unittest
from types import SimpleNamespace
import numpy as np
from declared_ship_orientation import state,rotate_frame
from motion import enu_basis,ecef


class DeclaredOrientationTests(unittest.TestCase):
    def test_physical_angular_history_preserves_anchor_and_differentiates_frame(self):
        t=np.linspace(-15,15,301);frame=np.broadcast_to(enu_basis(*np.radians([-37.8,144.9])),(len(t),3,3))
        base=dict(body_to_ecef=frame,frame_velocity=np.zeros_like(frame),position_ecef_m=np.ones((len(t),3)))
        actual=rotate_frame(base,t,0.,-.13,.002)
        plus=rotate_frame(base,t+.001,0.,-.13,.002)['body_to_ecef']
        minus=rotate_frame(base,t-.001,0.,-.13,.002)['body_to_ecef']
        np.testing.assert_allclose(actual['frame_velocity'],(plus-minus)/.002,atol=2e-13,rtol=0)
        np.testing.assert_array_equal(actual['body_to_ecef'][150],frame[150])
        np.testing.assert_array_equal(actual['position_ecef_m'],base['position_ecef_m'])
        np.testing.assert_allclose(actual['body_to_ecef'].transpose(0,2,1)@actual['body_to_ecef'],np.broadcast_to(np.eye(3),frame.shape),atol=6e-16,rtol=0)
        self.assertIs(rotate_frame(base,t,0.),base)

    def test_reported_course_removes_old_fitted_heading_and_preserves_translation(self):
        lat,lon=np.radians([-37.8,144.9]);local=enu_basis(lat,lon);origin=ecef(lat,lon,5.);velocity=local@np.array([1.,3.,0.])
        def sample(seconds,**kwargs):
            seconds=np.atleast_1d(seconds);return dict(seconds=seconds,latitude=np.full(len(seconds),lat),longitude=np.full(len(seconds),lon),
                position_ecef_m=origin+(seconds-1000)[:,None]*velocity,velocity_ecef_mps=np.broadcast_to(velocity,(len(seconds),3)))
        track=SimpleNamespace(seconds=np.array([1000.,1157.,1600.]),epoch=1000.,course=np.radians([1.,2.,3.]),sample=sample)
        def motion(amplitude):
            def reference(t):
                t=np.atleast_1d(t);angle=amplitude*(np.sin(.1*t)-np.sin(15.7));c,s=np.cos(angle),np.sin(angle)
                rotation=np.zeros((len(t),3,3));rotation[:,0,0]=rotation[:,1,1]=c;rotation[:,0,1]=-s;rotation[:,1,0]=s;rotation[:,2,2]=1
                return dict(position_ecef_m=origin+t[:,None]*velocity,velocity_ecef_mps=np.broadcast_to(velocity,(len(t),3)),body_to_ecef=local@rotation,frame_velocity=np.zeros_like(rotation))
            return SimpleNamespace(state=reference,epoch=1000.,track=track)
        first,second=motion(.1),motion(.8);t=np.linspace(10,590,301)
        a,b=state(first,t,'reported-course'),state(second,t,'reported-course')
        np.testing.assert_allclose(a['body_to_ecef'],b['body_to_ecef'],atol=1e-14,rtol=0)
        np.testing.assert_array_equal(a['position_ecef_m'],first.state(t)['position_ecef_m']);np.testing.assert_array_equal(a['velocity_ecef_mps'],first.state(t)['velocity_ecef_mps'])
        finite=(state(first,t+.02,'reported-course')['body_to_ecef']-state(first,t-.02,'reported-course')['body_to_ecef'])/.04
        np.testing.assert_allclose(a['frame_velocity'],finite,atol=2e-12,rtol=0)
        np.testing.assert_allclose(state(first,np.array([157.]),'reported-course')['body_to_ecef'],first.state(np.array([157.]))['body_to_ecef'],atol=1e-14,rtol=0)
        fixed=state(first,t,'fixed');np.testing.assert_allclose(fixed['body_to_ecef'],np.broadcast_to(local,(len(t),3,3)),atol=1e-14,rtol=0)
        np.testing.assert_array_equal(fixed['frame_velocity'],np.zeros((len(t),3,3)))
        delayed=state(first,t,'reported-course',5.)
        np.testing.assert_array_equal(delayed['position_ecef_m'],a['position_ecef_m'])
        np.testing.assert_allclose(state(first,np.array([157.]),'reported-course',5.)['body_to_ecef'],first.state(np.array([157.]))['body_to_ecef'],atol=1e-14,rtol=0)
        finite=(state(first,t+.02,'reported-course',5.)['body_to_ecef']-state(first,t-.02,'reported-course',5.)['body_to_ecef'])/.04
        np.testing.assert_allclose(delayed['frame_velocity'],finite,atol=2e-12,rtol=0)

if __name__=='__main__':unittest.main()
