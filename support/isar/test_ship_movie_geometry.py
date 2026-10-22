import unittest
import numpy as np
from ship_coherent_movie import packet,viewport_grid,clip_chart,frame_times
from imaging import geometry_from_state
from test_continuous_ship_fit import fixture
from types import MethodType


class MovieGeometryTests(unittest.TestCase):
    def test_60x_playback_at_24fps_preserves_physical_aperture_cadence(self):
        t=frame_times(21.,586.,24,60.)
        np.testing.assert_array_equal(np.diff(t),2.5)
        self.assertEqual(len(t),227);self.assertEqual(t[-1],586.)
        self.assertAlmostEqual(1-2.5/30,11/12)
        self.assertAlmostEqual((t[-1]-t[0])/((len(t)-1)/24),60.)
        self.assertEqual(len(frame_times(10.,9.,24,60.)),0)
        with self.assertRaises(ValueError):frame_times(0.,1.,0,60.)

    def test_chart_preserves_physical_rays_velocities_and_carrier(self):
        b,m=fixture()
        def geometry(self,c,t,first,information,sign=1.):
            g,d=geometry_from_state(self.state(t),self.tx[c],self.rx,self.rf[c],first,information)
            g[:,5,3]=2*np.pi*self.rf[c]/299792458.;g[:,6,3]=c;g[:,7,3]=sign;g[:,8,3]=1.
            return g,d
        m.geometry=MethodType(geometry,m)
        a=dict(time_s=b.t,meta=dict(first_range_sample=160),information=np.ones(len(b.t)),carrier_reference_path_m=np.zeros(len(b.t)),moments=np.zeros((2,len(b.t),64),complex))
        angle=.61;c,s=np.cos(angle),np.sin(angle);chart=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        original,_=packet(m,[a]*5,[(3,0,len(b.t),1.)]);rotated,_=packet(m,[a]*5,[(3,0,len(b.t),1.)],chart)
        q=np.array([53.,-87.,0.]);old=chart@q
        for start in [2,6]:
            before=np.einsum('tij,i->tj',original[:,start:start+3,:3],old)
            after=np.einsum('tij,i->tj',rotated[:,start:start+3,:3],q)
            np.testing.assert_allclose(after,before,rtol=0,atol=1e-5)
        np.testing.assert_array_equal(original[:,9],rotated[:,9])
        np.testing.assert_array_equal(original[:,5:9,3],rotated[:,5:9,3])

    def test_camera_uses_declared_plane_height_and_following_extent(self):
        b,m=fixture();m.local=m.rotation.T;t=b.t[::100];state=m.state(t);origin=state['position_ecef_m'][0]
        camera=dict(origin_ecef_m=origin.tolist(),enu_to_ecef=m.local.tolist(),image_plane_height_m=0.,following_extent_m=[240.,440.],geographic_bounds_m=[-150.,150.,-250.,250.])
        chart=clip_chart(m,t,camera);_,_,transforms,followers=viewport_grid(m,t,camera,1.,chart)
        for i,(matrix,shift) in enumerate(transforms):
            np.testing.assert_allclose(shift,(state['position_ecef_m'][i]-origin)@m.local[:,:2],rtol=0,atol=1e-10)
            self.assertAlmostEqual(followers[i][1]-followers[i][0],240.);self.assertAlmostEqual(followers[i][3]-followers[i][2],440.)

if __name__=='__main__':unittest.main()
