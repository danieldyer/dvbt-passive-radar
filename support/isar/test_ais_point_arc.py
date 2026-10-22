from copy import deepcopy
import unittest

import numpy as np

from ais import AisTrack
from ais_point_arc import AisPointGeometry,AisPointArc
from motion import ecef,enu_basis,excess_path
from test_physical_amplitude_arc import problem


def geometry(follow=True):
    t=np.array([0.,20.,40.,60.]);nan=np.full(4,np.nan)
    track=AisTrack(t,np.radians([-37.81,-37.8098,-37.80955,-37.8093]),
        np.radians([144.98,144.9801,144.9803,144.9806]),nan,nan,nan,nan,nan,nan,None,5.)
    rx=ecef(np.radians(-37.8104),np.radians(144.9874),40.)
    rotation=enu_basis(np.radians(-37.8104),np.radians(144.9874)).T
    assignment=[dict(site_id=0,transmitter_ecef_m=(rx+np.array([30000.+c*50,5000.,500.])@rotation).tolist()) for c in range(5)]
    return AisPointGeometry(np.zeros((5,3)),np.tile(np.eye(3),(5,1,1)),track,30.,rotation,rx,
        assignment,follow_course_change=follow)


class AisPointArcTests(unittest.TestCase):
    def test_prescribed_lever_arm_rotation_and_exact_time_derivatives(self):
        for follow in (False,True):
            g=geometry(follow);q=np.zeros(g.count);q[18:]=[.3,-.4,.8,.2]
            time=np.linspace(-2,2,9);p,v,a=g.trajectory(q,time);step=1e-3
            plus=g.trajectory(q,time+step);minus=g.trajectory(q,time-step)
            np.testing.assert_allclose((plus[0]-minus[0])/(2*step),v,atol=2e-8,rtol=2e-8)
            np.testing.assert_allclose((plus[1]-minus[1])/(2*step),a,atol=2e-8,rtol=2e-8)
            atmid=g.trajectory(q,[0.])[0][0]-g.reference(q[-1]*2.)
            np.testing.assert_allclose(atmid,q[18:21]*g.nuisance_sigma[:3],atol=2e-12)
            displacement=p-g.reference.sample(time+q[-1]*2.)
            np.testing.assert_allclose(np.sum(displacement[:,:2]**2,axis=1),2500.,atol=2e-10)

    def test_excess_range_rate_and_curvature_match_independent_time_differences(self):
        g=geometry();q=np.zeros(g.count);q[18:]=[.1,-.2,.8,.2];tx,rx=g.antennas_at(q)
        def paths(t):
            p=g.trajectory(q,[t])[0][0]
            return np.linalg.norm(p-tx,axis=1)+np.linalg.norm(p-rx)-np.linalg.norm(tx-rx,axis=1)
        step=.03;expected=g.prediction(q)
        np.testing.assert_allclose(paths(0),expected[:,0],atol=2e-11)
        np.testing.assert_allclose((paths(step)-paths(-step))/(2*step),expected[:,1],atol=2e-7,rtol=2e-7)
        np.testing.assert_allclose((paths(step)-2*paths(0)+paths(-step))/step**2,expected[:,2],atol=2e-7,rtol=2e-7)

    def test_original_row_response_includes_geometry_phase_gauge_and_symbol_rate(self):
        old,_=problem();f=deepcopy(old.field);g=geometry();channel=2;tx,rx=g.antennas_at(np.zeros(g.count))
        time=f.data['row_time']-old.midpoint;p,v,_=g.trajectory(np.zeros(g.count),time)
        centre=g.receiver+p@g.rotation;velocity=v@g.rotation
        transmitter=g.receiver+tx[channel]@g.rotation
        path,gradient=excess_path(centre,transmitter,g.receiver)
        f.data.update(centre_ecef_m=centre,centre_velocity_ecef_mps=velocity,centre_path_m=path,
            centre_omega=-f.k*.001024*np.sum(gradient*velocity,axis=1),
            transmitter_ecef_m=transmitter,receiver_ecef_m=g.receiver)
        b=AisPointArc(f,old.training,old.knots,g,channel,old.midpoint,stationary_paths=[])
        truth=np.zeros(g.count+5);truth[:18]=np.sin(np.arange(18))*.001;truth[18:22]=[.03,-.02,.7,.1]
        truth[g.count+channel]=.4
        direction=np.cos(np.arange(len(truth)))*.01;step=5e-4
        raw,jac=b.response(truth)
        finite=(b.response(truth+step*direction)[0]-b.response(truth-step*direction)[0])/(2*step)
        self.assertLess(np.linalg.norm(finite-jac@direction)/np.linalg.norm(finite),4e-5)


if __name__=='__main__':unittest.main()
