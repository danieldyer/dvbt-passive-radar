from copy import deepcopy
import unittest

import numpy as np

from ais_point_arc import AisPointArc
from kinematic_point_arc import KinematicPointGeometry
from motion import excess_path
from smooth_echo_cluster import jerk_factor
from test_ais_point_arc import geometry
from test_physical_amplitude_arc import problem


def kinematic_geometry():
    old=geometry(False)
    assignments=[dict(site_id=0,transmitter_ecef_m=(old.receiver+p@old.rotation).tolist()) for p in old.local_transmitters]
    return KinematicPointGeometry(old.measured,np.tile(np.eye(3),(5,1,1)),_track(),
        30.,old.rotation,old.receiver,assignments,half_width_s=2.)


def _track():
    from ais import AisTrack
    t=np.array([0.,20.,40.,60.]);nan=np.full(4,np.nan)
    return AisTrack(t,np.radians([-37.81,-37.8098,-37.80955,-37.8093]),
        np.radians([144.98,144.9801,144.9803,144.9806]),nan,nan,nan,nan,nan,nan,None,5.)


class KinematicPointTests(unittest.TestCase):
    def test_motion_prior_standardization_and_zero_midpoint_gauge(self):
        g=kinematic_geometry();q=np.sin(np.arange(g.count))*.1
        coefficients=g.correction_coefficients(q)
        jerk=jerk_factor(g.curve)
        energy=0.
        for c in coefficients.T:
            energy+=(g.curve([0.],1)@c)[0]**2/.5**2+(g.curve([0.],2)@c)[0]**2/.1**2+np.linalg.norm(jerk@c)**2/.05**2
        self.assertAlmostEqual(energy,q[22:]@q[22:],places=11)
        zero=q.copy();zero[22:]=0
        np.testing.assert_allclose(g.trajectory(q,[0.])[0],g.trajectory(zero,[0.])[0],atol=1e-12)

    def test_cartesian_velocity_and_acceleration_are_exact_time_derivatives(self):
        g=kinematic_geometry();q=np.sin(np.arange(g.count))*.05
        t=np.array([-.8,-.3,.2,.7]);step=1e-3
        p,v,a=g.trajectory(q,t);plus=g.trajectory(q,t+step);minus=g.trajectory(q,t-step)
        np.testing.assert_allclose((plus[0]-minus[0])/(2*step),v,atol=3e-8,rtol=3e-8)
        np.testing.assert_allclose((plus[1]-minus[1])/(2*step),a,atol=3e-8,rtol=3e-8)

    def test_original_row_range_carrier_and_doppler_derivatives(self):
        old,_=problem();f=deepcopy(old.field);g=kinematic_geometry();channel=2
        tx,rx=g.antennas_at(np.zeros(g.count));dt=f.data['row_time']-old.midpoint
        local,velocity,_=g.trajectory(np.zeros(g.count),dt)
        centre=g.receiver+local@g.rotation;v=velocity@g.rotation
        transmitter=g.receiver+tx[channel]@g.rotation
        path,gradient=excess_path(centre,transmitter,g.receiver)
        f.data.update(centre_ecef_m=centre,centre_velocity_ecef_mps=v,centre_path_m=path,
            centre_omega=-f.k*.001024*np.sum(gradient*v,axis=1),transmitter_ecef_m=transmitter,receiver_ecef_m=g.receiver)
        b=AisPointArc(f,old.training,old.knots,g,channel,old.midpoint,stationary_paths=[])
        p=np.sin(np.arange(g.count+5))*.002;p[20]=.7
        direction=np.cos(np.arange(len(p)))*.01
        raw,jac=b.response(p)
        position,velocity,_=g.trajectory(p[:g.count],dt)
        tx,rx=g.antennas_at(p[:g.count])
        length=np.linalg.norm(position-tx[channel],axis=1)+np.linalg.norm(position-rx,axis=1)-np.linalg.norm(tx[channel]-rx)
        unit=(position-tx[channel])/np.linalg.norm(position-tx[channel],axis=1)[:,None]
        unit+=(position-rx)/np.linalg.norm(position-rx,axis=1)[:,None]
        omega=-f.k*.001024*np.sum(unit*velocity,axis=1)
        delta=length-f.data['centre_path_m']
        midpoint=g.prediction(p[:g.count])[channel,0]
        rotation=np.exp(1j*(f.k*(midpoint-b.reference_path)+p[g.count+channel]))
        expected=f.path_response(delta,omega)*rotation
        np.testing.assert_allclose(raw,expected,atol=2e-8,rtol=2e-8)
        # Smaller differences of weak antenna columns lose precision when
        # Earth-sized ECEF transmitter coordinates are perturbed. Verify two
        # millimetre-scale probes instead of relaxing the derivative tolerance.
        for step in (.003,.01):
            finite=(b.response(p+step*direction)[0]-b.response(p-step*direction)[0])/(2*step)
            self.assertLess(np.linalg.norm(finite-jac@direction)/np.linalg.norm(finite),4e-5)


if __name__=='__main__':unittest.main()
