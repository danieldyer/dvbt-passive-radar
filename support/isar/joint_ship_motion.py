"""One continuous physical state and static calibration for final native BP."""
import json
from pathlib import Path
import numpy as np
from ais import read_csv,utc_seconds
from imaging import sites,motion_state,geometry_from_state
from motion import C,enu_basis,excess_path
from rigid_patch import RigidPatch
from ship_site_signature import assigned_rays
from spline_motion import SplineMotion,basis,evaluate_coefficients


class JointMotion:
    def __init__(self,path):
        self.path=Path(path);self.result=json.loads(self.path.read_text());self.model=self.result['base_model']
        self.epoch=utc_seconds(self.model['nominal_start_utc']);self.interval=list(self.result['interval_s']);self.motion_interval=tuple(self.interval);self.curve=SplineMotion(self.model)
        self.track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
        self.reference_tx,self.reference_rx,_=sites(Path(self.model['geometry_directory']))
        self.nominal_tx,self.rx=assigned_rays(Path('docs/data/calibration_20261016_sites'),'00002')
        self.tx=self.nominal_tx.copy();self.parameters=np.asarray(self.result['parameters']);counts=self.result['parameter_counts']
        n,nc=counts['path'],counts.get('cross',0);self.radial=self.parameters[:n];self.cross=self.parameters[n:n+nc]
        # Read old exploratory coefficients in the basis in which they were fitted.
        # Every new fit saves its explicit, correctly converted ENU basis.
        self.receiver_local=np.asarray(self.result.get('antenna_offset_basis_ecef',enu_basis(-37.81040454194318,144.98744506389454)))
        a=self.track.sample(np.array([self.epoch+157.]),max_gap_s=400.)
        self.local=enu_basis(a['latitude'][0],a['longitude'][0]);self.gauge_point=np.asarray(self.result['gauge_point_ecef_m'])
        if counts['antenna']:
            offsets=self.parameters[n+nc:n+nc+counts['antenna']].reshape(-1,3)@self.receiver_local.T
            self.tx+=offsets[self.result['transmitter_groups']]
        self.gains=np.asarray([complex(*z) for z in self.result['calibration_gains_real_imag']])
        if counts['phase']:self.gains[[0,1,2,4]]*=np.exp(1j*self.parameters[n+nc+counts['antenna']:])
        self.rf=np.array([177.5e6,184.5e6,191.625e6,219.5e6,226.5e6])
        delta=np.array([(excess_path(self.gauge_point[None],tx,self.rx)[0]-excess_path(self.gauge_point[None],old,self.rx)[0])[0]
            for tx,old in zip(self.tx,self.nominal_tx)])
        self.gains*=np.exp(2j*np.pi*self.rf/C*delta)

    def base(self,t):
        t=np.atleast_1d(t);return self.curve.apply(t,motion_state(self.track,self.epoch+t,self.model),self.reference_tx[3],self.reference_rx)

    def state(self,t):
        return self.state_at_basis(t,self.local)

    def state_at_basis(self,t,local):
        """Evaluate the declared radial/transverse coordinate basis exactly.

        Local likelihoods retain their aperture's tangent basis. A final
        whole-capture trajectory uses one fixed basis throughout instead.
        """
        t=np.atleast_1d(t);s=self.base(t)
        kin=RigidPatch(t,157.,s['position_ecef_m'],s['velocity_ecef_mps'],local,self.nominal_tx[3],self.rx)
        q,qd=kin.direction,kin.direction_rate
        path=evaluate_coefficients(t,self.motion_interval,self.result['path_spacing_seconds'],np.r_[self.radial,0.],anchor=157.)
        rate=evaluate_coefficients(t,self.motion_interval,self.result['path_spacing_seconds'],np.r_[self.radial,0.],derivative=1)
        position=s['position_ecef_m']+path[:,None]*q;velocity=s['velocity_ecef_mps']+path[:,None]*qd+rate[:,None]*q
        if len(self.cross):
            vector=np.cross(local[:,2],q);norm=np.linalg.norm(vector,axis=1);direction=vector/norm[:,None]
            vr=np.cross(local[:,2],qd);dr=(vr-direction*np.sum(direction*vr,axis=1)[:,None])/norm[:,None]
            offset=evaluate_coefficients(t,self.motion_interval,self.result['cross_spacing_seconds'],np.r_[self.cross,0.],anchor=157.)
            speed=evaluate_coefficients(t,self.motion_interval,self.result['cross_spacing_seconds'],np.r_[self.cross,0.],derivative=1)
            position+=offset[:,None]*direction;velocity+=offset[:,None]*dr+speed[:,None]*direction
        return dict(position_ecef_m=position,velocity_ecef_mps=velocity,body_to_ecef=s['body_to_ecef'],frame_velocity=s['frame_velocity'])

    def geometry(self,channel,t,first_sample,information,*,deramped=False,sign=1.):
        g,details=geometry_from_state(self.state(t),self.tx[channel],self.rx,self.rf[channel],first_sample,information)
        if deramped:g[:,9,1]=1.;g[:,9,2]=0.
        g[:,5,3]=2*np.pi*self.rf[channel]/C;g[:,6,3]=channel;g[:,7,3]=sign;g[:,8,3]=1.
        return g,details
