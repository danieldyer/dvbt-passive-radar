"""Reusable recorded-data windows for full-duration rigid-target autofocus.

Observation identities and known-waveform erasures are explicit. The raw
archive is memory mapped; only a single range crop/window is materialised.
The viewing geometry may be initialized from an earlier fitted curve, but
it is evaluated at the original acquisition epochs without time warps.
"""
import json
from pathlib import Path
import numpy as np

from ais import read_csv,utc_seconds
from cluster_phase import crop
from extract_full_aperture import load
from fixed_range_field import fixed_reference
from imaging import sites,motion_state,geometry_from_state
from midpoint_model import representative
from motion import C,enu_basis
from rigid_patch import RigidPatch,PatchBand
from spline_motion import SplineMotion
from waveform_kernels import read_kernels,bind_subset,KernelField
from report_measured_scatterer_phases import load_fields
from ship_site_signature import assigned_rays

J=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,0.]])

class Dataset:
    def __init__(self,model=Path('build/isar-20261015/full-capture-radar-motion.json'),assignment='00002'):
        self.model=json.loads(Path(model).read_text());self.epoch=utc_seconds(self.model['nominal_start_utc'])
        self.reference_tx,self.reference_rx,_=sites(Path(self.model['geometry_directory']))
        self.tx,self.rx=assigned_rays(Path('docs/data/calibration_20261016_sites'),assignment)
        self.track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
        self.records=load_fields(Path('build/isar-20261015/full-fields'),Path('docs/data/calibration_20261016_sites'))
        self.raw=[load(Path(f'build/isar-20261015/full-geographic-aperture/channel-{c}')) for c in range(5)]
        directory=Path('build/isar-20261017/waveform-kernels-full')
        _,self.kernel_records,self.samples=read_kernels(directory/'response.kernels');self.grid=np.load(directory/'rows.npz')
        self.assignment=assignment

    def state(self,t,model=None):
        model=self.model if model is None else model
        return SplineMotion(model).apply(t,motion_state(self.track,self.epoch+np.asarray(t),model),self.reference_tx[3],self.reference_rx)

    def window_from_description(self,description):
        """Replay the saved physical observer and source model verbatim."""
        return self.window(*description['interval_s'],model=description['source_model'],
            observer_ecef_m=description.get('fixed_observer_ecef_m'))

    def window(self,first,last,*,model=None,stationary=True,observer_ecef_m=None):
        mid=(first+last)/2;anchor=self.track.sample(np.array([self.epoch+mid]),max_gap_s=400.)
        local=enu_basis(anchor['latitude'][0],anchor['longitude'][0]);anchor_state=self.state(np.array([mid]),model)
        gauge=anchor_state['body_to_ecef'][0].T@local
        native=[];fast=[];erasures=[]
        for c in range(5):
            # All original rows in each 16-symbol bin must remain in the
            # declared motion support, including at the recording endpoints.
            observed=crop(self.records[c]['data'],first+.02,last-.02,np.arange(-8,9))
            f,_=fixed_reference(observed,self.raw[c],mid,np.arange(-16,17),omega_reference=0.,observer_ecef_m=observer_ecef_m)
            grid=self.grid[f'channel_{c}'];times=grid['time'];rows=f.data['row_time'];index=np.searchsorted(times,rows)
            previous=np.maximum(index-1,0);index=np.minimum(index,len(times)-1)
            index=np.where(abs(times[previous]-rows)<abs(times[index]-rows),previous,index)
            if np.max(abs(times[index]-rows))>2e-8:raise ValueError('Kernel epochs do not match observations')
            valid=np.asarray([r['valid'] for r in self.kernel_records[c]])
            missing=~valid[index//16];info=f.data['row_information']
            # Invalid oracle blocks are erasures of the known-template fit,
            # never invented pulses or a trial-dependent observation mask.
            # The raw archive/final native imager retain their original data.
            erasures.append(dict(channel=c,unavailable_rows=int(missing.sum()),unavailable_positive_information_rows=int(np.count_nonzero(missing&(info>0)))))
            if np.any(missing):
                from field import Field
                data=dict(f.data);data['row_information']=info*(~missing)
                data['information']=data['row_information'].reshape(-1,16).sum(axis=1)
                f=Field(data,f.pulse.samples)
            # bind_subset deliberately refuses missing kernels, so bind their
            # identities explicitly here after the declared fit erasure.
            from block_pulse import BlockPulse
            used,inverse=np.unique(index//16,return_inverse=True)
            f=KernelField(f,BlockPulse(self.samples[c][used]),inverse)
            a=representative(f);blocks=f.pulse.row_blocks.reshape(len(f.time),16)
            if np.any(blocks!=blocks[:,0,None]):raise ValueError('Averaging and waveform boundaries disagree')
            # A recording-wide epoch keeps every retained bin in the same
            # fold wherever it appears in overlapping apertures.
            a=KernelField(a,f.pulse.kernels,blocks[:,0]);train=np.floor((f.time-3.32)/.5).astype(int)%2==0
            for field,target in [(f,native),(a,fast)]:
                state=self.state(field.data['row_time'],model)
                kin=RigidPatch(field.data['row_time'],mid,state['position_ecef_m'],state['velocity_ecef_mps'],local,self.tx[3],self.rx,
                    reference_frame=state['body_to_ecef']@gauge,reference_frame_rate=state['frame_velocity']@gauge)
                target.append(PatchBand(field,kin,np.array([[0.,0.,20.]]),train,self.tx[c],self.rx))
        return dict(first=first,last=last,mid=mid,local_to_ecef=local,native=native,fast=fast,erasures=erasures,model=self.model if model is None else model)


def local_geometry(band,path,path_rate,angle,angle_rate,selected,stride):
    """Five row-local variables: path, rate, yaw, yaw rate, unused.

    Chain these through a spline basis on the CPU after the GPU adjoint.
    Angles are radians. The frame derivatives include all mixed terms.
    """
    kin=band.kinematics;f=band.field;c,s=np.cos(angle),np.sin(angle)
    r=np.zeros_like(kin.frame);r[:,0,0]=r[:,1,1]=c;r[:,0,1]=-s;r[:,1,0]=s;r[:,2,2]=1
    frame=kin.frame@r;fd=kin.frame_rate@r+frame@J*angle_rate[:,None,None]
    centre=kin.position+path[:,None]*kin.direction
    velocity=kin.velocity+path_rate[:,None]*kin.direction+path[:,None]*kin.direction_rate
    state=dict(position_ecef_m=centre,velocity_ecef_mps=velocity,body_to_ecef=frame,frame_velocity=fd)
    base,details=geometry_from_state(state,band.tx,band.rx,f.metadata['rf_hz'],0.,np.asarray(selected,float))
    base[:,9,0]=(details['path_m']-f.data['centre_path_m']+band.origin)*8e6/C
    phase=f.k*(details['path_m']-f.data.get('carrier_reference_path_m',f.data['centre_path_m']))
    base[:,9,1]=np.cos(phase);base[:,9,2]=np.sin(phase);base[:,5,3]=f.k
    result=np.zeros((stride,50,4),'<f4');n=len(centre);result[:n,:10]=base
    result[:n,10,:3]=kin.direction;result[:n,11,:3]=kin.direction_rate
    result[:n,19,:3]=kin.direction
    for axis in range(3):
        result[:n,28+axis,:3]=(frame@J)[:,:,axis]
        result[:n,31+axis,:3]=(fd@J)[:,:,axis]
        result[:n,39+axis,:3]=(frame@J)[:,:,axis]
    return result
