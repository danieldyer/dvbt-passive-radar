"""One smooth physical position/orientation and fixed calibration per capture."""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline,PchipInterpolator
from ais import read_csv
from flash_siting_report import EPOCH
from motion import C,enu_basis,excess_path
from imaging import geometry_from_state
from spline_motion import knots


class ContinuousShipMotion:
    def __init__(self,path):
        self.path=Path(path);self.result=json.loads(self.path.read_text())
        d=self.result;self.epoch=EPOCH;self.interval=d['interval_s'];self.motion_interval=tuple(self.interval)
        self.rx=np.asarray(d['receiver_ecef_m']);self.tx=np.asarray(d['transmitters_ecef_m']);self.rf=np.asarray(d['rf_hz'])
        self.rotation=np.asarray(d['ecef_to_enu']);self.local=self.rotation.T
        self.gains=np.asarray([complex(*z) for z in d['calibration_gains_real_imag']])
        self.track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
        self.position_curve=BSpline(d['position_knots_s'],np.asarray(d['position_coefficients_enu_m']),3,extrapolate=False)
        self.angle_curve=BSpline(d['angle_knots_s'],np.asarray(d['angle_coefficients_rad']),3,extrapolate=False)
        a=self.track.sample([EPOCH+d['anchor_s']],max_gap_s=400.)
        self.geographic_gauge=enu_basis(a['latitude'][0],a['longitude'][0]).T@self.rotation.T

    def geographic_frame(self,t):
        t=np.atleast_1d(t);a=self.track.sample(EPOCH+t,max_gap_s=400.)
        frame=enu_basis(a['latitude'],a['longitude'])@self.geographic_gauge
        step=.01;left=self.track.sample(EPOCH+t-step,max_gap_s=400.);right=self.track.sample(EPOCH+t+step,max_gap_s=400.)
        rate=(enu_basis(right['latitude'],right['longitude'])-enu_basis(left['latitude'],left['longitude']))/(2*step)
        return frame,rate@self.geographic_gauge

    def base(self,t):return self.state(t)

    def state(self,t):
        t=np.atleast_1d(t)
        if np.any((t<self.interval[0])|(t>self.interval[1])):raise ValueError('Do not extrapolate the fitted capture motion')
        position=self.rx+self.position_curve(t)@self.rotation;velocity=self.position_curve(t,1)@self.rotation
        frame,fd=self.geographic_frame(t);angle=self.angle_curve(t);rate=self.angle_curve(t,1)
        c,s=np.cos(angle),np.sin(angle);r=np.zeros((len(t),3,3));dr=np.zeros_like(r)
        r[:,0,0]=r[:,1,1]=c;r[:,1,0]=s;r[:,0,1]=-s;r[:,2,2]=1
        dr[:,0,0]=dr[:,1,1]=-s*rate;dr[:,1,0]=c*rate;dr[:,0,1]=-c*rate
        return dict(position_ecef_m=position,velocity_ecef_mps=velocity,body_to_ecef=frame@r,frame_velocity=fd@r+frame@dr)

    def geometry(self,channel,t,first_sample,information,*,deramped=False,sign=1.):
        g,details=geometry_from_state(self.state(t),self.tx[channel],self.rx,self.rf[channel],first_sample,information)
        if deramped:g[:,9,1]=1.;g[:,9,2]=0.
        g[:,5,3]=2*np.pi*self.rf[channel]/C;g[:,6,3]=channel;g[:,7,3]=sign;g[:,8,3]=1.
        return g,details


def initialise(cache,reference_fit,out,*,spacing_s=15.,angle_spacing_s=30.):
    """AIS supplies a broad initial track; the radar likelihood refines it."""
    cache=Path(cache);reference_fit=Path(reference_fit);out=Path(out)
    h=json.loads((cache/'native-header.json').read_text());plan=h['plan'];rx=np.asarray(plan['receiver_ecef_m']);r=np.asarray(plan['ecef_to_enu']);tx=np.asarray(plan['transmitters_ecef_m'])
    support=json.loads((cache/'block-4/aperture.json').read_text())['channels']
    # Model the union of native supports. Individual image windows still
    # require the intersection; the model must also cover edge bins in each
    # likelihood, without extrapolating a channel by a few milliseconds.
    interval=[min(c['interval_s'][0] for c in support),max(c['interval_s'][1] for c in support)]
    reference=json.loads((reference_fit/'analysis.json').read_text());anchor=reference['midpoint_s']
    render=json.loads((reference_fit/'render-fit.json').read_text())
    from render_injected_point import fitted_state
    current=fitted_state(np.array([anchor]),np.asarray(render['coarse_state_enu']),anchor,r,rx,render)
    track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    known=np.isfinite(track.course);course=PchipInterpolator(track.seconds[known]-EPOCH,np.unwrap(track.course[known]),extrapolate=False)
    sample=np.linspace(*interval,6001);ais=track.sample(EPOCH+sample,max_gap_s=400.)
    at=track.sample([EPOCH+anchor],max_gap_s=400.);geo=enu_basis(ais['latitude'],ais['longitude']);geo0=enu_basis(at['latitude'][0],at['longitude'][0])
    gauge=geo0.T@r.T;frame=geo@gauge
    theta=course(anchor)-course(sample);cs,sn=np.cos(theta),np.sin(theta)
    rot=np.zeros((len(sample),3,3));rot[:,0,0]=rot[:,1,1]=cs;rot[:,1,0]=sn;rot[:,0,1]=-sn;rot[:,2,2]=1
    offset=(current['position_ecef_m'][0]-at['position_ecef_m'][0])@r.T
    position=(ais['position_ecef_m']+frame@rot@offset-rx)@r.T
    grid=knots(interval,spacing_s);b=BSpline.design_matrix(sample,grid,3).toarray()
    coefficient=np.linalg.lstsq(b,position,rcond=None)[0]
    # A constant coordinate shift makes the already-reviewed reference epoch
    # exact without turning interpolated AIS positions into observations.
    predicted=BSpline(grid,coefficient,3)(anchor);wanted=(current['position_ecef_m'][0]-rx)@r.T;coefficient+=wanted-predicted
    # Reproduce the reviewed radar position AND velocity at the reference
    # epoch. Greville abscissae reproduce a linear correction exactly.
    greville=np.array([grid[j+1:j+4].mean() for j in range(len(coefficient))])
    dv=current['velocity_ecef_mps'][0]@r.T-BSpline(grid,coefficient,3)(anchor,1)
    coefficient+=(greville-anchor)[:,None]*dv
    ag=knots(interval,angle_spacing_s);ab=BSpline.design_matrix(sample,ag,3).toarray();ac=np.linalg.lstsq(ab,theta,rcond=None)[0]
    ac-=BSpline(ag,ac,3)(anchor)
    body=current['body_to_ecef'][0];body_rate=current['frame_velocity'][0]
    spin=(body.T@body_rate)[1,0]
    angle_nodes=np.array([ag[j+1:j+4].mean() for j in range(len(ac))])
    ac+=(angle_nodes-anchor)*(spin-BSpline(ag,ac,3)(anchor,1))
    # Sparse AIS positions/course do not determine carrier-accurate curvature.
    # Preserve the already-reviewed radar aperture as the local initializer,
    # rather than only matching its midpoint position and velocity. These
    # weights affect the INITIALIZER only; no AIS penalty enters the final
    # radar likelihood.
    rt=np.linspace(*reference['interval_s'],1201)
    rs=fitted_state(rt,np.asarray(render['coarse_state_enu']),anchor,r,rx,render)
    target=(rs['position_ecef_m']-rx)@r.T
    rb=BSpline.design_matrix(rt,grid,3).toarray()
    coefficient=np.linalg.lstsq(np.r_[b/50.,rb/.001],np.r_[(b@coefficient)/50.,target/.001],rcond=None)[0]
    coefficient+=wanted-BSpline(grid,coefficient,3)(anchor)
    dv=current['velocity_ecef_mps'][0]@r.T-BSpline(grid,coefficient,3)(anchor,1)
    coefficient+=(greville-anchor)[:,None]*dv
    body=np.einsum('ij,tjk->tik',r,rs['body_to_ecef'])
    target_angle=np.unwrap(np.arctan2(body[:,1,0],body[:,0,0]))
    rab=BSpline.design_matrix(rt,ag,3).toarray()
    ac=np.linalg.lstsq(np.r_[ab/np.radians(20.),rab/np.radians(.0001)],
        np.r_[(ab@ac)/np.radians(20.),target_angle/np.radians(.0001)],rcond=None)[0]
    ac-=BSpline(ag,ac,3)(anchor)
    brightness=reference['phase_histories']['brightness_model'];gain=np.asarray([np.asarray(c['coefficients'])[0].mean()*np.exp(1j*c['band_phase_rad']) for c in brightness['bands']])
    result=dict(model_kind='native-continuous-ship',complete=False,stage='AIS/radar reference initializer; full-capture radar optimisation pending',
        interval_s=interval,anchor_s=anchor,reference_fit=str(reference_fit),profiles_cache=str(cache),receiver_ecef_m=rx.tolist(),transmitters_ecef_m=tx.tolist(),
        ecef_to_enu=r.tolist(),rf_hz=[c['rf_hz'] for c in h['applied_channels']],
        calibration_gains_real_imag=[[z.real,z.imag] for z in gain],position_knots_s=grid.tolist(),position_coefficients_enu_m=coefficient.tolist(),
        angle_knots_s=ag.tolist(),angle_coefficients_rad=ac.tolist(),
        reference_points_east_north_up_m=reference['refined_points_east_north_up_m'],
        nominal_ais_antenna_offset_reference_enu_m=offset.tolist(),
        reference_aperture_initializer_scope='Reviewed radar position/attitude throughout its aperture; broad AIS initializer elsewhere. All refined with the same raw radar likelihood, without an AIS residual penalty.',
        scope='One C2 physical track and heading history, shared across every band. AIS is an initializer, not radar truth. No fitted band-dependent phase history.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    return result
