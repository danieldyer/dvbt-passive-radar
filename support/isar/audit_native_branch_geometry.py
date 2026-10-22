"""Compare the recently fitted strong branch with AIS and broad hull bounds.

Course is not heading. Hull envelopes below are explicitly conditional on
the stated course/heading tolerance, not a ship identification or multipath
model. No phase, range or image registration is fitted in this diagnostic.
"""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ais import read_csv
from flash_siting_report import EPOCH
from motion import C,ecef,excess_path
from native_folds import training_mask
from piecewise_ais import LocalAisReference
from recorded_injection_fields import load_fields
from render_injected_point import fitted_state
from report_measured_scatterer_phases import serial
from recovery_experiment import NAMES


def hull_offsets(heading,tolerance_deg):
    angle=np.asarray(heading)[...,None]+np.radians(np.linspace(-tolerance_deg,tolerance_deg,21))
    along=np.stack([np.sin(angle),np.cos(angle),np.zeros_like(angle)],axis=-1)
    across=np.stack([np.cos(angle),-np.sin(angle),np.zeros_like(angle)],axis=-1)
    return np.concatenate([a*along+b*across+np.array([0.,0.,z])
        for a in [-300.,300.] for b in [-25.,25.] for z in [0.,60.]],axis=-2)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    root=Path('docs/data/isar_20261018/native-spatial-validation');source=root/'residual-three-point-30s-native'
    model=json.loads((source/'analysis.json').read_text());render=json.loads((source/'render-fit.json').read_text())
    short=json.loads(Path('docs/data/isar_20261018/native-shared-radial-point-6s/analysis.json').read_text())
    fields,rotation,rx,tx,_,_=load_fields(Path(model['profiles']),offsets=np.arange(-16,33),fixed_epoch_s=161.,interval_s=(142.,172.))
    track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    reference=LocalAisReference(track,EPOCH+161.,rotation,rx,half_width_s=20.)
    prior=np.asarray(short['prior_mean_state_enu']);point=np.asarray(short['physical_state_enu'])[0]
    heading=np.arctan2(prior[1,0],prior[1,1]);axis=np.array([np.sin(heading),np.cos(heading),0.]);across=np.array([np.cos(heading),-np.sin(heading),0.])
    delta=point-prior[0];records=[]
    update=json.loads(Path('docs/data/calibration_20261016_sites/transmitter-update.json').read_text())
    sites={d['id']:ecef(np.radians(d['new']['latitude_deg']),np.radians(d['new']['longitude_deg']),d['new']['height_ellipsoid_m']) for d in update['site_changes']}
    config=json.loads(Path('recordings/live_mt_dandenong.json').read_text());site_records=[]
    strong_path=float(excess_path((rx+point@rotation)[None,:],tx[0],rx)[0][0])
    for name,antenna in sites.items():
        ais_path=float(excess_path((rx+prior[0]@rotation)[None,:],antenna,rx)[0][0]);bounds={}
        for tolerance in [0.,20.]:
            points=rx+(prior[0]+hull_offsets(heading,tolerance))@rotation
            high=float(excess_path(points,antenna,rx)[0].max())
            bounds[str(tolerance)]=dict(maximum_hull_path_m=high,strong_path_above_maximum_m=strong_path-high)
        site_records.append(dict(site=name,ais_excess_path_m=ais_path,hull_bounds=bounds))
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    corners=np.array([a*axis+b*across for a,b in [(-300,-25),(300,-25),(300,25),(-300,25),(-300,-25)]])
    axes[0].fill(corners[:,0],corners[:,1],color='#ddd',label='Generous course-aligned hull: ±300 m × ±25 m')
    bound=hull_offsets(heading,20.)
    axes[0].scatter(bound[:,0],bound[:,1],s=3,color='#999',label='±20° heading hypotheses')
    axes[0].plot(0,0,'+',color='#00aaaa',ms=12,label='Interpolated AIS antenna')
    axes[0].plot(*delta[:2],'x',color='black',ms=10,label='Conditional radial-fit point')
    east=np.arange(-400,401,2.);north=np.arange(-400,401,2.);xx,yy=np.meshgrid(east,north)
    grid=prior[0]+np.stack([xx,yy,np.zeros_like(xx)],axis=-1)
    length=excess_path(rx+grid.reshape(-1,3)@rotation,tx[0],rx)[0].reshape(xx.shape)
    axes[0].contour(xx,yy,length,levels=[strong_path],colors=['tab:red'],linewidths=1.)
    axes[0].set(xlabel='East from AIS antenna (m)',ylabel='North (m)',aspect='equal',title='Strong-branch equal-range contour versus AIS hull hypotheses')
    axes[0].legend(fontsize=8,loc='upper right')
    tolerances=np.arange(0.,61.)
    for name,antenna in sites.items():
        high=[excess_path(rx+(prior[0]+hull_offsets(heading,a))@rotation,antenna,rx)[0].max() for a in tolerances]
        axes[1].plot(tolerances,np.asarray(high)-strong_path,label=name)
    axes[1].axhline(0,color='tab:red',lw=1.);axes[1].set(xlabel='Allowed heading departure from course (degrees)',
        ylabel='Maximum hull path minus strong-branch path (m)',title='Exact two-leg paths; all three candidate transmitter sites')
    axes[1].legend();fig.savefig(out/'branch-geometry.png',dpi=160);plt.close(fig)
    panels=[]
    for c,f in enumerate(fields):
        t=f.time;state=reference.sample(t-161.);speed=reference.sample(t-161.,1);angle=np.arctan2(speed[:,0],speed[:,1])
        centre=rx+state@rotation;ais_path=excess_path(centre,tx[c],rx)[0];fixed=f.metadata['fixed_reference_path_m']
        physical=fitted_state(t,np.asarray(model['reference_state_enu']),model['midpoint_s'],rotation,rx,render)
        fitted_path=excess_path(physical['position_ecef_m'],tx[c],rx)[0]
        candidates=state[:,None,:]+hull_offsets(angle,20.)
        high=excess_path(rx+candidates.reshape(-1,3)@rotation,tx[c],rx)[0].reshape(len(t),-1).max(axis=1)
        y=f.data['value'];keep=training_mask(t,142.);w=f.data['information']*keep
        residual=y-(w@y)/w.sum()
        panels.append((t,np.abs(y)**2,np.abs(residual)**2,ais_path-fixed,high-fixed,fitted_path-fixed))
        records.append(dict(channel=c,rf_hz=f.metadata['rf_hz'],fitted_minus_ais_path_at_161_m=float(np.interp(161.,t,fitted_path-ais_path))))
    high=max(max(raw.max(),residual.max()) for _,raw,residual,*_ in panels);vmax=10*np.log10(high);vmin=vmax-50
    fig,axes=plt.subplots(5,2,figsize=(15,15),sharex=True,sharey=True,layout='constrained')
    for c,(t,raw,residual,ais_path,hull_max,fitted) in enumerate(panels):
        for ax,value,label in zip(axes[c],[raw,residual],['Original fixed-grid native means','Training-mean subtraction: display diagnostic only']):
            im=ax.imshow(10*np.log10(np.maximum(value,1e-30)),origin='lower',aspect='auto',
                extent=[fields[c].offsets[0]*C/8e6,fields[c].offsets[-1]*C/8e6,t[0],t[-1]],cmap='magma',vmin=vmin,vmax=vmax)
            ax.plot(ais_path,t,color='#00ffff',lw=1.,label='AIS antenna path')
            ax.plot(hull_max,t,color='#44ff88',lw=1.,ls='--',label='Maximum broad hull path, COG ±20°')
            ax.plot(fitted,t,color='white',lw=.8,ls=':',label='Recently fitted stronger branch')
            ax.set(title=NAMES[c]+' · '+label,xlabel='Path offset from fixed AIS observer at 161 s (m)',ylabel='Capture time (s)')
    axes[0,0].legend(fontsize=7,loc='lower right');fig.colorbar(im,ax=axes.ravel().tolist(),label='Native fixed-grid magnitude squared (dB; common fixed scale)',shrink=.8)
    fig.suptitle('Branch identification before further geometric autofocus\nBoth raw and diagnostic residual retain the same physical axes; no delay or phase correction fitted')
    fig.savefig(out/'native-branch-profiles.png',dpi=160);plt.close(fig)
    result=dict(scope=__doc__,complete=True,ship_identification_certified=False,interval_s=[142.,172.],
        nominal_epoch_s=161.,conditional_point_minus_ais_enu_m=delta,
        along_course_m=float(axis@delta),across_course_m=float(across@delta),heading_hypothesis_deg=float(np.degrees(heading)%360),
        physical_hull_assumption=dict(along_m=[-300,300],across_m=[-25,25],height_above_ais_m=[0,60]),
        transmitter_candidates=site_records,channels=records,
        conclusion='The recent stronger-branch fit is not a direct ship-point identification under these near-course hull hypotheses. Check range registration and the AIS-near branch; no multipath explanation is adopted.')
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    print('Wrote branch audit',out,flush=True)


if __name__=='__main__':main()
