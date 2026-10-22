"""Test whether a few range-separated points under-model the ship response.

Training-only native images provide candidates. Exact known-waveform GLS
orthogonal pursuit then selects common physical locations using range AND
slow-time/Doppler discrimination. Constant complex responses are frozen before
predicting excluded epochs. Local band calibration is a diagnostic comparison,
never a correction adopted for the full recording.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

from full_ship_field import load
from full_path_point import PointBand
from full_path_scene import SceneBand
from joint_ship_motion import JointMotion
from joint_support import select
from local_ship_motion import load_motion
from motion import enu_basis, excess_path
from rigid_patch import profile_static
from shared_scene import solve, phase_coherent
from report_measured_scatterer_phases import serial


def rss(gram,rhs,energy,coefficients):
    return float(sum(e-2*np.vdot(z,b).real+np.vdot(z,g@z).real for g,b,e,z in zip(gram,rhs,energy,coefficients)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True);p.add_argument('--images',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--candidates',type=int,default=80)
    p.add_argument('--points',type=int,default=24)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    d=json.loads(a.motion.read_text());native=load_motion(a.motion);motion=JointMotion(Path(d['motion']))
    fields,coordinates,_=load(Path(d['field']));reports=[]
    for root in sorted(a.images.glob('aperture-*')):
        meta=json.loads((root/'analysis.json').read_text())
        if 'training_epoch_selection' not in meta:raise ValueError('Candidate images must use training epochs only')
        first,last=meta['interval_s'];mid=(first+last)/2;z=np.load(root/'images.npz')
        state=native.state(np.array([mid]));ais=native.track.sample(np.array([native.epoch+mid]),max_gap_s=400.)
        local=enu_basis(ais['latitude'][0],ais['longitude'][0]);gauge=state['body_to_ecef'][0].T@local
        east,north=z['east_m'],z['north_m'];xx,yy=np.meshgrid(east,north);pixels=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,20.)]
        body=pixels@gauge.T;positions=state['position_ecef_m'][0]+body@state['body_to_ecef'][0].T
        path=excess_path(positions,native.tx[3],native.rx)[0];aispath=excess_path(ais['position_ecef_m'],native.tx[3],native.rx)[0][0]
        offset=(ais['position_ecef_m'][0]-state['position_ecef_m'][0])@local
        gate=(abs(path-aispath)<200)&(np.sum((pixels[:,:2]-offset[:2])**2,axis=1)<350**2)
        power=z['power'][:5];combined=z['coherent_positive_amplitude_glrt'];ranking=np.sum(power,axis=0)
        candidates=[]
        for image in [*power,combined]:
            at=np.flatnonzero((image==maximum_filter(image,size=9)).ravel()&gate)
            candidates.extend(at[np.argsort(image.ravel()[at])[-max(12,a.candidates//2):]])
        unique=np.unique(candidates);order=unique[np.argsort(ranking.ravel()[unique])[::-1]];chosen=[]
        for index in order:
            if all(np.linalg.norm(pixels[index,:2]-pixels[j,:2])>=8 for j in chosen):chosen.append(int(index))
            if len(chosen)>=a.candidates:break
        points=body[chosen];design=[];data=[];masks=[];statistics=[]
        for c,(full,coordinate) in enumerate(zip(fields,coordinates)):
            f=full.windowed(first,last);indices=np.searchsorted(full.time,f.time);np.testing.assert_array_equal(full.time[indices],f.time)
            coord={k:coordinate[k].reshape(-1,16)[indices].ravel() for k in ['native_time_s','native_information']}
            coord['training']=coordinate['training'][indices]
            if 'stationary_groups' in coordinate:coord['stationary_groups']=coordinate['stationary_groups'][indices]
            b=PointBand(f,coord,motion,d['point_body_m'],np.asarray(d['path_knots_s']),np.asarray(d['path_coefficients_m']),d.get('anchor_path_time_s',162.),0.,d['orientation'],angular_rate_offset_degps=d.get('angular_rate_offset_degps',0.),angular_acceleration_offset_degps2=d.get('angular_acceleration_offset_degps2',0.))
            band=SceneBand(b,points);h=band.design(np.zeros(b.b.shape[1]),c,False)
            ah=np.zeros_like(h);y=np.zeros_like(f.y)
            for mask in [b.training,~b.training]:
                ah[mask]=profile_static(h,f.root_information,mask,b.groups)[mask]
                y[mask]=profile_static(f.y,f.root_information,mask,b.groups)[mask]
            at=ah[b.training].reshape(-1,len(points));yt=y[b.training].ravel()
            ae=ah[~b.training].reshape(-1,len(points));ye=y[~b.training].ravel()
            design.append(at);data.append(yt)
            statistics.append((at.conj().T@at,at.conj().T@yt,float(np.vdot(yt,yt).real),
                               ae.conj().T@ae,ae.conj().T@ye,float(np.vdot(ye,ye).real)))
        trials=[]
        for support,selection in select(design,data,min(a.points,len(points)),tolerance=1e-8):
            count=len(support)
            if count not in [1,3,6,9,12,18,a.points]:continue
            gram=np.array([s[0][np.ix_(support,support)] for s in statistics]);rhs=np.array([s[1][support] for s in statistics]);energy=np.array([s[2] for s in statistics])
            testgram=np.array([s[3][np.ix_(support,support)] for s in statistics]);testrhs=np.array([s[4][support] for s in statistics]);testenergy=np.array([s[5] for s in statistics])
            independent=np.array([np.linalg.lstsq(x[:,support],y,rcond=1e-10)[0] for x,y in zip(design,data)])
            gains=np.exp(1j*np.angle(native.gains));common=np.linalg.lstsq(gram.sum(axis=0),np.einsum('c,cp->p',gains.conj(),rhs),rcond=1e-10)[0]
            models={'independent_constant_coefficients':independent,'frozen_global_band_phase_unit_response':gains[:,None]*common}
            conditions=[float(np.linalg.cond(g)) for g in gram];phase_report=None
            if max(conditions)<1e10:
                shared=solve(gram,rhs,energy);phase=phase_coherent(gram,rhs,energy)
                models['local_shared_complex_scene']=shared['gain'][:,None]*shared['scene']
                models['local_shared_point_phase']=phase['coefficient']
                phase_report=dict(maximum_gradient=phase['maximum_gradient'],converged=phase['converged'],gain_real_imag=np.c_[phase['gain'].real,phase['gain'].imag],point_phase_rad=phase['point_phase_rad'])
            prediction={}
            for label,coef in models.items():
                cost=rss(testgram,testrhs,testenergy,coef)
                prediction[label]=dict(unused_rss=cost,unused_energy=float(testenergy.sum()),explained_fraction=1-cost/testenergy.sum(),coefficient_real_imag=np.stack([coef.real,coef.imag],axis=-1))
            trial=dict(count=count,support=support,selection=selection,gram_condition_numbers=conditions,models=prediction,shared_phase_fit=phase_report)
            trials.append(trial);print(json.dumps(dict(interval_s=[first,last],count=count,explained={k:v['explained_fraction'] for k,v in prediction.items()})),flush=True)
        record=dict(interval_s=[first,last],candidate_points_body_m=points,candidate_points_east_north_up_m=pixels[chosen],trials=trials)
        reports.append(record);(a.output/'analysis.json').write_text(json.dumps(serial(dict(complete=False,scope=__doc__,motion=str(a.motion),windows=reports)),indent=2)+'\n')
    from matplotlib import pyplot as plt
    columns=min(5,len(reports));rows=(len(reports)+columns-1)//columns
    fig,axes=plt.subplots(2*rows,columns,figsize=(6*columns,11*rows),layout='constrained',squeeze=False)
    for index,record in enumerate(reports):
        row,column=2*(index//columns),index%columns
        for label in ['independent_constant_coefficients','frozen_global_band_phase_unit_response','local_shared_complex_scene','local_shared_point_phase']:
            eligible=[q for q in record['trials'] if label in q['models']]
            axes[row,column].plot([q['count'] for q in eligible],[100*q['models'][label]['explained_fraction'] for q in eligible],'.-',label=label.replace('_',' '))
        axes[row,column].legend(fontsize=7);axes[row,column].set(xlabel='Shared physical points',ylabel='Excluded full-crop energy predicted (%)',title=f"{record['interval_s'][0]:.1f}–{record['interval_s'][1]:.1f} s")
        points=np.asarray(record['candidate_points_east_north_up_m']);support=record['trials'][-1]['support'];axes[row+1,column].plot(points[:,0],points[:,1],'.',color='grey');axes[row+1,column].plot(points[support,0],points[support,1],'o',mfc='none',mec='#00aacc')
        axes[row+1,column].set(xlabel='East (m)',ylabel='North (m)',aspect='equal',title='Exact training-only joint OLS support')
    for index in range(len(reports),rows*columns):
        row,column=2*(index//columns),index%columns
        axes[row,column].set_axis_off();axes[row+1,column].set_axis_off()
    fig.suptitle('Extended-scene control: physical points need not be separated in range alone\nTraining selection and fitted constants frozen before excluded-data prediction; motion held fixed')
    fig.savefig(a.output/'extended-scene-transfer.png',dpi=140);plt.close(fig)
    (a.output/'analysis.json').write_text(json.dumps(serial(dict(complete=True,scope=__doc__,motion=str(a.motion),windows=reports)),indent=2)+'\n')


if __name__=='__main__':main()
