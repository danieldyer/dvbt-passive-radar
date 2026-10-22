"""Carrier-resolved slow angular-rate control with the exact waveform field.

Only physical rigid-frame rotation is changed. All points/bands share it;
the virtual reference point keeps the same physical path. Constant phases
and nonnegative brightness are refitted on training epochs. This is a local
conditional search, not an accepted heading or a completed autofocus result.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from full_scene_context import open_scene
from full_path_scene import ScenePath,warm_phases
from declared_ship_orientation import rotate_frame
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True);p.add_argument('--support',type=Path,required=True)
    p.add_argument('--extent-degps',type=float,default=.01);p.add_argument('--step-degps',type=float,default=.001)
    p.add_argument('--acceleration-grid-degps2',type=float,nargs='+',help='Instead screen these explicit physical angular accelerations; include zero')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    fit,motion,bands,saved=open_scene(a.motion,full_field=True);likelihood=ScenePath(bands)
    if bands[0].points.ndim!=2:raise ValueError('Use one isolated persistent-scene aperture')
    baseline=[dict(body_to_ecef=b.point.frame.copy(),frame_velocity=b.point.frame_rate.copy()) for b in bands]
    initial=np.r_[np.zeros(likelihood.n),np.angle([complex(*z) for z in fit['physical_gains_real_imag']]),np.asarray(fit['scene_point_phase_radians'])[1:]]
    acceleration=a.acceleration_grid_degps2 is not None
    count=round(a.extent_degps/a.step_degps)
    grid=np.asarray(a.acceleration_grid_degps2) if acceleration else np.arange(-count,count+1)*a.step_degps
    if not np.isfinite(grid).all() or not np.any(grid==0):raise ValueError('The physical angular screen needs finite values and a zero control')
    order=np.r_[0.,grid[grid!=0]];reports=[];best=None
    for variable in order:
        rate,angular_acceleration=(0.,variable) if acceleration else (variable,0.)
        for band,base in zip(bands,baseline):
            b=band.point;state=rotate_frame(base,b.t,fit['anchor_path_time_s'],rate,angular_acceleration)
            b.frame,b.frame_rate=state['body_to_ecef'],state['frame_velocity'];band.warm=None
        parameters,diagnostic=warm_phases(likelihood,initial,200)
        value,g,solutions=likelihood.objective(parameters,True);phi=np.r_[0.,parameters[likelihood.n+5:]]
        testing=[]
        for c,(band,solution) in enumerate(zip(bands,solutions)):
            test=band.evaluate(parameters[:likelihood.n],phi+parameters[likelihood.n+c],c,training=False,derivatives=False,frozen_amplitude=solution['amplitude'])
            testing.append({k:test[k] for k in ['cost','energy','explained_fraction']})
        record=dict(additional_angular_rate_degps=rate,additional_angular_acceleration_degps2=angular_acceleration,training_objective=value,
            excluded_explained_fraction=1-sum(t['cost'] for t in testing)/sum(t['energy'] for t in testing),
            excluded=testing,phase_optimization=diagnostic,seconds=time.monotonic()-started)
        reports.append(record)
        if diagnostic['numerical_convergence'] and (best is None or value<best[0]):
            best=(value,float(variable),parameters.copy(),[r['amplitude'].copy() for r in solutions],record,
                [{k:r[k] for k in ['cost','energy','explained_fraction']} for r in solutions])
        (a.output/'screen.json').write_text(json.dumps(serial(dict(complete=False,scope=__doc__,trials=reports)),indent=2)+'\n')
        print(json.dumps(serial(record)),flush=True)
    if best is None:raise ValueError('No angular phase-block control converged')
    value,variable,parameters,amplitudes,selected,training=best
    rate,angular_acceleration=(0.,variable) if acceleration else (variable,0.)
    beta=parameters[likelihood.n:likelihood.n+5];phi=np.r_[0.,parameters[likelihood.n+5:]]
    gain=np.exp(1j*beta);output=dict(fit)
    output.pop('parameters',None)
    output.update(scope=__doc__,angular_rate_offset_degps=float(fit.get('angular_rate_offset_degps',0.)+rate),
        angular_acceleration_offset_degps2=float(fit.get('angular_acceleration_offset_degps2',0.)+angular_acceleration),
        physical_gains_real_imag=np.c_[gain.real,gain.imag],scene_point_phase_radians=phi,
        source_fine_angular_screen=str(a.output/'screen.json'),initializer_only=True,numerical_convergence=False,
        accepted_as_vessel_motion=False,objective=value,unused_explained_fraction=selected['excluded_explained_fraction'],
        unused=selected['excluded'],training=training)
    for c,(band,amplitude) in enumerate(zip(bands,amplitudes)):
        saved[f'amplitude_coefficients_{c}']=amplitude
        saved[f'amplitude_{c}']=band.point.amplitude@amplitude
    np.savez_compressed(a.output/'trajectory.npz',**saved)
    (a.output/'analysis.json').write_text(json.dumps(serial(output),indent=2)+'\n')
    support=json.loads(a.support.read_text());support['phase_parameters']=np.r_[np.zeros(4),phi].tolist();support['band_phase_initial_radians']=beta.tolist()
    (a.output/'phase-initialized-support.json').write_text(json.dumps(support,indent=2)+'\n')
    result=dict(complete=True,scope=__doc__,motion=str(a.motion),trials=reports,selected_additional_rate_degps=rate,
        selected_additional_acceleration_degps2=angular_acceleration,
        selection='Lowest converged training likelihood; excluded data not used to select the angular correction',joint_motion_refinement_required=True)
    (a.output/'screen.json').write_text(json.dumps(serial(result),indent=2)+'\n')
    from matplotlib import pyplot as plt
    key='additional_angular_acceleration_degps2' if acceleration else 'additional_angular_rate_degps'
    ordered=sorted(reports,key=lambda r:r[key]);x=[1000*r[key] for r in ordered]
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
    ax.plot(x,[100*(1-r['training_objective']) for r in ordered],'.-',label='Training')
    ax.plot(x,[100*r['excluded_explained_fraction'] for r in ordered],'.-',label='Excluded')
    quantity='acceleration' if acceleration else 'rate';unit='second²' if acceleration else 'second'
    ax.axvline(1000*variable,color='grey',ls='--');ax.set(xlabel=f'Additional physical angular {quantity} (millidegrees / {unit})',
        ylabel='Full-crop energy predicted (%)',title=f'Fine rigid-frame {quantity} control: same fixed path/points; positive fading; constant RF calibration')
    ax.legend();fig.savefig(a.output/f'fine-angular-{quantity}.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
