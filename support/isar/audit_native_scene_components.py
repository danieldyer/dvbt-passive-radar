"""Full-aperture multiband phase closure with overlap uncertainty retained.

All motion, positions, calibration, phases and real visibility come from the
saved fit. The complete aperture projects its known nuisance design. For a tested
component, all other frozen source templates have unknown complex nuisance
coefficients; the exact Schur complement supplies its score and variance.
The resulting common-phase/independent-phase comparison is conditional on
the declared thermal covariance, not a universal target-SNR or identity test.
"""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from amplitude_arc import linear_basis
from coherent_amplitude_glrt import statistic
from field import average
from fixed_range_field import stationary_profile_design
from native_folds import training_mask
from recorded_injection_fields import load_fields
from recovery_experiment import NAMES
from render_injected_point import fitted_state
from report_measured_scatterer_phases import serial
from stationary import Stationary,component_statistics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--diagnostic-folds',action='store_true',help='Reproduce legacy periodic subsets; these do not describe the continuous image aperture')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    saved=json.loads((args.fit/'analysis.json').read_text());render=json.loads((args.fit/'render-fit.json').read_text())
    first,last=saved['interval_s'];mid=render['midpoint_s'];state=np.asarray(render['coarse_state_enu'])
    points=np.asarray(saved['refined_points_east_north_up_m']);brightness=saved['phase_histories']['brightness_model']
    observed=saved['observed_range_samples']
    fields,rotation,rx,tx,_,_=load_fields(Path(saved['profiles']),offsets=np.arange(observed[0],observed[1]+1),fixed_epoch_s=mid,interval_s=(first,last))
    scores=[];information=[];condition=[]
    competing=None;competing_render=None
    if saved.get('nuisance_fit'):
        competing_path=Path(saved['nuisance_fit'])
        competing=json.loads((competing_path/'analysis.json').read_text())
        competing_render=json.loads((competing_path/'render-fit.json').read_text())
    for c,f in enumerate(fields):
        time=f.data['row_time'];motion=fitted_state(time,state,mid,rotation,rx,render)
        band=brightness['bands'][c];basis=linear_basis(time,np.asarray(band['knots_s']));coefficient=np.asarray(band['coefficients'])
        components=[]
        for j,q in enumerate(points):
            position=motion['position_ecef_m']+motion['body_to_ecef']@q
            velocity=motion['velocity_ecef_mps']+motion['frame_velocity']@q
            raw=f.raw_trajectory(position,velocity,position_offset=position-f.data['centre_ecef_m'])
            gain=np.exp(1j*(band['band_phase_rad']+brightness['point_phase_rad'][j]))
            value=f.apply(average(raw*(basis@coefficient[j])[:,None]*gain,f.data['row_information'],f.metadata['block_rows'])[0])
            components.append(value.ravel())
        design=stationary_profile_design(f)
        if competing is not None:
            from frozen_native_scene import component
            extra=[component(f,rotation,rx,competing_render,competing['phase_histories']['brightness_model'],
                competing['refined_points_east_north_up_m'],j) for j in saved['nuisance_components']]
            design=np.column_stack([design,*extra])
        h=np.column_stack(components);cs=[];ci=[];cc=[]
        groups=[np.ones(len(f.time),bool)]
        labels=['Complete aperture']
        if args.diagnostic_folds:
            train=training_mask(f.time,first);groups=[train,~train];labels=['Legacy fitting subset','Legacy unused subset']
        for keep in groups:
            selected=np.repeat(keep,len(f.offsets));nuisance=Stationary(design,selected)
            z,i,k=component_statistics(h[selected],f.y.ravel()[selected],nuisance);cs.append(z);ci.append(i);cc.append(k)
        scores.append(cs);information.append(ci);condition.append(cc)
        print('Verified native component scores',NAMES[c],flush=True)
    scores=np.asarray(scores);information=np.asarray(information);folds=[]
    for f,label in enumerate(labels):
        z=scores[:,f]/np.sqrt(information[:,f]);positive,phase=statistic(z);independent=np.sum(abs(z)**2,axis=0)
        ratio=positive/independent
        folds.append(dict(fold=label,positive_amplitude_to_independent_ratio=ratio,
            ratio_db=10*np.log10(ratio),band_phase_error_deg=np.degrees(np.angle(z)),
            band_independent_score=abs(z)**2,band_schur_information=information[:,f],
            global_reflector_phase_rad=phase,band_template_condition=np.asarray(condition)[:,f]))
    fig,axes=plt.subplots(2,1,figsize=(12,9),layout='constrained')
    for c in range(5):axes[0].plot(np.arange(len(points)),folds[-1]['band_phase_error_deg'][c],'o-',label=NAMES[c])
    axes[0].set(xticks=np.arange(len(points)),xlabel='Component index',ylabel=labels[-1]+' phase error (degrees)',ylim=(-180,180),title='Frozen native predictions; stationary and other components profiled as nuisances');axes[0].legend(ncol=5)
    for fold in folds:axes[1].plot(np.arange(len(points)),fold['ratio_db'],'o-',label=fold['fold'])
    axes[1].set(xticks=np.arange(len(points)),xlabel='Component index',ylabel='Common-phase / independent-phase statistic (dB)',title='Exact overlap variance; constant phase hypotheses compared, no correction applied');axes[1].legend()
    fig.savefig(out/'component-phase-closure.png',dpi=160);plt.close(fig)
    result=dict(scope=__doc__,complete=True,source_fit=str(args.fit),interval_s=[first,last],points_east_north_up_m=points,
        folds=folds,estimation_samples_reused=(saved.get('fit_samples','diagnostic-subset')=='all' or not args.diagnostic_folds),
        interpretation='Conditional model residual diagnostic; not an independent recovery test or an optimal-SNR certificate.',
        ship_scatterer_identities_certified=False)
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    rows=''.join(f'<tr><td>{j}</td><td>{q[0]:+.2f}, {q[1]:+.2f}</td><td>{folds[-1]["positive_amplitude_to_independent_ratio"][j]:.2%}</td><td>{folds[-1]["ratio_db"][j]:.3f}</td></tr>' for j,q in enumerate(points))
    html=f'''<!doctype html><html><meta charset="utf-8"><title>Native scene component phase closure</title><style>body{{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px}}p{{line-height:1.5}}img{{width:100%}}td,th{{padding:8px}}</style><h1>Several native component phase histories</h1><p>Source coordinates, shared motion, constant band calibration and positive visibility are frozen from the declared fit. Each component's matched score accounts for overlap with all the other fitted components and stationary profiles through an exact Schur complement. This is a conditional residual comparison of common-phase and independent-phase models, not independent validation, a target-SNR bound or a claim that all candidate locations are on the ship.</p><table><tr><th>Component</th><th>East, north from anchor (m)</th><th>{labels[-1]} common / independent</th><th>Ratio (dB)</th></tr>{rows}</table><img src="component-phase-closure.png"><p><a href="../phase-histories/phase-histories.png">Quarter-second phase histories</a> · <a href="analysis.json">Native scores and overlap information</a></p></html>'''
    (out/'index.html').write_text(html);print(labels[-1],'coherent ratios',serial(folds[-1]['positive_amplitude_to_independent_ratio']),flush=True)


if __name__=='__main__':main()
