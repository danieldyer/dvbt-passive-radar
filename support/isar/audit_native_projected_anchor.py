"""Excluded-sample phase transfer and temporal support of the fixed anchor.

Every physical/motion/visibility/band-phase parameter is frozen from training.
The constant-profile projection is a known linear nuisance elimination on each
fold; there is no per-band phase correction or target fit on excluded data.
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
from rigid_patch import RigidPatch
from stationary import Stationary
from report_measured_scatterer_phases import serial


def support(time,weight):
    cumulative=np.r_[0.,np.cumsum(weight)];total=cumulative[-1];best=None
    for first in range(len(time)):
        end=int(np.searchsorted(cumulative,cumulative[first]+.9*total))
        if end>len(time):break
        candidate=[float(time[first]),float(time[max(first,end-1)])]
        if best is None or candidate[1]-candidate[0]<best[1]-best[0]:best=candidate
    mean=float(np.dot(time,weight)/total);variance=float(np.dot((time-mean)**2,weight)/total)
    return dict(shortest_90_percent_energy_interval_s=best,
        shortest_90_percent_duration_s=best[1]-best[0],energy_mean_time_s=mean,
        moment_equivalent_rectangular_duration_s=float(np.sqrt(12*variance)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,default=Path('docs/data/isar_20261018/native-spatial-validation/ais-near-projected-likelihood-brightness-full-source'))
    p.add_argument('--fit',type=Path,default=Path('docs/data/isar_20261018/native-spatial-validation/ais-near-three-point-30s'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    image=json.loads((a.image/'analysis.json').read_text());fit=json.loads((a.fit/'render-fit.json').read_text());model=json.loads((a.fit/'analysis.json').read_text())
    first,last=image['interval_s'];mid=fit['midpoint_s'];state=np.asarray(fit['coarse_state_enu']);parameters=np.asarray(fit['rigid_parameters'])
    fields,rotation,rx,tx,_,_=load_fields(Path(image['source_profiles']),offsets=np.arange(-12,13),fixed_epoch_s=mid,interval_s=[first,last])
    gain=np.asarray(fit['fits']['constant_velocity_gls']['coefficients_real_imag'])[:,0];gain=gain[:,0]+1j*gain[:,1]
    brightness=model['phase_histories']['brightness_model'];scores=[];information=[];phases=[];supports=[]
    fig,axs=plt.subplots(5,1,figsize=(11,10),sharex=True,layout='constrained')
    for c,f in enumerate(fields):
        t=f.data['row_time'];dt=t-mid
        kin=RigidPatch(t,mid,rx+(state[0]+dt[:,None]*state[1]+.5*dt[:,None]**2*state[2])@rotation,
            (state[1]+dt[:,None]*state[2])@rotation,rotation.T,np.asarray(fit['reference_transmitter_ecef_m']),rx,
            translation_direction=fit.get('translation_direction_ecef'))
        offset,velocity=kin.point_state(parameters,np.zeros(3),origin=f.data['centre_ecef_m'])
        raw=f.raw_trajectory(f.data['centre_ecef_m']+offset,velocity,position_offset=offset)
        coefficient=np.asarray(brightness['bands'][c]['coefficients'])[0];knots=np.asarray(brightness['bands'][c]['knots_s'])
        visibility=linear_basis(t,knots)@coefficient/coefficient.mean()
        h=f.apply(average(raw*visibility[:,None],f.data['row_information'],16)[0])
        energy=np.sum(abs(h*gain[c])**2,axis=1);s=support(f.time,energy);supports.append(s)
        density=energy/energy.sum()/np.median(np.diff(f.time));axs[c].plot(f.time,density)
        axs[c].axvspan(*s['shortest_90_percent_energy_interval_s'],color='C1',alpha=.2)
        axs[c].set(ylabel=f'{NAMES[c]}\nenergy density (1/s)',title=f"Shortest 90% energy interval: {s['shortest_90_percent_duration_s']:.2f} s; moment-equivalent duration: {s['moment_equivalent_rectangular_duration_s']:.2f} s")
        train=training_mask(f.time,model['interval_s'][0]);band_score=[];band_information=[]
        for mask in [np.ones(len(f.time),bool),train,~train]:
            selected=np.repeat(mask,len(f.offsets));nu=Stationary(stationary_profile_design(f),selected)
            template=nu.remove(h.ravel()[selected]);data=nu.remove(f.y.ravel()[selected])
            band_score.append(np.vdot(template,data));band_information.append(np.vdot(template,template).real)
        scores.append(band_score);information.append(band_information)
    axs[-1].set_xlabel('Capture time (s)');fig.suptitle('Information support of the training-fitted anchor visibility\nArea is one per band; orange interval contains 90% of predicted component energy; no observation is renormalised')
    fig.savefig(a.output/'anchor-visibility-support.png',dpi=150);plt.close(fig)
    scores,information=np.asarray(scores),np.asarray(information);folds=[]
    for j,label in enumerate(['all','training','excluded']):
        calibrated=scores[:,j]*np.exp(-1j*np.angle(gain))/np.sqrt(information[:,j])
        positive,phase=statistic(calibrated[:,None]);independent=np.sum(abs(calibrated)**2)
        coherent=abs(np.vdot(gain,scores[:,j]))**2/np.dot(abs(gain)**2,information[:,j])
        ratio=float(positive[0]/independent)
        folds.append(dict(fold=label,positive_amplitude_to_independent_ratio=ratio,
            positive_amplitude_to_independent_db=float(10*np.log10(ratio)),
            fixed_spectrum_to_independent_ratio=float(coherent/independent),
            calibrated_band_phase_deg=np.degrees(np.angle(calibrated)),
            independent_statistic=float(independent),scores_real_imag=np.c_[scores[:,j].real,scores[:,j].imag],information=information[:,j]))
    result=dict(scope=__doc__,complete=True,folds=folds,source_support_by_band=supports,
        target_parameters_refit_on_excluded=False,stationary_operation='Known fixed linear projection separately on each fold; target and data receive the same projection',
        full_duration_ship_focus_certified=False)
    (a.output/'anchor-transfer.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    rows=''.join(f'<tr><td>{v["fold"]}</td><td>{v["positive_amplitude_to_independent_ratio"]:.2%}</td><td>{v["positive_amplitude_to_independent_db"]:.3f}</td><td>'+', '.join(f'{x:+.1f}°' for x in v['calibrated_band_phase_deg'])+'</td></tr>' for v in folds)
    (a.output/'anchor-transfer.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Native anchor transfer</title><style>body{{font:16px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:0 24px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}}</style>
<h1>Native-symbol excluded-data phase transfer</h1><table><tr><th>Fold</th><th>Shared-phase positive-amplitude / independent-band statistic</th><th>Ratio (dB)</th><th>Calibrated phases: Seven / SBS / Nine / Ten / ABC</th></tr>{rows}</table>
<p>Physical motion, real visibility and band calibration stay frozen from training. Each fold receives the same known constant-profile projection in its data and template. This profiles the stationary nuisance for that fold; it does not fit a target phase or trajectory. The ratio is a conditional coherent-model comparison, not a universal target-SNR bound or proof of globally unique motion.</p>
<img src="anchor-visibility-support.png"><p>The nominal interval is almost 30 seconds, but the current template's predicted signal energy is not uniform throughout it. The 90% interval and time-variance equivalent quantify that distinction. These are properties of the training-fitted component model, not a claim that the physical ship disappears outside those intervals.</p>
<p><a href="index.html">Image and point-spread validation</a> · <a href="anchor-transfer.json">Native numerical results</a>.</p></html>''')
    print(json.dumps(serial(dict(folds=folds,support=supports))),flush=True)


if __name__=='__main__':main()
