"""Publish physical CV constraints, errors, prediction and bound diagnostics."""
import argparse,json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from recorded_injection_fields import load_fields
from recovery_experiment import Experiment
from report_measured_scatterer_phases import serial
from validate_recorded_injection import metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('docs/data/injection_20261018'))
    args=parser.parse_args();root=args.root;out=root/'constant-velocity-study';out.mkdir(exist_ok=True)
    names=['Seven','SBS','Nine','Ten','ABC'];cases=[]
    for directory in ['constant-velocity','constant-velocity-acceleration-1mm','constant-velocity-acceleration-3mm','constant-velocity-acceleration-3mm-wide','constant-velocity-plus20db']:
        p=root/directory/'analysis.json'
        if p.exists():
            x=json.loads(p.read_text())
            if x['complete']:cases.append((directory,x))
    fields,rotation,rx,tx,state,mid=load_fields(Path('build/injection-20261018/deterministic-baseline/profiles.bin'))
    state[2]=np.zeros(3);train=[f.time<161.5 for f in fields]
    ex=Experiment(fields,rotation,rx,tx,state,mid,training=train)
    comparisons={};phase={}
    for directory,x in cases:
        with np.load(root/directory/'observations.npz') as saved:clean=[saved[f'injected_{c}'] for c in range(5)]
        records={}
        for label,fit in x['fits'].items():
            actual=[];true=[];testing_actual=[];testing_true=[]
            for c,f in enumerate(fields):
                a=np.asarray(fit['coefficients_real_imag'][c])[0];a=complex(*a)
                p=ex.templates(np.asarray(fit['parameters']),c)[:,0]*a
                testing=np.repeat(~train[c],len(f.offsets))
                actual.append(p);true.append(clean[c]);testing_actual.append(p[testing]);testing_true.append(clean[c][testing])
                if label=='constant_velocity_gls':
                    cross=np.sum(clean[c].reshape(f.y.shape).conj()*p.reshape(f.y.shape),axis=1)
                    phase[directory,c]=np.degrees(np.unwrap(np.angle(cross)))
            parameters=np.asarray(fit['parameters'])
            active=fit.get('active_bound_coordinates',[])
            if not active:
                half=fit.get('velocity_halfwidth_mps',1.)
                active=np.flatnonzero(np.isclose(abs(parameters[:4]),[10.,10.,half/.5,half/.5],atol=1e-5)).tolist()
            records[label]=dict(full=metrics(np.concatenate(actual),np.concatenate(true)),
                excluded=metrics(np.concatenate(testing_actual),np.concatenate(testing_true)),active_bounds=active,
                converged=fit['success'],position_error_m=x['scores'][label]['horizontal_error_at_epoch_m'],
                velocity_error_mps=x['scores'][label]['velocity_error_at_epoch_mps'])
        comparisons[directory]=records
    colors=dict(constant_velocity='#2c7fb8',free_acceleration='#c66a36',constant_velocity_gls='#32805a')
    labels=dict(constant_velocity='Constant velocity',free_acceleration='Free acceleration',constant_velocity_gls='Constant velocity + covariance')
    curve=[('constant-velocity',0.),('constant-velocity-acceleration-1mm',1.),('constant-velocity-acceleration-3mm-wide',3.)]
    curve=[(name,a) for name,a in curve if name in comparisons]
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for method in colors:
        x=[a for _,a in curve];values=[comparisons[name][method] for name,_ in curve]
        for ax,key in zip(axes,['position_error_m','velocity_error_mps','loss']):
            y=[v[key] if key!='loss' else v['excluded']['coherence_loss_db'] for v in values]
            ax.plot(x,y,'o-',label=labels[method],color=colors[method])
    for ax,title in zip(axes,['Position error at 157 s (m)','Velocity error at 157 s (m/s)','Frozen five-band template loss (dB)']):
        ax.set(xlabel='True physical east acceleration (mm/s²)',ylabel=title);ax.set_yscale('log')
    axes[0].legend(fontsize=8);fig.suptitle('Same point strength and recorded background; 19.5 s fitting, 10.5 s excluded\nThe 3 mm/s² case uses wider velocity bounds; no truth-seeded fits')
    fig.savefig(out/'constraint-tradeoff.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(5,1,figsize=(11,11),sharex=True,layout='constrained')
    for c,(ax,f) in enumerate(zip(axes,fields)):
        for name,label in [('constant-velocity','0 mm/s²'),('constant-velocity-acceleration-1mm','1 mm/s²'),('constant-velocity-acceleration-3mm-wide','3 mm/s², wider bounds')]:
            if (name,c) in phase:ax.plot(f.time,phase[name,c],label=label)
        ax.axvline(161.5,color='#777',ls='--');ax.axhline(0,color='#aaa',lw=.5)
        ax.set(ylabel=f'{names[c]}\nphase error (°)')
    axes[0].legend(fontsize=9);axes[-1].set_xlabel('Capture time (s)')
    fig.suptitle('Constant-velocity + covariance: frozen predicted point phase versus truth\nBand coefficients are fitted on training data; no rephasing on excluded data')
    fig.savefig(out/'frozen-phase-errors.png',dpi=150);plt.close(fig)
    rows=[]
    for directory,records in comparisons.items():
        for label,z in records.items():
            flag=('Bound: '+str(z['active_bounds'])) if z['active_bounds'] else ('Interior' if z['converged'] else 'Not converged')
            rows.append(f'<tr><td>{directory}</td><td>{labels[label]}</td><td>{z["position_error_m"]:.2f}</td><td>{z["velocity_error_mps"]:.4f}</td><td>{z["excluded"]["coherence_loss_db"]:.5f}</td><td>{flag}</td></tr>')
    information=json.loads((root/'constant-velocity/conditional-position-information.json').read_text())
    result=dict(scope=__doc__,comparison=comparisons,conditional_position_information=information,
        production_changed=False,coherent_ship_recovery_accepted=False)
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Constant-velocity recovery study</title>
<style>body{{font:16px/1.6 system-ui;max-width:1350px;margin:30px auto;padding:0 24px}}img{{width:100%}}table{{border-collapse:collapse;font-size:14px}}td,th{{padding:8px 10px;border-bottom:1px solid #ddd;text-align:left}}.note{{padding:16px;background:#eef6fa}}</style>
<h1>Does constant velocity resolve the injected-point ambiguity?</h1>
<p class="note">It substantially improves velocity recovery when the assumption is true, but does not remove the weak geographic direction. Small true accelerations can instead be absorbed as erroneous velocity. Neither a smaller position error nor a sharper-looking image is a sufficient acceptance criterion.</p>
<p>The point is 500 m along the equal-range contour from the AIS reference. All five bands retain the original recorded background, exact symbol epochs, C0/C1 waveform responses and fixed stationary nuisance. Acquisition sees the sum only, starts at the coarse AIS position/velocity and uses a zero-acceleration physical reference. The injected position, velocity and complex band coefficients are hidden. The final 10.5 s is excluded from acquisition, covariance estimation, fitting and band calibration. Covariance spectral resolution uses excluded-frequency likelihood on the training residual. Each band has a constant complex coefficient, with one global complex alignment removed only by the reported energy-loss statistic.</p>
<p>The 1 and 3 mm/s² accelerations are physical east accelerations. They are not corrections to the AIS interpolation. The optional +20 dB control changes point power, not recorded background or fitting weights by hand. It is a sensitivity control, not a replacement acceptance test.</p>
<img src="constraint-tradeoff.png" alt="Position, velocity and frozen template errors versus true acceleration">
<table><tr><th>Case</th><th>Model</th><th>Position error (m)</th><th>Velocity error (m/s)</th><th>Excluded combined loss (dB)</th><th>Bounds/convergence</th></tr>{''.join(rows)}</table>
<p>The original ±1 m/s velocity-correction bounds are reached in the 3 mm/s² constant-velocity fits. The wider control permits ±3 m/s and reports active bounds explicitly. A bound-limited result is not accepted as recovered motion. The free-acceleration model allows ±0.04 m/s² planar correction.</p>
<h2>Phase prediction across channels</h2><img src="frozen-phase-errors.png" alt="Fitted point phase error by band, without excluded-data phase correction">
<p>These curves compare the fitted point kernel against the independently generated point. They are evaluation diagnostics, not phase corrections applied to the mixed scene. The five-band loss concatenates the frozen complex predictions before scoring, so differential band-phase errors remain visible.</p>
<h2>What the information calculation says</h2>
<p>For the zero-acceleration point, after eliminating unknown velocity and band coefficients, the nominal thermal weak-direction position standard deviation is {information['thermal']['principal_sigma_m'][-1]:.2f} m. Under the covariance learned from the actual residual, its working-model conditional CRLB is {information['covariance']['principal_sigma_m'][-1]:.2f} m. The 68 m error is about 1.6 times that latter scale. The covariance assumes a Gaussian block-circulant background with independent Fourier vectors; this number is not an established fundamental bound for the arbitrary real scene. Independent coloured-noise recovery tests are being used to test estimator efficiency under that model.</p>
<p>The geometry has two physical transmitter centres, with four bands at one centre and ABC at the other, about 326 m apart. The limited angular diversity leaves position and velocity strongly coupled. Multiple frequencies at one centre add measurement precision, rather than independent geometric look directions.</p>
<h2>Ship attempt and rejected changes</h2>
<p>The same CV acquisition was also applied to the uninjected ship return. Its initial fits reach bounds or fail to converge; the frozen one-point prediction improves ABC but worsens the other four bands relative to a static-only comparator. This is not an accepted five-band ship-motion fit. More flexible neighbouring-echo and range-projection controls also failed to improve the point's geographic error. Those experimental nuisance projections are not applied to production data.</p>
<p><a href="analysis.json">All comparisons</a> · <a href="../separated-500m/bias-explanation/index.html">Quantitative background-displacement mechanism</a> · <a href="../separated-500m/index.html">Paired profile/IQ control</a>. Reproduction: <code>support/isar/validate_constant_velocity.py</code>, <code>support/isar/report_constant_velocity.py</code>. Production DSP and original captures are unchanged.</p></html>''')


if __name__=='__main__':main()
