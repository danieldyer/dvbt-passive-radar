"""Conditional PGA test on recursively followed quadratic motion branches.

This does not establish scatterer identity. A per-band correction is measured
and saved, never silently adopted as coherent multiband motion or calibration.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gpga import fit as gpga
from higher_order_autofocus import HigherOrderPhase
from quadratic_motion_kernel import QuadraticMotionKernel
from recovery_experiment import Experiment,NAMES,load_fields
from report_measured_scatterer_phases import serial


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();out=args.input;source=json.loads((out/'analysis.json').read_text())
    fields,rot,rx,tx,state,mid=load_fields(149.,179.,8);ex=Experiment(fields,rot,rx,tx,state,mid)
    fig,axes=plt.subplots(5,3,figsize=(16,14),layout='constrained');records=[]
    for c,f in enumerate(fields):
        entry=next(e for e in source['recorded'] if e['channel']==c and e['duration_s']==30.)
        saved=entry['continuation'] or entry;k=QuadraticMotionKernel(f,ex.training[c],ex.designs[c])
        p=np.asarray(saved['parameters']);r,_,alpha,h=k.evaluate(p,f.y.ravel())
        beta=np.linalg.lstsq(k.design[k.selected],f.y.ravel()[k.selected]-h[k.selected]*alpha,rcond=1e-10)[0]
        clean=(f.y.ravel()-k.design@beta).reshape(f.y.shape);h=h.reshape(f.y.shape)
        norm=np.sum(abs(h)**2,axis=1);z=np.sum(h.conj()*clean,axis=1)/norm
        fitted=gpga(f.time,z[:,None],norm,k.training,bandwidths_hz=(1.,.5,.25,.125),iterations_per_bandwidth=16)
        use=k.training
        phi,refinement=HigherOrderPhase(f.time,z,norm,use).fit(fitted['phase_radians'])
        corrected=z*np.exp(-1j*phi);test=~use
        estimate=lambda v:np.sum(norm[use]*v[use])/norm[use].sum()
        before,after=estimate(z),estimate(corrected)
        score=lambda v:abs(np.sum(norm[test]*v[test]))**2/norm[test].sum()
        error=lambda v,a:np.sum(norm[test]*abs(v[test]-a)**2)
        gain=10*np.log10(score(corrected)/score(z));reduction=100*(1-error(corrected,after)/error(z,before))
        record=dict(name=NAMES[c],parameters=p,phase_rad=phi,time_s=f.time,training=use,
            continuation_cost_replay_error=abs(np.vdot(r,r).real-saved['training_rss']),
            raw_matched_coefficient_real_imag=np.c_[z.real,z.imag],conditional_information=norm,
            unused_coherent_power_gain_db=gain,unused_prediction_error_reduction_percent=reduction,
            before_unused_coherent_statistic=score(z),after_unused_coherent_statistic=score(corrected),
            gpga_converged=fitted['converged'],constrained_refinement=refinement,
            phase_constraint='Training-information weighted constant/rate/curvature fixed DURING refinement. Constant is a phase gauge; fixing motion is a model constraint.',
            scope='Conditional one-history phase diagnostic. Static nuisance frozen from training. No reassociation, antenna fit, phase unwrapping or calibrated interband phase claim.')
        records.append(record)
        axes[c,0].plot(f.time,np.degrees(phi));axes[c,0].set(title=NAMES[c]+' · higher-order correction',ylabel='Phase (°)',xlabel='Capture time (s)')
        axes[c,1].plot(f.time[test],np.degrees(np.angle(z[test]*before.conjugate())),'.',ms=3)
        axes[c,1].set(title='Original measured residual',ylim=(-180,180),ylabel='Wrapped phase (°)',xlabel='Unused epoch (s)')
        axes[c,2].plot(f.time[test],np.degrees(np.angle(corrected[test]*after.conjugate())),'.',ms=3)
        axes[c,2].set(title=f'PGA: {gain:+.2f} dB unused coherent power',ylim=(-180,180),ylabel='Wrapped phase (°)',xlabel='Unused epoch (s)')
        print(NAMES[c],'PGA + constrained refinement gain',gain,'unused prediction improvement',reduction,'gradient',refinement['maximum_gradient'],flush=True)
    fig.suptitle('Recursive branch + training-only higher-order PGA · original recording\nPhase concentration is measured; scatterer identity and calibrated multiband coherence remain unproved.')
    fig.savefig(out/'04-real-pga.png',dpi=150);plt.close(fig)
    (out/'continued-autofocus.json').write_text(json.dumps(serial(dict(scope=__doc__,channels=records)),indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for c in range(5):
        entries=[e for e in source['recorded'] if e['channel']==c]
        time=[e['duration_s'] for e in entries]
        axes[0].plot(time,[e['parameters'][1] for e in entries],'o-',label=NAMES[c])
        axes[0].plot(time,[(e['continuation'] or e)['parameters'][1] for e in entries],'--')
        axes[1].plot(time,[(e['continuation']['test_rss']/e['test_rss']-1)*100 if e['continuation'] else 0. for e in entries],'o-')
    axes[0].set(xlabel='Coherent window (s)',ylabel='Excess-path rate (m/s)',title='Solid: global candidate · dashed: recursive continuation');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Coherent window (s)',ylabel='Continuation unused RSS excess (%)',title='Following a branch need not select the strongest echo')
    fig.savefig(out/'05-branch-continuation.png',dpi=150);plt.close(fig)
    body='<h2>Recursive continuation and real-data PGA</h2><img src="05-branch-continuation.png"><img src="04-real-pga.png">'
    body+='<p>Dashed curves follow the short-window branch as the aperture grows. The independently acquired maximum can prefer another coherent return. This demonstrates an association hazard; it does not identify which material scatterer each branch represents. PGA initializes a degree-eight smooth complex-likelihood refinement. Acquired rate/curvature are held fixed during that refinement, rather than stripping a quadratic phase after fitting. Fixing motion is a constraint, not a phase gauge. The unchanged conditional noise metric and frozen unused epochs are retained.</p><ul>'
    for r in records:body+=f'<li>{r["name"]}: unused coherent power {r["unused_coherent_power_gain_db"]:+.2f} dB; unused prediction error reduction {r["unused_prediction_error_reduction_percent"]:+.2f}%; constrained gradient {r["constrained_refinement"]["maximum_gradient"]:.2g}.</li>'
    body+='</ul><p>These conditional phase gains are not geometric motion estimates or proof of calibrated cross-band coherence. Motion phase and receiver phase must be distinguished before any correction is used for geographic formation. <a href="continued-autofocus.json">Measurements and saved corrections</a>.</p>'
    path=out/'index.html';html=path.read_text()
    start=html.find('<h2>Recursive continuation and real-data PGA</h2>')
    finish=html.find('<h2>Literature and next gate</h2>')
    if start>=0:html=html[:start]+html[finish:]
    html=html.replace('<h2>Literature and next gate</h2>',body+'<h2>Literature and next gate</h2>')
    html=html.replace('Constant, linear and quadratic components are explicitly removed so this control cannot silently change image origin or quadratic motion.',
        'For the RMS comparison, constant/linear/quadratic components are projected out of both the truth and estimate. This is an error diagnostic, not a post-fit correction applied to the real data.')
    html=html.replace('Reported PGA gain before that additional quadratic gauge removal:',
        'Held-out gain for the original GPGA correction, before the higher-order-only diagnostic projection:')
    path.write_text(html)


if __name__=='__main__':main()
