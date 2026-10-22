"""Review measured candidate trajectories with explicit unused-time tests."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_ais_phase import NAMES,COLOURS
from fit_measured_phase_arcs import make_bands
from polynomial_phase_arc import PolynomialPhaseArc
from report_measured_scatterer_phases import load_fields,serial


def selected(arc,family):
    degree=arc['selected_'+family+'_degree']
    return next(m for m in arc['models'] if m['degree']==degree)[family]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/piecewise-measured-phases/fits.json'))
    args=parser.parse_args();out=args.source.parent;data=json.loads(args.source.read_text())
    if not data['complete']:raise ValueError('Wait for completed fits')
    fields=load_fields(Path('build/isar-20261015/full-fields'),Path('docs/data/calibration_20261016_sites'))
    arcs=data['arcs'];reproduced=0;maximum=0.;summaries=[]
    for arc in arcs:
        first,last=arc['interval_s'];bands=make_bands(fields,first,last,arc['origin_samples'])
        for model in arc['models']:
            degree=model['degree']
            for fitter,record in [(PolynomialPhaseArc(bands,first,last,degree),model['common']),
                *[(PolynomialPhaseArc([b],first,last,degree),r) for b,r in zip(bands,model['independent'])]]:
                residual,_=fitter.evaluate(np.asarray(record['parameters']));cost=float(residual@residual*fitter.energy)
                error=abs(cost-record['cost'])/max(1.,record['cost']);maximum=max(maximum,error);reproduced+=1
        common=selected(arc,'common');independent=selected(arc,'independent')
        common_power=sum(r['testing']['matched_power'] or 0 for r in common['scores'])
        independent_power=sum(r['scores'][0]['testing']['matched_power'] or 0 for r in independent)
        summaries.append(dict(window=arc['window'],point=arc['point'],interval_s=arc['interval_s'],
            common_degree=common['degree'],independent_degree=independent[0]['degree'],
            final_error_ratio=arc['selected_final_error_ratio_common_to_independent'],
            common_matched_power_vs_independent_db=float(10*np.log10(common_power/independent_power)) if common_power*independent_power>0 else None,
            common_phase_rms_deg=[r['testing']['phase_rms_deg'] for r in common['scores']],
            independent_phase_rms_deg=[r['scores'][0]['testing']['phase_rms_deg'] for r in independent],
            common_projected_gradient=common['maximum_projected_gradient'],common_bound_indices=common['bound_indices']))
    if maximum>1e-10:raise ValueError(f'Saved likelihood reproduction failed {maximum}')
    labels=[f"{a['window']}-{a['point']} {a['interval_s'][0]:g}–{a['interval_s'][1]:g}" for a in arcs]
    fig,axes=plt.subplots(1,3,figsize=(15,16),layout='constrained',gridspec_kw={'width_ratios':[1,1,.9]})
    matrices=[np.array([r[key] for r in summaries],float) for key in ('independent_phase_rms_deg','common_phase_rms_deg')]
    upper=max(90.,float(np.ceil(np.nanmax(matrices)/10)*10))
    for ax,matrix,title in zip(axes[:2],matrices,['Independent per-band path','One shared path']):
        image=ax.imshow(matrix,aspect='auto',cmap='viridis_r',vmin=0,vmax=upper)
        ax.set_xticks(range(5),NAMES,rotation=30);ax.set_yticks(range(len(arcs)),labels,fontsize=7)
        ax.set_title(title)
        for i in range(len(arcs)):
            for j in range(5):
                if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',fontsize=6,color='black' if matrix[i,j]<upper*.45 else 'white')
    y=np.arange(len(arcs));ratio=np.array([r['final_error_ratio'] for r in summaries]);axes[2].plot(ratio,y,'o',ms=4)
    axes[2].axvline(1,color='#555',ls='--',lw=1);axes[2].set(ylim=(len(arcs)-.5,-.5),xlabel='Shared / independent prediction error',title='Final unused-quarter error')
    axes[2].set_yticks(y,labels,fontsize=7);fig.colorbar(image,ax=axes[:2],label='Final unused-quarter phase RMS (degrees)',shrink=.6)
    fig.suptitle('Thirty-eight two-second candidate arcs: phase compatibility tested on unused samples\n'
        'Quadratic/cubic degree chosen on a separate validation quarter. Brightness and constant phase fit on training only.',fontsize=13)
    fig.savefig(out/'01-unused-prediction.png',dpi=165);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
    for row,point in enumerate(('B','C')):
        for col,first in enumerate((163.,164.,165.)):
            ax=axes[row,col];arc=next(a for a in arcs if a['window']=='W3' and a['point']==point and a['interval_s'][0]==first)
            bands=make_bands(fields,*arc['interval_s'],arc['origin_samples']);ind=selected(arc,'independent');common=selected(arc,'common')
            grid=np.linspace(*arc['interval_s'],180)
            for ch,(band,fit,colour) in enumerate(zip(bands,ind,COLOURS)):
                model=PolynomialPhaseArc([band],*arc['interval_s'],fit['degree']);p=np.array(fit['parameters'])
                path=model.trajectory(p,grid);ax.plot(grid,100*path,color=colour,lw=1.5,label=NAMES[ch])
                t=np.asarray(fit['prediction'][0]['time_s']);use=np.asarray(fit['scores'][0]['testing']['use'])
                innovation=np.radians(np.asarray(fit['prediction'][0]['phase_residual_deg']))
                observed=model.trajectory(p,t)-innovation/band.k
                ax.plot(t[use],100*observed[use],'.',color=colour,ms=2,alpha=.5)
            model=PolynomialPhaseArc(bands,*arc['interval_s'],common['degree'])
            ax.plot(grid,100*model.trajectory(common['parameters'],grid),color='black',ls='--',lw=1.7,label='Shared fit')
            ax.set(title=f"{point} · {first:g}–{first+2:g} s",xlabel='Capture time (s)',ylabel='Residual path relative to midpoint (cm)')
    axes[0,0].legend(ncol=3,fontsize=8)
    fig.suptitle('Independent short phase trajectories around the flash\n'
        'Solid: independent fits. Dashed: shared fit. Dots: final-quarter measured circular innovations around each fit.',fontsize=13)
    fig.savefig(out/'02-flash-local-trajectories.png',dpi=170);plt.close(fig)
    rows=''.join(f'<tr><td>{label}</td><td>{s["common_degree"]}/{s["independent_degree"]}</td><td>{s["final_error_ratio"]:.3f}</td><td>{s["common_matched_power_vs_independent_db"]:+.3f}</td><td>{s["common_projected_gradient"]:.2g}</td><td>{s["common_bound_indices"]}</td></tr>' for label,s in zip(labels,summaries) if s['common_matched_power_vs_independent_db'] is not None)
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Piecewise measured ship phase trajectories</title>
<style>body{{font:16px/1.5 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:6px;border:1px solid #ddd}}.note{{background:#eef4f8;padding:15px}}</style>
<h1>Short measured trajectories: independent versus shared paths</h1>
<p class="note">This extends the <a href="../measured-scatterer-phases/index.html">selected-point measurements</a> to 38 overlapping two-second arcs. It fits the original complex samples with each carrier’s wavenumber and exact saved original-row phase averaging. There is no new clutter fit, multipath model, antenna adjustment or arbitrary phase screen.</p>
<p>Every candidate has five constant scattering phases and nonnegative real brightness. The independent model has five polynomial paths; the shared model has one path plus the nominal geometric interband difference. A repeating 125 ms partition assigns half the data to fitting, one quarter to choosing quadratic/cubic degree and the last quarter to frozen prediction. Brightness on unused blocks is interpolated from training values; it is never re-estimated there. These observations have previously been inspected, so this is internal validation.</p>
<h2>The useful flash interval</h2><p>The central cell B at 164–166 seconds is predicted about equally well with a shared trajectory and independent trajectories. Its final-quarter shared/independent squared-error ratio is {next(s for s in summaries if s['window']=='W3' and s['point']=='B' and s['interval_s'][0]==164.)['final_error_ratio']:.3f}. The neighbouring cells do not all pass the same comparison. This is a reason to resolve overlapping range structure jointly before treating scalar-cell phase as a material point measurement.</p>
<a href="02-flash-local-trajectories.png"><img src="02-flash-local-trajectories.png" alt="Local fitted trajectories and unused measured innovations around the ship flash"></a>
<p>Each arc has an independent constant phase origin; curves are not connected between arcs. Dots use the measured circular innovation relative to the displayed independent fit, so they retain a wavelength ambiguity. The earlier unfitted phase plots provide a separate view of that ambiguity. These are phase-equivalent paths of resolution cells, not yet a physical ship motion solution. Range-envelope shape is deliberately not fitted in this first comparison.</p>
<h2>All local candidates</h2><a href="01-unused-prediction.png"><img src="01-unused-prediction.png" alt="Independent and shared trajectory prediction for every candidate arc"></a>
<p>A prediction-error ratio near one means sharing a trajectory does not materially worsen this local prediction. It does not prove a correct scatterer association, antenna location or low phase error: the phase-RMS panels retain weak and inconsistent candidates. Likewise, convergence of a polynomial fit is not a physical calibration.</p>
<table><tr><th>Local candidate</th><th>Shared/independent degree</th><th>Unused error ratio</th><th>Shared matched power versus independent (dB)</th><th>Projected gradient</th><th>Bounds</th></tr>{rows}</table>
<p>All {reproduced} selected-degree-family objectives reproduce from their saved parameters; maximum relative discrepancy {maximum:.3g}. The exact profiled residual derivative, unequal carriers, phase gauges, missing original rows, unused-data isolation and band-specific drift are covered by numerical tests. <a href="tests.log">Test results</a>.</p>
<p><a href="fits.json">All fits and starts</a> · <a href="analysis.json">Summary and reproduction evidence</a> · <a href="../shared-echo-cluster/index.html">Joint range-resolved echo follow-up</a> · <a href="../index.html">Analysis index</a></p></html>''')
    summary=dict(scope=__doc__,arcs=summaries,reproduced_objectives=reproduced,maximum_objective_error=maximum,
        source_sha256={str(args.source):hashlib.sha256(args.source.read_bytes()).hexdigest(),str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (out/'analysis.json').write_text(json.dumps(serial(summary),indent=2,allow_nan=False));print(out/'index.html',flush=True)


if __name__=='__main__':main()
