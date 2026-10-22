"""Compare equal-sized carrier searches for direct and HH-assisted primaries."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_ais_phase import NAMES
from report_joint_carrier_cycles import height,unused


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direct',type=Path,default=Path('docs/data/isar_20261017/joint-carrier-cycles/fits.json'))
    parser.add_argument('--sea',type=Path,default=Path('docs/data/isar_20261017/joint-carrier-cycles-hh/fits.json'))
    parser.add_argument('--assignment',default='00002')
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/carrier-cycle-propagation'))
    args=parser.parse_args();paths=[args.direct,args.sea];data=[json.loads(p.read_text()) for p in paths]
    if not all(d['complete'] for d in data):raise ValueError('Complete both model searches first')
    for key in ('common_starts','integer_candidates','intervals'):
        if data[0][key]!=data[1][key]:raise ValueError('Search scopes differ: '+key)
    if [d['primary_propagation'] for d in data]!=['direct','transmit']:
        raise ValueError('Expected direct and transmit-side HH models')
    groups=[next(g for g in d['groups'] if g['assignment']==args.assignment) for d in data]
    chosen=[min(g['common_trials'],key=lambda r:r['cost']) for g in groups]
    controls=[min([*g['independent_trials'],g['relaxed_control']],key=lambda r:r['cost']) for g in groups]
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    result=dict(accepted_calibration=False,assignment=args.assignment,
        integer_candidates=data[0]['integer_candidates'],refined_starts=data[0]['common_starts'],
        model_names=['Direct primary','Direct + transmit-side HH reflection'],groups=[],
        source_sha256={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in [*paths,Path(__file__)]})
    for label,group,row,control in zip(result['model_names'],groups,chosen,controls):
        result['groups'].append(dict(model=label,cost=row['cost'],unused_error=unused(row),
            strict=row['audit']['strict_local_minimum'],strict_starts=sum(r['audit']['strict_local_minimum'] for r in group['common_trials']),
            independent_cost=control['cost'],independent_unused_error=unused(control),
            maximum_tx_up_correction_m=height(row),
            maximum_tx_horizontal_correction_m=float(np.linalg.norm(np.asarray(row['antenna_fit']['transmitter_offsets_enu_m'])[:,:2],axis=1).max()),
            rx_correction_enu_m=row['antenna_fit'].get('receiver_offset_enu_m'),bound_indices=row['bound_indices']))
    result['hh_unused_error_change_percent']=100*(unused(chosen[1])/unused(chosen[0])-1)
    result['hh_training_error_change_percent']=100*(chosen[1]['cost']/chosen[0]['cost']-1)
    fig,axes=plt.subplots(1,3,figsize=(15,5.4),layout='constrained')
    colours=['#24699b','#bd6b22']
    for i,(group,row,label,colour) in enumerate(zip(groups,chosen,result['model_names'],colours)):
        for j,trial in enumerate(group['common_trials']):
            marker='o' if trial['audit']['strict_local_minimum'] else 'x';x=i+(j-(len(group['common_trials'])-1)/2)*.06
            values=[trial['cost'],100*(unused(trial)/unused(chosen[0])-1),height(trial)]
            for ax,y in zip(axes,values):
                ax.plot(x,y,marker,color=colour if trial is row else '#aaa',ms=8 if trial is row else 5)
        for ax in axes:ax.set_xticks([0,1],['Direct','Direct + HH'])
    axes[0].set(ylabel='Original training objective',title='All refined carrier starts')
    axes[1].set(ylabel='Unused error relative to selected direct fit (%)',title='Frozen prediction coefficients');axes[1].axhline(0,color='#bbb',lw=.7)
    axes[2].set(ylabel='Largest |TX height correction| (m)',title='Physical plausibility remains separate')
    fig.suptitle('Equal search size: 64 quadratic lattice candidates and six raw-data fits per model\nColour: lowest training objective · grey: other starts · ×: numerical audit incomplete',fontsize=12)
    fig.savefig(out/'01-direct-vs-hh.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for interval,label in enumerate(('F01','Flash')):
        base=np.array([p['heldout_baseline_error'] for p in chosen[0]['prediction'][interval]])
        for row,name,colour in zip(chosen,result['model_names'],colours):
            errors=np.array([p['heldout_error'] for p in row['prediction'][interval]])
            axes[interval,0].plot(NAMES,100*(1-errors/base),'o-',color=colour,label=name)
        for channel,name in enumerate(NAMES):
            values=[row['primary_observables'][interval][channel][2] for row in chosen]
            axes[interval,1].plot(range(2),values,'o-',label=name)
        axes[interval,0].set(title=label+' · unused complex-data prediction',ylabel='Explained energy relative to stationary baseline (%)');axes[interval,0].legend(fontsize=8)
        axes[interval,1].set(title=label+' · fitted physical primary curvature',ylabel='Geometric excess-path curvature (m/s²)',xticks=[0,1],xticklabels=['Direct model','Direct + HH model']);axes[interval,1].legend(fontsize=8)
    fig.suptitle('A change in fitted physical motion need not imply a comparable prediction gain\nCurvature is the direct geometric primary path; HH interference contributes additional observed phase',fontsize=12)
    fig.savefig(out/'02-prediction-and-motion.png',dpi=170);plt.close(fig)
    rows=''.join(f"<tr><td>{r['model']}</td><td>{r['cost']:.3f}</td><td>{r['unused_error']:.3f}</td><td>{r['strict']}</td><td>{r['maximum_tx_up_correction_m']:.2f} m</td><td>{r['maximum_tx_horizontal_correction_m']:.2f} m</td></tr>" for r in result['groups'])
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Carrier search: direct versus HH sea assistance</title>
<style>body{{font:16px/1.55 system-ui;max-width:1200px;margin:30px auto;padding:0 20px}}img{{width:100%}}td,th{{padding:10px;border:1px solid #ddd;text-align:left}}table{{border-collapse:collapse}}</style>
<h1>Direct versus HH sea-assisted scattering after broader carrier search</h1>
<p>No calibration is accepted. For assignment {args.assignment}, the selected HH model changes unused-sample error by {result['hh_unused_error_change_percent']:+.3f}% relative to the selected direct model, and training objective by {result['hh_training_error_change_percent']:+.3f}%. Lower is better. Both use the same observations, masks, parameter count, correlated antenna priors and bounds. Each fits six starts selected from 64 integer-cycle proposals, generated from its own full local quadratic. This equal effort does not establish global optimality.</p>
<table><tr><th>Model</th><th>Training objective</th><th>Unused error</th><th>Strict selected fit</th><th>Largest |TX up| correction</th><th>Largest TX horizontal correction</th></tr>{rows}</table>
<p>The HH hypothesis is a coherent sum of the direct ship echo and its transmit-side sea-assisted ray, with curved-water path lengths, spreading and Fresnel factors. Both rays share the same physical primary and real brightness function. There is no fitted sea-ray phase or amplitude. Smooth-water relative permittivity 80, conductivity 4 S/m and the existing sea datum are assumptions; sea roughness, tide and ship bistatic scattering differences are not measured. A direct ocean echo alone is a different hypothesis.</p>
<img src="01-direct-vs-hh.png" alt="Equal-effort direct and HH comparison">
<img src="02-prediction-and-motion.png" alt="Per-band prediction and physical motion comparison">
<p>The phase-cycle search removes an avoidable optimization limitation, but lower waveform error alone cannot validate antenna coordinates, identify the physical scatterer, or establish a full-capture motion trajectory. Broad uncertainty, active bounds and alternative scene explanations remain visible in the individual reports. These same inspected data are used for all comparisons; unused-block coefficients stay frozen.</p>
<h2>Relevant prior art and the next methodological boundary</h2>
<p><a href="https://cris.tau.ac.il/en/publications/joint-self-calibration-of-receiver-geometry-timing-and-target-pos/">Weiss et al., 2026, joint multistatic self-calibration</a> distinguish measurement information from location/timing priors and characterize geometric gauge freedoms before sharpness refinement. Their common-transmitter, multiple-receiver model is different from this multiband passive setup; it supports checking observability rather than treating optimizer convergence as a position survey. The exact rotation gauge for our own forward operator is already covered by the synthetic tests.</p>
<p><a href="https://iris.uniroma1.it/handle/11573/1339039">Pisciottano, Cristallini and Pastina, 2019</a> demonstrate maritime DVB-T/DVB-S backprojection with known ship motion. This supports exact bistatic backprojection as the image-formation reference, but does not establish unknown antenna calibration or a common scattering phase across our five VHF bands.</p>
<p><a href="https://ira.lib.polyu.edu.hk/handle/10397/99517">He et al., 2023</a> use long-observation translational motion, Doppler-history velocity estimation and subsequent image fusion for GNSS maritime imaging. That motivates recovering persistent motion from multiple intervals independently of the brightest flash. Image fusion in that study is not evidence that arbitrary inter-band phase alignment is physically coherent. The next calibration evidence must therefore come from stable, spatially diverse trajectories and explicit prediction across them; the inspected flash alone remains insufficient in these fits.</p>
<p><a href="../joint-carrier-cycles/index.html">Direct cycle search</a> · <a href="../joint-carrier-cycles-hh/index.html">HH cycle search</a> · <a href="analysis.json">Numerical comparison</a> · <a href="../index.html">Analysis index</a></p></html>'''
    (out/'index.html').write_text(page);(out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
