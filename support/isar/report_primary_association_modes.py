"""Review competing physical-primary modes without joining target identities."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv
from compare_ais_phase import NAMES,COLOURS
from component_phase import physical_primary_measurements
from fit_physical_primary_association import assignment_code,geometry_for,make_model,prepare_window
from flash_geometry import assignments,observables
from joint_physical_echoes import JointPhysicalEchoes
from measured_phase import range_rate_map
from motion import C
from nearby_echo_control import load,ROOT
from report_fixed_grid_guided_return import phase_comparison
from report_measured_scatterer_phases import serial
from ship_site_signature import licensed_options


def trajectory(g,p,time,midpoint):
    position,velocity,acceleration,tx,rx=g.state(np.asarray(p)[:28]);dt=np.asarray(time)-midpoint
    return np.array([observables(position+t*velocity+.5*t*t*acceleration,
        velocity+t*acceleration,acceleration,tx,rx) for t in dt])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'physical-primary-moving-alternatives/fits.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=json.loads(args.source.read_text())
    if not source['complete']:raise ValueError('Wait for completed alternative association fits')
    if args.output.exists():raise ValueError('Use a fresh directory')
    old=json.loads(Path(source['source']).read_text());empirical=json.loads(Path(old['source']).read_text())
    control=load();track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    options=assignments(licensed_options(control['geometry']));out=args.output;out.mkdir(parents=True)
    errors=[];reviews=[];arc_models=[];arc_parameters=[];maps=[];centres=[]
    for group in source['groups']:
        window=empirical['windows'][group['window_index']];first,last=group['interval_s'];midpoint=(first+last)/2
        bands,parts,models,parameters,baseline=prepare_window(control,window)
        assignment=next(a for a in options if assignment_code(a)==group['assignment'])
        g=geometry_for(control,assignment,midpoint,track);model=make_model(bands,models,control,g,midpoint)
        original=next(v for v in old['groups'] if v['window_index']==group['window_index'])
        variants=[('Original selected mode',group['previous_best']),
            *[(f'Alternative {t["proposal_index"]} · {"".join(t["labels"])} start',t['steps'][-1]) for t in group['trials']]]
        candidates=[];candidates_centres=[]
        for label,fit in variants:
            p=np.asarray(fit['parameters']);r,_=model.evaluate(p);errors.append(abs(r@r*model.energy-fit['cost']))
            rec=[physical_primary_measurements(c,p[index]) for c,index in zip(model.clusters,model.indices)]
            m=phase_comparison(rec,bands,first,last);time=np.linspace(first,last,101);value=trajectory(g,p,time,midpoint)
            candidates.append(dict(label=label,measurement=m,time_s=time,physical_model_observables=value,
                position_offset_enu_m=p[18:21]*g.nuisance_sigma[:3],
                velocity_offset_enu_mps=p[21:24]*g.nuisance_sigma[3:6],
                acceleration_offset_enu_mps2=p[24:27]*g.nuisance_sigma[6:9],
                cost=fit['cost'],gradient=fit['audit']['projected_gradient'],bounds=fit['audit']['bound_indices'],
                unused_change_from_independent_percent=fit['unused_change_from_independent_percent']))
            obs=g.prediction(p[:28]);candidates_centres.append([dict(offset_m=obs[c,0]-np.interp(midpoint,b.field.data['row_time'],b.field.data['centre_path_m']),
                rate_mps=obs[c,1]) for c,b in enumerate(bands)])
        for t in group['trials']:
            for fit in t['steps']:
                r,_=model.evaluate(np.asarray(fit['parameters']));errors.append(abs(r@r*model.energy-fit['cost']))
        rates=np.linspace(-3,3,301)
        maps.append([range_rate_map(b.field.time,b.field.data['value'],b.field.data['information'],b.field.k,rates)/b.field.pulse.samples[64].real for b in bands])
        centres.append(candidates_centres);reviews.append(dict(interval_s=group['interval_s'],variants=candidates))
        arc_models.append(model);arc_parameters.append(np.asarray(group['previous_best']['parameters']))
    joint=JointPhysicalEchoes(arc_models);p=np.zeros(joint.count)
    for index,local in zip(joint.indices,arc_parameters):p[index]=local
    r,_=joint.evaluate(p);joint_cost=float(r@r*joint.energy)
    individual=sum(m.objective(q)[0]*m.energy for m,q in zip(arc_models,arc_parameters))
    if abs(joint_cost-individual)>1e-7:raise ValueError('Joint ownership differs at zero shared antenna offset')
    fig,axes=plt.subplots(sum(len(r['variants']) for r in reviews),3,figsize=(17,12),layout='constrained')
    row=0
    for review in reviews:
        for variant in review['variants']:
            m=variant['measurement'];reference=m['channels'][2]['measured_path_change_m']
            for c,(ch,colour) in enumerate(zip(m['channels'],COLOURS)):
                axes[row,0].plot(m['time_s'],ch['measured_path_change_m'],'.-',color=colour,ms=2,lw=.8,label=NAMES[c])
                if c!=2:axes[row,1].plot(m['time_s'],100*(ch['measured_path_change_m']-reference),'.-',color=colour,ms=2,lw=.8)
                axes[row,2].plot(ch['time_s'],10*np.log10(np.maximum(ch['power_to_noise'],1e-12)),color=colour,lw=.8)
            axes[row,0].set(title=f'{review["interval_s"]} · {variant["label"]}',ylabel='Measured path change (m)')
            axes[row,1].set(title='Channel minus Nine',ylabel='Measured path-change difference (cm)')
            axes[row,2].set(title='Conditional measurement strength',ylabel='Power / conditional noise (dB)')
            for ax in axes[row]:ax.set(xlabel='Capture time (s)',xlim=review['interval_s'])
            row+=1
    axes[0,0].legend(ncol=3,fontsize=8)
    fig.suptitle('Retain competing primary associations before fitting a continuous vessel track\n'
        'Measured complex coefficients after frozen neighbour subtraction; constant origins only. No slope/curvature alignment.',fontsize=12)
    fig.savefig(out/'01-competing-measured-paths.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained');colours=('#0072b2','#d55e00')
    boundary=reviews[0]['interval_s'][1];gaps=[]
    for number,review in enumerate(reviews):
        for s,variant in enumerate(review['variants']):
            value=np.asarray(variant['physical_model_observables']);colour=colours[s%2];style='-' if number==0 else '--'
            label=f'{review["interval_s"]} · '+variant['label']
            axes[0].plot(variant['time_s'],value[:,2,0],style,color=colour,label=label)
            axes[1].plot(variant['time_s'],value[:,2,1],style,color=colour,label=label)
            offset=np.asarray(variant['position_offset_enu_m']);axes[2].plot(offset[0],offset[1],'o' if number==0 else '^',color=colour,label=label)
            axes[2].annotate(f'{review["interval_s"][0]:g} s',offset[:2],xytext=(5,4),textcoords='offset points',fontsize=8)
    axes[0].set(title='Competing fitted range envelopes',xlabel='Capture time (s)',ylabel='Nine excess path (m)')
    axes[1].set(title='Competing fitted phase-derived rates',xlabel='Capture time (s)',ylabel='Nine excess-path rate (m/s)')
    axes[2].set(title='Conditional midpoint offsets from AIS',xlabel='East (m)',ylabel='North (m)',aspect='equal')
    axes[2].plot([-350,350,350,-350,-350],[-350,-350,350,350,-350],':',color='#aaa');axes[2].plot(0,0,'k+',ms=8)
    axes[0].axvline(boundary,color='#aaa',lw=.7);axes[1].axvline(boundary,color='#aaa',lw=.7)
    axes[0].legend(fontsize=7)
    fig.suptitle('A good local phase fit does not by itself establish target identity or ship motion\n'
        'These are fitted conditional trajectories, not independently measured geographic tracks. No antenna coordinates are adopted.',fontsize=11)
    fig.savefig(out/'02-mode-continuity.png',dpi=150);plt.close(fig)
    for a in reviews[0]['variants']:
        for b in reviews[1]['variants']:
            gaps.append(dict(first=a['label'],second=b['label'],boundary_s=boundary,
                fitted_boundary_range_gap_m=np.asarray(b['physical_model_observables'])[0,:,0]-np.asarray(a['physical_model_observables'])[-1,:,0]))
    fig,axes=plt.subplots(len(reviews),5,figsize=(21,8),layout='constrained',sharex=True,sharey=True)
    vmax=np.ceil(10*np.log10(max(m.max() for row in maps for m in row)))
    for row,(review,row_maps,row_centres) in enumerate(zip(reviews,maps,centres)):
        for c,(ax,power) in enumerate(zip(axes[row],row_maps)):
            offsets=control['bands'][c].field.offsets*C/8e6
            image=ax.imshow(10*np.log10(np.maximum(power,1e-8)),origin='lower',aspect='auto',cmap='magma',
                vmin=0,vmax=vmax,extent=[offsets[0]-C/16e6,offsets[-1]+C/16e6,-3,3],interpolation='nearest')
            for s,selection in enumerate(row_centres):
                point=selection[c];ax.plot(point['offset_m'],point['rate_mps'],'s' if s==0 else 'o',mfc='none',mec='#00e5ff' if s==0 else '#72ff64',ms=9)
            ax.set(title=f'{NAMES[c]} · {review["interval_s"]}',xlabel='Archived reference offset (m)',xlim=(-100,100))
            if c==0:ax.set_ylabel('Excess-path rate (m/s)')
    fig.colorbar(image,ax=axes.ravel().tolist(),label='Original coherent cell power / conditional noise (dB)',shrink=.8)
    fig.suptitle('Competing physical primary centres in unchanged original maps\nCyan square: original mode. Green circle: moving-return alternative. Neither is an asserted material-point identity.',fontsize=12)
    fig.savefig(out/'00-selected-modes.png',dpi=150);plt.close(fig)
    if max(errors)>1e-7:raise ValueError('Original objectives changed')
    files=[args.source,Path(source['source']),Path(old['source']),Path(__file__),Path('support/isar/joint_physical_echoes.py')]
    result=dict(complete=True,accepted_calibration=False,production_changed=False,reviews=reviews,
        fitted_boundary_gaps=gaps,original_row_intervals_disjoint=True,joint_data_rows=sum(joint.data_rows),
        joint_zero_antenna_cost=joint_cost,individual_cost=individual,reproduced_objectives=len(errors),
        maximum_absolute_cost_error=max(errors),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    rows=''.join(f'<tr><td>{r["interval_s"]}</td><td>{v["label"]}</td><td>{v["cost"]:.2f}</td><td>{v["unused_change_from_independent_percent"]:+.3f}%</td><td>{v["gradient"]:.3g}</td><td>{v["bounds"]}</td></tr>' for r in reviews for v in r['variants'])
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Competing scatterer associations</title>
<style>body{{font:16px/1.6 system-ui;max-width:1550px;margin:30px auto;padding:0 20px}}img{{width:100%}}td,th{{border:1px solid #ddd;padding:6px}}table{{border-collapse:collapse}}</style>
<h1>Keep distinct scatterer associations before joining a track</h1>
<p>Two non-overlapping two-second windows each retain their original minimum-cold-cost primary and a separately refined all-E initializer. The latter is a competing start, not a declaration that E is the same hull point in all bands. One physical primary still shares range, phase and symbol Doppler; A/C/D/nearby neighbours remain independent. No new guide, kinematic gate, fitted phase correction or data normalization enters the likelihood.</p>
<img src="00-selected-modes.png" alt="Selected candidate modes in the original maps">
<table><tr><th>Window</th><th>Mode</th><th>Training cost</th><th>Unused error vs empirical control</th><th>Projected gradient</th><th>Active guard indices</th></tr>{rows}</table>
<img src="01-competing-measured-paths.png" alt="Measured competing candidate phase trajectories"><img src="02-mode-continuity.png" alt="Competing fitted range and rate trajectories">
<p>Changing which return occupies the physical-primary role can create an apparent range/motion jump even when each local likelihood settles. The envelope/rate curves here are fitted hypotheses, with no uncertainty bars, not evidence that a vessel accelerated or moved between their conditional geographic positions. Priors/bounds and nuisance ambiguity remain material. Close unused prediction alone does not identify a common material point. These modes must be retained or rejected by a joint physical scene/continuity test, rather than stitched by minimum cost. No coordinates or continuous ship motion are adopted.</p>
<p>The two windows have disjoint native row supports. Their combined likelihood contains {sum(joint.data_rows)} real data residuals; its zero-shared-antenna objective equals the sum of individual objectives to {abs(joint_cost-individual):.3g}. Geometry priors are counted once for shared antennas and once per local scene state; each independent neighbour jerk prior remains. This checks joint likelihood ownership, not antenna calibration. {len(errors)} original objectives reproduce with maximum absolute discrepancy {max(errors):.3g}.</p>
<p><a href="analysis.json">Modes, measured arrays and boundary-gap diagnostics</a> · <a href="../physical-primary-coherence-review/index.html">Common physical RF-phase control</a> · <a href="../coherent-primary-and-joint-tests.log">99 numerical tests</a>.</p></html>''')
    print('report',out/'index.html','joint real data rows',sum(joint.data_rows),flush=True)


if __name__=='__main__':main()
