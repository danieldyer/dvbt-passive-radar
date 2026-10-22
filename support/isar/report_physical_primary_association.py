"""Replay the physical-primary association and plot measurements, not fitted phases."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
import numpy as np

from ais import read_csv
from compare_ais_phase import NAMES,COLOURS
from component_phase import physical_primary_measurements
from fit_physical_primary_association import (assignment_code,geometry_for,make_model,prepare_window,free_parameters)
from flash_geometry import assignments
from joint_path_information import expected_jacobian,functional_bounds
from measured_phase import range_rate_map
from motion import C
from nearby_echo_control import load,ROOT
from report_fixed_grid_guided_return import phase_comparison
from report_measured_scatterer_phases import serial
from report_multiscatterer_phase_selections import measurement_records
from ship_site_signature import licensed_options


def physical_path_change(g,parameters,time,midpoint):
    p,v,a,tx,rx=g.state(parameters);dt=np.asarray(time)-midpoint
    point=p+dt[:,None]*v+.5*dt[:,None]**2*a
    def distance(vector):return np.sqrt(np.sum(vector**2,axis=-1))
    value=distance(point[:,None,:]-tx[None])+distance(point-rx)[:,None]
    origin=distance(p-tx)+distance(p-rx)
    return value-origin


def path_information(model,parameters,midpoint):
    free=free_parameters(model,None,nuisance_free=True)
    transform=np.eye(model.count)[:,free]
    matrix,norm=expected_jacobian(model,parameters,transform)
    geometry=model.geometry;time=np.linspace(midpoint-1,midpoint+1,101)
    derivative=[]
    for i in range(18,27):
        p=np.asarray(parameters[:geometry.count],complex);p[i]+=1e-20j
        derivative.append(physical_path_change(geometry,p,time,midpoint).imag/1e-20)
    functionals=np.moveaxis(np.asarray(derivative),0,-1).transpose(1,0,2).reshape(5*len(time),9)
    bounds=functional_bounds(matrix,norm,np.arange(9),functionals)
    bounds.update(time_s=time,standard_deviation_m=bounds.pop('standard_deviation').reshape(5,len(time)),
        estimable_by_channel=bounds.pop('estimable').reshape(5,len(time)),
        control='Antennas, redundant AIS delay and two inherited Ten C nuisance coefficients fixed exactly as during this association control. Every other nonlinear nuisance, active brightness and stationary profile marginalized. No motion or jerk prior used as information.')
    return bounds


def primary_records(model,p):
    raw=[];conditional=[];retention=[]
    for cluster,index in zip(model.clusters,model.indices):
        b=cluster.base;rec=physical_primary_measurements(cluster,p[index]);conditional.append(rec)
        unit,_=cluster.response_blocks(p[index])[0];unit=b.reduced(unit[:,:,None])[:,0].reshape(b.field.y.shape)
        norm=np.sum(abs(unit)**2,axis=1);z=np.divide(np.sum(unit.conj()*b.field.y,axis=1),norm,
            out=np.zeros(len(norm),complex),where=norm>0)
        raw.append(dict(time_s=rec['time_s'],physical_path_change_m=rec['physical_path_change_m'],
            conditional_information=rec['conditional_information'],
            complex_coefficient=z*np.exp(-1j*b.field.k*rec['physical_path_change_m'])))
        value=cluster.evaluate(p[index]);k=len(b.knots);parts=[]
        for s in range(cluster.components):
            h=value['design'][:,s*k:(s+1)*k]@value['amplitude'][s*k:(s+1)*k]
            before=np.sum(abs(h[b.selected])**2);after=np.sum(abs(b.nuisance.remove(h[b.selected]))**2)
            parts.append(float(after/max(before,1e-300)))
        retention.append(parts)
    return raw,conditional,retention


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'physical-primary-association/fits.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=json.loads(args.source.read_text())
    if not source['complete']:raise ValueError('Wait for both complete association trials')
    if args.output.exists():raise ValueError('Use a fresh report directory')
    control=load();empirical=json.loads(Path(source['source']).read_text())
    track_path=Path('docs/data/isar_20261013/rio-delta-ais.csv')
    track=read_csv(track_path,reference_height_ellipsoid_m=5.,mmsi=563052200)
    options=assignments(licensed_options(control['geometry']));out=args.output;out.mkdir(parents=True)
    reviews=[];errors=[]
    for number,group in enumerate(source['groups']):
        window=empirical['windows'][group['window_index']];first,last=window['interval_s'];midpoint=(first+last)/2
        bands,parts,independent,independent_p,independent_fits=prepare_window(control,window)
        assignment=next(a for a in options if assignment_code(a)==group['assignment'])
        geometry=geometry_for(control,assignment,midpoint,track);model=make_model(bands,independent,control,geometry,midpoint)
        for trial in group['trials']:
            for step in trial['steps']:
                r,_=model.evaluate(np.asarray(step['parameters']));errors.append(abs(r@r*model.energy-step['cost']))
        selected=group['trials'][group['best_index']];step=selected['steps'][-1];p=np.asarray(step['parameters'])
        raw,conditional,retention=primary_records(model,p)
        controls=[measurement_records(m,q) for m,q in zip(independent,independent_p)]
        choices=[1 if s=='B' else 4 for s in selected['initialization_labels']]
        variants=[]
        for label,records in (
                ('Independent candidate · original cells',[v[0][s] for v,s in zip(controls,choices)]),
                ('Independent candidate · neighbours subtracted',[v[1][s] for v,s in zip(controls,choices)]),
                ('Physical primary · original cells',raw),
                ('Physical primary · neighbours subtracted',conditional)):
            variants.append(dict(label=label,measurement=phase_comparison(records,bands,first,last)))
        fig,axes=plt.subplots(3,4,figsize=(20,11),layout='constrained',sharex=True)
        for col,variant in enumerate(variants):
            m=variant['measurement'];reference=m['channels'][2]['measured_path_change_m']
            for c,(ch,colour) in enumerate(zip(m['channels'],COLOURS)):
                path=ch['measured_path_change_m']
                axes[0,col].plot(m['time_s'],path,'.-',color=colour,lw=.8,ms=2,label=NAMES[c])
                if c!=2:axes[1,col].plot(m['time_s'],100*(path-reference),'.-',color=colour,lw=.8,ms=2)
                axes[2,col].plot(ch['time_s'],10*np.log10(np.maximum(ch['power_to_noise'],1e-12)),color=colour,lw=.8)
            axes[0,col].set(title=variant['label'],ylabel='Measured path change (m)')
            axes[1,col].set(ylabel='Channel minus Nine (cm)');axes[1,col].axhline(0,color='#bbb',lw=.6)
            axes[2,col].set(ylabel='Conditional measurement strength (dB)',xlabel='Capture time (s)')
            for ax in axes[:,col]:
                ax.set_xlim(first,last);ax.xaxis.set_major_locator(MultipleLocator(.5));ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.axvline(m['anchor_time_s'],color='#bbb',lw=.6,ls=':')
        axes[0,0].legend(ncol=3,fontsize=8)
        fig.suptitle(f'{first:g}–{last:g} s · test a physical point without forcing neighbour identities\n'
            'All plotted curves are complex measurements. Constant origins only; RF phase divided by each band’s own wavenumber.\n'
            f'Original-data unused error change {step["unused_change_from_independent_percent"]:+.3f}%; antenna assignment {group["assignment"]} held nominal, not calibrated.',fontsize=12)
        phase_plot=f'{number+1:02d}-measured-phase-comparison.png';fig.savefig(out/phase_plot,dpi=145);plt.close(fig)
        fig,axes=plt.subplots(1,5,figsize=(21,4.8),layout='constrained',sharex=True,sharey=True)
        rates=np.linspace(-3,3,301);maps=[range_rate_map(b.field.time,b.field.data['value'],b.field.data['information'],b.field.k,rates)/b.field.pulse.samples[64].real for b in bands]
        vmax=np.ceil(10*np.log10(max(m.max() for m in maps)));obs=geometry.prediction(p[:geometry.count])
        for c,(ax,power,b,cluster,index) in enumerate(zip(axes,maps,bands,model.clusters,model.indices)):
            offsets=b.field.offsets*C/8e6
            image=ax.imshow(10*np.log10(np.maximum(power,1e-8)),origin='lower',aspect='auto',cmap='magma',vmin=0,vmax=vmax,
                extent=[offsets[0]-C/16e6,offsets[-1]+C/16e6,-3,3],interpolation='nearest')
            centre=np.interp(midpoint,b.field.data['row_time'],b.field.data['centre_path_m'])
            ax.plot(obs[c,0]-centre,obs[c,1],'s',mec='#00e5ff',mfc='none',ms=11)
            ax.annotate('Physical P',(obs[c,0]-centre,obs[c,1]),xytext=(5,7),textcoords='offset points',color='#00e5ff',fontsize=8)
            for s,(arc,block) in enumerate(zip(cluster.nuisance_arcs,cluster.nuisance_slices)):
                q=p[index[block]];curve=arc.base.curve;rows=b.field.data['row_time']
                path=arc.known_path+curve(rows)@q[2:]
                rate=arc.known_rate+curve(rows,1)@q[2:]-b.field.data['centre_omega']/(b.field.k*.001024)
                xy=q[0]*C/8e6+np.interp(midpoint,rows,path),np.interp(midpoint,rows,rate)
                ax.plot(*xy,'o',mec='#fff',mfc='none',ms=6);ax.annotate(('A','C','D','Nearby')[s],xy,xytext=(4,4),textcoords='offset points',color='white',fontsize=7)
            ax.set(title=NAMES[c],xlabel='Archived reference offset (m)',xlim=(-320,640))
        axes[0].set_ylabel('Excess-path rate (m/s)')
        fig.colorbar(image,ax=axes,label='Original coherent cell power / conditional noise (dB)',shrink=.8)
        fig.suptitle(f'{first:g}–{last:g} s · physical primary P and independent nuisance centres\n'
            'Original observations before subtraction. Cyan centre uses one physical trajectory; white neighbours are not assumed to be ship points.',fontsize=12)
        selection_plot=f'{number+1:02d}-selected-points.png';fig.savefig(out/selection_plot,dpi=150);plt.close(fig)
        info=path_information(model,p,midpoint);fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
        for c,colour in enumerate(COLOURS):
            axes[0].plot(info['time_s'],np.degrees(bands[c].field.k*info['standard_deviation_m'][c]),color=colour,label=NAMES[c])
        axes[0].set(xlabel='Capture time (s)',ylabel='Conditional σ of path phase change (°)',title='Data-only bound after nuisance marginalization')
        axes[0].legend(ncol=3,fontsize=8)
        for s,label in enumerate(('Primary P','A','C','D','Nearby')):
            axes[1].plot(NAMES,np.maximum(np.asarray(retention)[:,s],1e-16),'o-',label=label)
        axes[1].set(yscale='log',ylabel='Retained fitted component energy fraction',title='Retention after stationary-profile projection');axes[1].legend(ncol=3,fontsize=8)
        fig.suptitle('Uncertainty and nuisance overlap are separate from conditional phase-measurement strength\n'
            f'Data-only Cartesian target rank {info["target_rank"]}/9; nominal antennas and the inherited fixed Ten C control are not estimated here.',fontsize=11)
        information_plot=f'{number+1:02d}-information-and-retention.png';fig.savefig(out/information_plot,dpi=145);plt.close(fig)
        reviews.append(dict(interval_s=window['interval_s'],assignment=group['assignment'],initialization_labels=selected['initialization_labels'],
            phase_plot=phase_plot,selection_plot=selection_plot,information_plot=information_plot,
            variants=variants,path_information=info,static_projection_retention=retention,
            selected_trial=group['best_index'],selected_audit=step['audit'],selected_cost=step['cost'],
            unused_error_change_percent=step['unused_change_from_independent_percent'],diagnostics=step['diagnostics']))
    if max(errors)>1e-7:raise ValueError('Fitted original likelihood changed')
    files=[args.source,Path(source['source']),Path(__file__),*[Path('support/isar')/(n+'.py') for n in
        ('fit_physical_primary_association','physical_echo_cluster','physical_amplitude_arc','component_phase',
         'joint_path_information','report_multiscatterer_phase_selections','report_fixed_grid_guided_return')]]
    result=dict(complete=True,accepted_calibration=False,production_changed=False,reviews=reviews,
        reproduced_objectives=len(errors),maximum_absolute_cost_error=max(errors),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    rows=''.join(f'<tr><td>{r["interval_s"]}</td><td>{r["initialization_labels"]}</td><td>{r["unused_error_change_percent"]:+.4f}%</td><td>{r["selected_audit"]["projected_gradient"]:.3g}</td><td>{r["selected_audit"]["bound_indices"]}</td><td>{r["path_information"]["target_rank"]}/9</td></tr>' for r in reviews)
    images=''.join(f'<h2>{r["interval_s"]}</h2><img src="{r["selection_plot"]}" alt="Selected physical primary and independent nuisance returns"><img src="{r["phase_plot"]}" alt="Measured physical-primary phase histories"><img src="{r["information_plot"]}" alt="Data-only information and stationary overlap">' for r in reviews)
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Physical-primary phase association</title>
<style>body{{font:16px/1.6 system-ui;max-width:1600px;margin:30px auto;padding:0 20px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:6px;border:1px solid #ddd}}</style>
<h1>Can one physical point explain the measured five-band trajectory?</h1>
<p>This control replaces matching B/E labels with one physical primary P. All 32 B/E choices supply initializers for the same model family; the two best training scores receive original-data refinement. A/C remain independent quadratics, D and the unselected nearby component retain independent splines. Antennas are fixed at assignment 00002 for this association experiment. No multipath model, fitted heading, phase screen or per-band range offset is introduced.</p>
<p>The primary uses one Cartesian position/velocity/acceleration, with exact bistatic envelope delay, carrier phase and symbol Doppler. Each band retains its independent constant scattering phase. This is not yet a calibrated coherent physical RF-phase fit. Weak/absent primary brightness is allowed. Only original complete bins and their native waveform kernels enter; stationary background, brightness knots, noise covariance and unused-block split are unchanged. Nuisance-freezing steps initialize only; the displayed final results release nuisance trajectories, except the same two inherited fixed Ten C coefficients used in the independent control.</p>
<table><tr><th>Window</th><th>Acquisition labels only</th><th>Unused error vs independent</th><th>Projected gradient</th><th>Bound indices</th><th>Data-only target rank</th></tr>{rows}</table>
<p>The original-cell and neighbour-subtracted curves are measured complex coefficients; fitted path curves are not substituted. Only constant origins are removed, separately for each diagnostic. Path-change differences are wavelength normalized; the conditional strength excludes neighbour uncertainty. The information panel instead marginalizes all estimated nonlinear nuisance and active real brightness without prior rows. Full Cartesian position need not be estimable even when physical path-change functionals are. Static-projection retention makes near-stationary overlap explicit.</p>
{images}<p>{len(errors)} final/frozen-step objectives reproduce with maximum absolute discrepancy {max(errors):.3g}. Close unused error on this two-second, previously inspected aperture is exploratory evidence only. The nominal transmitter assignment, common-scatterer identity across a longer aperture and antenna coordinates are not established. All 32 licensed assignments and disjoint candidate episodes remain required for calibration.</p>
<p><a href="analysis.json">Measured arrays and diagnostics</a> · <a href="../multiscatterer-phase-review-complete/index.html">All five independently selected candidate phases</a> · <a href="../multiscatterer-primary-tests.log">Numerical tests</a>.</p></html>''')
    snapshot=out/'sources';snapshot.mkdir()
    for p in files:
        if p.suffix=='.py':shutil.copyfile(p,snapshot/p.name)
    print('report',out/'index.html','objectives',len(errors),'max error',max(errors),flush=True)


if __name__=='__main__':main()
