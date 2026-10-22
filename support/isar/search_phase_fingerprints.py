"""Use four-band measured trajectories to query the separate ABC range field.

Guide trajectories are estimated from Seven/SBS/Nine/Ten training samples.
The ABC search fits a constant phase and real brightness only. Its temporal
path coefficients remain fixed to the guide; all candidate ranges are shown.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_ais_phase import NAMES,COLOURS
from field import average
from fit_shared_echo_cluster import prepare
from measured_phase import local_average
from phase_fingerprint import measure
from report_measured_scatterer_phases import load_fields,serial
from shared_echo_cluster import SharedEchoCluster
from shared_echo_cluster import ReferencedAmplitudeArc
from motion import C
from field import Field
from background_reference import field_reference
from report_background_extrapolation import original_values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/shared-echo-cluster-context/fits.json'))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/phase-fingerprint-search'))
    parser.add_argument('--background-reference',type=Path,help='Fixed disjoint-time residual reference; no local background fit')
    parser.add_argument('--background-aggregate',type=int,choices=[0,1,2],default=0)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);output=out/'search.json'
    if output.exists():raise ValueError('Use a fresh output directory')
    source=json.loads(args.source.read_text());first,last=source['interval_s'];labels=source['labels'];origins=source['selected_origins_samples'];n=len(labels)
    geometry=Path('docs/data/calibration_20261016_sites');fields=load_fields(Path('build/isar-20261015/full-fields'),geometry)
    bands,partition=prepare(fields,geometry,first,last,range_extent=source.get('range_extent_samples',8))
    references=[]
    if args.background_reference:
        corrected=[]
        for ch,b in enumerate(bands):
            path=args.background_reference/f'channel-{ch}-means.npz';references.append(path)
            with np.load(path) as reference:
                mean=field_reference(b.field,reference['aggregates'][args.background_aggregate],int(reference['first_range_sample']))
            values=b.field.data['value']-original_values(b.field,mean)
            field=Field({**b.field.data,'value':values},b.field.pulse.samples)
            corrected.append(ReferencedAmplitudeArc(field,b.training,b.knots,first,last,b.known_path,b.known_rate))
        bands=corrected
    model=SharedEchoCluster(bands[:4],origins,origin_radius=source['origin_radius_samples']);seed=np.zeros(model.count)
    coefficients=[];energies=[]
    for ch,record in enumerate(source['independent'][:4]):
        fit=record['trials'][record['best_index']];p=np.asarray(fit['parameters'])
        seed[ch*2*n:(ch+1)*2*n]=p[:2*n];coefficients.append(p[-2*n:]);energies.append(fit['data_energy'])
    seed[-2*n:]=np.average(coefficients,axis=0,weights=energies);started=time.perf_counter()
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,interval_s=[first,last],guide_fits=[],channels=[],
        source=str(args.source),guide_channels=NAMES[:4],searched_channel='ABC',
        background_reference=str(args.background_reference) if args.background_reference else None,
        background_aggregate=args.background_aggregate if args.background_reference else None,
        background_policy='Subtract only the fixed archived residual mean from the declared disjoint period, through the exact saved C0/C1 observation operator. No local fit, scaling or phase alignment. Reference uncertainty is not treated as additional white noise; this is a controlled mean-transfer diagnostic.' if args.background_reference else 'No new background subtraction.',
        guide_policy='Fit original range/phase data from the first four bands only, using the saved independently fitted first-four-band starts. No ABC values enter the guide objective or start ranking.',
        search_policy='Per range cell: one globally optimized constant phase and positive real training brightness. Fixed guide trajectory. Final-quarter prediction freezes phase and interpolates brightness from training. No candidate selected using final-quarter statistics.',
        source_sha256={str(args.source):hashlib.sha256(args.source.read_bytes()).hexdigest()})
    paths=[Path(__file__),Path('support/isar/phase_fingerprint.py'),Path('support/isar/test_phase_fingerprint.py'),Path('support/isar/positive_phase.py'),
        Path('support/isar/shared_echo_cluster.py'),Path('support/isar/fit_shared_echo_cluster.py'),Path('support/isar/field.py'),Path('support/isar/amplitude_pair.py'),
        geometry/'geometry-prior.json',geometry/'sites-main.json',geometry/'acma-transmitters.json']
    paths += [f['source'] for f in fields]
    paths += references+[Path('support/isar/background_reference.py'),Path('support/isar/report_background_extrapolation.py')]
    snapshot=out/'sources';snapshot.mkdir(exist_ok=True)
    for p in paths:
        with p.open('rb') as f:result['source_sha256'][str(p)]=hashlib.file_digest(f,'sha256').hexdigest()
        if p.suffix=='.py':shutil.copyfile(p,snapshot/p.name)
    def save():
        result['seconds']=time.perf_counter()-started;output.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for start in range(2):
        initial=seed.copy()
        if start:
            for c in range(4):initial[2*n*c+np.arange(n)*2+1]+=.3*np.cos(np.arange(n)+c)
        fit=model.fit(initial,240);fit['start']=start
        fit['prediction']=model.prediction(np.asarray(fit['parameters']),[p==3 for p in partition[:4]])
        result['guide_fits'].append(fit);save()
        print('four-band guide',start,'cost',round(fit['cost'],2),'gradient',fit['maximum_projected_gradient'],flush=True)
    selected=int(np.argmin([r['cost'] for r in result['guide_fits']]));result['selected_guide']=selected
    parameters=np.asarray(result['guide_fits'][selected]['parameters']);guide=parameters[-2*n:].reshape(n,2)
    result['guide_path_coefficients_m']=dict(zip(labels,guide.tolist()))
    for ch,(band,part) in enumerate(zip(bands,partition)):
        d=band.field.data;weight=8192*d['information']/band.field.pulse.samples[64].real
        centre=int(np.flatnonzero(band.field.offsets==0)[0]);channel=dict(name=NAMES[ch],channel=ch,offset_samples=band.field.offsets,
            time_s=band.field.time,guides=[])
        for label,coefficients in zip(labels,guide):
            raw,_=band.response(np.r_[0.,0.,coefficients])
            h=average(raw,d['row_information'],band.field.metadata['block_rows'])[0][:,centre]
            scores=measure(band.field.time,d['value'],weight,h,band.training,part==3)
            channel['guides'].append(dict(label=label,scores=scores,template_real=h.real,template_imag=h.imag))
        result['channels'].append(channel);save();print('searched',NAMES[ch],flush=True)
    result['complete']=True;save()
    abc=result['channels'][4];offset=np.asarray(abc['offset_samples']);metres=offset*C/8e6
    fig,axes=plt.subplots(2,2,figsize=(15,9),layout='constrained')
    for ax,g in zip(axes.flat,abc['guides']):
        score=np.array([r['coherent_training_score'] for r in g['scores']]);rms=np.array([r['testing_phase_rms_deg'] for r in g['scores']],float)
        ax.plot(metres,10*np.log10(np.maximum(score,1e-12)),'.-',color='#0072b2',label='Training coherent score')
        second=ax.twinx();second.plot(metres,rms,'.-',color='#d55e00',label='Unused phase RMS');second.set_ylabel('Unused phase RMS (degrees)',color='#d55e00');second.set_ylim(0,180)
        ax.set(title='Fixed four-band trajectory '+g['label'],xlabel='ABC range offset from saved AIS reference (m)',ylabel='Training coherent score / conditional noise (dB)')
        for sample in (0,6,13,14,15):ax.axvline(sample*C/8e6,color='#999',lw=.5,ls=':')
    fig.suptitle('Does ABC contain the same measured phase fingerprint at another range?\n'
        'Guide coefficients fixed from the other four bands. Constant phase fitted on training only; orange is unused prediction.',fontsize=13)
    fig.savefig(out/'01-abc-range-search.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(19,8),layout='constrained',sharex=True,sharey=True)
    grid=np.arange(first+.15,last-.149,.05)
    for row,label in enumerate(('B','D')):
        for ch,(channel,band) in enumerate(zip(result['channels'],bands)):
            g=next(g for g in channel['guides'] if g['label']==label);h=np.asarray(g['template_real'])+1j*np.asarray(g['template_imag'])
            phase=np.array([r['phase_rad'] for r in g['scores']]);aligned=band.field.data['value']/h[:,None]*np.exp(-1j*phase)
            values=[];valid=[]
            info=8192*band.field.data['information']*abs(h)**2/band.field.pulse.samples[64].real
            for j in range(len(offset)):
                epoch,z,var=local_average(band.field.time,aligned[:,j],info/8192,grid,.25,1.,0.)
                values.append(np.angle(z));valid.append(abs(z)**2/var>=4)
            values=np.ma.array(values,mask=~np.array(valid))
            cmap=plt.get_cmap('twilight_shifted').copy();cmap.set_bad('#ddd')
            image=axes[row,ch].imshow(np.degrees(values),origin='lower',aspect='auto',
                extent=[grid[0]-.025,grid[-1]+.025,metres[0]-C/16e6,metres[-1]+C/16e6],cmap=cmap,vmin=-180,vmax=180)
            axes[row,ch].set_title(NAMES[ch]+' · guide '+label)
            if ch==0:axes[row,ch].set_ylabel('Range offset (m)')
            if row==1:axes[row,ch].set_xlabel('Capture time (s)')
    fig.colorbar(image,ax=axes,label='Measured phase after fixed trajectory and constant training phase (degrees)',shrink=.7)
    fig.suptitle('Fixed-trajectory phase maps, with no fitted slope or curvature in ABC\n'
        '0.25 s complex display averages; grey marks power/conditional noise below 4.',fontsize=13)
    fig.savefig(out/'02-fixed-trajectory-phase.png',dpi=165);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained',sharex=True)
    for row,label in enumerate(('B','D')):
        g=next(g for g in abc['guides'] if g['label']==label)
        for col,sample in enumerate((13,14,15)):
            j=int(np.flatnonzero(offset==sample)[0]);score=g['scores'][j];t=np.asarray(abc['time_s']);phase=np.degrees(score['phase_residual_rad']);test=np.asarray(score['testing'])
            axes[row,col].plot(t,phase,'.',color='#777',ms=3,alpha=.4,label='All measured bins')
            axes[row,col].plot(t[test],phase[test],'.',color='#d55e00',ms=4,label='Final unused quarter')
            axes[row,col].axhline(0,color='#333',lw=.7)
            rms=score['testing_phase_rms_deg'];text='undefined' if rms is None else f'{rms:.1f}°'
            axes[row,col].set(title=f"ABC cell {sample} ({sample*C/8e6:+.0f} m), guide {label}\nUnused phase RMS {text}",ylabel='Measured phase residual (degrees)',xlabel='Capture time (s)',ylim=(-180,180))
    axes[0,0].legend(fontsize=8);fig.savefig(out/'03-abc-delayed-candidates.png',dpi=170);plt.close(fig)
    rows=''
    for label in ('B','D'):
        g=next(g for g in abc['guides'] if g['label']==label)
        for sample in (0,6,11,12,13,14,15,16):
            score=g['scores'][int(np.flatnonzero(offset==sample)[0])];rms=score['testing_phase_rms_deg']
            rows+=f'<tr><td>{label}</td><td>{sample}</td><td>{10*np.log10(max(score["coherent_training_score"],1e-12)):.2f}</td><td>{rms if rms is not None else "undefined"}</td><td>{score["testing_explained_fraction"]}</td></tr>'
    background_note=(f'<p><b>Disjoint residual-reference control:</b> {result["background_policy"]} Aggregate {args.background_aggregate}: '+('30–90 s','100–130 s','190–220 s')[args.background_aggregate]+'. This reference changes neither the source archive nor production processing.</p>') if args.background_reference else '<p>No additional stationary background is subtracted in this baseline.</p>'
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Measured phase fingerprint search</title>
<style>body{{font:16px/1.5 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}}img{{width:100%}}td,th{{padding:7px;border:1px solid #ddd}}table{{border-collapse:collapse}}</style>
<h1>Use four channels’ measured phase histories to search ABC</h1>
{background_note}
<p>Seven, SBS, Nine and Ten alone determine the guide path coefficients from the original training samples. Each ABC range cell then fits one constant phase and changing positive real brightness, with no fitted rate, curvature or phase screen. The final quarter of ABC time samples supplies prediction statistics. No sea/multipath hypothesis enters.</p>
<img src="01-abc-range-search.png" alt="ABC coherent score and unused phase error versus range">
<p>Strong coherent energy alone does not prove a matching trajectory: a brighter mismatched return can still have a larger score than a weak correct one. The unused phase errors and full range profiles are therefore shown together. These data were previously inspected, so this is internal validation rather than a blind external experiment.</p>
<img src="02-fixed-trajectory-phase.png" alt="Measured residual phase after each fixed guide trajectory">
<p>Phase maps remove the constant phase fitted on training at each range, then display 0.25 s complex averages. They do not remove a slope or curvature. Grey indicates low conditional thermal information, not a measured zero phase. Range cells can contain unresolved mixtures; the guide is not assumed to describe every cell.</p>
<img src="03-abc-delayed-candidates.png" alt="Measured ABC phase at nearby delayed-return candidates">
<table><tr><th>Guide</th><th>ABC range sample</th><th>Training coherent score (dB)</th><th>Unused phase RMS (degrees)</th><th>Unused explained fraction</th></tr>{rows}</table>
<p><a href="search.json">All guide fits, original complex-template values and every range-cell result</a> · <a href="tests.log">Numerical tests</a> · <a href="../multiple-scatterer-review/index.html">Previous scene controls</a> · <a href="../index.html">Analysis index</a></p></html>''')
    print(out/'index.html',flush=True)


if __name__=='__main__':main()
