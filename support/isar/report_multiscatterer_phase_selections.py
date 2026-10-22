"""Mark every selected return and show independently measured five-band phases.

No common motion is fitted here. Original-cell and training-frozen conditional
measurements are shown side by side, so subtraction cannot masquerade as raw
cross-band agreement. Candidate labels are not material-point identifications.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
import numpy as np

from compare_ais_phase import NAMES,COLOURS
from component_phase import component_measurements
from fit_physical_primary_association import prepare_window
from measured_phase import range_rate_map
from motion import C
from nearby_echo_control import load,ROOT
from report_fixed_grid_guided_return import phase_comparison
from report_measured_scatterer_phases import serial


LABELS=('A','B','C','D','E')
MARKERS=('#35ddff','#9bff67','#ffb94d','#ed8dff','#ffffff')


def relative_point_paths(first,second):
    """Compare measured point deltas with a single constant origin per band."""
    time=np.asarray(first['time_s']);np.testing.assert_array_equal(time,second['time_s'])
    delta=np.array([a['measured_path_change_m']-b['measured_path_change_m']
        for a,b in zip(first['channels'],second['channels'])])
    valid=np.isfinite(delta);count=valid.sum(axis=0)
    # Prefer a central epoch with the largest simultaneously visible subset.
    # Missing bands stay missing; no phase bridge is supplied by another band.
    eligible=np.flatnonzero(count==count.max())
    anchor=int(eligible[np.argmin(abs(time[eligible]-(time[0]+time[-1])/2))])
    usable=valid[:,anchor];delta-=delta[:,anchor,None]
    delta[~usable]=np.nan
    return dict(time_s=time,anchor_time_s=time[anchor],connected_channels=usable,relative_path_change_m=delta)


def measurement_records(model,parameters):
    """Two diagnostics from identical rows and candidate templates."""
    cluster=model.clusters[0];local=np.asarray(parameters)[model.indices[0]];b=cluster.base
    conditional=component_measurements(cluster,local);raw=[];retention=[]
    value=cluster.evaluate(local);k=len(b.knots)
    for record,(unit,_) in zip(conditional,cluster.response_blocks(local)):
        h=b.reduced(unit[:,:,None])[:,0].reshape(b.field.y.shape)
        norm=np.sum(abs(h)**2,axis=1)
        z=np.divide(np.sum(h.conj()*b.field.y,axis=1),norm,
            out=np.zeros(len(norm),complex),where=norm>0)
        raw.append(dict(time_s=record['time_s'],physical_path_change_m=record['physical_path_change_m'],
            complex_coefficient=z*np.exp(-1j*b.field.k*record['physical_path_change_m']),
            conditional_information=record['conditional_information']))
        s=len(retention);part=value['design'][:,s*k:(s+1)*k]@value['amplitude'][s*k:(s+1)*k]
        before=np.sum(abs(part[b.selected])**2);after=np.sum(abs(b.nuisance.remove(part[b.selected]))**2)
        retention.append(float(after/max(before,1e-300)))
    return raw,conditional,retention


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'piecewise-nearby-common-paths/fits.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    if out.exists() and (not (out/'analysis.json').exists() or
            json.loads((out/'analysis.json').read_text()).get('source')!=str(args.source)):
        raise ValueError('Only an existing report of the same source may be refreshed')
    source=json.loads(args.source.read_text());control=load();out.mkdir(parents=True,exist_ok=True)
    reviews=[];maps=[];selected=[];errors=[]
    for w in source['windows']:
        bands,parts,models,parameters,fits=prepare_window(control,w);first,last=w['interval_s'];midpoint=(first+last)/2
        measurements=[measurement_records(m,p) for m,p in zip(models,parameters)]
        points=[];chosen=[];window_maps=[]
        for c,(m,p,fit) in enumerate(zip(models,parameters,fits)):
            r,_=m.evaluate(p);errors.append(abs(r@r*m.energy-fit['cost']))
            cluster=m.clusters[0];local=p[m.indices[0]];f=cluster.base.field;channel=[]
            for s in range(5):
                arc=cluster.arcs[s];q=local[cluster.slices[s]]
                rate=arc.known_rate+arc.base.rate_basis@q[2:]-f.data['centre_omega']/(f.k*.001024)
                path=arc.known_path+arc.base.basis@q[2:]
                channel.append(dict(label=LABELS[s],offset_m=q[0]*C/8e6+np.interp(midpoint,f.data['row_time'],path),
                    path_rate_mps=float(np.interp(midpoint,f.data['row_time'],rate))))
            chosen.append(channel)
            rates=np.linspace(-3,3,301)
            window_maps.append(range_rate_map(f.time,f.data['value'],f.data['information'],f.k,rates)/f.pulse.samples[64].real)
        fig,axes=plt.subplots(5,4,figsize=(21,16),layout='constrained',sharex=True)
        for s,label in enumerate(LABELS):
            raw=phase_comparison([r[0][s] for r in measurements],bands,first,last)
            conditional=phase_comparison([r[1][s] for r in measurements],bands,first,last)
            points.append(dict(label=label,original_cells=raw,conditional_cells=conditional))
            for col,measurement in ((0,raw),(1,conditional)):
                for c,(channel,colour) in enumerate(zip(measurement['channels'],COLOURS)):
                    axes[s,col].plot(measurement['time_s'],channel['measured_path_change_m'],'.-',color=colour,lw=.8,ms=2,label=NAMES[c])
                axes[s,col].axvline(measurement['anchor_time_s'],color='#bbb',lw=.6,ls=':')
                axes[s,col].set(title=f'{label} · '+('Original cells' if col==0 else 'Neighbours subtracted'),ylabel='Measured path change (m)')
            reference=conditional['channels'][2]['measured_path_change_m']
            for c,(channel,colour) in enumerate(zip(conditional['channels'],COLOURS)):
                if c!=2:
                    axes[s,2].plot(conditional['time_s'],100*(channel['measured_path_change_m']-reference),'.-',color=colour,lw=.8,ms=2)
                axes[s,3].plot(channel['time_s'],10*np.log10(np.maximum(channel['power_to_noise'],1e-12)),color=colour,lw=.8)
            axes[s,2].axhline(0,color='#bbb',lw=.6)
            axes[s,2].set(title=f'{label} · Channel minus Nine',ylabel='Measured path-change difference (cm)')
            axes[s,3].axhline(10*np.log10(4.),color='#999',lw=.6,ls=':')
            axes[s,3].set(title=f'{label} · Conditional measurement strength',ylabel='Power / conditional noise (dB)')
            for ax in axes[s]:
                ax.set_xlim(first,last);ax.xaxis.set_major_locator(MultipleLocator(.5));ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            if s==0:axes[s,0].legend(ncol=3,fontsize=8)
        for ax in axes[-1]:ax.set_xlabel('Capture time (s)')
        gradients=[f['maximum_projected_gradient'] for f in fits]
        unsettled=[NAMES[c] for c,g in enumerate(gradients) if g>1e-5]
        warning=('Unsettled fit: '+', '.join(unsettled)+'. These conditional curves are diagnostic only.') if unsettled else 'Independent fits only: no shared path, rate or curvature constraint.'
        fig.suptitle(f'{first:g}–{last:g} s · five selected returns, independently measured in each RF band\n'
            'Each phase uses its own RF wavenumber. Only constant origins removed; no fitted slope/curvature subtracted.\n'
            '250 ms measurements at 50 ms steps overlap. Gaps remain disconnected below the 6 dB display threshold.\n'+warning,
            fontsize=12,color='#b00020' if unsettled else '#222')
        name=f'{len(reviews)+1:02d}-all-point-trajectories.png';fig.savefig(out/name,dpi=145);plt.close(fig)
        phase_name=f'{len(reviews)+1:02d}-rf-phase-trajectories.png'
        fig,axes=plt.subplots(5,2,figsize=(12.5,15),layout='constrained',sharex=True)
        for s,point in enumerate(points):
            for col,key in enumerate(('original_cells','conditional_cells')):
                m=point[key]
                for c,(ch,colour) in enumerate(zip(m['channels'],COLOURS)):
                    axes[s,col].plot(m['time_s'],-np.degrees(bands[c].field.k*ch['measured_path_change_m']),
                        '.-',color=colour,lw=.8,ms=2,label=NAMES[c])
                axes[s,col].set(title=f'{point["label"]} · '+('Original cells' if col==0 else 'Neighbours subtracted'),
                    ylabel='Measured RF phase change (°)',xlim=(first,last))
                axes[s,col].xaxis.set_major_locator(MultipleLocator(.5))
                axes[s,col].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axes[0,0].legend(ncol=3,fontsize=8)
        for ax in axes[-1]:ax.set_xlabel('Capture time (s)')
        fig.suptitle(f'{first:g}–{last:g} s · measured RF phases before wavelength normalization\n'
            'Constant origins only; the same path motion gives different phase slopes at different carrier frequencies.\n'+warning,
            fontsize=11,color='#b00020' if unsettled else '#222')
        fig.savefig(out/phase_name,dpi=145);plt.close(fig)
        pair_name=f'{len(reviews)+1:02d}-between-point-differences.png';pair_records=[]
        fig,axes=plt.subplots(4,2,figsize=(13,12),layout='constrained',sharex=True)
        for row,s in enumerate((0,2,3,4)):
            item=dict(pair=f'{LABELS[s]}−B')
            for col,key in enumerate(('original_cells','conditional_cells')):
                pair=relative_point_paths(points[s][key],points[1][key]);item[key]=pair
                for c,(value,colour) in enumerate(zip(pair['relative_path_change_m'],COLOURS)):
                    axes[row,col].plot(pair['time_s'],100*value,'.-',color=colour,lw=.8,ms=2,label=NAMES[c])
                axes[row,col].axhline(0,color='#bbb',lw=.6)
                axes[row,col].axvline(pair['anchor_time_s'],color='#bbb',lw=.6,ls=':')
                axes[row,col].set(title=f'{LABELS[s]}−B · '+('Original cells' if col==0 else 'Neighbours subtracted'),
                    ylabel='Within-band path-change difference (cm)',xlim=(first,last))
                axes[row,col].xaxis.set_major_locator(MultipleLocator(.5));axes[row,col].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            pair_records.append(item)
        axes[0,0].legend(ncol=3,fontsize=8)
        for ax in axes[-1]:ax.set_xlabel('Capture time (s)')
        fig.suptitle(f'{first:g}–{last:g} s · between-point measured trajectory differences\n'
            'Subtract B within each RF band, then remove one constant at a shared epoch. No rates are aligned.\n'+warning,
            fontsize=11,color='#b00020' if unsettled else '#222')
        fig.savefig(out/pair_name,dpi=145);plt.close(fig)
        reviews.append(dict(interval_s=w['interval_s'],plot=name,phase_plot=phase_name,pair_plot=pair_name,
            points=points,between_point_differences=pair_records,
            static_projection_retention=[r[2] for r in measurements],
            independent_projected_gradients=gradients,unsettled_channels=unsettled,
            component_to_sum_energy_ratios=[f['component_to_sum_energy_ratio'][0] for f in fits]))
        maps.append(window_maps);selected.append(chosen)
    if max(errors)>1e-7:raise ValueError('Saved original-data likelihood changed')
    vmax=np.ceil(10*np.log10(max(m.max() for row in maps for m in row)))
    fig,axes=plt.subplots(len(reviews),5,figsize=(21,4.3*len(reviews)),layout='constrained',sharex=True,sharey=True)
    for row,(review,row_maps,choices) in enumerate(zip(reviews,maps,selected)):
        for c,(power,points) in enumerate(zip(row_maps,choices)):
            f=control['bands'][c].field;offset=f.offsets*C/8e6;ax=axes[row,c]
            image=ax.imshow(10*np.log10(np.maximum(power,1e-8)),origin='lower',aspect='auto',cmap='magma',
                vmin=0,vmax=vmax,extent=[offset[0]-C/16e6,offset[-1]+C/16e6,-3,3],interpolation='nearest')
            for point,colour in zip(points,MARKERS):
                xy=point['offset_m'],point['path_rate_mps'];ax.plot(*xy,'s',mec=colour,mfc='none',ms=8,lw=.8)
                ax.annotate(point['label'],xy,xytext=(5,5),textcoords='offset points',color=colour,fontsize=9,
                    bbox=dict(facecolor='#111',edgecolor='none',alpha=.6,pad=.5))
            ax.set(title=f'{NAMES[c]} · {review["interval_s"][0]:g}–{review["interval_s"][1]:g} s',xlim=(-320,640))
            if row==len(reviews)-1:ax.set_xlabel('Offset from archived fixed reference (m)')
            if c==0:ax.set_ylabel('Excess-path rate (m/s)')
    fig.colorbar(image,ax=axes.ravel().tolist(),label='Original coherent cell power / conditional noise (dB)',shrink=.8)
    fig.suptitle('All selected A–E candidate centres in the original five-band observations\n'
        'No notch, fitted stationary profile or neighbour subtraction in these maps. Labels are acquisition identities, not confirmed common hull points.',fontsize=12)
    fig.savefig(out/'00-selected-points.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,len(reviews),figsize=(17,4.8),layout='constrained',sharey=True)
    for ax,review in zip(axes,reviews):
        retained=np.asarray(review['static_projection_retention'])
        for s,(label,colour) in enumerate(zip(LABELS,('#007ea8','#639f00','#be7600','#9e46b2','#555'))):
            ax.plot(NAMES,np.maximum(retained[:,s],1e-16),'o-',color=colour,label=label)
        ax.set(title=f'{review["interval_s"][0]:g}–{review["interval_s"][1]:g} s',yscale='log',
            ylabel='Fitted component energy retained after static projection')
    axes[0].legend(ncol=5,fontsize=9)
    fig.suptitle('Stationary overlap: conditional brightness is not independent moving-point information\n'
        'The measured-phase strength panels freeze nuisance subtraction; this retention audit uses the actual training projection.',fontsize=11)
    fig.savefig(out/'04-stationary-overlap.png',dpi=150);plt.close(fig)
    files=[args.source,Path(__file__),*[Path('support/isar')/(n+'.py') for n in
        ('component_phase','measured_phase','report_fixed_grid_guided_return','fit_physical_primary_association',
         'fit_piecewise_nearby_paths','phase_window','nearby_echo_control')]]
    result=dict(complete=True,accepted_calibration=False,production_changed=False,source=str(args.source),
        reviews=reviews,selected_points=selected,reproduced_objectives=len(errors),maximum_absolute_cost_error=max(errors),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    with (out/'measured-trajectories.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(('window_first_s','window_last_s','point','measurement','channel',
            'capture_time_s','measured_path_change_m','rf_phase_change_deg'))
        for review in reviews:
            for point in review['points']:
                for key in ('original_cells','conditional_cells'):
                    m=point[key]
                    for c,ch in enumerate(m['channels']):
                        for t,p in zip(m['time_s'],ch['measured_path_change_m']):
                            writer.writerow((*review['interval_s'],point['label'],key,NAMES[c],t,p,-np.degrees(control['bands'][c].field.k*p)))
    images=''.join(f'<h2>{r["interval_s"][0]:g}–{r["interval_s"][1]:g} seconds</h2><img src="{r["plot"]}" alt="Five candidates and measured interband trajectories"><p><a href="{r["phase_plot"]}">Measured RF phase in degrees</a> · <a href="{r["pair_plot"]}">Between-point measured trajectory differences</a>. Selected projected gradients: {r["independent_projected_gradients"]}. Component-energy / sum-energy ratios: {r["component_to_sum_energy_ratios"]}.</p><img src="{r["pair_plot"]}" alt="Within-band point differences compared across channels">' for r in reviews)
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Multiple-scatterer phase measurements</title>
<style>body{{font:16px/1.6 system-ui;max-width:1600px;margin:30px auto;padding:0 20px}}img{{width:100%}}.note{{background:#fff1d3;padding:14px}}</style>
<h1>Selected points and measured five-band phase trajectories</h1>
<p>Five candidates A–E are marked in every channel over three matching two-second windows. The curves come from complex measurements, not substituted fitted motion. RF phase is converted to excess-path change separately for each wavelength; this is the quantity to compare across frequencies. Only constant phase origins are removed. No transmitter/receiver calibration, heading, sea reflection or multipath model is fitted.</p>
<img src="00-selected-points.png" alt="All selected points in original range-rate maps">
<div class="note">The same letter is not established to be the same material point across bands. In the middle window, B in Seven and SBS is nearly stationary while E is moving. A/C and parts of B are close to stationary clutter, and D is a more distant return whose hull association is unknown. The final window includes an unsettled, strongly cancelling SBS fit; its conditional curves must not be used as calibration evidence.</div>
<p>A bright, nearly flat conditional phase is not sufficient evidence of a precise moving-point trajectory: its fitted response can overlap almost entirely with the stationary nuisance profile. The following panel quantifies that overlap. It is a projection audit, not a noise renormalization or correction applied to the data.</p><img src="04-stationary-overlap.png" alt="Candidate energy retention after actual stationary projection">
<p>The first trajectory column uses the original cells without fitted stationary/neighbour subtraction. The second uses the same templates after training-frozen subtraction of the other candidates and stationary profile. The third compares conditional path changes with Nine without removing a rate or curvature difference. The last shows conditional measurement strength. A connected phase curve is shown only above the 6 dB display threshold; no integer cycles are guessed through fades. The two diagnostics use separately selected constant origins for readability. Their absolute levels should not be compared.</p>
{images}
<p>All five channels are fitted independently. Conditional noise figures freeze the fitted nuisance responses and therefore exclude nuisance uncertainty and association error. The 250 ms complex measurement windows overlap at 50 ms spacing, and the three two-second apertures overlap; these points are not independent calibration observations. The original row likelihood, waveform kernels, C0/C1 references, range-noise metric and training/testing split are preserved. {len(errors)} selected original likelihoods reproduce with maximum absolute discrepancy {max(errors):.3g}.</p>
<p><a href="analysis.json">Measured arrays, selected cells and provenance</a> · <a href="measured-trajectories.csv">CSV</a> · <a href="../piecewise-nearby-review/index.html">Common-path and physical-pair controls</a>.</p></html>''')
    snapshot=out/'sources';snapshot.mkdir(exist_ok=True)
    for p in files:
        if p.suffix=='.py':shutil.copyfile(p,snapshot/p.name)
    print('report',out/'index.html',flush=True)


if __name__=='__main__':main()
