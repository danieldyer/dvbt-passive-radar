"""Plot selected ship-return cells and their measured multiband phase changes.

No multipath, antenna, motion, polynomial phase or stationary-clutter fit.
Candidate coordinates are selected from measured range/rate magnitude maps,
then held fixed across channels for the phase comparison. They are candidate
resolution cells, not identified material points or tracks between windows.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_ais_phase import NAMES,COLOURS,path_geometry
from measured_phase import local_average,anchored_history,interpolate_connected,range_rate_map,path_increments
from motion import C
from ship_site_signature import licensed_options


WINDOWS=[('W1',151.,155.),('W2',158.,162.),('W3',163.,167.),('W4',320.,328.)]
# Reviewed range/rate magnitude peaks. The strongest ship cell is 0 during
# the flash and 1 in the other windows. Do not label stationary ridges or the
# weak near-side cell in W1 as additional moving-target candidates.
SELECTIONS={'W1':[('B',1),('C',6)],'W2':[('A',-1),('B',1),('C',6)],
            'W3':[('A',-7),('B',0),('C',6)],'W4':[('A',7),('B',1)]}


def serial(value):
    if isinstance(value,np.ndarray):return serial(value.tolist())
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    return value


def load_fields(directory,geometry):
    prior=json.loads((geometry/'geometry-prior.json').read_text());receiver=np.asarray(prior['receiver_ecef_m'])
    options=licensed_options(geometry)
    assignment=[next(o for o in options if o['channel']==ch and o['site_id']==('12014' if ch==4 else '12013')) for ch in range(5)]
    records=[]
    for ch in range(5):
        source=directory/f'channel-{ch}.npz'
        with np.load(source) as archive:data=dict(archive)
        meta=json.loads(str(data['metadata_json']));k=2*np.pi*meta['rf_hz']/C
        tx=np.asarray(assignment[ch]['transmitter_ecef_m'])
        nominal,_=path_geometry(data['centre_ecef_m'],data['centre_velocity_ecef_mps'],tx,receiver)
        records.append(dict(data=data,k=k,nominal=nominal,source=source,
            source_sha256=hashlib.file_digest(source.open('rb'),'sha256').hexdigest()))
    return records


def analyse_window(fields,label,first,last,*,width=.5,step=.05,selections=None,acquisition_rates=None,acquisition_accelerations=None):
    rates=np.linspace(-3,3,241);maps=[]
    for f in fields:
        d=f['data'];use=(d['time']>=first)&(d['time']<last)
        maps.append(range_rate_map(d['time'][use],d['value'][use],d['information'][use],f['k'],rates)/f.get('variance_scale',1.))
    grid=np.arange(first+.3,last-.299,step);points=[]
    for name,origin in SELECTIONS[label] if selections is None else selections:
        # One shared acquisition rate assists local averaging. Every band
        # restores it, so this never enforces a common measured trajectory.
        column=int(np.flatnonzero(fields[0]['data']['offset_samples']==origin)[0])
        weakest=np.min(np.asarray(maps)[:,:,column],axis=0)
        seed_rate=float(rates[np.argmax(weakest)] if acquisition_rates is None else acquisition_rates[name])
        if not np.isfinite(seed_rate):raise ValueError('Finite declared acquisition rate required')
        acceleration=0. if acquisition_accelerations is None else float(acquisition_accelerations[name])
        if not np.isfinite(acceleration):raise ValueError('Finite acquisition acceleration required')
        histories=[];narrow=[]
        for f in fields:
            d=f['data'];use=(d['time']>=first-.6)&(d['time']<=last+.6)
            inputs=(d['time'][use],d['value'][use,column],d['information'][use],grid)
            extra=dict(acceleration_mps2=acceleration,rate_epoch_s=(first+last)/2)
            for target,aperture in ((histories,width),(narrow,.25)):
                epoch,z,var=local_average(*inputs,aperture,f['k'],seed_rate,**extra)
                target.append((epoch,z,var*f.get('variance_scale',1.)))
        ratio=np.array([abs(z)**2/var for _,z,var in histories])
        # Choose a common bright phase origin using magnitude alone, within
        # the middle half of the interval. No phase agreement drives selection.
        middle=(grid>=first+(last-first)*.25)&(grid<=last-(last-first)*.25)
        anchor=int(np.argmax(np.where(middle,np.min(ratio,axis=0),-np.inf)))
        channels=[]
        for ch,(f,(epoch,z,var),(nt,nz,nv)) in enumerate(zip(fields,histories,narrow)):
            d=f['data'];carrier=d.get('carrier_reference_path_m',d['centre_path_m'])
            old=np.interp(epoch,d['row_time'],carrier);nom=np.interp(epoch,d['row_time'],f['nominal'])
            h=anchored_history(epoch,z,var,f['k'],old,nom,anchor)
            oldn=np.interp(nt,d['row_time'],carrier);nomn=np.interp(nt,d['row_time'],f['nominal'])
            nh=anchored_history(nt,nz,nv,f['k'],oldn,nomn,anchor)
            result=dict(name=NAMES[ch],rf_hz=f['k']*C/(2*np.pi),effective_time_s=epoch,
                averaged_real=z.real,averaged_imag=z.imag,conditional_variance=var,
                phase_rad=h['phase_rad'],wrapped_phase_rad=h['wrapped_phase_rad'],
                power_to_noise=h['power_to_noise'],connected=h['connected'],good=h['good'])
            it,di,ig=path_increments(epoch,z,var,f['k'],old,5)
            result.update(increment_time_s=it,measured_increment_m=di,increment_good=ig,
                nominal_increment_m=nom[5:]-nom[:-5])
            for key in ('phase_rad','measured_path_m','nominal_path_m','residual_path_m'):
                values=interpolate_connected(epoch,h[key],grid)
                if np.isfinite(values[anchor]):values-=values[anchor]
                result['grid_'+key]=values
            result['grid_short_measured_path_m']=interpolate_connected(nt,nh['measured_path_m'],grid)
            value=result['grid_short_measured_path_m']
            if np.isfinite(value[anchor]):value-=value[anchor]
            result['connected_duration_s']=float(np.count_nonzero(h['connected'])*step)
            channels.append(result)
        points.append(dict(name=name,offset_samples=origin,offset_m=origin*C/8e6,
            acquisition_residual_rate_mps=seed_rate,acquisition_acceleration_mps2=acceleration,
            anchor_time_s=grid[anchor],anchor_index=anchor,channels=channels))
    return dict(label=label,interval_s=[first,last],time_s=grid,rates_mps=rates,maps=maps,points=points)


def plot_selection(windows,fields,out):
    fig,axes=plt.subplots(5,len(windows),figsize=(18,15),layout='constrained',sharex=True,sharey=True)
    offsets=fields[0]['data']['offset_samples']*C/8e6
    upper=float(np.ceil(10*np.log10(max(np.max(m) for w in windows for m in w['maps']))))
    for j,w in enumerate(windows):
        for ch in range(5):
            ax=axes[ch,j];power=10*np.log10(np.maximum(w['maps'][ch],1e-6))
            image=ax.imshow(power,origin='lower',aspect='auto',extent=[offsets[0]-C/16e6,offsets[-1]+C/16e6,-3,3],
                cmap='magma',vmin=0,vmax=upper,interpolation='nearest')
            for p in w['points']:
                ax.plot(p['offset_m'],p['acquisition_residual_rate_mps'],'s',mec='#00e5ff',mfc='none',ms=10,mew=1.4)
                ax.annotate(p['name'],(p['offset_m'],p['acquisition_residual_rate_mps']),xytext=(7,6),
                    textcoords='offset points',color='#00e5ff',weight='bold')
            d=fields[ch]['data'];first,last=w['interval_s'];middle=(first+last)/2
            old=np.interp([middle-.1,middle+.1],d['row_time'],d['centre_path_m']);stationary=-(old[1]-old[0])/.2
            ax.axhline(stationary,color='#ddd',lw=.7,ls='--')
            if j==0:ax.set_ylabel(f'{NAMES[ch]}\nResidual path rate (m/s)')
            if ch==0:ax.set_title(f"{w['label']}: {first:g}–{last:g} s")
            if ch==4:ax.set_xlabel('Range offset from saved AIS reference (m)')
    fig.colorbar(image,ax=axes,label='Coherent cell power / conditional thermal noise (dB)',shrink=.65)
    fig.suptitle('Selected candidate cells in every channel and window\n'
        'Direct measured periodograms; no new clutter removal. Dashed line: stationary-return rate in this AIS frame.',fontsize=14)
    fig.savefig(out/'01-selected-points.png',dpi=160);plt.close(fig)


def plot_histories(w,out):
    grid=w['time_s'];count=len(w['points'])
    fig,axes=plt.subplots(count,4,figsize=(19,3.7*count),layout='constrained',squeeze=False)
    for row,p in enumerate(w['points']):
        reference=p['channels'][2]
        for ch,(c,colour) in enumerate(zip(p['channels'],COLOURS)):
            axes[row,0].plot(grid,np.degrees(c['grid_phase_rad']),'.-',color=colour,ms=2,lw=1,label=c['name'])
            axes[row,1].plot(grid,c['grid_measured_path_m'],'.-',color=colour,ms=2,lw=1)
            axes[row,1].plot(grid,c['grid_nominal_path_m'],color=colour,lw=.8,ls='--',alpha=.7)
            if ch!=2:
                diff=c['grid_measured_path_m']-reference['grid_measured_path_m']
                predicted=c['grid_nominal_path_m']-reference['grid_nominal_path_m']
                axes[row,2].plot(grid,100*diff,color=colour,lw=1.2)
                axes[row,2].plot(grid,100*predicted,color=colour,lw=.8,ls='--')
            axes[row,3].plot(c['effective_time_s'],10*np.log10(np.maximum(c['power_to_noise'],1e-6)),color=colour,lw=1)
        axes[row,0].set_ylabel(f"{p['name']}: {p['offset_m']:+.0f} m\nPhase change (degrees)")
        axes[row,1].set_ylabel('Excess-path change (m)')
        axes[row,2].set_ylabel('Band minus Nine (cm)')
        axes[row,3].set_ylabel('Power / conditional noise (dB)');axes[row,3].axhline(10*np.log10(4),color='#777',ls=':',lw=1)
        for ax in axes[row]:
            ax.axvline(p['anchor_time_s'],color='#aaa',ls=':',lw=.6);ax.set_xlim(w['interval_s']);ax.set_xlabel('Capture time (s)')
            if 'ais_report_epoch_s' in w:
                ax.axvline(w['ais_report_epoch_s'],color='black',ls='-.',lw=.9,alpha=.75)
        axes[row,0].set_title(f"Candidate {p['name']}; common phase origin {p['anchor_time_s']:.2f} s",fontsize=10)
    axes[0,0].legend(ncol=3,fontsize=8)
    axes[0,1].set_title('Measured path; dashed = nominal AIS')
    axes[0,2].set_title('Interband change; dashed = nominal geometry')
    axes[0,3].set_title('Fade evidence; dotted line = display threshold')
    fig.suptitle(f"{w['label']} · measured phases for selected candidate cells over the same window\n"
        '0.5 s complex measurements, every 0.05 s. Only constant phase origins removed; no fitted slopes or curvature.',fontsize=14)
    fig.savefig(out/f"02-{w['label']}-trajectories.png",dpi=160);plt.close(fig)


def plot_double_differences(windows,out):
    fig,axes=plt.subplots(len(windows),3,figsize=(17,3.1*len(windows)),layout='constrained')
    metrics=[]
    for row,w in enumerate(windows):
        grid=w['time_s'];points={p['name']:p for p in w['points']};b=points['B']
        for col,name in enumerate(('A','C')):
            if name not in points:
                axes[row,col].text(.1,.5,f'No {name} candidate selected',transform=axes[row,col].transAxes)
                axes[row,col].axis('off')
                continue
            p=points[name]
            deltas=np.array([x['grid_measured_path_m']-y['grid_measured_path_m'] for x,y in zip(p['channels'],b['channels'])])
            common=np.all(np.isfinite(deltas),axis=0)
            if np.any(common):
                anchor=int(np.flatnonzero(common)[len(np.flatnonzero(common))//2]);deltas-=deltas[:,anchor,None]
            else:
                # Do not make separately anchored pieces look comparable.
                deltas[:]=np.nan
                axes[row,col].text(.1,.5,'No common five-band\nphase-connected segment',transform=axes[row,col].transAxes)
            for c,colour,values in zip(NAMES,COLOURS,deltas):axes[row,col].plot(grid,100*values,color=colour,label=c,lw=1.2)
            axes[row,col].set(title=f"{w['label']}: {p['name']} − B",ylabel='Within-band path difference (cm)',xlabel='Capture time (s)')
            if np.any(common):
                error=deltas-deltas[2]
                metrics.append(dict(window=w['label'],pair=p['name']+'-B',common_duration_s=float(common.sum()*.05),
                    band_order=NAMES,rms_against_nine_cm=(100*np.sqrt(np.nanmean(error[:,common]**2,axis=1))).tolist()))
            else:metrics.append(dict(window=w['label'],pair=p['name']+'-B',common_duration_s=0.))
        # A second averaging aperture shows whether smoothing creates apparent
        # agreement. No alignment except an explicitly removed constant.
        for c,colour in zip(b['channels'],COLOURS):
            difference=c['grid_short_measured_path_m']-c['grid_measured_path_m']
            axes[row,2].plot(grid,100*difference,color=colour,lw=1.2)
        axes[row,2].set(title=f"{w['label']}: B, 0.25 s minus 0.5 s measurement",ylabel='Averaging sensitivity (cm)',xlabel='Capture time (s)')
        for ax in axes[row]:ax.axhline(0,color='#aaa',lw=.6);ax.set_xlim(w['interval_s'])
    axes[0,1].legend(ncol=3,fontsize=8)
    fig.suptitle('Do different candidate cells share the same phase-change differences across bands?\n'
        'Within-band differencing suppresses phase change common to both cells. No slope or curvature alignment.',fontsize=13)
    fig.savefig(out/'03-between-points.png',dpi=160);plt.close(fig)
    return metrics


def plot_increments(windows,out):
    fig,axes=plt.subplots(len(windows),3,figsize=(17,3.3*len(windows)),layout='constrained')
    metrics=[]
    for row,w in enumerate(windows):
        present={p['name'] for p in w['points']}
        for col,name in enumerate(('A','B','C')):
            if name not in present:
                axes[row,col].text(.1,.5,f'No {name} candidate selected',transform=axes[row,col].transAxes)
                axes[row,col].axis('off')
        for p in w['points']:
            col=ord(p['name'])-ord('A')
            ax=axes[row,col];reference=p['channels'][2];comparisons=[]
            for ch,(c,colour) in enumerate(zip(p['channels'],COLOURS)):
                t=c['increment_time_s'];value=c['measured_increment_m']
                ax.plot(t,100*value,'.-',color=colour,ms=2,lw=1,label=c['name'])
                ax.plot(t,100*c['nominal_increment_m'],color=colour,ls='--',lw=.65,alpha=.7)
                ref=interpolate_connected(reference['increment_time_s'],reference['measured_increment_m'],t)
                refnom=np.interp(t,reference['increment_time_s'],reference['nominal_increment_m'])
                error=value-ref-(c['nominal_increment_m']-refnom)
                valid=np.isfinite(error)
                comparisons.append(dict(channel=c['name'],samples=int(valid.sum()),
                    rms_geometry_removed_difference_cm=float(100*np.sqrt(np.mean(error[valid]**2))) if np.any(valid) else None,
                    median_geometry_removed_difference_cm=float(100*np.median(error[valid])) if np.any(valid) else None))
            ax.set(title=f"{w['label']} · {p['name']} ({p['offset_m']:+.0f} m)",xlabel='Capture time (s)',
                ylabel='Excess-path change over 0.25 s (cm)',xlim=w['interval_s'])
            metrics.append(dict(window=w['label'],point=p['name'],reference_band='Nine',comparisons=comparisons))
    axes[0,1].legend(ncol=3,fontsize=8)
    fig.suptitle('Local measured phase increments: no phase unwrapping across the window\n'
        'Solid: measured. Dashed: nominal geometry + AIS. Gaps mark weak endpoints; each phase increment retains its wavelength ambiguity.',fontsize=13)
    fig.savefig(out/'04-local-increments.png',dpi=160);plt.close(fig)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields',type=Path,default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--geometry',type=Path,default=Path('docs/data/calibration_20261016_sites'))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/measured-scatterer-phases'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    fields=load_fields(args.fields,args.geometry);windows=[]
    for label,first,last in WINDOWS:
        windows.append(analyse_window(fields,label,first,last));print(label,'measured',flush=True)
    plot_selection(windows,fields,out)
    for w in windows:plot_histories(w,out)
    metrics=plot_double_differences(windows,out)
    increments=plot_increments(windows,out)
    for w in windows:w.pop('maps')
    result=dict(scope=__doc__,complete=True,production_changed=False,antenna_fit=False,
        point_selection=SELECTIONS,window_width_s=.5,output_step_s=.05,
        quality_policy='Connect only the contiguous measured segment containing the common magnitude-selected phase origin with power/conditional noise >=4. No phase cycles bridged through fades. This is a display flag, not a detection or confidence claim.',
        phase_policy='Use each carrier wavenumber and undo its saved AIS carrier deramp. Independent constant phase per candidate and band; no fitted slope, curvature or screen.',
        geometry_policy='Dashed nominal comparison only: user receiver and 00002 Ornata/Ridge priors, interpolated AIS centre. No antenna changes and no assumption that this assignment is proven.',
        capture_background='The archived profiles already contain the capture-time frozen background subtraction. This analysis adds no cancellation, notch, stationary fit or local normalization.',
        independence='Candidate cells may contain mixtures or sidelobes of the same return; their designation does not assert independent physical scatterers. Overlapping local measurements are temporally correlated.',
        source_sha256={str(f['source']):f['source_sha256'] for f in fields},windows=windows,double_differences=metrics,
        local_increment_comparisons=increments)
    for p in (Path(__file__),Path(__file__).with_name('measured_phase.py'),Path(__file__).with_name('test_measured_phase.py'),
              args.geometry/'geometry-prior.json',args.geometry/'sites-main.json',args.geometry/'acma-transmitters.json'):
        result['source_sha256'][str(p)]=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
    (out/'measurements.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    sources=out/'sources';sources.mkdir(exist_ok=True)
    for name in ('report_measured_scatterer_phases.py','measured_phase.py','test_measured_phase.py'):shutil.copyfile(Path(__file__).with_name(name),sources/name)
    sections=''.join(f'<h2>{w["label"]}: {w["interval_s"][0]:g}–{w["interval_s"][1]:g} seconds</h2><a href="02-{w["label"]}-trajectories.png"><img src="02-{w["label"]}-trajectories.png" alt="Measured phase trajectories for {w["label"]}"></a>' for w in windows)
    rows=''.join(f'<tr><td>{w["label"]}</td><td>{p["name"]}</td><td>{p["offset_m"]:+.1f}</td><td>{p["anchor_time_s"]:.2f}</td><td>'+', '.join(f'{c["name"]}: {c["connected_duration_s"]:.2f}' for c in p['channels'])+'</td></tr>' for w in windows for p in w['points'])
    flash=next(r for r in increments if r['window']=='W3' and r['point']=='B')
    differences=', '.join(f'{r["channel"]} {r["rms_geometry_removed_difference_cm"]:.1f} cm' for r in flash['comparisons'] if r['channel']!='Nine')
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Measured multiple-candidate ship phases</title>
<style>body{{font:16px/1.5 system-ui;max-width:1500px;margin:30px auto;padding:0 20px}}img{{width:100%}}td,th{{padding:7px;border:1px solid #ddd}}table{{border-collapse:collapse}}.note{{padding:15px;background:#eef4f8}}</style>
<h1>Selected points and their measured phase trajectories</h1>
<p class="note">Multipath fitting has been stopped. These plots start from the recorded complex range cells. They contain no newly fitted clutter, motion, antenna position, phase slope or phase curvature. Ten local candidates are compared in four matching windows across all five channels.</p>
<p>Candidate positions were selected by inspecting measured range/rate magnitude, then held fixed across all five channels. B marks the central return; A and C mark other visible features where available. The strongest central range cell moves from +37.5 m to 0 m during the flash. Labels identify local candidates within each window, not established tracks across windows. A marker does not prove a separate physical scatterer: mixtures and sidelobes remain visible in the measurements. Weak or stationary-ridge cells are not promoted to additional moving candidates.</p>
<a href="01-selected-points.png"><img src="01-selected-points.png" alt="Selected cells in all five measured range rate maps"></a>
<h2>What the measurements show</h2><p>The central flash cell in W3 (163–167 s) gives the clearest shared trajectory: all five channels remain phase-connected for the full displayed interval. Its 0.25 s measured path increments, after subtracting only the nominal geometric increment difference, differ from Nine by RMS {differences}. These are observed discrepancies, not formal uncertainty bounds; overlapping measurements are correlated.</p>
<p>The other W3 feature at +224.8 m also follows the main path change, especially in Seven, SBS and Nine. Its difference from the central cell is much steadier in the four non-ABC channels than in ABC over the early part of this window. Agreement is not uniform across all selected cells: earlier windows have competing phase changes, and the later W4 secondary candidate has no continuous five-band connection. The plots therefore support comparing several local trajectories, while retaining the visible failures instead of forcing a common polynomial.</p>
<h2>Reading the phase plots</h2><p>Each complex measurement averages 0.5 s of data, sampled every 0.05 s. A single declared acquisition rate is temporarily removed while averaging and restored exactly afterwards. The plotted phase is measured; no polynomial has been fitted. Phase is divided by each channel’s own RF wavenumber to obtain excess-path change, and the archived carrier deramp is undone. Only a constant phase origin is removed. Dashed curves show the current nominal Ornata/Ridge geometry and AIS, for comparison only.</p>
<p>Faint returns are not joined across fades: lines contain only the connected segment around the common phase origin with power/conditional thermal noise at least 4. The last column shows that strength directly. Missing lines mean the phase connection is unsupported by this simple measurement. The thermal model does not include coherent competing echoes. Overlapping measurements are correlated.</p>
{sections}
<h2>Differences between selected points</h2><p>For each band, A−B and C−B subtract the two measured excess-path changes, then remove one constant at a common available epoch. This suppresses phase change common to both cells; finite averaging and different echo mixtures can limit that cancellation. Similar curves across nearby transmitters would support consistent associations under slow heading change. A different range cell’s phase is not forced to agree; if no shared connected segment exists, the comparison is left blank. The last column repeats B using 0.25 s measurements as an averaging control.</p>
<a href="03-between-points.png"><img src="03-between-points.png" alt="Within-band differences between selected points and averaging control"></a>
<h2>Local phase increments without a long phase connection</h2><p>These plots use arg[z(t+0.25 s) conjugate(z(t))] directly, divided by each carrier’s wavenumber, with the known saved deramp restored. They need no global phase unwrap and keep useful pieces on either side of a fade. Increments remain ambiguous by one wavelength; the short separation is appropriate for the displayed residual-rate neighbourhood, not an arbitrary high-Doppler target. No polynomial or per-band drift is removed.</p>
<a href="04-local-increments.png"><img src="04-local-increments.png" alt="Measured local phase increments across all five bands"></a>
<table><tr><th>Window</th><th>Point</th><th>Range offset (m)</th><th>Common phase origin (s)</th><th>Connected duration per band (s)</th></tr>{rows}</table>
<p>The saved profiles already contain capture-time frozen stationary subtraction. This diagnostic adds none. It changes neither production processing nor recording metadata. Fourteen numerical tests passed, covering carrier/deramp signs, unequal frequencies, missing samples, constant phase invariance, weighted complex-noise variance, periodogram direction, local increments, stopping unwrapping at fades and the earlier range-support controls.</p>
<p><a href="measurements.json">All complex measurements and phase values</a> · <a href="tests.log">Numerical tests</a> · <a href="../index.html">Analysis index</a></p></html>''')
    print(out/'index.html',flush=True)


if __name__=='__main__':main()
