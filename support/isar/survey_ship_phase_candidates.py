"""Acquire more local ship-neighbourhood phase candidates by magnitude alone.

Survey four-second windows every 32 seconds. Find two separated local peaks
supported in all five channels after profiling static clutter. Rank windows
by the weaker peak's worst-band power, never phase agreement. Re-extract and
plot measured phases with the selected acquisition rates fully restored.
These are acquisition candidates, not CFAR detections or hull associations.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter

from compare_ais_phase import COLOURS, NAMES
from cluster_phase import crop
from extract_full_aperture import load as load_aperture
from fixed_range_field import fixed_reference
from phase_acquisition import projected_periodogram
from motion import C
from report_measured_scatterer_phases import analyse_window, load_fields, serial


def acquire(fields, first, last, *, accelerations=(0.,)):
    rates = np.linspace(-3., 3., 241)
    inputs = []
    stationary = []
    for f in fields:
        d = f['data']
        use = (d['time'] >= first)&(d['time'] < last)
        t, y, w = d['time'][use], d['value'][use], d['information'][use]
        inputs.append((t,y,w,f['k'],rates,float(d['pulse_samples'][64].real)))
        old = np.interp([(first+last)/2-.1, (first+last)/2+.1], d['row_time'], d['centre_path_m'])
        stationary.append(-(old[1]-old[0])/.2)
    shape=(len(rates),len(fields[0]['data']['offset_samples']))
    score=np.full(shape,-np.inf);maps=np.zeros((len(fields),*shape));acceleration_map=np.zeros(shape)
    for acceleration in accelerations:
        trial=np.array([projected_periodogram(*data,acceleration=acceleration,epoch_s=(first+last)/2)[0] for data in inputs])
        joint=np.min(trial,axis=0);better=joint>score
        score[better]=joint[better];acceleration_map[better]=acceleration
        maps=np.where(better[None,:,:],trial,maps)
    peaks = (score == maximum_filter(score, size=(5,3), mode='constant', cval=-np.inf))
    peaks[[0,-1], :] = False
    peaks[:, [0,-1]] = False
    positions = sorted(zip(*np.nonzero(peaks)), key=lambda p:-score[p])
    selected = []
    for i,j in positions:
        # Three raw range samples give >100 m separation, reducing duplicate
        # selections of one compressed main lobe. This does not prove two points.
        if any(abs(j-p['column'])<3 for p in selected):
            continue
        selected.append(dict(column=int(j), offset_samples=int(fields[0]['data']['offset_samples'][j]),
            acquisition_rate_mps=float(rates[i]), worst_band_power=float(score[i,j]),
            acquisition_acceleration_mps2=float(acceleration_map[i,j]),
            band_power=[float(m[i,j]) for m in maps]))
        if len(selected)==2:break
    return dict(interval_s=[first,last], candidates=selected,
        score=float(min(p['worst_band_power'] for p in selected)) if len(selected)==2 else 0.,
        stationary_rates_mps=stationary,acceleration_hypotheses_mps2=list(accelerations)), np.asarray(maps), rates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extent',type=int,default=16,choices=range(8,17))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/additional-phase-candidates-chirp'))
    args=parser.parse_args()
    root=Path('docs/data/isar_20261017')
    out=args.output;out.mkdir(exist_ok=True)
    if (out/'analysis.json').exists():raise ValueError('Use a new acquisition report directory')
    fields=load_fields(Path('build/isar-20261015/full-fields'),Path('docs/data/calibration_20261016_sites'))
    apertures=[load_aperture(Path(f'build/isar-20261015/full-geographic-aperture/channel-{c}')) for c in range(5)]
    survey=[];maps={};expanded={};context_errors=[]
    for first in np.arange(100.,581.,32.):
        context=[]
        for c,f in enumerate(fields):
            base=crop(f['data'],first-.8,first+4.8,np.arange(-8,9))
            offsets=np.arange(-args.extent,args.extent+1)
            wide,_=fixed_reference(base,apertures[c],first+2,offsets)
            # A separately re-extracted narrow fixed grid must reproduce the
            # interior exactly; moving-grid observations are intentionally different.
            narrow,_=fixed_reference(base,apertures[c],first+2,np.arange(-8,9))
            original=np.isin(wide.offsets,narrow.offsets)
            np.testing.assert_allclose(wide.data['value'][:,original],narrow.data['value'],rtol=1e-12,atol=1e-12)
            context_errors.append(float(np.max(abs(wide.data['value'][:,original]-narrow.data['value']))))
            nominal=np.interp(wide.data['row_time'],f['data']['row_time'],f['nominal'])
            context.append({**f,'data':wide.data,'nominal':nominal,'variance_scale':float(wide.pulse.samples[64].real)})
        expanded[first]=context
        entry,power,rates=acquire(context,first,first+4,accelerations=np.linspace(-.5,.5,21))
        survey.append(entry);maps[first]=power
        print(first,'weakest shared peak dB',round(10*np.log10(max(entry['score'],1e-30)),2),flush=True)
    ranked=sorted(survey,key=lambda e:-e['score'])
    chosen=[]
    for entry in ranked:
        if entry['score']<100 or any(abs(entry['interval_s'][0]-p['interval_s'][0])<32 for p in chosen):continue
        chosen.append(entry)
        if len(chosen)==4:break
    windows=[]
    for i,entry in enumerate(sorted(chosen,key=lambda e:e['interval_s'][0])):
        points=sorted(entry['candidates'],key=lambda p:p['offset_samples'])
        labels=['A','B']
        acquisitions=dict(zip(labels,[p['acquisition_rate_mps'] for p in points]))
        accelerations=dict(zip(labels,[p['acquisition_acceleration_mps2'] for p in points]))
        context=expanded[entry['interval_s'][0]]
        w=analyse_window(context,f'S{i+1}',*entry['interval_s'],
            selections=list(zip(labels,[p['offset_samples'] for p in points])),acquisition_rates=acquisitions,
            acquisition_accelerations=accelerations)
        w['magnitude_acquisition']=entry
        observation_directory=out/'observations';observation_directory.mkdir(exist_ok=True)
        for c,f in enumerate(context):
            np.savez(observation_directory/f"{w['label']}-channel-{c}.npz",**f['data'],nominal_path_m=f['nominal'])
        windows.append(w)
    if not windows:raise ValueError('No two-peak window meets the declared acquisition strength')
    columns=len(windows)
    fig,axes=plt.subplots(5,columns,figsize=(4.6*columns,13),layout='constrained',squeeze=False)
    offsets=np.arange(-args.extent,args.extent+1)*C/8e6
    upper=np.ceil(10*np.log10(max(maps[w['interval_s'][0]].max() for w in windows)))
    for j,w in enumerate(windows):
        first,last=w['interval_s'];power=maps[first]
        for c,ax in enumerate(axes[:,j]):
            image=ax.imshow(10*np.log10(np.maximum(power[c],1e-6)),origin='lower',aspect='auto',
                extent=[offsets[0]-C/16e6,offsets[-1]+C/16e6,-3,3],cmap='magma',vmin=0,vmax=upper,interpolation='nearest')
            ax.axhline(w['magnitude_acquisition']['stationary_rates_mps'][c],color='#ccc',ls='--',lw=.7)
            for p in w['points']:
                ax.plot(p['offset_m'],p['acquisition_residual_rate_mps'],'s',mfc='none',mec='#00e5ff',ms=10)
                ax.annotate(p['name'],(p['offset_m'],p['acquisition_residual_rate_mps']),xytext=(6,5),textcoords='offset points',color='#00e5ff')
            if j==0:ax.set_ylabel(NAMES[c]+'\nPhysical path rate (m/s)')
            if c==0:ax.set_title(f'{first:g}–{last:g} s')
            if c==4:ax.set_xlabel('Offset from saved AIS range reference (m)')
    fig.colorbar(image,ax=axes,label='Profiled chirp likelihood gain / conditional noise (dB)',shrink=.65)
    fig.suptitle('Additional candidates selected only by range/rate magnitude in all five bands\n'
        'Dashed: stationary ridge. Acquisition profiles one static coefficient per cell; phase plots use unchanged fixed-grid data.',fontsize=12)
    fig.savefig(out/'01-selected-points.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(len(windows),4,figsize=(19,3.4*len(windows)),layout='constrained',squeeze=False)
    for row,w in enumerate(windows):
        t=w['time_s'];a,b=w['points']
        for point,col in ((a,0),(b,1)):
            for ch,colour in zip(point['channels'],COLOURS):
                axes[row,col].plot(t,ch['grid_measured_path_m'],color=colour,lw=1.15,label=ch['name'])
            axes[row,col].set(title=f"{w['label']} · {point['name']} · {point['offset_m']:+.0f} m",ylabel='Measured path change (m)')
        delta=np.asarray([x['grid_measured_path_m']-y['grid_measured_path_m'] for x,y in zip(a['channels'],b['channels'])])
        good=np.all(np.isfinite(delta),axis=0)
        if good.any():
            ids=np.flatnonzero(good);anchor=ids[len(ids)//2];delta-=delta[:,anchor,None]
            for value,colour in zip(delta,COLOURS):axes[row,2].plot(t,100*value,color=colour,lw=1.15)
            w['difference_common_interval_s']=[t[ids[0]],t[ids[-1]]]
            w['difference_from_nine_rms_cm']=100*np.sqrt(np.mean((delta[:,good]-delta[2,good])**2,axis=1))
        else:
            axes[row,2].text(.5,.5,'No common five-band\nphase-connected epoch',ha='center',transform=axes[row,2].transAxes)
        axes[row,2].set(title='A − B, translation-suppressed',ylabel='Within-band path difference (cm)')
        for ch,colour in zip(b['channels'],COLOURS):
            axes[row,3].plot(ch['effective_time_s'],10*np.log10(np.maximum(ch['power_to_noise'],1e-6)),color=colour,lw=1.15)
        axes[row,3].axhline(10*np.log10(4),color='#999',ls=':',lw=.7)
        axes[row,3].set(title='B: fade evidence',ylabel='Power / conditional noise (dB)')
        for ax in axes[row]:ax.set(xlim=w['interval_s'],xlabel='Capture time (s)')
    axes[0,0].legend(ncol=3,fontsize=8)
    fig.suptitle('Additional independently acquired cells: measured five-band phase-derived paths\n'
        '0.5 s complex measurements every 50 ms; acquisition phase restored; constant origins only. '
        'No phase-based candidate ranking.',fontsize=12)
    fig.savefig(out/'02-measured-trajectories.png',dpi=150);plt.close(fig)
    for w in windows:w.pop('maps')
    result=dict(scope=__doc__,complete=True,production_changed=False,accepted_calibration=False,
        acquisition_policy='Four-second fixed-grid apertures at 100+32n seconds. Conditional chirp likelihood profiles one constant complex static coefficient and uses retained information and known pulse variance. Common physical acceleration searched from -0.5 to +0.5 m/s² at 0.05 spacing. Worst-band ranking; local 2D peaks; three-bin range separation; weaker peak >=20 dB. Every surveyed window is recorded. Acquisition only; plotted phases use original fixed-grid observations and restore the full acquisition polynomial. This is not a calibrated detector or independent significance test.',
        survey=survey,windows=windows,range_extent_samples=args.extent,
        original_inner_cells_maximum_reextraction_error=max(context_errors),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path('support/isar/report_measured_scatterer_phases.py'),Path('support/isar/measured_phase.py'),
             Path('support/isar/recorded_field_context.py'),Path('support/isar/phase_acquisition.py'),
             Path('support/isar/fixed_range_field.py'))},
        input_field_sha256={str(f['source']):f['source_sha256'] for f in fields})
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    print('chosen windows',[w['interval_s'] for w in windows],flush=True)


if __name__=='__main__':main()
