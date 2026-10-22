"""Independent-band acquisition on the current native replay, before geometry.

This is a diagnostic, not a phase correction. A wide fixed range crop avoids
forcing the acquired echo into the AIS-centred crop. A range-collapsed chirp
bank supplies seeds; final scores and predictions use the exact original-row
range migration, waveform kernels, symbol moments and stationary projection.
Only training epochs select delay, rate, curvature and complex coefficients.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from chirp_acquisition import scores
from coherent_arc import CoherentArc
from field import FS
from motion import C
from recorded_injection_fields import load_fields
from recovery_experiment import NAMES
from report_measured_scatterer_phases import serial
from fixed_range_field import stationary_profile_design


def acquire(arc, *, delay_limit=450., count=3,origin_bounds=None):
    f=arc.field;keep=arc.training;r=len(f.offsets);dt=f.time[keep]-arc.epoch
    centre=float(np.mean(f.offsets))*C/FS
    delays=(centre+np.arange(-delay_limit,delay_limit+C/FS/8,C/FS/2) if origin_bounds is None else
        np.arange(origin_bounds[0],origin_bounds[1]+.125,.5)*C/FS)
    incoherent=[];templates=[]
    clean=arc.clean.reshape(-1,r)
    for delay in delays:
        h=arc.template([delay*FS/C,0.,0.]).reshape(f.y.shape)[keep]
        templates.append(h)
        incoherent.append(np.sum(abs(np.sum(h.conj()*clean,axis=1))**2/
                                 np.maximum(np.sum(abs(h)**2,axis=1),1e-100)))
    # Edge maxima must be reported too: an edge peak diagnoses an inadequate
    # range crop; excluding it can silently select a weaker interior return.
    maxima=np.r_[0,find_peaks(incoherent)[0],len(delays)-1]
    maxima=sorted(maxima,key=lambda i:-incoherent[i])[:count]
    rate,acceleration=np.meshgrid(np.linspace(-1.,1.,101),np.linspace(-.1,.1,201),indexing='ij')
    rate,acceleration=rate.ravel(),acceleration.ravel();proposals=[]
    for index in maxima:
        power=scores(templates[index],clean,arc.nuisance.q,dt,f.k,rate,acceleration)
        accepted=[]
        for i in np.argsort(-power):
            if any(abs(rate[i]-rate[j])<.06 and abs(acceleration[i]-acceleration[j])<.006 for j in accepted):continue
            accepted.append(i)
            proposals.append(dict(delay_m=delays[index],rate_mps=rate[i],acceleration_mps2=acceleration[i],
                approximate_score=power[i],seed=[delays[index]*FS/C,rate[i]*arc.scale,.5*acceleration[i]*arc.scale**2]))
            if len(accepted)==2:break
    return sorted(proposals,key=lambda p:-p['approximate_score'])[:count],delays,np.asarray(incoherent)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--first',type=float,default=150.)
    parser.add_argument('--last',type=float,default=168.)
    parser.add_argument('--count',type=int,default=3)
    parser.add_argument('--range-centre-m',type=float,default=0.)
    parser.add_argument('--fixed-reference',action='store_true',help='Control: fixed absolute range/phase observer and constant-profile stationary nuisance')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    # Leave room for the stationary-delay basis and 64-tap kernel support.
    offsets=np.arange(-12,13)+int(round(args.range_centre_m*FS/C))
    fixed_epoch=(args.first+args.last)/2 if args.fixed_reference else None
    fields,*_=load_fields(Path('build/injection-20261018/deterministic-baseline/profiles.bin'),offsets=offsets,fixed_epoch_s=fixed_epoch)
    result=dict(scope=__doc__,interval_s=[args.first,args.last],offset_samples=offsets,fixed_reference_epoch_s=fixed_epoch,channels=[],complete=False)
    def save():(out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(5,3,figsize=(16,15),layout='constrained')
    for c,original in enumerate(fields):
        f=original.windowed(args.first,args.last)
        keep=np.floor((f.time-args.first)/2).astype(int)%4!=1
        arc=CoherentArc(f,keep,stationary_design=stationary_profile_design(f)) if args.fixed_reference else CoherentArc(f,keep)
        proposals,delay,power=acquire(arc,count=args.count)
        save();print('Acquired',NAMES[c],serial(proposals),flush=True)
        fit=arc.fit(origin_bounds=(float(offsets.min()-1),float(offsets.max()+1)),rate_bound=3.,seeds=[p['seed'] for p in proposals])
        diagnostic=arc.diagnostics(fit)
        p=fit.parameters
        record=dict(channel=c,name=NAMES[c],epoch_s=arc.epoch,scale_s=arc.scale,proposals=proposals,
            parameters=p,objective=fit.objective,gradient=fit.gradient,trials=fit.trials,
            origin_m=p[0]*C/FS,rate_mps=p[2]/arc.scale,acceleration_mps2=2*p[3]/arc.scale**2,
            prediction={k:v for k,v in diagnostic.items() if k not in ('time_s','training','testing','phase_residual_deg','amplitude','matched_information','model_path_increment_m','model_residual_phase_rad')})
        result['channels'].append(record);save()
        np.savez(out/f'channel-{c}-histories.npz',time_s=f.time,training=keep,
            phase_residual_deg=diagnostic['phase_residual_deg'],amplitude=diagnostic['amplitude'],
            matched_information=diagnostic['matched_information'],path_increment_m=diagnostic['model_path_increment_m'],
            delay_grid_m=delay,incoherent_score=power)
        axes[c,0].plot(delay,10*np.log10(np.maximum(power,1e-30)));axes[c,0].axvline(record['origin_m'],color='tab:red')
        axes[c,0].set(title=NAMES[c]+' · acquisition',ylabel='Noncoherent acquisition score (dB)',xlabel='Delay from AIS prediction (m)')
        t=f.time;phase=np.asarray(diagnostic['phase_residual_deg']);strength=np.asarray(diagnostic['matched_information'])*np.asarray(diagnostic['amplitude'])**2
        # Diagnostics retain all phase samples, including fades; colour shows
        # their measured matched strength rather than hiding bad phase points.
        axes[c,1].scatter(t[keep],phase[keep],c=np.log10(np.maximum(strength[keep],1e-30)),s=3,cmap='viridis')
        axes[c,1].scatter(t[~keep],phase[~keep],s=5,color='tab:red',label='Excluded')
        axes[c,1].set(ylim=(-180,180),ylabel='Phase residual (degrees)',xlabel='Capture time (s)',title=f"Exact quadratic fit · prediction {record['prediction']['test_complex_prediction_fraction']:.1%}")
        axes[c,2].plot(t,np.asarray(diagnostic['model_path_increment_m']));axes[c,2].set(title='Fitted residual path from AIS',xlabel='Capture time (s)',ylabel='Path change excluding constant (m)')
        print('Exact',NAMES[c],serial({k:record[k] for k in ('origin_m','rate_mps','acceleration_mps2','objective','gradient','prediction')}),flush=True)
        fig.savefig(out/'independent-band-chirps.png',dpi=140)
    result['complete']=True;save();plt.close(fig)


if __name__=='__main__':main()
