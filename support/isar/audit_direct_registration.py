"""Measure native direct-path peak lags without applying any range correction."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from recorded_injection_fields import read_profiles
from report_measured_scatterer_phases import serial
from recovery_experiment import NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profiles',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    header,channels=read_profiles(args.profiles)
    if header['first_range_sample']!=0 or header['range_count']!=1024 or header['plan'].get('cancel_direct',True):
        raise ValueError('Require full native lag support with direct cancellation explicitly disabled')
    lag=np.arange(-32,33);fig,axes=plt.subplots(5,2,figsize=(13,13),layout='constrained');records=[]
    for c in range(5):
        ch=channels[c];use=ch['rows'][:,4]!=0;w=ch['information'][use];value=ch['values'][0,use][:,lag%1024]
        mean=np.sum(w[:,None]*value,axis=0)/w.sum();peak=int(lag[np.argmax(abs(mean))]);native=lag[np.argmax(abs(value),axis=1)]
        pulse=np.sum(w[:,None]*ch['pulse'][use],axis=0)/w.sum();reference=pulse[lag+64]
        axes[c,0].plot(lag,20*np.log10(np.maximum(abs(mean),1e-15)),label='Uncancelled native C0 mean')
        axes[c,0].plot(lag,20*np.log10(np.maximum(abs(reference),1e-15)),ls='--',label='Known waveform pulse')
        axes[c,0].axvline(0,color='#777',lw=.7);axes[c,0].set(title=NAMES[c],xlabel='Lag (8 MS/s samples)',ylabel='Correlation magnitude (dB)',ylim=(-100,5))
        axes[c,1].plot(ch['rows'][use,1],native,'.',ms=2);axes[c,1].set(xlabel='Capture time (s)',ylabel='Native integer peak lag (samples)',ylim=(-2,2))
        unique,count=np.unique(native,return_counts=True)
        records.append(dict(channel=c,name=NAMES[c],coherent_mean_peak_lag_samples=peak,
            native_peak_histogram=[dict(lag_samples=int(a),rows=int(b)) for a,b in zip(unique,count)],
            direct_mean_magnitude_at_zero=float(abs(mean[lag==0][0])),
            peak_resolution_samples=1,interpretation='Integer-lag diagnostic only; no sub-sample calibration or nuisance correction applied'))
    axes[0,0].legend(fontsize=8);fig.suptitle('Production matched compressor, direct cancellation disabled\nAn approximately 12-sample common range-origin error would be visible here')
    fig.savefig(args.output/'direct-registration.png',dpi=160);plt.close(fig)
    result=dict(scope=__doc__,complete=True,profiles=str(args.profiles),channels=records,
        sample_rate_hz=8000000,metres_per_lag_sample=299792458/8000000,range_correction_applied=False)
    (args.output/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(serial(records)),flush=True)


if __name__=='__main__':main()
