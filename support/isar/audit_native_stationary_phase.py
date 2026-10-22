"""Training-selected stationary-cell phase controls from the continuous replay.

Only constant phase origins estimated on training are removed for plotting.
No phase interpolation, correction, whitening or motion fit is applied. Cells
are separated in range and exclude the two ship-return branches. Low measured
phase variation here constrains a common instrumental-phase explanation; it
does not establish the physical origin of every selected residual response.
"""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from field import average
from native_folds import training_mask
from recorded_injection_fields import read_profiles
from recovery_experiment import NAMES
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    first,last=142.,172.
    header,channels=read_profiles(Path('build/isar-20261018/current-native-full/profiles.bin'),interval_s=(first,last))
    ranges=(header['first_range_sample']+np.arange(header['range_count']))*299792458./8e6
    domain=(ranges>=9000)&(ranges<=18500)&((ranges<13200)|(ranges>15400))
    result=dict(scope=__doc__,complete=True,interval_s=[first,last],phase_correction_applied=False,
        cell_selection='Three strongest training DC means, separated by at least 600 m, outside 13.2–15.4 km ship neighbourhood',channels=[])
    fig,axes=plt.subplots(5,1,figsize=(13,12),sharex=True,layout='constrained')
    for c,ch in sorted(channels.items()):
        n=len(ch['rows'])//16*16;w=ch['information'][:n]*ch['rows'][:n,4]
        x,weight=average(ch['values'][0,:n],w,16);time,_=average(ch['rows'][:n,1],w,16)
        zero=weight==0;time[zero]=average(ch['rows'][:n,1],np.ones(n),16)[0][zero]
        use=(time>=first)&(time<last);x=x[use];weight=weight[use];time=time[use]
        train=training_mask(time,first);mu=np.sum(weight[train,None]*x[train],axis=0)/weight[train].sum()
        score=8192*weight[train].sum()*abs(mu)**2;selected=[]
        for i in np.argsort(-score):
            if not domain[i] or any(abs(ranges[i]-ranges[j])<600 for j in selected):continue
            selected.append(i)
            if len(selected)==3:break
        ordinal=np.floor((time-first)/.25).astype(int);starts=np.r_[0,np.flatnonzero(np.diff(ordinal))+1]
        block_weight=np.add.reduceat(weight,starts);block_time=np.add.reduceat(time*weight,starts)/block_weight
        block_mean=np.add.reduceat(x[:,selected]*weight[:,None],starts,axis=0)/block_weight[:,None]
        error=np.angle(block_mean*mu[selected].conj());test=~training_mask(block_time,first)
        records=[]
        for j,i in enumerate(selected):
            rms=float(np.degrees(np.sqrt(np.average(error[test,j]**2,weights=block_weight[test]))))
            records.append(dict(range_m=ranges[i],training_dc_power_db=10*np.log10(score[i]),
                excluded_quarter_second_phase_rms_deg=rms))
            axes[c].plot(block_time,np.degrees(error[:,j]),'.-',ms=3,label=f'{ranges[i]/1000:.2f} km; excluded RMS {rms:.1f}°')
        axes[c].set(title=NAMES[c],ylabel='Phase from frozen origin (°)',ylim=(-180,180));axes[c].legend(loc='upper right',fontsize=8)
        result['channels'].append(dict(channel=c,name=NAMES[c],cells=records))
        np.savez(out/f'channel-{c}.npz',time_s=block_time,testing=test,phase_error_deg=np.degrees(error),range_m=ranges[selected])
        print(NAMES[c],serial(records),flush=True)
    axes[-1].set_xlabel('Capture time (s)')
    fig.suptitle('Independent stationary-cell controls; original C0 complex samples and weights\nQuarter-second phase means; constants selected/estimated on training only; no phase tracking')
    fig.savefig(out/'stationary-phase-controls.png',dpi=160);plt.close(fig)
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
