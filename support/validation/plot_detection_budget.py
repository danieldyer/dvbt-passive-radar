#!/usr/bin/env python3
"""Plot the qualified model cases and measured CPU/GPU costs separately."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run(bound,full,benchmark,output):
    b,f,t=(json.loads(p.read_text()) for p in (bound,full,benchmark))
    output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for name in ('white','smooth_20db','strong_narrowband'):
        rows=[r for r in b['results'] if r['noise']==name]
        axes[0].scatter([r['doppler_hz'] for r in rows],[r['loss_including_cp_clock_db'] for r in rows],s=14,alpha=.6,label=name)
    axes[0].scatter([r['doppler_hz'] for r in f['results']],[r['loss_including_cp_clock_dc_db'] for r in f['results']],marker='x',s=65,c='black',label='Full 3 s CPI')
    axes[0].axhline(1,color='red',ls='--',label='1 dB criterion')
    axes[0].set(xlabel='Doppler [Hz]',ylabel='Conditional model loss [dB]',title='Known template and diagonal noise model',ylim=(0,1.1));axes[0].legend(fontsize=8)
    labels=['Noise estimate','Finite CAF','CV focus','Noise calibration','Standardization'];keys=['gpu_noise_ms_per_s','gpu_profiles_ms_per_s','gpu_focus_ms_per_s','calibration_cpu_ms_per_s','standardization_cpu_ms_per_s']
    axes[1].barh(labels,[np.mean([run[k] for run in t['matched_runs']]) for k in keys],color=['#365b9d']*3+['#b15a32']*2)
    axes[1].set(xlabel='Stage time [ms per signal-second]',title='Blue: device timestamps; orange: CPU stages')
    fig.savefig(output/'efficiency_and_cost.png',dpi=180);fig.savefig(output/'efficiency_and_cost.pdf');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bound',type=Path);p.add_argument('full',type=Path);p.add_argument('benchmark',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.bound,a.full,a.benchmark,a.output)
