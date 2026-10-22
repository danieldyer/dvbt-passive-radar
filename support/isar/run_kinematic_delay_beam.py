"""Compare finite-beam kinematic acquisitions on the full delay/rate cube."""
import argparse,json,time
from pathlib import Path
import numpy as np
from kinematic_delay_beam import track
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--acquisition',type=Path,required=True);p.add_argument('--beams',type=int,nargs='+',default=[2048,8192])
    p.add_argument('--gate-offsets-m',type=float,nargs=2,default=[250.,650.],help='Declared delay gate relative to the saved coarse prediction')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);d=np.load(a.acquisition);started=time.monotonic();reports=[]
    if a.gate_offsets_m[0]>=a.gate_offsets_m[1]:raise ValueError('Delay gate must be ordered')
    gate=d['predicted'][:,0,None]+np.asarray(a.gate_offsets_m)[None,:]
    for beam in a.beams:
        result=track(d['centres_s'],d['ranges_m'],d['rates_mps'],d['power'],gate,beam=beam)
        result['seconds']=time.monotonic()-started;reports.append(result)
        np.savez_compressed(a.output/f'track-{beam}.npz',time_s=d['centres_s'],**{k:result[k] for k in ['path_m','rate_mps']})
        (a.output/'analysis.json').write_text(json.dumps(serial(dict(complete=False,scope=__doc__,source=str(a.acquisition),trials=reports)),indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['path_m','rate_mps']}),flush=True)
    (a.output/'analysis.json').write_text(json.dumps(serial(dict(complete=True,scope=__doc__,source=str(a.acquisition),gate_offsets_m=a.gate_offsets_m,trials=reports)),indent=2)+'\n')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(15,11),layout='constrained',sharex=True);t=d['centres_s']
    for r in reports:
        axes[0].plot(t,r['path_m'],label=f'{r["beam"]} states');axes[1].plot(t,r['rate_mps']);axes[2].plot(t,r['path_m']-reports[-1]['path_m'])
    axes[0].legend();axes[0].set_ylabel('Excess path (m)');axes[1].set_ylabel('Path rate (m/s)');axes[2].set(xlabel='Capture time (s)',ylabel='Path minus largest beam (m)')
    fig.suptitle('Kinematic track-before-detect acquisition; full delay/rate likelihood\nContinuous states; exact linked path/rate propagation; Gaussian interval-acceleration prior sigma 0.03 m/s²; |acceleration| ≤ 0.15 m/s²')
    fig.savefig(a.output/'kinematic-acquisition.png',dpi=140);plt.close(fig)
if __name__=='__main__':main()
