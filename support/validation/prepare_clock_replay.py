"""Export original feedback-removed timing observations for a clock-only audit.

Measurement noise is estimated from original-epoch second differences in the
first sixty seconds. Clock curvature at one symbol is treated as noise. This is a
white measurement-noise model for the diagnostic, not a real-data CRLB claim.
"""
import argparse,json,struct
from pathlib import Path
import numpy as np
from audit_capture_clock import clock_at


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metrology',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    description=json.loads((a.metrology/'analysis.json').read_text());fs=description['coordinate_rate_hz']
    with np.load(a.metrology/'coordinates.npz') as f:
        segments=f['segments'];rows=[f[f'original_rows_{c}'] for c in range(len(description['channels']))]
        t=f['time_s'];variance=[]
        data=np.empty(sum(len(r) for r in rows),dtype=[('time','<f8'),('offset','<f8'),('variance','<f8'),('channel','<u8')]);at=0
        for c,r in enumerate(rows):
            n=r[:,1]*fs;offset,rate=clock_at(segments,n)
            nominal=np.median(np.diff(r[:,0]));shift=r[:,0]-r[0,0]-np.arange(len(r))*nominal
            correction=(offset-(r[:,6]-shift)*(fs/8e6)*rate)
            correction-=correction[0]
            initial=r[:,1]<r[0,1]+60
            noise=float(np.mean(np.diff(correction[initial],n=2)**2)/6)
            if noise<=0:raise ValueError('No positive white timing-noise estimate')
            variance.append(noise/fs**2)
            view=data[at:at+len(r)];view['time']=r[:,1];view['offset']=correction;view['variance']=noise;view['channel']=c;at+=len(r)
        data=data[np.argsort(data['time'],kind='stable')]
        with (a.output/'observations.bin').open('xb') as f:
            f.write(b'CLKOBS01'+struct.pack('<IdQ',len(rows),fs,len(data)));data.tofile(f)
    result=dict(complete=True,scope=__doc__,metrology=str(a.metrology),observations=len(data),noise_fit_interval_s=[float(t[0]),float(t[0]+60)],
        estimated_white_timing_sigma_ns=(np.sqrt(variance)*1e9).tolist(),measurement_covariance_is_fisher_bound=False,
        noise_model_validation_required=True)
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
