"""Retain a fixed-grid, full-capture complex field for continuous phase fitting.

Only the existing range-compressed archive is read. The main delay branch
stays inside one fixed range crop for the full recording; no per-window phase
origin or target-dependent observation mask is introduced. Original native
epochs/weights are retained beside representative bins for averaging checks.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from joint_ship_motion import JointMotion
from ship_window import Dataset
from motion import excess_path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--observer-path-offset-m',type=float,default=450.);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();motion=JointMotion(a.motion);dataset=Dataset();at=np.array([157.]);s=dataset.state(at,motion.result['base_model'])
    _,gradient=excess_path(s['position_ecef_m'],dataset.tx[3],dataset.rx);g=gradient[0];up=motion.local[:,2];g-=up*(g@up);observer=s['position_ecef_m'][0]+g/(g@g)*a.observer_path_offset_m
    print(json.dumps(dict(stage='constructing-fixed-full-capture-field',observer_ecef_m=observer.tolist())),flush=True)
    interval=list(motion.result['interval_s'])
    # The exported known-waveform archive stops a few OFDM symbols before
    # the last raw profile. Its end is explicit fit support, not a new erasure
    # invented from a trial motion or a reason to discard original data.
    interval[1]=min(interval[1],min(dataset.grid[f'channel_{c}']['time'][-1] for c in range(5)))
    window=dataset.window(*interval,model=motion.result['base_model'],observer_ecef_m=observer);records=[]
    for c,(native,fast) in enumerate(zip(window['native'],window['fast'])):
        out=a.output/f'channel-{c}';out.mkdir();f=fast.field;kernel=f.pulse.kernels
        np.savez_compressed(out/'field.npz',**f.data)
        np.save(out/'mean-pulse.npy',f.pulse.samples);np.save(out/'kernel-samples.npy',kernel.samples)
        np.save(out/'kernel-coefficients.npy',kernel.coefficients);np.save(out/'kernel-derivatives.npy',kernel.derivative_coefficients)
        np.savez_compressed(out/'epochs.npz',native_time_s=native.field.data['row_time'],native_information=native.field.data['row_information'],
            row_blocks=f.pulse.row_blocks,training=fast.training)
        record=dict(channel=c,rows=len(f.time),native_rows=len(native.field.data['row_time']),range_cells=len(f.offsets),
            kernel_radius=kernel.radius,kernel_degree=kernel.degree,seconds=time.monotonic()-started)
        records.append(record);print(json.dumps(record),flush=True)
        (a.output/'analysis.json').write_text(json.dumps(dict(complete=False,motion=str(a.motion),observer_ecef_m=observer.tolist(),scope=__doc__,channels=records),indent=2)+'\n')
    (a.output/'analysis.json').write_text(json.dumps(dict(complete=True,motion=str(a.motion),observer_ecef_m=observer.tolist(),scope=__doc__,channels=records,
        interval_s=interval,source_curve_interval_s=motion.result['interval_s'],known_waveform_erasures=window['erasures'],seconds=time.monotonic()-started),indent=2)+'\n')
if __name__=='__main__':main()
