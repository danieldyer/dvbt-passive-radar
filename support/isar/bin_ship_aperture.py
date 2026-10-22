"""Validated short coherent means for offline native-symbol ship BP.

Multiply both original moments by the exact fitted centre carrier phasor,
then take information-weighted means on the unchanged range sample grid.
The pixel-dependent carrier phase and symbol Doppler remain in backprojection.
This is not a retiming or image warp. Original epochs and bin support are saved.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from extract_full_aperture import load
from field import average
from joint_ship_motion import JointMotion
from motion import C,excess_path
from scene_dictionary import fingerprint


def coherent_means(motion,channel,aperture,first,last,block=4):
    """Yield short means without materialising a full capture or window."""
    for begin in range(first,last,512*block):
        end=min(last,begin+512*block);t=aperture['time_s'][begin:end]
        w=aperture['information'][begin:end]*aperture['valid'][begin:end]
        state=motion.state(t);path=excess_path(state['position_ecef_m'],motion.tx[channel],motion.rx)[0]
        phase=np.exp(2j*np.pi*motion.rf[channel]/C*path);chunk=np.asarray(aperture['moments'][:,begin:end],dtype=complex)*phase[None,:,None]
        values=[]
        for moment in range(2):
            mean,information=average(chunk[moment],w,block);values.append(mean)
        tm,_=average(t,w,block);fallback=average(t,np.ones(len(t)),block)[0];tm=np.where(information>0,tm,fallback)
        at=np.arange(0,len(t),block)
        reference=excess_path(motion.state(tm)['position_ecef_m'],motion.tx[channel],motion.rx)[0]
        yield tm,information,np.asarray(values,dtype='<c8'),t[at],t[np.minimum(at+block-1,len(t)-1)],reference


def prepare(motion,directory,block=4):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);description=dict(motion=str(motion.path),motion_sha256=fingerprint(motion.path),block_rows=block,channels=[],complete=False)
    for c in range(5):
        out=directory/f'channel-{c}';out.mkdir(exist_ok=True);manifest=out/'metadata.json'
        if manifest.exists():
            existing=json.loads(manifest.read_text())
            if existing['motion_sha256']!=description['motion_sha256'] or existing['block_rows']!=block:raise ValueError('Binned aperture belongs to a different motion')
            if existing['complete']:description['channels'].append(existing);continue
        a=load(Path(f'build/isar-20261015/full-geographic-aperture/channel-{c}'));source_meta=json.loads(a['metadata_json'])
        keep=np.flatnonzero((a['time_s']>=motion.interval[0])&(a['time_s']<=motion.interval[1]));start=keep[0];stop=keep[-1]+1
        count=(len(keep)+block-1)//block;data=np.lib.format.open_memmap(out/'moments.npy',mode='w+',dtype='<c8',shape=(2,count,a['moments'].shape[2]))
        times=[];weights=[];first=[];last=[];references=[];cursor=0;started=time.monotonic()
        for tm,information,chunk,begin_time,end_time,reference in coherent_means(motion,c,a,start,stop,block):
            data[:,cursor:cursor+len(tm)]=chunk
            times.append(tm);weights.append(information);first.append(begin_time);last.append(end_time);references.append(reference)
            cursor+=len(tm)
        data.flush();times=np.concatenate(times);weights=np.concatenate(weights)
        np.savez(out/'coordinates.npz',time_s=times,information=weights,first_time_s=np.concatenate(first),last_time_s=np.concatenate(last),carrier_reference_path_m=np.concatenate(references))
        record=dict(channel=c,block_rows=block,rows=count,original_rows=len(keep),first_range_sample=source_meta['first_range_sample'],
            ranges=a['moments'].shape[2],motion_sha256=description['motion_sha256'],rf_hz=float(motion.rf[c]),interval_s=[float(first[0][0]),float(last[-1][-1])],seconds=time.monotonic()-started,
            centre_carrier_removed=True,range_grid_shifted=False,complete=True)
        manifest.write_text(json.dumps(record,indent=2)+'\n');description['channels'].append(record);print(json.dumps(record),flush=True)
    if fingerprint(motion.path)!=description['motion_sha256']:raise ValueError('Motion changed while preparing the aperture')
    description['complete']=True;(directory/'aperture.json').write_text(json.dumps(description,indent=2)+'\n');return description


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--block-rows',type=int,default=4);a=p.parse_args()
    if a.block_rows not in [1,2,4,8,16]:raise ValueError('Choose a small declared coherent block')
    prepare(JointMotion(a.motion),a.output,a.block_rows)
if __name__=='__main__':main()
