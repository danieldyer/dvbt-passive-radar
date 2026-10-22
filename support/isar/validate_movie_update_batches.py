"""Compare GPU scheduling on identical recorded-data coherent apertures."""
import argparse,json,tempfile,time
from pathlib import Path
import numpy as np
from channel_update import ChannelUpdates
from local_ship_motion import load_motion
from projected_aperture import apply
from ship_coherent_movie import load_apertures,plans,packet,camera_description,clip_chart,viewport_grid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True);p.add_argument('--aperture',type=Path,required=True)
    p.add_argument('--executable',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();motion=load_motion(a.motion);apertures,_=load_apertures(a.aperture);apertures=apply(apertures,motion.rf)
    times=np.array([157.,159.5]);camera=camera_description(motion);chart=clip_chart(motion,times,camera)
    along,port,_,_=viewport_grid(motion,times,camera,1.,chart,('following',));xx,yy=np.meshgrid(along,port)
    pixels=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size),np.zeros(xx.size)]
    plan=plans(apertures,times,30.);packets=[];old=None
    for frame in range(len(times)):
        current=[(int(first[frame]),int(last[frame])) for first,last in plan];selected=[]
        for c,(first,last) in enumerate(current):
            if old is None:selected.append((c,first,last,1.))
            else:selected.extend([(c,old[c][0],first,-1.),(c,old[c][1],last,1.)])
        packets.append(packet(motion,apertures,selected,chart));old=current
    outcomes=[];summary=[]
    with tempfile.TemporaryDirectory(prefix='movie-update-validation-',dir='build') as temporary:
        for active,batch in [(False,256),(True,256)]:
            worker=ChannelUpdates(pixels,Path(temporary)/str(active),active_channels=active,executable=a.executable)
            images=np.zeros((5,len(pixels)),complex);information=np.zeros(5);snapshots=[];started=time.monotonic()
            try:
                for geometry,moments in packets:
                    for first in range(0,len(geometry),batch):
                        last=min(first+batch,len(geometry));worker.accumulate(geometry[first:last],moments[:,first:last],images,information)
                    snapshots.append((images.copy(),information.copy()))
            finally:worker.close()
            outcomes.append(snapshots);summary.append(dict(active_channels=active,batch_rows=batch,seconds=time.monotonic()-started,transferred_image_bytes=worker.transferred_bytes))
            print(json.dumps(summary[-1]),flush=True)
    checks=[]
    for frame,(reference,actual) in enumerate(zip(*outcomes)):
        z,w=reference;other,other_w=actual;strong=abs(z)>.1*abs(z).max(axis=1)[:,None]
        error=float(np.linalg.norm(other-z)/np.linalg.norm(z));phase=float(np.max(abs(np.angle(other[strong]/z[strong])))*180/np.pi)
        power=float(np.max(abs(20*np.log10(abs(other[strong]/z[strong])))))
        np.testing.assert_allclose(other_w,w,rtol=1e-12,atol=1e-25)
        if error>2e-4 or phase>.02 or power>.002:raise AssertionError('GPU batching changes coherent scores')
        checks.append(dict(frame=frame,relative_complex_error=error,maximum_strong_cell_phase_difference_deg=phase,maximum_strong_cell_power_difference_db=power))
    report=dict(complete=True,pixels=len(pixels),frame_centres_s=times.tolist(),coherent_seconds=30.,schedules=summary,checks=checks,
        scope='Identical original samples, physical motion, carrier phases and per-channel information; only packet scheduling and inactive-image transport differ')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
