"""Full-aperture, five-channel north-up ship ISAR movie caches.

All five complex images are updated from entering/leaving observations; their
static calibrated complex sum precedes detection. Every viewport pixel is
backprojected. Power projection and colour mapping are presentation only.
"""
import argparse,json,math,struct,subprocess,time
from pathlib import Path
import numpy as np
from channel_update import ChannelUpdates
from geographic_movie import frame_transform,body_corners,project_power
from joint_ship_motion import JointMotion
from local_ship_motion import load_motion
from scene_dictionary import fingerprint
from gpu_coherent_statistics import Statistics

FPS=30;SIZES={'following':(512,512),'geographic':(512,768)}


def frame_times(first,last,fps,playback_speed):
    """Separate physical aperture cadence from encoded playback cadence."""
    if fps<=0 or not np.isfinite(playback_speed) or playback_speed<=0:raise ValueError('Playback frame rate and speed must be positive')
    hop=playback_speed/fps
    return first+np.arange(max(0,int(np.floor((last-first)/hop+1e-9))+1))*hop


def lossless(path,size,fps=FPS):
    return subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pixel_format','gray16le','-video_size',f'{size[0]}x{size[1]}',
        '-framerate',str(fps),'-i','-','-an','-c:v','ffv1','-level','3','-coder','1','-context','1','-g','1','-threads','2','-n',str(path)],stdin=subprocess.PIPE)


def load_apertures(directory):
    description=json.loads((directory/'aperture.json').read_text())
    if not description['complete']:raise ValueError('Finish the binned aperture first')
    result=[]
    for c in range(5):
        d=directory/f'channel-{c}';meta=json.loads((d/'metadata.json').read_text())
        with np.load(d/'coordinates.npz') as source:a=dict(source)
        a.update(meta=meta,moments=np.load(d/'moments.npy',mmap_mode='r'));result.append(a)
    return result,description


def plans(apertures,centres,seconds):
    output=[]
    for a in apertures:
        period=float(np.median(np.diff(a['time_s'])));count=round(seconds/period)
        start=np.arange(len(a['time_s'])-count+1);mid=(a['first_time_s'][start]+a['last_time_s'][start+count-1])/2
        at=np.clip(np.searchsorted(mid,centres),0,len(mid)-1);before=np.maximum(at-1,0);at=np.where(abs(mid[before]-centres)<abs(mid[at]-centres),before,at)
        if np.max(abs(mid[at]-centres))>period:raise ValueError('Requested frame lacks a full coherent window')
        output.append((at,at+count))
    return output


def packet(motion,apertures,selections,chart=None):
    geometry=[];samples=[]
    for c,first,last,sign in selections:
        if first==last:continue
        a=apertures[c];t=a['time_s'][first:last]
        g,details=motion.geometry(c,t,a['meta']['first_range_sample'],a['information'][first:last],sign=sign)
        if chart is not None:
            # An orthogonal coordinate chart, fixed for this clip. Rotating
            # the pixel basis and both physical frame derivatives together
            # leaves every physical ray and phase exactly unchanged.
            for first_axis in [2,6]:
                g[:,first_axis:first_axis+3,:3]=np.einsum('ij,tjk->tik',chart.T,g[:,first_axis:first_axis+3,:3])
        phase=2*np.pi*motion.rf[c]/299792458.*(details['path_m']-a['carrier_reference_path_m'][first:last])
        g[:,9,1]=np.cos(phase);g[:,9,2]=np.sin(phase)
        geometry.append(g);samples.append(a['moments'][:,first:last])
    if not geometry:return None
    return np.concatenate(geometry),np.concatenate(samples,axis=1)


def camera_description(motion):
    t=np.linspace(*motion.interval,1201);state=motion.state(t);origin=motion.base(np.array([157.]))['position_ecef_m'][0];local=motion.local
    centre=(state['position_ecef_m']-origin)@local;limits=np.c_[centre[:,:2].min(axis=0)-400,centre[:,:2].max(axis=0)+400]
    centre=limits.mean(axis=1);width=max(limits[0,1]-limits[0,0],(limits[1,1]-limits[1,0])*2/3);height=width*1.5
    bounds=[centre[0]-width/2,centre[0]+width/2,centre[1]-height/2,centre[1]+height/2]
    return dict(origin_ecef_m=origin.tolist(),enu_to_ecef=local.tolist(),following_extent_m=[400.,600.],geographic_bounds_m=bounds,fixed_north_up=True,image_plane_height_m=0.)


def clip_chart(motion,times,camera):
    """Choose a compact computational chart, without changing the camera."""
    body=motion.state(np.array([times[len(times)//2]]))['body_to_ecef'][0]
    rotation=body.T@np.asarray(camera['enu_to_ecef'])
    angle=np.arctan2(rotation[1,0],rotation[0,0]);c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def viewport_grid(motion,times,camera,spacing,chart=None,views=('following','geographic')):
    state=motion.state(times);origin=np.asarray(camera['origin_ecef_m']);local=np.asarray(camera['enu_to_ecef'])
    centre=(state['position_ecef_m']-origin)@local;transforms=[];followers=[];corners=[]
    for i in range(len(times)):
        body=state['body_to_ecef'][i] if chart is None else state['body_to_ecef'][i]@chart
        transform=frame_transform(state['position_ecef_m'][i],body,origin,local,height=camera.get('image_plane_height_m',20.))
        east,north=np.asarray(camera['following_extent_m'])/2
        bounds=[centre[i,0]-east,centre[i,0]+east,centre[i,1]-north,centre[i,1]+north]
        transforms.append(transform);followers.append(bounds)
        if 'following' in views:corners.append(body_corners(bounds,*transform))
        if 'geographic' in views:corners.append(body_corners(camera['geographic_bounds_m'],*transform))
    corners=np.concatenate(corners);minimum=np.floor((corners.min(axis=0)-4*spacing)/spacing)*spacing;maximum=np.ceil((corners.max(axis=0)+4*spacing)/spacing)*spacing
    along,port=[np.arange(lo,hi+spacing*.01,spacing) for lo,hi in zip(minimum,maximum)]
    return along,port,transforms,followers


def clip(motion,apertures,times,camera,out,spacing,seconds,combination='fixed-spectrum',fps=FPS,snapshot_first=False,views=('following','geographic')):
    out.mkdir(parents=True,exist_ok=False)
    chart=clip_chart(motion,times,camera)
    along,port,transforms,followers=viewport_grid(motion,times,camera,spacing,chart,views);xx,yy=np.meshgrid(along,port)
    pixels=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,camera.get('image_plane_height_m',20.)),np.zeros(xx.size)]
    # Permit the finer grid needed for multi-band power fringes while keeping
    # the five complex rolling images and update packets below a few GiB.
    if len(pixels)>8_000_000:raise ValueError('Use shorter clips or a smaller camera extent to stay within the bounded science-grid allocation')
    plan=plans(apertures,times,seconds);num=np.zeros((5,len(pixels)),complex);information=np.zeros(5)
    worker=ChannelUpdates(pixels,out/'worker');encoders={name:lossless(out/f'{name}.mkv',(3*SIZES[name][0],2*SIZES[name][1]),fps) for name in views}
    statistics=Statistics(out/'statistics-worker') if combination=='positive-amplitude-glrt' else None
    hist=np.zeros(65536,dtype=np.uint64);peak=-np.inf;actual=[];started=time.monotonic();old=None
    progress_every=max(1,round(5/np.median(np.diff(times)))) if len(times)>1 else 1
    try:
        for frame,t in enumerate(times):
            current=[(int(begin[frame]),int(end[frame])) for begin,end in plan]
            selections=[]
            for c,(first,last) in enumerate(current):
                if old is None:selections.append((c,first,last,1.))
                else:
                    previous_first,previous_last=old[c]
                    if first<previous_first or last<previous_last:raise ValueError('Frame windows must progress monotonically')
                    selections.extend([(c,previous_first,first,-1.),(c,previous_last,last,1.)])
            data=packet(motion,apertures,selections,chart)
            if data is not None:
                geometry,moments=data
                for begin in range(0,len(geometry),256):
                    stop=min(begin+256,len(geometry));worker.accumulate(geometry[begin:stop],moments[:,begin:stop],num,information)
            if np.any(information<=0):raise ValueError('A channel has no coherent support')
            # Assert the rolling information against the original input, so a
            # dropped/duplicated update cannot masquerade as image contrast.
            expected=np.asarray([a['information'][first:last].astype('<f4').astype(float).sum() for a,(first,last) in zip(apertures,current)])
            np.testing.assert_allclose(information,expected,rtol=1e-10,atol=1e-25)
            independent=8192*abs(num)**2/information[:,None]
            if statistics is None:
                total=np.sum(abs(motion.gains)**2*information);combined=8192*abs(np.einsum('c,cp->p',motion.gains.conj(),num))**2/total
            else:
                whitened=np.asarray(num,dtype=np.complex64)
                factor=(np.sqrt(8192/information)*np.exp(-1j*np.angle(motion.gains))).astype(np.complex64)
                whitened*=factor[:,None];combined,_=statistics.evaluate(whitened)
            power=np.r_[independent,combined[None]].reshape(6,len(port),len(along))
            if snapshot_first and frame==0:
                np.savez_compressed(out/'first-frame-science.npz',along_m=along,port_m=port,chart_to_body=chart,
                    complex_score=num.reshape(5,len(port),len(along)),information=information,gains=motion.gains,power=power,time_s=t)
            for name,bounds in [('following',followers[frame]),('geographic',camera['geographic_bounds_m'])]:
                if name not in views:continue
                size=SIZES[name];panels=[]
                for c in range(6):
                    displayed=project_power(power[c],along,port,bounds,*transforms[frame],size)
                    if not np.isfinite(displayed).all() or np.any(displayed<0):raise ValueError('Unformed pixel in the displayed viewport')
                    db=10*np.log10(np.maximum(displayed,1e-30));peak=max(peak,float(db.max()));code=np.rint(np.clip((db+64)*64,0,65535)).astype('<u2')
                    hist+=np.bincount(code.ravel(),minlength=65536).astype(np.uint64);panels.append(code)
                tiled=np.r_[np.c_[panels[0],panels[1],panels[2]],np.c_[panels[3],panels[4],panels[5]]]
                encoders[name].stdin.write(tiled.tobytes())
            actual.append([[float(a['first_time_s'][first]),float(a['last_time_s'][last-1])] for a,(first,last) in zip(apertures,current)])
            old=current
            if frame%min(150,progress_every)==0 or frame+1==len(times):print(json.dumps(dict(clip=out.name,frame=frame,frames=len(times),pixels=len(pixels),seconds=time.monotonic()-started)),flush=True)
        for encoder in encoders.values():
            encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError('Scientific display cache encoding failed')
    finally:
        if statistics is not None:statistics.close()
        worker.close()
        for encoder in encoders.values():
            if encoder.poll() is None:encoder.kill();encoder.wait()
    np.save(out/'histogram.npy',hist)
    record=dict(complete=True,frame_centres_s=times.tolist(),actual_apertures_s=actual,maximum_display_power_db=peak,
        pixels=len(pixels),body_grid=[along[[0,-1]].tolist(),port[[0,-1]].tolist()],spacing_m=spacing,chart_to_body=chart.tolist(),seconds=time.monotonic()-started)
    (out/'clip.json').write_text(json.dumps(record,indent=2)+'\n')
    # The deterministic grid is recorded above; its upload packet is scratch.
    (out/'worker'/'pixels.bin').unlink()
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--aperture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=30.);p.add_argument('--spacing-m',type=float,default=1.);p.add_argument('--clip-seconds',type=float,default=30.)
    p.add_argument('--fps',type=int,default=30,help='Encoded playback frame rate')
    p.add_argument('--playback-speed',type=float,default=1.,help='Capture seconds per playback second; scientific frame hop is speed/fps')
    p.add_argument('--camera',type=Path,help='Explicit fixed north-up camera; independent of fitted attitude')
    p.add_argument('--stationary-profile',action='store_true',help='Profile one whole-record stationary range vector, retaining the original information scale')
    p.add_argument('--save-first-frame-science',action='store_true',help='Save complex scores and information for a single numerical comparison frame')
    p.add_argument('--views',nargs='+',choices=list(SIZES),default=list(SIZES),help='Form only the requested camera views')
    p.add_argument('--combination',choices=['fixed-spectrum','positive-amplitude-glrt'],default='fixed-spectrum')
    p.add_argument('--interval',type=float,nargs=2,help='Optional short validation render with full coherent apertures')
    a=p.parse_args();motion=load_motion(a.motion);apertures,description=load_apertures(a.aperture)
    if len(set(a.views))!=len(a.views):raise ValueError('Each camera view may be requested only once')
    if a.fps<=0 or not np.isfinite(a.playback_speed) or a.playback_speed<=0:raise ValueError('Playback frame rate and speed must be positive')
    hop=a.playback_speed/a.fps
    if not all('carrier_reference_path_m' in b for b in apertures):raise ValueError('Binned aperture must retain its known carrier reference')
    earliest=[];latest=[]
    for b in apertures:
        period=np.median(np.diff(b['time_s']));count=round(a.seconds/period)
        earliest.append((b['first_time_s'][0]+b['last_time_s'][count-1])/2);latest.append((b['first_time_s'][-count]+b['last_time_s'][-1])/2)
    first=math.ceil(max(earliest)*a.fps)/a.fps;last=math.floor(min(latest)*a.fps)/a.fps
    if a.interval:first=max(first,a.interval[0]);last=min(last,a.interval[1])
    frames=frame_times(first,last,a.fps,a.playback_speed)
    if len(frames)==0:raise ValueError('Requested interval has no complete coherent windows')
    camera=json.loads(a.camera.read_text()) if a.camera else camera_description(motion)
    if a.stationary_profile:
        from projected_aperture import apply
        apertures=apply(apertures,motion.rf)
    per_clip=max(1,round(a.clip_seconds/hop));maximum_pixels=0;segments=[];begin=0
    while begin<len(frames):
        stop=min(begin+per_clip,len(frames))
        while True:
            at=frames[begin:stop]
            along,port,_,_=viewport_grid(motion,at,camera,a.spacing_m,clip_chart(motion,at,camera),a.views);pixels=len(along)*len(port)
            if pixels<=8_000_000:break
            if stop-begin==1:raise ValueError(f'Camera needs {pixels} science pixels at a single frame; reduce its extent')
            stop=begin+max(1,(stop-begin)//2)
        maximum_pixels=max(maximum_pixels,pixels);segments.append((begin,stop));begin=stop
    a.output.mkdir(parents=True,exist_ok=True);manifest=a.output/'movie.json'
    result=dict(motion=str(a.motion),motion_sha256=fingerprint(a.motion),aperture=str(a.aperture),carrier_reference_motion=description['motion'],carrier_reference_sha256=description['motion_sha256'],fps=a.fps,coherent_seconds=a.seconds,
        frame_centres_s=frames.tolist(),camera=camera,sizes={name:list(SIZES[name]) for name in a.views},channels=['Seven','SBS','Nine','Ten','ABC','Coherent GLRT' if a.combination=='positive-amplitude-glrt' else 'Coherent sum'],combination_statistic=a.combination,
        cache_db_step=1/64,maximum_science_grid_pixels=maximum_pixels,complete=False,scope=__doc__)
    result['stationary_profile_projection']=a.stationary_profile
    result.update(playback_speed=a.playback_speed,frame_hop_capture_s=hop,nominal_overlap_fraction=1-hop/a.seconds)
    result['clip_frame_intervals']=[list(span) for span in segments]
    if a.stationary_profile:
        result['stationary_projection_norm_scope']='Original information retained; matched-template Schur norm is not applied to these display powers'
        result['maximum_stationary_norm_loss_bound_db']=-10*np.log10(1-max(a.seconds/np.ptp(b['time_s']) for b in apertures))
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        for key in ['motion_sha256','frame_centres_s','camera','coherent_seconds','combination_statistic','stationary_profile_projection','fps','playback_speed','clip_frame_intervals','sizes']:
            if previous.get(key,'fixed-spectrum' if key=='combination_statistic' else None)!=result[key]:raise ValueError('Existing movie cache belongs to a different formation')
    manifest.write_text(json.dumps(result,indent=2)+'\n')
    records=[]
    for index,(begin,stop) in enumerate(segments):
        out=a.output/f'clip-{index:03d}';saved=out/'clip.json'
        records.append(json.loads(saved.read_text()) if saved.exists() else clip(motion,apertures,frames[begin:stop],camera,out,a.spacing_m,a.seconds,a.combination,a.fps,a.save_first_frame_science and index==0,a.views))
    result.update(complete=True,clips=len(records));manifest.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
