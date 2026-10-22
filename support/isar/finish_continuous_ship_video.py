"""Finish, render and validate a bounded full-capture ship video job."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def run(script,*args):
    command=[sys.executable,str(Path(__file__).with_name(script)),*map(str,args)]
    print(json.dumps(dict(stage=script,command=command)),flush=True);subprocess.run(command,check=True)


def validate_video(path,frames,fps):
    r=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate,avg_frame_rate,nb_frames,duration','-show_entries','frame=key_frame,pict_type','-of','json',str(path)],check=True,capture_output=True,text=True)
    d=json.loads(r.stdout);s=d['streams'][0]
    if s['r_frame_rate']!=f'{fps}/1' or int(s['nb_frames'])!=frames or len(d['frames'])!=frames:raise ValueError('Final video cadence/count differs from the scientific frames')
    if any(f['key_frame']!=1 or f['pict_type']!='I' for f in d['frames']):raise ValueError('QuickTime scrubbing requires every frame to be intra')
    return dict(file=str(path),frames=frames,fps=fps,duration_s=float(s['duration']),all_frames_intra=True,bytes=path.stat().st_size)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',type=Path,required=True);p.add_argument('--aperture',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--wait-pid',type=int)
    p.add_argument('--fps',type=int,default=24);p.add_argument('--playback-speed',type=float,default=60.);a=p.parse_args()
    analysis=a.fit/'analysis.json';motion=a.fit/'motion.json';last_notice=0
    while not analysis.exists() or not motion.exists():
        if a.wait_pid is None:raise ValueError('Finish the motion fit before rendering, or declare its running process')
        try:os.kill(a.wait_pid,0)
        except ProcessLookupError:raise RuntimeError('Motion fit exited without completed output')
        now=time.monotonic()
        if now-last_notice>30:print(json.dumps(dict(stage='waiting-for-radar-fit',fit=str(a.fit))),flush=True);last_notice=now
        time.sleep(5)
    fit=json.loads(analysis.read_text())
    if not fit['complete'] or fit['maximum_kinematic_constraint_violation']>1e-7:raise ValueError('Motion computation incomplete or physical bounds violated')
    a.output.mkdir(parents=True,exist_ok=True)
    run('report_continuous_ship_motion.py','--motion',motion,'--output',a.output/'motion-review')
    label='Shared physical radar motion; fixed band calibration'
    if not fit.get('numerical_convergence'):label+='; finite-iteration fit'
    preview=a.output/'following-preview';preview_cache=a.cache.with_name(a.cache.name+'-following-preview')
    if not (preview/'videos.json').exists():
        run('ship_coherent_movie.py','--motion',motion,'--aperture',a.aperture,'--output',preview_cache,'--seconds',30,'--spacing-m',1,
            '--clip-seconds',30,'--fps',a.fps,'--playback-speed',a.playback_speed,'--stationary-profile','--views','following')
        run('encode_ship_coherent_movie.py','--cache',preview_cache,'--output',preview,'--label',label)
        print(json.dumps(dict(stage='full-capture-following-preview-ready',output=str(preview))),flush=True)
    run('ship_coherent_movie.py','--motion',motion,'--aperture',a.aperture,'--output',a.cache,'--seconds',30,'--spacing-m',1,
        '--clip-seconds',30,'--fps',a.fps,'--playback-speed',a.playback_speed,'--stationary-profile')
    run('encode_ship_coherent_movie.py','--cache',a.cache,'--output',a.output,'--label',label)
    videos=json.loads((a.output/'videos.json').read_text());records=[validate_video(Path(d['path']),videos['frames'],a.fps) for d in videos['files']]
    validation=dict(complete=True,files=records,coherent_seconds=30.,playback_speed=a.playback_speed,frame_hop_capture_s=a.playback_speed/a.fps,
        overlap_fraction=1-a.playback_speed/(30*a.fps),motion_numerically_converged=fit.get('numerical_convergence'),scale_db=videos['scale_db'],
        motion_scope='Conditional radar motion and static inter-band calibration; not proof of a uniquely located or resolved hull')
    (a.output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    html='<!doctype html><meta charset="utf-8"><title>Full-capture ship ISAR</title><style>body{font:17px system-ui;max-width:1600px;margin:2em auto}video,img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Full-capture ship ISAR</h1>'
    html+=f'<p>Five individual channels and their calibrated coherent sum. North up; cyan AIS antenna track. 30-second coherent windows, {a.fps} fps, {a.playback_speed:g}× playback. Fixed colour scale across the sequence; every video frame is intra for scrubbing.</p>'
    if (a.output/'antenna-geometry-review.json').exists():
        html+='<p>Antenna positions are fixed supplied priors; the working site allocation has not been conclusively calibrated. <a href="antenna-geometry-review.json">Exact geometry and outstanding uncertainty</a>.</p>'
    for d in records:
        name=Path(d['file']).name;html+=f'<h2>{"Ship-following" if "following" in name else "Fixed geographic"} view</h2><video controls loop preload="metadata" src="{name}"></video><p><a href="{name}">Open video</a></p>'
    html+='<p><a href="motion-review/index.html">Physical trajectory and numerical fit review</a></p><pre>'+json.dumps(validation,indent=2)+'</pre>'
    (a.output/'index.html').write_text(html);print(json.dumps(validation,indent=2),flush=True)

if __name__=='__main__':main()
