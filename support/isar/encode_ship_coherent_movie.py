"""Fixed-scale, all-intra north-up six-panel ship ISAR videos with AIS."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import matplotlib
from PIL import Image,ImageDraw,ImageFont
from ais_video_overlay import AisOverlay,CYAN
from channel_update import read_exact
from local_ship_motion import load_motion
from scene_dictionary import fingerprint


class Overlay(AisOverlay):
    def __init__(self,motion,description,times):
        self.camera=description['camera'];origin=np.asarray(self.camera['origin_ecef_m']);frame=np.asarray(self.camera['enu_to_ecef'])
        start=times[0]-description['coherent_seconds']/2;stop=times[-1]+description['coherent_seconds']/2
        t=np.linspace(start,stop,int(np.ceil((stop-start)*2))+1);a=motion.track.sample(motion.epoch+t,max_gap_s=400.)
        self.path=((a['position_ecef_m']-origin)@frame)[:,:2];self.dashed=a['bracketing_interval_s']>60;self.dash_on=np.floor((t-t[0])/4).astype(int)%2==0
        self.current=((motion.track.sample(motion.epoch+times,max_gap_s=400.)['position_ecef_m']-origin)@frame)[:,:2]
        use=(motion.track.seconds>=motion.epoch+start)&(motion.track.seconds<=motion.epoch+stop)
        self.reports=((motion.track.sample(motion.track.seconds[use],max_gap_s=400.)['position_ecef_m']-origin)@frame)[:,:2]
        self.centres=((motion.state(times)['position_ecef_m']-origin)@frame)[:,:2];self.fixed_layers={}


def encoder(path,size,fps):
    return subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r',str(fps),'-i','-',
        '-an','-c:v','libx264','-preset','fast','-crf','17','-g','1','-bf','0','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart','-n',str(path)],stdin=subprocess.PIPE)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache','--input',dest='cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scale-db',nargs=2,type=float);p.add_argument('--label',default='Joint radar motion; static band calibration');a=p.parse_args()
    description=json.loads((a.cache/'movie.json').read_text())
    if not description['complete']:raise ValueError('Finish all clips before fixing the full-sequence colour scale')
    if fingerprint(Path(description['motion']))!=description['motion_sha256']:raise ValueError('Motion changed since image formation')
    motion=load_motion(Path(description['motion']));paths=sorted(a.cache.glob('clip-*/clip.json'));records=[json.loads(p.read_text()) for p in paths]
    times=np.concatenate([r['frame_centres_s'] for r in records])
    np.testing.assert_array_equal(times,description['frame_centres_s'])
    fps=description['fps'];hop=description.get('frame_hop_capture_s',1/fps)
    if len(times)>1 and max(abs(np.diff(times)-hop))>1e-8:raise ValueError('Frames do not have the declared capture cadence')
    histogram=sum(np.load(p.parent/'histogram.npy') for p in paths);grouped=histogram.reshape(-1,16).sum(axis=1)
    level=(np.arange(len(grouped))+.5)/4-64;mode=float(level[np.argmax(np.where((level>-25)&(level<35),grouped,0))])
    low,high=a.scale_db if a.scale_db else [mode-1.5,np.ceil((max(r['maximum_display_power_db'] for r in records)+.1)*10)/10]
    codes=np.arange(65536)/64-64;lut=np.rint(matplotlib.colormaps['magma'](np.clip((codes-low)/(high-low),0,1))[:,:3]*255).astype('u1')
    font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),20);small=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),16)
    overlay=Overlay(motion,description,times);a.output.mkdir(parents=True,exist_ok=True)
    summary=dict(complete=False,frames=len(times),fps=fps,coherent_seconds=description['coherent_seconds'],scale_db=[low,high],background_mode_db=mode,playback_speed=description.get('playback_speed',1.),frame_hop_capture_s=hop,
        motion=description['motion'],motion_sha256=description['motion_sha256'],files=[],encoding='H.264, every frame intra, no B frames, faststart',
        scientific_scope='Five complex matched images formed before power and display projection; sixth panel uses the manifest-declared calibrated coherent statistic',combination_statistic=description.get('combination_statistic','fixed-spectrum'),label=a.label)
    sizes=description['sizes']
    for name,(width,height) in sizes.items():
        size=(3*width+100,2*(height+34)+180);path=a.output/f'ship-{name}-all-five-and-coherent-{description["coherent_seconds"]:g}s-{fps:g}fps.mp4'
        out=encoder(path,size,fps);base=Image.new('RGB',size,'#111217');draw=ImageDraw.Draw(base)
        title='Ship-comoving viewport; north up' if name=='following' else 'Fixed geographic viewport; north up'
        draw.text((24,12),title,font=font,fill='white');draw.text((24,42),f'{description["coherent_seconds"]:g} s coherent window · {description.get("playback_speed",1.):g}× playback · '+a.label,font=small,fill='#dddddd')
        draw.text((24,size[1]-58),'Cyan: interpolated AIS antenna track; squares: actual reports; dashed across >60 s gaps',font=small,fill=CYAN)
        draw.text((24,size[1]-30),f'Fixed scale {low:.1f} to {high:.1f} dB; no Doppler notch; east right / north up',font=small,fill='#dddddd')
        for c,title in enumerate(description['channels']):draw.text((20+(c%3)*width,82+(c//3)*(height+34)),title,font=font,fill='white')
        y0=112;bar=np.rint(matplotlib.colormaps['magma'](np.linspace(1,0,2*height+34))[:,:3]*255).astype('u1')
        base.paste(Image.fromarray(np.repeat(bar[:,None],16,axis=1)),(3*width+26,y0))
        for fraction in [0.,.25,.5,.75,1.]:draw.text((3*width+47,y0+fraction*(2*height+12)),f'{high-fraction*(high-low):.0f}',font=small,fill='white')
        count=0
        try:
            for source,record in zip(paths,records):
                decoder=subprocess.Popen(['ffmpeg','-v','error','-threads','2','-i',str(source.parent/f'{name}.mkv'),'-f','rawvideo','-pix_fmt','gray16le','-'],stdout=subprocess.PIPE)
                for t in record['frame_centres_s']:
                    raw=np.frombuffer(read_exact(decoder.stdout,3*width*2*height*2),'<u2').reshape(2*height,3*width);frame=base.copy()
                    for c in range(6):
                        pane=Image.fromarray(lut[raw[(c//3)*height:(c//3+1)*height,(c%3)*width:(c%3+1)*width]])
                        overlay.draw(pane,name,count);frame.paste(pane,(20+(c%3)*width,y0+(c//3)*(height+34)))
                    draw=ImageDraw.Draw(frame);draw.text((size[0]-340,12),f'Capture {t:.3f} s',font=font,fill='white');out.stdin.write(frame.tobytes())
                    if count in [0,len(times)//4,len(times)//2,3*len(times)//4,len(times)-1]:frame.save(a.output/f'{name}-frame-{count:05d}.png')
                    count+=1
                if decoder.stdout.read(1) or decoder.wait()!=0:raise ValueError('Unexpected lossless-cache length')
                print(json.dumps(dict(video=name,encoded_frames=count,frames=len(times))),flush=True)
            out.stdin.close()
            if out.wait()!=0 or count!=len(times):raise RuntimeError('Final video encode failed')
        except BaseException:out.kill();out.wait();raise
        summary['files'].append(dict(path=str(path),bytes=path.stat().st_size,size=size));summary['complete']=len(summary['files'])==len(sizes)
        (a.output/'videos.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
