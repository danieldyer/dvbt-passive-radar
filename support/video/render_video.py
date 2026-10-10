#!/usr/bin/env python3
"""Render recorded DVB-T I/Q to an unpaced, timestamped range-Doppler video."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ENCODINGS = {
    'h264': ('.mp4', ('-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p')),
    # Apple's hardware codec encodes every frame independently for random access.
    'prores': ('.mov', ('-c:v','prores_videotoolbox','-profile:v','standard','-allow_sw','0',
                       '-vf','scale=out_color_matrix=bt709:out_range=tv','-pix_fmt','p210le',
                       '-color_range','tv','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709',
                       '-bsf:v','prores_metadata=color_primaries=bt709:color_trc=bt709:colorspace=bt709')),
}


def read_exact(source, size, allow_eof=False):
    result = bytearray(size)
    position = 0
    while position < size:
        count = source.readinto(memoryview(result)[position:])
        if not count:
            if allow_eof and not position:
                return None
            raise ValueError('Truncated RDM frame')
        position += count
    return result


def read_frames(source):
    previous_sequence = 0
    previous_end = -math.inf
    while True:
        prefix = read_exact(source, 16, allow_eof=True)
        if prefix is None:
            return
        magic, length, size = struct.unpack('<4sIQ', prefix)
        if magic != b'RDM1' or not 0 < length <= 1024*1024 or size > 2**31:
            raise ValueError('Invalid RDM framing; stdout must contain only binary maps')
        meta = json.loads(read_exact(source, length))
        nr, nf = meta['ranges'], meta['frequencies']
        if meta['format'] != 'cf32le' or meta['order'] != 'range,doppler' or size != nr*nf*8:
            raise ValueError('Unsupported RDM layout')
        if meta['sequence'] != previous_sequence+1 or meta['time_end_s'] < previous_end:
            raise ValueError('Missing, reordered or non-monotonic RDM frame')
        previous_sequence, previous_end = meta['sequence'], meta['time_end_s']
        values = np.frombuffer(read_exact(source, size), dtype='<c8').reshape(nr, nf)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite radar map')
        yield meta, values


def power_preview(values, width, height):
    """Pool complex-map power before resizing; narrow Doppler peaks survive."""
    rd = max(1, math.ceil(values.shape[0]/width))
    fd = max(1, math.ceil(values.shape[1]/height))
    pooled = np.empty((math.ceil(values.shape[0]/rd),math.ceil(values.shape[1]/fd)),dtype=np.float32)
    for first in range(0,pooled.shape[0],32):
        last=min(first+32,pooled.shape[0])
        z=values[first*rd:min(last*rd,values.shape[0])]
        power=z.real*z.real
        power+=z.imag*z.imag
        if fd>1:power=np.maximum.reduceat(power,np.arange(0,power.shape[1],fd),axis=1)
        if rd>1:power=np.maximum.reduceat(power,np.arange(0,power.shape[0],rd),axis=0)
        pooled[first:last]=power
    return (10*np.log10(np.maximum(pooled,1e-30))).T[::-1]


def clock_label(seconds):
    minutes = int(seconds//60)
    return f'{minutes:02d}:{seconds-60*minutes:06.3f}'


class Renderer:
    def __init__(self, args):
        self.args = args
        self.width, self.height = 1920, 1080
        self.box = (118, 196, 1742, 936)
        paths = [ROOT/'build/vendor/imgui-1.91.9b/misc/fonts/Roboto-Medium.ttf',
                 Path('/System/Library/Fonts/Supplemental/Arial.ttf'),
                 Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
        font = next((p for p in paths if p.exists()), None)
        self.fonts = {size: ImageFont.truetype(str(font), size) if font else ImageFont.load_default(size=size)
                      for size in (18, 21, 24, 28, 38)}
        stops = np.array([[.018,.032,.07],[.08,.22,.40],[.08,.65,.66],[.94,.65,.20],[1,.98,.82]])
        x = np.linspace(0, 4, 4096)
        self.colours = np.stack([np.interp(x, np.arange(5), stops[:,c]) for c in range(3)],axis=-1)
        self.colours = np.round(self.colours*255).astype(np.uint8)
        self.db_min, self.db_max = args.db_min, args.db_max
        self.base = self.background()
        self.map_layer = None
        self.meta = None

    def text(self, draw, xy, text, size=24, colour='#c2d0df', anchor=None):
        draw.text(xy, text, fill=colour, font=self.fonts[size], anchor=anchor)

    def background(self):
        image = Image.new('RGB', (self.width,self.height), '#0c131e')
        draw = ImageDraw.Draw(image)
        self.text(draw,(42,28),'DVB-T  /  PASSIVE RADAR',28,'#55d6c1')
        title='MATCHED-FILTER DETECTION' if self.args.backend=='matched' else 'RESIDUAL CROSS-SPECTRUM'
        self.text(draw,(485,31),title,24,'#899bad')
        draw.line((42,78,1878,78),fill='#304050',width=1)
        x0,y0,x1,y1 = self.box
        draw.rectangle(self.box, fill='#040813', outline='#415568')
        for km in range(0, int(self.args.range_km)+1, max(1, math.ceil(self.args.range_km/6))):
            x = x0+(x1-x0)*km/self.args.range_km
            self.text(draw,(x,y1+18),str(km),24,anchor='mt')
        self.text(draw,((x0+x1)/2,y1+61),'Bistatic excess path length [km]',28,anchor='mt')
        for hz in np.linspace(-self.args.doppler_hz, self.args.doppler_hz, 5):
            y = y1-(y1-y0)*(hz+self.args.doppler_hz)/(2*self.args.doppler_hz)
            self.text(draw,(x0-18,y),f'{hz:+g}',24,anchor='rm')
        self.text(draw,(x0-18,y0-27),'Hz',21,anchor='rm')
        bar = Image.fromarray(self.colours[np.linspace(4095,0,y1-y0).astype(int)][:,None,:]).resize((24,y1-y0))
        image.paste(bar,(1775,y0))
        for db in np.linspace(self.db_min,self.db_max,7):
            y = y1-(db-self.db_min)/(self.db_max-self.db_min)*(y1-y0)
            self.text(draw,(1812,y),f'{db:.0f}',21,anchor='lm')
        self.text(draw,(1770,y0-35),'Power [dB]',21)
        scale = 'Auto power scale' if self.args.auto_scale else 'Fixed power scale'
        dc = 'Stationary / DC removed' if self.args.remove_dc else 'DC retained'
        self.text(draw,(42,1030),f'{scale}  ·  {dc}  ·  Positive Doppler upward  ·  No grid overlay',18,'#899bad')
        return image

    def set_map(self, meta, values):
        x0,y0,x1,y1 = self.box
        db = power_preview(values, x1-x0, y1-y0)
        if self.args.auto_scale:
            self.db_max = float(db.max())
            self.db_min = self.db_max-(self.args.db_max-self.args.db_min)
            self.base = self.background()
        index = np.clip((db-self.db_min)/(self.db_max-self.db_min)*4095,0,4095).astype(np.int32)
        self.map_layer = Image.fromarray(self.colours[index]).resize((x1-x0,y1-y0),Image.Resampling.NEAREST)
        self.meta = dict(meta,display_db_min=self.db_min,display_db_max=self.db_max,
                         map_peak_db=float(db.max()))

    def frame(self, recording_time, duration, valid=True):
        image = self.base.copy()
        x0,y0,x1,y1 = self.box
        if self.map_layer is not None and valid:
            image.paste(self.map_layer,(x0,y0))
        draw = ImageDraw.Draw(image)
        self.text(draw,(42,104),f'{clock_label(recording_time)}  /  {clock_label(self.args.start+duration)}',38,'#eef5f8')
        if self.meta is not None and valid:
            m=self.meta
            state='SLIDING WINDOW' if m['integration_s'] >= self.args.integration-.02 else 'WINDOW FILL / COHERENCE LIMIT'
            self.text(draw,(685,106),f"{m['integration_s']:.3f} s coherent   ·   {state}",28,'#55d6c1')
            detail='Linear range migration corrected' if m.get('motion_model')=='constant_bistatic_path_rate' else f"coherence {m['coherence']:.5f}"
            self.text(draw,(685,148),f"Window {m['time_start_s']:.3f}–{m['time_end_s']:.3f} s  |  bins {m['frequency_step_hz']:.4f} Hz  |  {detail}",21,'#aabaca')
        else:
            self.text(draw,((x0+x1)/2,(y0+y1)/2),'Acquiring / no complete validated map',28,'#aabaca',anchor='mm')
            self.text(draw,(685,112),f'{self.args.integration:g} s sliding coherent window',28,'#55d6c1')
        progress=max(0,min(1,(recording_time-self.args.start)/duration))
        draw.rectangle((42,1063,1878,1067),fill='#253545')
        draw.rectangle((42,1063,42+1836*progress,1067),fill='#55d6c1')
        return image


def duration_from_metadata(path, start, seconds):
    m=json.loads(path.read_text())
    available=m['input_samples']/m['input_sample_rate']-start
    return min(seconds,available) if seconds>0 else available


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('input',type=Path,help='Float32 stereo I/Q WAV recording')
    p.add_argument('--output',type=Path,required=True,help='New video path; existing outputs are not overwritten')
    p.add_argument('--encoding',choices=tuple(ENCODINGS),default='h264',
                   help='h264: compact MP4; prores: hardware ProRes 422 MOV for QuickTime scrubbing on Apple Silicon')
    p.add_argument('--integration',type=float,default=3,help='Trailing coherent window in seconds')
    p.add_argument('--range-km',type=float,default=30,help='Excess path extent, starting at zero')
    p.add_argument('--doppler-hz',type=float,default=200,help='Symmetric Doppler limit')
    p.add_argument('--fps',type=float,help='Frames per second; default is a whole number of OFDM symbols per frame, at least 30 fps')
    p.add_argument('--remove-dc',action=argparse.BooleanOptionalAction,default=True,help='Suppress stationary returns using the selected detector')
    p.add_argument('--auto-scale',action='store_true',help='Follow each map peak; db-max minus db-min sets the displayed dynamic range')
    p.add_argument('--db-min',type=float,default=-100)
    p.add_argument('--db-max',type=float,default=-40)
    p.add_argument('--start',type=float,default=0)
    p.add_argument('--seconds',type=float,default=0,help='Zero processes the entire remaining recording')
    p.add_argument('--offset-hz',type=float,default=4625000)
    p.add_argument('--guard',type=int,choices=(4,8,16,32),default=16)
    p.add_argument('--response-order',type=int,choices=range(33),default=8)
    p.add_argument('--compute',choices=('vulkan','cpu'),default='vulkan')
    p.add_argument('--backend',choices=('matched','legacy'),default='matched',help='Production matched detector or previous Hann processor for comparisons')
    p.add_argument('--recording-rf-hz',type=float,default=553000000)
    p.add_argument('--channel-rf-hz',type=float,default=557625000,help='Physical transmitter RF, independent of NCO and receiver CFO')
    p.add_argument('--processor',type=Path,help='Override the selected backend executable')
    p.add_argument('--ffmpeg',default='ffmpeg')
    args=p.parse_args(argv)
    if args.fps is None:
        interval=(8192+8192/args.guard)/8e6
        args.fps=1/(max(1,math.floor(1/(30*interval)))*interval)
    for key in ('integration','range_km','doppler_hz','fps','db_min','db_max','start','seconds','offset_hz','recording_rf_hz','channel_rf_hz'):
        if not math.isfinite(getattr(args,key)):p.error(f'--{key.replace("_","-")} must be finite')
    if not .5<=args.integration<=3600 or not 0<args.range_km<=300 or not 0<args.doppler_hz<=400 or not .1<=args.fps<=60:
        p.error('Invalid integration, range, Doppler or frame rate')
    if args.start<0 or args.seconds<0 or args.db_min>=args.db_max:p.error('Invalid time interval or colour scale')
    if args.backend=='matched' and (args.compute!='vulkan' or args.integration>10 or args.range_km>30 or args.range_km<1 or args.doppler_hz>200 or args.doppler_hz<4/args.integration):
        p.error('Matched detection requires Vulkan, CPI <=10 s, range 1..30 km and Doppler 4/CPI..200 Hz')
    if args.processor is None:args.processor=ROOT/'build/release'/('dvbt_detect' if args.backend=='matched' else 'dvbt_reconstruct')
    extension,_=ENCODINGS[args.encoding]
    if args.output.suffix.lower()!=extension:p.error(f'--encoding {args.encoding} requires a {extension} output')
    return args


def processor_command(args,work):
    common=[str(args.processor.resolve()),str(args.input.resolve()),'--output',str(work/'pipeline'),
            '--integration-seconds',str(args.integration),'--range-metres',str(args.range_km*1000),
            '--doppler-hz',str(args.doppler_hz),'--map-stream','-',
            '--response-order',str(args.response_order),'--start',str(args.start),'--seconds',str(args.seconds)]
    if args.backend=='matched':
        common+=['--recording-rf-hz',str(args.recording_rf_hz),
                 '--channel',f'{args.channel_rf_hz},{args.offset_hz},{args.guard}',
                 '--update-seconds',str(1/args.fps),'--stream-only']
        if not args.remove_dc:common.append('--keep-dc')
    else:
        interval=(8192+8192/args.guard)/8e6
        decimation=max(1,min(4,int(.44/(args.doppler_hz*interval))))
        common+=['--history-seconds',str(args.integration),'--map-interval',str(1/args.fps),
                 '--slow-decimation',str(decimation),'--offset-hz',str(args.offset_hz),
                 '--guard',str(args.guard),'--compute',args.compute]
        if args.remove_dc:common.append('--remove-dc')
    return common


def main(argv=None):
    args=arguments(argv)
    if not args.input.is_file():raise ValueError(f'Recording not found: {args.input}')
    if not args.processor.is_file():raise ValueError(f'Build the Release {args.processor.name} executable first')
    ffmpeg=shutil.which(args.ffmpeg)
    if not ffmpeg:raise ValueError('FFmpeg is required for video export')
    args.output=args.output.resolve()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    work=args.output.with_suffix('.processing')
    for path in (args.output,args.output.with_suffix('.json'),args.output.with_suffix('.png'),work):
        if path.exists():raise ValueError(f'Output already exists: {path}')
    work.mkdir()
    env=dict(os.environ,TMP=str(work),XDG_CACHE_HOME=str(ROOT/'build/cache'),XDG_CONFIG_HOME=str(ROOT/'build/config'))
    command=processor_command(args,work)
    renderer=Renderer(args)
    # Read the recording duration from the same checked WAV parser as the DSP;
    # its opening diagnostic includes both rate and actual recovered sample count.
    # A small local header scan provides only the display duration before EOF.
    duration=recording_duration(args.input)-args.start
    if args.seconds:duration=min(duration,args.seconds)
    if duration<=0:raise ValueError('Start is beyond the recording')
    video_frames=math.ceil(duration*args.fps-1e-9)
    extension,codec_options=ENCODINGS[args.encoding]
    temporary=work/f'video.partial{extension}'
    encoder_command=[ffmpeg,'-nostdin','-hide_banner','-loglevel','warning','-f','rawvideo','-pixel_format','rgb24',
                     '-video_size',f'{renderer.width}x{renderer.height}','-framerate',str(args.fps),'-i','pipe:0',
                     '-an',*codec_options,'-movflags','+faststart',
                     '-metadata','title=DVB-T residual range–Doppler, sliding coherent integration',str(temporary)]
    encoder=None;processor=None;frame_index=0;map_records=[];poster_saved=False
    started=time.monotonic()
    try:
        with (work/'pipeline.log').open('w') as log,(work/'encoder.log').open('w') as encoder_log:
            processor=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log,env=env,cwd=ROOT)
            encoder=subprocess.Popen(encoder_command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=encoder_log)
            def emit_until(end):
                nonlocal frame_index,poster_saved
                while frame_index<video_frames and args.start+frame_index/args.fps < end-1e-10:
                    timestamp=args.start+frame_index/args.fps
                    valid=renderer.meta is not None and timestamp-renderer.meta['time_end_s']<=max(.5,2/args.fps)
                    image=renderer.frame(timestamp,duration,valid)
                    encoder.stdin.write(image.tobytes())
                    if not poster_saved and timestamp>=args.start+min(60,duration/2) and valid:
                        image.save(args.output.with_suffix('.png'));poster_saved=True
                    frame_index+=1
            for meta,values in read_frames(processor.stdout):
                emit_until(meta['time_end_s'])
                renderer.set_map(meta,values)
                # The sidecar describes display frames; scientific calibration
                # arrays remain in RDM output and are not repeated thousands of times.
                map_records.append({k:v for k,v in renderer.meta.items() if k not in ('noise_range_factor','noise_doppler_power')})
                if len(map_records)%25==0:
                    print(f"Recording {meta['time_end_s']:.1f}/{args.start+duration:.1f} s | CPI {meta['integration_s']:.3f} s | elapsed {time.monotonic()-started:.1f} s",flush=True)
            if processor.wait()!=0:raise RuntimeError(f'Pipeline failed; see {work/"pipeline.log"}')
            if not map_records:raise RuntimeError('The pipeline produced no valid maps')
            checked_duration=duration_from_metadata(work/'pipeline/metadata.json',args.start,args.seconds)
            if abs(duration-checked_duration)>1e-7:raise RuntimeError('Renderer and DSP recording durations disagree')
            emit_until(args.start+duration+1/args.fps)
            encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError(f'Video encoding failed; see {work/"encoder.log"}')
        temporary.replace(args.output)
        report={'input':str(args.input.resolve()),'output':str(args.output),'recording_start_s':args.start,
                'recording_duration_s':duration,'video_frames':frame_index,'fps':args.fps,
                'video_duration_s':frame_index/args.fps,'requested_integration_s':args.integration,
                'range_km':[0,args.range_km],'doppler_hz':[-args.doppler_hz,args.doppler_hz],
                'power_limits_db':[args.db_min,args.db_max],'elapsed_s':time.monotonic()-started,
                'paced':False,'remove_dc':args.remove_dc,'auto_scale':args.auto_scale,
                'encoding':args.encoding,'backend':args.backend,'encoder_command':encoder_command,'processor_command':command,'maps':map_records}
        args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'Video: {args.output}\n{frame_index} frames; {duration:.3f} recording seconds; {report["elapsed_s"]:.2f} seconds to render',flush=True)
    finally:
        for process in (processor,encoder):
            if process is not None and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()


def recording_duration(path):
    """RIFF/RF64 float I/Q duration, including the recorder's exact 32-bit wrap."""
    length=path.stat().st_size
    with path.open('rb') as f:
        magic,riff_size,wave=struct.unpack('<4sI4s',f.read(12))
        if magic not in (b'RIFF',b'RF64') or wave!=b'WAVE':raise ValueError('Expected RIFF/RF64 WAV')
        sample_rate=None;rf64_bytes=None
        while f.tell()+8<=length:
            tag,size=struct.unpack('<4sI',f.read(8));position=f.tell()
            if tag==b'ds64' and size>=28:
                _,rf64_bytes,_=struct.unpack('<QQQ',f.read(24))
            elif tag==b'fmt ' and size>=16:
                fmt,channels,sample_rate,byte_rate,align,bits=struct.unpack('<HHIIHH',f.read(16))
                if (fmt,channels,align,bits)!=(3,2,8,32) or sample_rate==0 or byte_rate!=sample_rate*8:
                    raise ValueError('Expected float32 stereo I/Q WAV')
            elif tag==b'data':
                if sample_rate is None:raise ValueError('Missing WAV format')
                data_bytes=size
                if magic==b'RF64':
                    if rf64_bytes is None:raise ValueError('Missing RF64 length')
                    data_bytes=rf64_bytes
                elif length-position>0xffffffff and (length-position)&0xffffffff==size and (length-8)&0xffffffff==riff_size:
                    data_bytes=length-position
                if data_bytes>length-position or data_bytes%8:raise ValueError('Invalid WAV data length')
                return data_bytes/(8*sample_rate)
            if position+size+(size&1)>length:raise ValueError('Invalid WAV chunk')
            f.seek(position+size+(size&1))
    raise ValueError('Missing WAV sample data')


if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError,struct.error,KeyboardInterrupt) as error:
        print(f'Error: {error}',file=sys.stderr);sys.exit(1)
