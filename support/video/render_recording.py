#!/usr/bin/env python3
"""Render all metadata-defined DVB-T channels with fixed noise-floor/peak scales.

One production invocation shares acquisition/channelisation/clock correction.
Only pooled display images are spooled; scientific complex maps are untouched.
The two display passes measure the whole-recording maximum before encoding.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time
import zlib

import numpy as np
import render_video as video


def write_preview(stream, meta, db,compression=1):
    m={k:v for k,v in meta.items() if k not in ('noise_range_factor','noise_doppler_power')}
    m['display_shape']=list(db.shape)
    header=json.dumps(m,separators=(',',':')).encode()
    # float16 dB has <=0.016 dB error at the strongest usual returns, and much
    # less at the noise floor. This is a visualization cache, never a DSP input.
    payload=zlib.compress(np.asarray(db,dtype='<f2').tobytes(),compression)
    stream.write(struct.pack('<II',len(header),len(payload)))
    stream.write(header);stream.write(payload)


def read_previews(stream):
    while True:
        prefix=video.read_exact(stream,8,allow_eof=True)
        if prefix is None:return
        length,size=struct.unpack('<II',prefix)
        if not 0<length<=1024*1024 or size>32*1024*1024:raise ValueError('Invalid display cache record')
        meta=json.loads(video.read_exact(stream,length))
        raw=zlib.decompress(video.read_exact(stream,size))
        db=np.frombuffer(raw,dtype='<f2').reshape(meta['display_shape']).astype(np.float32)
        yield meta,db


def fingerprint(input,metadata):
    s=input.stat()
    return {'input':str(input.resolve()),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,
            'metadata_sha256':hashlib.sha256(metadata.read_bytes()).hexdigest()}


def read_pooled(source):
    previous={}
    while True:
        prefix=video.read_exact(source,16,allow_eof=True)
        if prefix is None:return
        magic,length,size=struct.unpack('<4sIQ',prefix)
        if magic!=b'RDP1' or not 0<length<=1024*1024 or size>32*1024*1024:raise ValueError('Invalid pooled preview framing')
        meta=json.loads(video.read_exact(source,length));shape=meta['display_shape']
        if meta['preview_format']!='power_f32le' or meta['preview_order']!='doppler_down,range' or size!=math.prod(shape)*4:
            raise ValueError('Invalid pooled preview layout')
        c=meta['channel'];sequence,end=previous.get(c,(0,-math.inf))
        if meta['sequence']!=sequence+1 or meta['time_end_s']<end:raise ValueError('Missing pooled preview')
        previous[c]=meta['sequence'],meta['time_end_s']
        power=np.frombuffer(video.read_exact(source,size),dtype='<f4').reshape(shape)
        db=10*np.log10(np.maximum(power,1e-45))
        if not np.isfinite(db).all():raise ValueError('Nonfinite preview')
        yield meta,db


def collect(args,signals,work,command=None,source_duration=None):
    duration=(video.recording_duration(args.input) if source_duration is None else source_duration)-args.start
    if args.seconds:duration=min(duration,args.seconds)
    if duration<=0:raise ValueError('Start is beyond recording')
    if command is None:command=video.processor_command(args,work)
    states=[{'id':s['id'],'name':s['name'],'rf_hz':s['frequency_hz'],'maps':0,'peak_db':-math.inf,
             'noise_medians':[],'spool':str(work/(s['id']+'.display'))} for s in signals]
    streams=[];processor=None;reducer=None;started=time.monotonic();progress=-10
    env=dict(os.environ,TMP=str(work),XDG_CACHE_HOME=str(video.ROOT/'build/cache'),XDG_CONFIG_HOME=str(video.ROOT/'build/config'))
    try:
        streams=[Path(s['spool']).open('xb') for s in states]
        with (work/'pipeline.log').open('w') as log,(work/'preview.log').open('w') as preview_log:
            processor=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log,env=env,cwd=video.ROOT)
            source=processor.stdout
            if args.preview_backend=='cpu':
                reducer=subprocess.Popen([str(args.preview_processor)],stdin=processor.stdout,stdout=subprocess.PIPE,stderr=preview_log)
                processor.stdout.close();source=reducer.stdout
            for meta,db in read_pooled(source):
                c=meta['channel'];s=states[c]
                if meta['physical_rf_hz']!=s['rf_hz']:raise ValueError('Map RF disagrees with metadata channel order')
                peak=float(db.max())
                s['peak_db']=max(s['peak_db'],peak);s['maps']+=1
                # Exclude stationary-suppression notch and the first 0.6 km.
                rows=np.linspace(args.doppler_hz,-args.doppler_hz,db.shape[0])
                background=db[np.abs(rows)>3/args.integration,int(math.ceil(.6/args.range_km*db.shape[1])):]
                s['noise_medians'].append(float(np.median(background[::4,::4])))
                write_preview(streams[c],dict(meta,map_peak_db=peak),db,args.cache_compression)
                if meta['time_end_s']>=progress+10:
                    progress=meta['time_end_s']
                    print(f"Collect {progress:.1f}/{args.start+duration:.1f} s | maps {[s['maps'] for s in states]} | elapsed {time.monotonic()-started:.1f} s",flush=True)
            if reducer is not None and reducer.wait()!=0:raise RuntimeError(f'Preview reducer failed: {work/"preview.log"}')
            if processor.wait()!=0:raise RuntimeError(f'Pipeline failed: {work/"pipeline.log"}')
        if any(s['maps']==0 for s in states):raise RuntimeError('A metadata channel produced no validated maps')
        checked=video.duration_from_metadata(work/'pipeline/metadata.json',args.start,args.seconds)
        if abs(checked-duration)>1e-7:raise RuntimeError('DSP and display WAV durations disagree')
        for s in states:
            noise=s.pop('noise_medians')
            median=float(np.median(noise))
            s['noise_median_db']=median;s['noise_median_p10_p90_db']=[float(np.percentile(noise,p)) for p in (10,90)]
            s['power_limits_db']=[median-1.5,s['peak_db']+.25]
        manifest={'source':fingerprint(args.input,args.metadata),'processor_command':command,
                  'recording_start_s':args.start,'recording_duration_s':duration,'fps':args.fps,
                  'integration_s':args.integration,'range_km':args.range_km,'doppler_hz':args.doppler_hz,
                  'remove_dc':args.remove_dc,'signals':states,'collect_elapsed_s':time.monotonic()-started,
                  'colour_scale':'Fixed per channel: whole-recording median displayed background minus 1.5 dB, full maximum plus 0.25 dB'}
        (work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        return manifest
    finally:
        for stream in streams:stream.close()
        for process in (reducer,processor):
            if process is not None and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()


def encode(args,manifest,state):
    suffix,codec=video.ENCODINGS[args.encoding]
    output=args.output/(state['id']+suffix)
    if output.exists():
        report=json.loads(output.with_suffix('.json').read_text())
        if report['source']!=manifest['source']:raise ValueError('Completed video belongs to another recording')
        print(f'Already complete: {output}',flush=True);return
    for path in (output.with_suffix('.json'),output.with_suffix('.png')):
        if path.exists():raise ValueError(f'Unfinished output already exists: {path}')
    args.db_min,args.db_max=state['power_limits_db'];args.auto_scale=False
    args.signal_name=f"{state['name']}  ·  {state['rf_hz']/1e6:g} MHz"
    renderer=video.Renderer(args)
    duration=manifest['recording_duration_s'];frames=math.ceil(duration*args.fps-1e-9)
    temporary=args.work/(state['id']+'.partial'+suffix)
    command=[shutil.which(args.ffmpeg),'-nostdin','-hide_banner','-loglevel','warning','-f','rawvideo',
             '-pixel_format','rgb24','-video_size','1920x1080','-framerate',str(args.fps),'-i','pipe:0','-an',
             *codec,'-movflags','+faststart','-metadata',f'title={args.signal_name} passive radar',str(temporary)]
    encoder=None;index=0;records=[];poster=False;started=time.monotonic();progress=-10
    try:
        with (args.work/(state['id']+'.encoder.log')).open('w') as log,Path(state['spool']).open('rb') as spool:
            encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
            def emit_until(end):
                nonlocal index,poster
                while index<frames and args.start+index/args.fps<end-1e-10:
                    t=args.start+index/args.fps
                    valid=renderer.meta is not None and t-renderer.meta['time_end_s']<=max(.5,2/args.fps)
                    image=renderer.frame(t,duration,valid)
                    encoder.stdin.write(image.tobytes())
                    if not poster and t>=args.start+min(60,duration/2) and valid:
                        image.save(output.with_suffix('.png'));poster=True
                    index+=1
            for meta,db in read_previews(spool):
                emit_until(meta['time_end_s']);renderer.set_display_map(meta,db)
                records.append(dict(meta,display_db_min=args.db_min,display_db_max=args.db_max))
                if meta['time_end_s']>=progress+30:
                    progress=meta['time_end_s'];print(f"Encode {state['id']} {progress:.1f}/{args.start+duration:.1f} s | elapsed {time.monotonic()-started:.1f} s",flush=True)
            emit_until(args.start+duration+1/args.fps)
            encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError(f'Encoder failed for {state["id"]}')
        if len(records)!=state['maps'] or index!=frames:raise RuntimeError('Incomplete display cache or video')
        report={k:v for k,v in manifest.items() if k!='signals'}
        report.update(signal=state,output=str(output),video_frames=index,video_duration_s=index/args.fps,
                      power_limits_db=state['power_limits_db'],auto_scale=False,paced=False,encoding=args.encoding,
                      encoder_command=command,encode_elapsed_s=time.monotonic()-started,maps=records,
                      upper_clipped_maps=sum(m['map_peak_db']>args.db_max for m in records))
        output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
        temporary.replace(output)
        # Only this tool's disposable cache is removed after successful export.
        Path(state['spool']).unlink()
        print(f'Video: {output} ({index} frames)',flush=True)
    finally:
        if encoder is not None and encoder.poll() is None:
            encoder.terminate()
            try:encoder.wait(timeout=10)
            except subprocess.TimeoutExpired:encoder.kill();encoder.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='Directory for one video per signal')
    p.add_argument('--encoding',choices=tuple(video.ENCODINGS),default='prores')
    p.add_argument('--start',type=float,default=0);p.add_argument('--seconds',type=float,default=0)
    p.add_argument('--integration',type=float,default=3);p.add_argument('--doppler-hz',type=float,default=100)
    p.add_argument('--range-km',type=float,default=30);p.add_argument('--fps',type=float)
    p.add_argument('--remove-dc',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--ffmpeg',default='ffmpeg');p.add_argument('--processor',type=Path)
    p.add_argument('--preview-backend',choices=('gpu','cpu'),default='gpu',help='GPU keeps full complex maps off the display pipe; CPU is a cross-check')
    p.add_argument('--preview-processor',type=Path,default=video.ROOT/'build/release/rdm_preview')
    p.add_argument('--cache-compression',type=int,choices=range(10),default=0,help='Zero is fastest; one uses less scratch disk')
    p.add_argument('--resume',action='store_true',help='Encode already collected display caches')
    p.add_argument('--collect-only',action='store_true',help='Prepare fixed-scale caches without encoding yet')
    p.add_argument('--encode-workers',type=int,default=3,help='Independent video exports in parallel; use one for serial validation')
    args=p.parse_args();args.input=args.input.resolve();args.metadata=args.metadata.resolve();args.output=args.output.resolve()
    description=json.loads(args.metadata.read_text());signals=description['signals']
    if args.encode_workers<1 or args.encode_workers>len(signals):p.error('Encode workers must be between one and the number of signals')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+',s['id']) for s in signals):p.error('Signal ids must be safe filenames')
    if not shutil.which(args.ffmpeg):p.error('FFmpeg is required')
    if args.processor is None:args.processor=video.ROOT/'build/release'/('dvbt_preview' if args.preview_backend=='gpu' else 'dvbt_detect')
    if not args.resume and not args.processor.is_file():p.error(f'Build the optional {args.processor.name} target first')
    if args.preview_backend=='cpu' and not args.preview_processor.is_file():p.error('Build the optional rdm_preview support target first')
    # Reuse checked video/DSP option bounds, including a symbol-aligned frame rate.
    checked=video.arguments([str(args.input),'--output','check.mov' if args.encoding=='prores' else 'check.mp4',
                             '--encoding',args.encoding,'--integration',str(args.integration),'--doppler-hz',str(args.doppler_hz),
                             '--range-km',str(args.range_km),'--start',str(args.start),'--seconds',str(args.seconds),
                             '--guard',str(signals[0]['guard_denominator']),*(['--fps',str(args.fps)] if args.fps else [])])
    args.fps=checked.fps;args.backend='matched';args.response_order=8;args.select_channel=None
    args.output.mkdir(parents=True,exist_ok=True);args.work=args.output/'processing'
    if args.resume:
        manifest=json.loads((args.work/'manifest.json').read_text())
        if fingerprint(args.input,args.metadata)!=manifest['source']:raise ValueError('Source or metadata changed since collection')
        duration=video.recording_duration(args.input)-args.start
        if args.seconds:duration=min(duration,args.seconds)
        if abs(duration-manifest['recording_duration_s'])>1e-7:raise ValueError('Resume duration disagrees with collection')
        for key in ('integration','range_km','doppler_hz','fps','remove_dc','start'):
            field={'integration':'integration_s','start':'recording_start_s'}.get(key,key)
            if getattr(args,key)!=manifest[field]:raise ValueError('Resume settings disagree with collection')
    else:
        args.work.mkdir() # Refuse to overwrite a preceding export/cache.
        manifest=collect(args,signals,args.work)
    if not args.collect_only:
        if args.encode_workers==1:
            for state in manifest['signals']:encode(args,manifest,state)
        else:
            with ProcessPoolExecutor(max_workers=args.encode_workers) as pool:
                jobs=[pool.submit(encode,args,manifest,state) for state in manifest['signals']]
                for job in jobs:job.result()


if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError,struct.error,KeyboardInterrupt) as error:
        import sys
        print(f'Error: {error}',file=sys.stderr);sys.exit(1)
