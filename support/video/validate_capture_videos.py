#!/usr/bin/env python3
"""Check exported live-profile maps, MOV framing and AVFoundation exact seeks."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import struct
import subprocess

import numpy as np


def admitted_ends(rows, interval, required, update):
    count=0;last=-math.inf
    for j,row in enumerate(rows):
        if j and (row[6]!=rows[j-1,6] or abs(row[1]-rows[j-1,1]-interval)>interval*.01):count=0
        if not row[5]:count=0;continue
        count+=1
        if count>=required and row[1]-last+interval*1e-8>=update:
            yield row[1];last=row[1]


def mov_atoms(path):
    result={};total=path.stat().st_size
    with path.open('rb') as source:
        while source.tell()<total:
            offset=source.tell();size,kind=struct.unpack('>I4s',source.read(8))
            header=8
            if size==1:size,=struct.unpack('>Q',source.read(8));header=16
            if size==0:size=total-offset
            if size<header or offset+size>total:raise ValueError('Invalid MOV atom length')
            result.setdefault(kind.decode(),offset);source.seek(offset+size)
    if result.get('moov',total)>=result.get('mdat',0):raise ValueError('MOV index is not at the front')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('videos',type=Path)
    p.add_argument('--arrays',type=Path,required=True);p.add_argument('--quicktime-check',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    arrays=np.load(args.arrays);manifest=json.loads((args.videos/'processing/manifest.json').read_text())
    results=[]
    for c,signal in enumerate(manifest['signals']):
        path=args.videos/(signal['id']+'.mov');report=json.loads(path.with_suffix('.json').read_text())
        maps=report['maps'];rows=arrays[signal['id']+'_rows']
        dt=round(float(np.median(np.diff(rows[:,1])))*8e6)/8e6
        required=math.ceil(manifest['integration_s']/dt)
        expected=np.array(list(admitted_ends(rows,dt,required,1/manifest['fps'])))
        expected=expected[(expected>=manifest['recording_start_s']) & (expected<=manifest['recording_start_s']+manifest['recording_duration_s'])]
        np.testing.assert_allclose([m['time_end_s'] for m in maps],expected,rtol=0,atol=1e-10)
        for m in maps:
            assert m['samples']==required and m['physical_rf_hz']==signal['rf_hz']
            assert abs(m['integration_s']-required*dt)<1e-12
            assert abs(m['time_end_s']-m['time_start_s']-(required-1)*dt)<1e-12
            assert abs(m['phase_epoch_s']-(m['time_start_s']+m['time_end_s'])/2-4095.5/8e6)<1e-12
            assert m['window']=='inverse_noise' and m['source_kind']=='live_profiles'
            assert 0<m['effective_integration_s']<=m['integration_s']+1e-6
            assert m['map_peak_db']<signal['power_limits_db'][1]
        assert report['upper_clipped_maps']==0 and report['power_limits_db']==signal['power_limits_db']
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=codec_name,profile,width,height,avg_frame_rate,nb_frames,color_space,color_transfer,color_primaries:format=duration,size',
            '-of','json',str(path)]))
        stream=probe['streams'][0]
        assert stream['codec_name']=='prores' and (stream['width'],stream['height'])==(1920,1080)
        assert int(stream['nb_frames'])==report['video_frames']==math.ceil(manifest['recording_duration_s']*manifest['fps']-1e-9)
        assert abs(float(Fraction(stream['avg_frame_rate']))-manifest['fps'])<1e-9
        assert abs(float(probe['format']['duration'])-report['video_duration_s'])<1e-6
        for key in ('color_space','color_transfer','color_primaries'):assert stream[key]=='bt709'
        flags=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_packets',
            '-show_entries','packet=flags','-of','csv=p=0',str(path)]).decode().splitlines()
        assert len(flags)==report['video_frames'] and all('K' in flag for flag in flags)
        seek=json.loads(subprocess.check_output([str(args.quicktime_check.resolve()),str(path.resolve())]))
        assert len(seek['seeks'])==34
        results.append(dict(id=signal['id'],output=str(path.resolve()),frames=report['video_frames'],maps=len(maps),
            first_map_s=maps[0]['time_end_s'],last_map_s=maps[-1]['time_end_s'],
            effective_integration_range_s=[min(m['effective_integration_s'] for m in maps),max(m['effective_integration_s'] for m in maps)],
            all_frames_independently_decodable=True,atoms=mov_atoms(path),probe=probe,quicktime=seek))
        print('Validated:',signal['id'],flush=True)
    args.output.write_text(json.dumps(dict(videos=results),indent=2)+'\n')


if __name__=='__main__':main()
