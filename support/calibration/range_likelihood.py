"""Retain the complex range aperture for a moving calibration candidate.

The carrier deramp is reversible. A configurable range neighbourhood retains
the delay-envelope information which a single complex history discards.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

from phase_history import C, FS, TAU, index_archive, exact

OFFSETS = np.arange(-4,5)


def interpolate_patch(profiles, coordinates, offsets=OFFSETS):
    centre = np.floor(coordinates).astype(int)
    indices = centre[:,None]+np.arange(-31,33)
    delta = coordinates[:,None]-indices
    weights = np.sinc(delta)*np.i0(8*np.sqrt(np.maximum(0,1-(delta/32)**2)))/np.i0(8)
    weights /= weights.sum(axis=1)[:,None]
    samples = indices[:,None,:]+offsets[None,:,None]
    if samples.min() < -64 or samples.max() > 928:
        raise ValueError('Patch outside recorded range support')
    return np.einsum('nrt,nt->nr',profiles[np.arange(len(profiles))[:,None,None],samples%1024],weights)


def rebin_patch(time, values, information, start, end, step=.02):
    bins = np.floor((time-start)/step).astype(int)
    n = int(np.ceil((end-start)/step))
    keep = (bins>=0)&(bins<n)
    t,z,w,b = time[keep],values[keep],information[keep],bins[keep]
    weight = np.bincount(b,weights=w,minlength=n)
    count = np.bincount(b,minlength=n)
    norm = np.maximum(weight,1e-30)
    mean = np.array([(np.bincount(b,weights=w*column.real,minlength=n)+
                      1j*np.bincount(b,weights=w*column.imag,minlength=n))/norm for column in z.T]).T
    average_time = np.bincount(b,weights=w*t,minlength=n)/norm
    residual = z-mean[b]
    variance = np.array([np.bincount(b,weights=w*w*abs(column)**2,minlength=n)/norm**2*count/np.maximum(count-1,1)
                         for column in residual.T]).T
    # Diagnostic only: correlation in high-rate innovations, rather than an
    # assumption that adjacent matched-filter range samples are independent.
    innovations = residual*np.sqrt(w[:,None])
    covariance = innovations.T@innovations.conj()/len(z)
    good = (count>1)&np.all(variance>0,axis=1)
    return {'time':average_time[good],'value':mean[good],'variance':variance[good],
            'information':weight[good],'rows':count[good],
            'innovation_range_covariance':covariance,
            'row_time':t,'row_value':z,'row_information':w}


def extract(capture, archive, description, channel, offsets=OFFSETS):
    seed = description['channels'][channel]
    start,end = description['start'],description['end']
    times,values,information = [],[],[]
    with capture.open('rb') as f:
        for position,ch,first,last in archive['records']:
            if ch!=channel or last<start-.002 or first>end:
                continue
            f.seek(position);tag,length,size = struct.unpack('<4sIQ',exact(f,16))
            meta = json.loads(exact(f,length));n=meta['count']
            if tag!=b'PROF' or meta['moments']!=2 or meta['size']!=1024:
                raise ValueError('Unsupported matched profile shape')
            data = np.frombuffer(exact(f,n*2048*8),'<c8').reshape(2,n,1024)
            rows = np.asarray(meta['rows'],dtype=float)
            t = rows[:,1]+4095.5/FS
            keep = (t>=start)&(t<end)&(rows[:,5]!=0);t=t[keep]
            if not len(t):
                continue
            scale=seed.get('scale',15)
            u = (t-seed['epoch'])/scale
            path = np.polynomial.polynomial.polyval(u,seed['path'])
            speed = np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyder(seed['path']))/scale
            doppler = -seed['rf_hz']*speed/C
            zero,one = [interpolate_patch(plane[keep],path*FS/C,offsets) for plane in data]
            z = (zero-1j*TAU*doppler[:,None]*.001024*one)*np.exp(1j*TAU*seed['rf_hz']/C*(path-seed['path'][0]))[:,None]
            times.append(t);values.append(z);information.append(np.full(len(t),meta['information']))
    return rebin_patch(np.concatenate(times),np.concatenate(values),np.concatenate(information),start,end)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    parser.add_argument('--histories',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--labels',nargs='+',default=['A','B','C','E'])
    parser.add_argument('--range-radius',type=int,default=4,
                        help='Retained range samples either side of the common source seed (1–12)')
    args = parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if not 1<=args.range_radius<=12:
        parser.error('Range radius must be between 1 and 12 samples')
    offsets=np.arange(-args.range_radius,args.range_radius+1)
    archive = index_archive(args.capture,args.histories/'archive-index.json')
    for label in args.labels:
        description = json.loads((args.histories/(label+'.json')).read_text())
        for channel in range(5):
            path=args.output/f'{label}-{channel}.npz'
            data=extract(args.capture,archive,description,channel,offsets)
            np.savez_compressed(path,**data,offset_samples=offsets)
            print(label,channel,'rows',len(data['row_time']),'bins',len(data['time']),flush=True)


if __name__=='__main__':
    main()
