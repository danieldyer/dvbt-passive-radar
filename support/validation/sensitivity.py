#!/usr/bin/env python3
"""Offline sensitivity audit of DVBTSAR1 spectra; never alters pipeline outputs."""
import argparse
import json
from pathlib import Path
import struct
import numpy as np

NFFT, BINS, FS, DT = 8192, 6817, 8e6, 8704/8e6
C, RF = 299792458., 557625000.
K = np.arange(BINS)-3408
ROW = np.dtype([('sample','<u8'),('phase','<f8'),('frequency','<f8'),('modulo','<u4'),('discontinuity','<u4')])


def db(power):
    return float(10*np.log10(max(float(power),1e-300)))


def blocks(path):
    """Read checked records; include context for checks, emit only new rows."""
    size=path.stat().st_size
    with path.open('rb') as f:
        if f.read(8)!=b'DVBTSAR1':raise ValueError('Not a DVBTSAR1 file')
        rate,decimation,channels,flags,origin,rf,budget=struct.unpack('<4IQdd',f.read(40))
        if channels!=1 or rate!=decimation*FS:raise ValueError('Expected one 8-MS/s channel')
        channel=struct.unpack('<ddII',f.read(24));count=0
        while f.tell()<size:
            tag,length=struct.unpack('<4sQ',f.read(12));start=f.tell()
            if start+length>size:raise ValueError('Truncated SAR record')
            if tag==b'SPEC':
                index,segment,rows,first,last,frames,reference_origin,error=struct.unpack('<6Idd',f.read(40))
                if not 0<=first<last<=rows<=1020 or length!=40+32*rows+92*frames+8*BINS+20*rows*BINS:
                    raise ValueError('Invalid SPEC dimensions')
                coordinates=np.frombuffer(f.read(rows*32),ROW).copy()
                f.seek(frames*92,1);pos=f.tell()
                h=np.memmap(path,dtype='<c8',mode='r',offset=pos,shape=(BINS,))
                pos+=BINS*8
                y=np.memmap(path,dtype='<c8',mode='r',offset=pos,shape=(rows,BINS))
                pos+=rows*BINS*8
                s=np.memmap(path,dtype='<c8',mode='r',offset=pos,shape=(rows,BINS))
                pos+=rows*BINS*8
                valid=np.memmap(path,dtype='<u4',mode='r',offset=pos,shape=(rows,BINS))
                yield dict(index=index,segment=segment,first=first,last=last,h=h,y=y,s=s,valid=valid,
                           coordinates=coordinates,origin=origin,rate=rate,decimation=decimation,
                           reference_origin=reference_origin,error=error)
                count+=1
            elif tag==b'DONE':
                if length!=8 or struct.unpack('<Q',f.read(8))[0]!=count or start+length!=size:
                    raise ValueError('Invalid completion record')
                return
            f.seek(start+length)
    raise ValueError('Missing DONE')


def profiles(spectrum,ranges=801):
    z=np.zeros((len(spectrum),NFFT),dtype=np.complex64)
    z[:,K%NFFT]=spectrum
    return np.fft.ifft(z,axis=1)[:,:ranges].astype(np.complex64)


def summarize(path,output):
    before=[];after=[];flat=[];stationary=[];times=[];records=[];noise=[];reference_power=[];discarded=0;replay_error=[]
    for b in blocks(path):
        x=np.asarray(b['s'],dtype=np.complex128)*np.exp(-2j*np.pi*K*b['reference_origin']/NFFT)
        y=np.asarray(b['y'],dtype=np.complex128);h=np.asarray(b['h'],dtype=np.complex128)
        mask=np.asarray(b['valid'],dtype=bool);predicted=x*h;error=np.where(mask,y-predicted,0)
        ratio=np.sum(np.abs(error)**2)/np.sum(np.abs(y[mask])**2)
        if abs(ratio-b['error'])>3e-6:raise ValueError('CPU replay disagrees with GPU residual statistic')
        replay_error.append(abs(ratio-b['error']))
        sl=slice(b['first'],b['last']);complete=mask[sl].all(axis=1)
        discarded+=int((~complete).sum())
        y=y[sl][complete];x=x[sl][complete];predicted=predicted[sl][complete];error=error[sl][complete]
        t=(b['origin']+b['coordinates']['sample'][sl][complete]*b['decimation'])/b['rate']
        power=np.abs(x)**2;raw=y*np.conj(x);cross=error*np.conj(x)
        # Calibrate the flattened ambiguity function to the same coherent
        # voltage response as the cross-spectrum for a flat unit channel.
        z=error/x*power.mean()
        # A noise-like proxy: first differences suppress stationary transfer.
        # Moving returns, ICI, tracking error and other residuals also contribute.
        noise.append(np.median(np.abs(np.diff(y/x,axis=0))**2,axis=0)/(2*np.log(2)))
        reference_power.append(power.mean(axis=0))
        coherent=np.sum(raw,axis=0)/np.sum(power,axis=0)
        static=(coherent-h)*power
        den=np.sum(np.abs(predicted)**2)
        a=np.vdot(predicted,y)/den;ae=np.vdot(predicted,error)/den
        noise_proxy=noise[-1];p=power.mean(axis=0)
        cross_efficiency=flat_efficiency=None
        if np.all(noise_proxy>0):
            optimal=np.sum(abs(h)**2/noise_proxy)
            cross_efficiency=float(abs(np.sum(h*p))**2/(np.sum(noise_proxy*p*p)*optimal))
            flat_efficiency=float(abs(np.sum(h))**2/(noise_proxy.sum()*optimal))
        records.append(dict(time_start_s=float(t[0]),time_end_s=float(t[-1]),
            total_suppression_db=db(np.sum(np.abs(y)**2)/np.sum(np.abs(error)**2)),
            zero_delay_coherent_suppression_db=db(abs(raw.sum())**2/abs(cross.sum())**2),
            predicted_template_suppression_db=db(abs(a)**2/abs(ae)**2),
            diagonal_proxy_cross_efficiency=cross_efficiency,
            diagonal_proxy_flat_efficiency=flat_efficiency,
            white_noise_oracle_efficiency=float(abs(np.sum(h*p))**2/(p.sum()*np.sum(abs(h)**2*p))),
            pilot_flatten_white_noise_efficiency=float(power.size**2/(power.sum()*np.sum(1/power))),
            projected_before_real=float(a.real),projected_before_imag=float(a.imag),
            projected_after_real=float(ae.real),projected_after_imag=float(ae.imag)))
        before.append(profiles(raw));after.append(profiles(cross));flat.append(profiles(z));stationary.append(profiles(static));times.append(t)
    cache=dict(times=np.concatenate(times),before=np.concatenate(before),after=np.concatenate(after),
               flat=np.concatenate(flat),stationary=np.concatenate(stationary),
               noise_ratio=np.median(noise,axis=0),reference_power=np.mean(reference_power,axis=0))
    if not np.allclose(np.diff(cache['times']),DT,atol=1e-7,rtol=0):raise ValueError('Nonuniform emitted time grid')
    np.savez(output,**cache)
    summary=dict(source=str(path),cache=str(output),rows=len(cache['times']),batches=records,
                 discarded_boundary_rows=discarded,max_gpu_cpu_energy_fraction_error=max(replay_error),
                 noise_ratio_median=float(np.median(cache['noise_ratio'])))
    output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def window(name,n):
    t=np.arange(n)/n
    if name=='rect':return np.ones(n)
    if name=='hann':return .5-.5*np.cos(2*np.pi*t)
    if name=='hamming':return .54-.46*np.cos(2*np.pi*t)
    if name=='blackman_harris':return .35875-.48829*np.cos(2*np.pi*t)+.14128*np.cos(4*np.pi*t)-.01168*np.cos(6*np.pi*t)
    raise ValueError(name)


def map_at(rows,seconds,name='hann',remove_dc=False):
    n=int(seconds/DT);a=np.asarray(rows[-n:],dtype=np.complex128);w=window(name,n)
    if remove_dc:a=a-(w@a/w.sum())[None,:]
    transform=np.fft.fft(a*w[:,None],axis=0)/w.sum()
    return np.fft.fftshift(transform,axes=0),np.fft.fftshift(np.fft.fftfreq(n,DT))


def maps(cache_path,output):
    a=np.load(cache_path);result=[]
    for seconds in (1.,10.):
        raw,f=map_at(a['before'],seconds);residual,_=map_at(a['after'],seconds)
        dc,_=map_at(a['after'],seconds,remove_dc=True)
        flat,_=map_at(a['flat'],seconds,remove_dc=True)
        static,_=map_at(a['stationary'],seconds,remove_dc=True)
        zero=int(np.argmin(abs(f)));ghost=int(np.argmin(abs(f-1/(4*DT))))
        # Expected pilot ghost is 8192/12 samples; allow the two nearest range bins.
        ranges=slice(681,685);all_power=np.abs(dc)**2
        background=(abs(f)>30)&(abs(f)<280)&(abs(f-1/(4*DT))>5)&(abs(f+1/(4*DT))>5)
        cell_noise=float(np.median(all_power[background,100:780])/np.log(2))
        ghost_peak=float(np.max(abs(dc[ghost-1:ghost+2,ranges])**2))
        static_peak=float(np.max(abs(static[ghost-1:ghost+2,ranges])**2))
        record=dict(cpi_s=seconds,actual_cpi_s=int(seconds/DT)*DT,
            zero_delay_zero_doppler_suppression_db=db(abs(raw[zero,0])**2/abs(residual[zero,0])**2),
            raw_direct_peak_power=float(abs(raw[zero,0])**2),
            residual_direct_peak_power=float(abs(residual[zero,0])**2),
            ghost_frequency_hz=float(f[ghost]),ghost_power=ghost_peak,
            ghost_relative_raw_direct_db=db(ghost_peak/abs(raw[zero,0])**2),
            flat_ghost_change_db=db(np.max(abs(flat[ghost-1:ghost+2,ranges])**2)/ghost_peak),
            stationary_ghost_fraction=static_peak/ghost_peak,
            background_power_proxy=cell_noise,ghost_above_background_db=db(ghost_peak/cell_noise))
        result.append(record)
        if seconds==1:
            keep=abs(f)<=300
            np.savez(output.with_name(output.stem+'-maps.npz'),frequency=f[keep],raw=raw[keep],residual=residual[keep],dc=dc[keep],flat=flat[keep],static=static[keep])
    output.write_text(json.dumps(result,indent=2)+'\n');return result


def theory(output):
    from scipy.signal import find_peaks
    from scipy.optimize import brentq, minimize_scalar
    results=dict(windows=[],dc=[],motion=[],doppler_ici=[],cpi=[])
    n=8192;time=(np.arange(n)+.5)/n-.5
    for name in ('rect','hann','hamming','blackman_harris'):
        w=window(name,n);response=abs(np.fft.fft(w,n*64))/w.sum()
        peaks=find_peaks(response[:n*32])[0]
        results['windows'].append(dict(name=name,enbw=n*np.sum(w*w)/w.sum()**2,
            white_noise_loss_db=db(n*np.sum(w*w)/w.sum()**2),peak_sidelobe_db=20*float(np.log10(response[peaks].max()))))
    w=window('hann',n)
    for cycles in (.05,.1,.25,.5,1.,2.,3.,5.):
        s=np.exp(2j*np.pi*cycles*np.arange(n)/n);c=np.sum(w*np.conj(s))/w.sum()
        g=w*(np.conj(s)-c)
        snr=abs(np.sum(g*s))**2/np.sum(abs(g)**2)
        optimum=n-abs(s.sum())**2/n
        results['dc'].append(dict(frequency_times_cpi=cycles,voltage_gain=float(abs(np.sum(g*s))/w.sum()),
            snr_loss_vs_known_white_db=db(snr/n),loss_vs_optimum_dc_nuisance_db=db(snr/optimum)))
    for frequency in (0.,10.,50.,100.,200.,300.):
        results['doppler_ici'].append(dict(doppler_hz=frequency,diagonal_ofdm_loss_db=-db(np.sinc(frequency*NFFT/FS)**2)))
        speed=C/RF*frequency
        for seconds in (.125,.25,.5,1.,2.,4.,8.,10.):
            walk=BINS*FS/NFFT*speed/C*seconds
            gain=abs(np.sum(w*np.sinc(walk*time))/w.sum())**2
            results['motion'].append(dict(doppler_hz=frequency,cpi_s=seconds,range_walk_metres=speed*seconds,
                fixed_range_gain=float(gain),fixed_range_loss_db=-db(gain),gain_relative_one_second_rect=seconds*gain/1.5))
    for name in ('rect','hann'):
        w=window(name,n)
        def gain(chirp):return abs(np.sum(w*np.exp(1j*np.pi*chirp*time*time))/w.sum())**2
        limit=brentq(lambda x:gain(x)-10**(-.1),.01,10)
        results['cpi'].append(dict(window=name,one_db_acceleration_limit_hz_per_s_times_s_squared=limit))
    w=window('hann',n)
    def migration_gain(walk):return abs(np.sum(w*np.sinc(walk*time))/w.sum())**2
    best=minimize_scalar(lambda u:-u*migration_gain(u),bounds=(.01,20),method='bounded')
    one_db=brentq(lambda u:migration_gain(u)-10**(-.1),.01,10)
    results['constant_velocity_cpi']=dict(optimum_bandwidth_times_delay_walk=float(best.x),
        one_db_bandwidth_times_delay_walk=float(one_db),
        hann_optimum_cpi_s_by_doppler={str(f):float(best.x*RF/(BINS*FS/NFFT*f)) for f in (25,50,100,200,300)})
    output.write_text(json.dumps(results,indent=2)+'\n');return results


def windows(cache_path, output):
    a=np.load(cache_path);rows=a['after'];results=[]
    for seconds in (1.,10.):
        for name in ('rect','hann','hamming','blackman_harris'):
            m,f=map_at(rows,seconds,name,remove_dc=True)
            region=(abs(f)>5)&(abs(f)<=200)
            p=abs(m[region,100:780])**2
            near=(abs(f)>2)&(abs(f)<20)
            results.append(dict(cpi_s=seconds,window=name,
                background_median_power_proxy=float(np.median(p)/np.log(2)),
                background_p99_power=float(np.quantile(p,.99)),
                near_origin_near_dc_p99_power=float(np.quantile(abs(m[near,:40])**2,.99))))
    output.write_text(json.dumps(results,indent=2)+'\n');return results


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('summarize');s.add_argument('input',type=Path);s.add_argument('--output',type=Path,required=True)
    m=sub.add_parser('maps');m.add_argument('input',type=Path);m.add_argument('--output',type=Path,required=True)
    t=sub.add_parser('theory');t.add_argument('--output',type=Path,required=True)
    w=sub.add_parser('windows');w.add_argument('input',type=Path);w.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.command=='summarize':r=summarize(a.input,a.output)
    elif a.command=='maps':r=maps(a.input,a.output)
    elif a.command=='windows':r=windows(a.input,a.output)
    else:r=theory(a.output)
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
