"""Bounded-memory full-capture caches from unchanged native C0/C1 profiles.

Four-symbol means support rolling video backprojection. Sixteen-symbol means
and their exact native epochs support the continuous motion likelihood.
No carrier correction, retiming, sample selection or clutter suppression is
applied here. The original archive remains the numerical reference.
"""
import argparse,hashlib,json,struct,time
from pathlib import Path
import numpy as np
from recorded_injection_fields import exact
from field import average,Field,interpolate,FS
from motion import C,excess_path
from block_pulse import BlockPulse
from ais import read_csv
from flash_siting_report import EPOCH


def records(source,*,payload):
    while True:
        tag,length,size=struct.unpack('<4sIQ',exact(source,16));meta=json.loads(exact(source,length))
        if tag==b'DONE':
            if size or source.read(1):raise ValueError('Invalid native archive completion')
            return
        if tag!=b'PROF':raise ValueError('Unexpected native archive record')
        yield meta,exact(source,size) if payload else size
        if not payload:source.seek(size,1)


def header(source):
    if exact(source,8)!=b'INJPROF1':raise ValueError('Expected native profile archive')
    length,=struct.unpack('<I',exact(source,4));raw=exact(source,length)
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def prepare(path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);path=Path(path);started=time.monotonic()
    identity=[str(path.resolve()),path.stat().st_size,path.stat().st_mtime_ns]
    with path.open('rb') as source:
        h,digest=header(source);counts={}
        for m,size in records(source,payload=False):
            n=len(m['rows']);c=m['channel']
            if size!=n*(2*h['range_count']+129)*8:raise ValueError('Native record dimensions changed')
            counts[c]=counts.get(c,0)+n
    ranges=h['range_count'];channels=len(h['applied_channels'])
    if set(counts)!=set(range(channels)):raise ValueError('Missing native channel')
    retained={c:n//16*16 for c,n in counts.items()};cache={};pending={};native_cursor={c:0 for c in counts}
    for block in (4,16):
        for c in counts:
            directory=out/f'block-{block}'/f'channel-{c}';directory.mkdir(parents=True)
            n=retained[c]//block
            value=dict(directory=directory,cursor=0,moments=np.lib.format.open_memmap(directory/'moments.npy',mode='w+',dtype='<c8',shape=(2,n,ranges)))
            for key in ['time_s','information','first_time_s','last_time_s']:
                value[key]=np.empty(n,float)
            if block==16:
                value['pulse']=np.lib.format.open_memmap(directory/'pulses.npy',mode='w+',dtype='<c8',shape=(n,129))
                value['native_time_s']=np.empty(retained[c],float);value['native_information']=np.empty(retained[c],float)
            cache[block,c]=value;pending[block,c]=None
    with path.open('rb') as source:
        actual,check=header(source)
        if check!=digest:raise ValueError('Source header changed')
        for index,(m,payload) in enumerate(records(source,payload=True)):
            c=m['channel'];n=len(m['rows']);row=np.asarray(m['rows'],float)
            raw=np.frombuffer(payload,'<c8');values=raw[:2*n*ranges].reshape(n,2,ranges).transpose(1,0,2);pulse=raw[2*n*ranges:].reshape(n,129)
            use=min(n,max(0,retained[c]-native_cursor[c]));section=slice(0,use)
            t=row[section,1];weight=np.full(use,m['information'])*row[section,4]
            if np.any(weight<0) or not np.isfinite(weight).all():raise ValueError('Invalid native information')
            native=cache[16,c];start=native_cursor[c]
            native['native_time_s'][start:start+use]=t;native['native_information'][start:start+use]=weight
            native_cursor[c]+=n
            if not use:continue
            for block in (4,16):
                key=block,c;data=cache[key];parts=(values[:,section],pulse[section],t,weight)
                if pending[key] is not None:
                    old=pending[key];parts=(np.concatenate([old[0],parts[0]],axis=1),*[np.concatenate([a,b]) for a,b in zip(old[1:],parts[1:])])
                v,p,at,w=parts;count=len(at)//block*block
                if count:
                    stop=data['cursor']+count//block;dest=slice(data['cursor'],stop)
                    result,total=average(v[:,:count].transpose(1,0,2),w[:count],block)
                    data['moments'][:,dest]=result.transpose(1,0,2)
                    epoch,_=average(at[:count],w[:count],block)
                    epoch[total==0]=at[:count].reshape(-1,block).mean(axis=1)[total==0]
                    data['time_s'][dest]=epoch;data['information'][dest]=total
                    data['first_time_s'][dest]=at[:count:block];data['last_time_s'][dest]=at[block-1:count:block]
                    if block==16:data['pulse'][dest]=average(p[:count],w[:count],block)[0]
                    data['cursor']=stop
                pending[key]=(v[:,count:].copy(),p[count:].copy(),at[count:].copy(),w[count:].copy()) if count<len(at) else None
            if index%1000==0:print('Cached native record',index,'seconds',round(time.monotonic()-started,1),flush=True)
    if identity!=[str(path.resolve()),path.stat().st_size,path.stat().st_mtime_ns]:raise ValueError('Native source changed during preparation')
    reference=out/'carrier-reference.json';reference.write_text(json.dumps(dict(carrier_path_m=0.,phase_operation='Identity; original C0/C1 phase retained'),indent=2)+'\n')
    reference_digest=hashlib.sha256(reference.read_bytes()).hexdigest()
    for block in (4,16):
        descriptions=[]
        for c in counts:
            d=cache[block,c];n=retained[c]//block
            if d['cursor']!=n or pending[block,c] is not None:raise ValueError('Incomplete native observation groups')
            if np.any(np.diff(d['time_s'])<=0):raise ValueError('Reversed native mean epochs')
            d['moments'].flush();coordinates={k:d[k] for k in ['time_s','information','first_time_s','last_time_s']}
            np.savez(d['directory']/'coordinates.npz',**coordinates,carrier_reference_path_m=np.zeros(n))
            if block==16:
                d['pulse'].flush();np.savez(d['directory']/'native-epochs.npz',time_s=d['native_time_s'],information=d['native_information'])
            meta=dict(complete=True,channel=c,rows=n,original_rows=counts[c],retained_native_rows=retained[c],
                block_rows=block,ranges=ranges,first_range_sample=h['first_range_sample'],
                interval_s=[float(d['first_time_s'][0]),float(d['last_time_s'][-1])],
                channel_configuration=h['applied_channels'][c],rf_hz=h['applied_channels'][c]['rf_hz'],
                centre_carrier_removed=False,range_grid_shifted=False,source_identity=identity)
            (d['directory']/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');descriptions.append(meta)
        description=dict(complete=True,scope=__doc__,block_rows=block,channels=descriptions,
            motion=str(reference),motion_sha256=reference_digest,source_header_sha256=digest,source_identity=identity)
        (out/f'block-{block}'/'aperture.json').write_text(json.dumps(description,indent=2)+'\n')
    (out/'native-header.json').write_text(json.dumps(h,indent=2)+'\n')
    (out/'analysis.json').write_text(json.dumps(dict(complete=True,scope=__doc__,source_identity=identity,
        source_header_sha256=digest,counts=counts,retained=retained,seconds=time.monotonic()-started),indent=2)+'\n')
    print('Prepared capture',out,flush=True)
    return h


def prepare_field(cache,out):
    cache=Path(cache);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    h=json.loads((cache/'native-header.json').read_text());plan=h['plan'];rx=np.asarray(plan['receiver_ecef_m']);tx=np.asarray(plan['transmitters_ecef_m']);rotation=np.asarray(plan['ecef_to_enu'])
    track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    example=np.load(cache/'block-16/channel-0/coordinates.npz');first,last=float(example['time_s'][0]),float(example['time_s'][-1])
    t=np.linspace(first,last,1001);position=track.sample(EPOCH+t,max_gap_s=400.)['position_ecef_m']
    path=excess_path(position,tx[3],rx)[0];target=(path.min()+path.max())/2
    observer=position[np.argmin(abs(path-target))].copy();offsets=np.arange(-16,17);records=[]
    for c in range(len(tx)):
        source=cache/f'block-16/channel-{c}';directory=out/f'channel-{c}';directory.mkdir()
        meta=json.loads((source/'metadata.json').read_text());coordinate=dict(np.load(source/'coordinates.npz'));native=dict(np.load(source/'native-epochs.npz'))
        times=coordinate['time_s'];information=coordinate['information'];n=len(times);moments=np.load(source/'moments.npy',mmap_mode='r')
        pulse=np.load(source/'pulses.npy',mmap_mode='r');mean=np.average(pulse,axis=0,weights=information)
        reference=float(excess_path(observer[None],tx[c],rx)[0][0]);rf=meta['rf_hz'];values=np.empty((n,len(offsets)),complex)
        for start in range(0,n,512):
            stop=min(start+512,n);q=np.full(stop-start,reference*FS/C-meta['first_range_sample'])
            values[start:stop]=interpolate(moments[:,start:stop],q,offsets)[0]*np.exp(2j*np.pi*rf/C*reference)
        metadata=dict(channel=c,rf_hz=rf,block_rows=1,centre_prior='One fixed physical observer for the complete capture',
            operator='Original information-weighted 16-symbol means, fixed linear range interpolation and constant carrier reference; no fitted data transform')
        data=dict(time=times,row_time=times,information=information,row_information=information,value=values,offset_samples=offsets,
            centre_ecef_m=np.broadcast_to(observer,(n,3)),centre_velocity_ecef_mps=np.zeros((n,3)),
            body_to_ecef=np.broadcast_to(rotation.T,(n,3,3)),centre_path_m=np.full(n,reference),centre_omega=np.zeros(n),
            transmitter_ecef_m=tx[c],receiver_ecef_m=rx,metadata_json=json.dumps(metadata))
        f=Field(data,mean);kernel=BlockPulse(pulse)
        np.savez_compressed(directory/'field.npz',**data);np.save(directory/'mean-pulse.npy',mean)
        np.save(directory/'kernel-samples.npy',kernel.samples);np.save(directory/'kernel-coefficients.npy',kernel.coefficients);np.save(directory/'kernel-derivatives.npy',kernel.derivative_coefficients)
        np.savez_compressed(directory/'epochs.npz',native_time_s=native['time_s'],native_information=native['information'],row_blocks=np.arange(n),training=np.ones(n,bool))
        record=dict(channel=c,rows=n,native_rows=len(native['time_s']),range_cells=len(offsets),kernel_radius=kernel.radius,kernel_degree=kernel.degree,
            covariance_min_eigenvalue=float(f.range_eigenvalues.min()),covariance_condition=float(f.range_eigenvalues.max()/f.range_eigenvalues.min()))
        records.append(record);print('Prepared global field',record,flush=True)
    (out/'analysis.json').write_text(json.dumps(dict(complete=True,scope=__doc__,profiles_cache=str(cache),
        observer_ecef_m=observer.tolist(),interval_s=[first,last],channels=records,fit_samples='all'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profiles',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--field-output',type=Path)
    args=p.parse_args()
    if args.profiles:prepare(args.profiles,args.output)
    elif not (args.output/'analysis.json').exists():raise ValueError('Declare a source or a completed native cache')
    if args.field_output:prepare_field(args.output,args.field_output)


if __name__=='__main__':main()
