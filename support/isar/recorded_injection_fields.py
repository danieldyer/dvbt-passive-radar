"""Read the paired IQ replay without changing native epochs or complex profiles."""
import json
from pathlib import Path
import struct

import numpy as np

from ais import read_csv
from block_pulse import BlockPulse
from field import Field, observation, average, interpolate, FS, OFFSETS
from flash_siting_report import EPOCH
from piecewise_ais import LocalAisReference
from waveform_kernels import KernelField


def exact(stream, count):
    value = stream.read(count)
    if len(value) != count:
        raise ValueError('Truncated injection export')
    return value


def read_profiles(path, *, interval_s=None):
    """Read native profiles, optionally retaining only a bounded time context.

    Skipped payloads are never loaded. The first retained row is aligned to
    its original 16-symbol group, so validation folds cannot change when a
    long archive is read one aperture at a time. Context records are retained
    around both edges; callers still select their exact scientific aperture.
    """
    if interval_s is not None and (len(interval_s)!=2 or not np.isfinite(interval_s).all() or not interval_s[0]<interval_s[1]):
        raise ValueError('A finite ordered read interval is required')
    channels = {};seen={}
    with Path(path).open('rb') as source:
        if exact(source, 8) != b'INJPROF1':
            raise ValueError('Expected an injection profile export')
        length, = struct.unpack('<I', exact(source, 4))
        header = json.loads(exact(source, length))
        size = header['range_count']
        while True:
            tag, length, payload = struct.unpack('<4sIQ', exact(source, 16))
            meta = json.loads(exact(source, length))
            if tag == b'DONE':
                if payload or source.read(1):
                    raise ValueError('Invalid injection completion record')
                break
            if tag != b'PROF' or payload != len(meta['rows'])*(2*size+129)*8:
                raise ValueError('Invalid injection profile record')
            n = len(meta['rows'])
            if not n:raise ValueError('Empty native profile record')
            rows = np.asarray(meta['rows'], float);channel=meta['channel']
            first_row=seen.get(channel,0);seen[channel]=first_row+n
            if interval_s is not None and (rows[-1,1]<interval_s[0]-.05 or rows[0,1]>=interval_s[1]+.05):
                source.seek(payload,1);continue
            values = np.frombuffer(exact(source, n*2*size*8), '<c8').reshape(n, 2, size).transpose(1, 0, 2)
            pulse = np.frombuffer(exact(source, n*129*8), '<c8').reshape(n, 129)
            ch = channels.setdefault(channel, dict(values=[], pulse=[], rows=[], information=[],first_row_index=first_row))
            ch['values'].append(values)
            ch['pulse'].append(pulse)
            ch['rows'].append(rows)
            ch['information'].append(np.full(n, meta['information']))
    for ch in channels.values():
        ch['values'] = np.concatenate(ch['values'], axis=1)
        for key in ['pulse', 'rows', 'information']:
            ch[key] = np.concatenate(ch[key])
        if interval_s is not None:
            trim=(-ch['first_row_index'])%16
            ch['values']=ch['values'][:,trim:]
            for key in ['pulse','rows','information']:ch[key]=ch[key][trim:]
            ch['first_row_index']+=trim
        if np.any(np.diff(ch['rows'][:, 1]) <= 0):
            raise ValueError('Duplicate or reversed native observation epochs')
    return header, channels


def write_profiles(path,header,channels):
    """Write an explicit synthetic/control export without replacing a file."""
    with Path(path).open('xb') as output:
        text=json.dumps(header).encode();output.write(b'INJPROF1'+struct.pack('<I',len(text))+text)
        for c,ch in sorted(channels.items()):
            n=len(ch['rows'])
            if ch['values'].shape!=(2,n,header['range_count']) or ch['pulse'].shape!=(n,129) or ch['information'].shape!=(n,):
                raise ValueError('Synthetic native payload dimensions disagree')
            boundaries=np.r_[0,np.flatnonzero(np.diff(ch['information'])!=0)+1,n]
            for a,b in zip(boundaries[:-1],boundaries[1:]):
                for first in range(a,b,512):
                    use=slice(first,min(first+512,b))
                    meta=json.dumps(dict(channel=c,rows=ch['rows'][use].tolist(),information=float(ch['information'][first]))).encode()
                    values=np.asarray(ch['values'][:,use].transpose(1,0,2),dtype='<c8').tobytes()
                    pulse=np.asarray(ch['pulse'][use],dtype='<c8').tobytes()
                    output.write(struct.pack('<4sIQ',b'PROF',len(meta),len(values)+len(pulse))+meta+values+pulse)
        output.write(struct.pack('<4sIQ',b'DONE',2,0)+b'{}')


def load_fields(path, stride=1, *, offsets=OFFSETS, fixed_epoch_s=None, interval_s=None):
    header, channels = read_profiles(path,interval_s=interval_s)
    plan = header['plan']
    rx = np.asarray(plan['receiver_ecef_m'])
    tx = np.asarray(plan['transmitters_ecef_m'])
    rotation = np.asarray(plan['ecef_to_enu'])
    track = read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),
                     reference_height_ellipsoid_m=5., mmsi=563052200)
    mid = plan['midpoint_s'] if interval_s is None else float(np.mean(interval_s))
    reference = LocalAisReference(track, EPOCH+mid, rotation, rx)
    state = [reference(0., i) for i in range(3)]  # Declared coarse AIS prior, not injected truth.
    fields = []
    for c in range(len(tx)):
        ch = channels[c]
        n = len(ch['information'])//16*16
        info = ch['information'][:n]*ch['rows'][:n, 4]
        meta = dict(channel=c, channel_configuration=header['applied_channels'][c],
                    first_range_sample=header['first_range_sample'])
        aperture = dict(time_s=ch['rows'][:n, 1], information=ch['information'][:n],
                        valid=ch['rows'][:n, 4] != 0, moments=ch['values'][:, :n], metadata_json=json.dumps(meta))
        data = observation(aperture, track, EPOCH, tx[c], rx, 'course-prior', offsets=offsets)
        kernels, weights = average(ch['pulse'][:n], info, 16)
        pulse = np.average(kernels, axis=0, weights=weights)
        base = Field(data, pulse)
        if fixed_epoch_s is not None:
            from fixed_range_field import fixed_reference
            base, _ = fixed_reference(base, aperture, fixed_epoch_s, offsets)
        from recovery_experiment import thin_field
        # Bind by native row identity before selecting complete bins.
        field = KernelField(base, BlockPulse(kernels), np.repeat(np.arange(len(kernels)), 16))
        if interval_s is not None:field=field.windowed(*interval_s)
        fields.append(thin_field(field, stride))
    return fields, rotation, rx, tx, state, mid


def common_projection(reference, channel, first_range_sample):
    """Apply a fixed reference's linear range/phase/binning map to native C0/C1.

    This does not interpolate in time or move a measurement. Actual native
    epochs remain separate and must be validated by the caller. The reference
    coordinates only define a common linear observation map for paired tests.
    """
    d=reference.data;n=len(d['row_time'])
    if len(channel['rows'])<n:raise ValueError('Missing native symbols for the common projection')
    values=np.empty((n,len(reference.offsets)),complex)
    for first in range(0,n,512):
        section=slice(first,min(first+512,n))
        coordinate=d['centre_path_m'][section]*FS/299792458.-first_range_sample
        zero,one=interpolate(channel['values'][:,section],coordinate,reference.offsets)
        phase=d.get('carrier_reference_path_m',d['centre_path_m'])[section]
        values[section]=(zero-1j*d['centre_omega'][section,None]*one)*np.exp(1j*reference.k*phase[:,None])
    mean,_=average(values,d['row_information'],reference.metadata['block_rows'])
    return reference.apply(mean).ravel()
