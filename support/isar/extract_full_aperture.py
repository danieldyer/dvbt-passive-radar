"""Stream unchanged archived profile crops into bounded-memory NPY arrays.

This is an offline working format. Original epochs, validity, both moments and
filter information are retained. No fitted phase, averaging or DC notch.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

from aperture import C, FS, exact


def records(source, index, channel):
    for offset, c, first, last in index['records']:
        if c != channel:
            continue
        source.seek(offset)
        tag, size, payload = struct.unpack('<4sIQ', exact(source, 16))
        metadata = json.loads(exact(source, size))
        if tag != b'PROF' or metadata['moments'] != 2 or metadata['size'] != 1024:
            raise ValueError('Unsupported profile archive')
        yield metadata


def extract(capture, index, directory, channel, first, last):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory/'metadata.json'
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        if (meta['source'] != str(capture.resolve()) or meta['first_range_sample'] != first
                or meta['last_range_sample'] != last or not meta['complete']):
            raise ValueError('Existing extraction has different inputs or is incomplete')
        return meta
    with capture.open('rb') as source:
        metadata = list(records(source, index, channel))
    if len({m['segment'] for m in metadata}) != 1:
        raise ValueError('Split acquisition segments before forming a coherent aperture')
    counts = [m['count'] for m in metadata]
    count = sum(counts)
    rows = np.concatenate([np.array(m['rows'], dtype=float) for m in metadata])
    time = rows[:, 1]+4095.5/FS
    if np.any(np.diff(time) <= 0):
        raise ValueError('Non-monotone original profile epochs')
    info = np.repeat([m['information'] for m in metadata], counts)
    if np.any(info <= 0) or not np.isfinite(info).all():
        raise ValueError('Invalid original information')
    moments = np.lib.format.open_memmap(directory/'moments.npy', mode='w+', dtype='<c8',
        shape=(2, count, last-first+1))
    with capture.open('rb') as source:
        start = 0
        for m in records(source, index, channel):
            n = m['count']
            original = np.frombuffer(exact(source, 2*n*1024*8), '<c8').reshape(2, n, 1024)
            moments[:, start:start+n] = original[:, :, first:last+1]
            start += n
    moments.flush()
    np.savez(directory/'coordinates.npz', time_s=time, rows=rows, valid=rows[:, 5] != 0,
        information=info, record_index=np.repeat(np.arange(len(metadata)), counts))
    result = {'source': str(capture.resolve()), 'channel': channel, 'rows': count,
        'channel_configuration': index['header']['applied_channels'][channel],
        'first_range_sample': first, 'last_range_sample': last, 'sample_rate_hz': FS,
        'symbol_duration_s': 8192/FS, 'interval_s': time[[0, -1]].tolist(),
        'source_records': [{k: v for k, v in m.items() if k != 'rows'} for m in metadata],
        'operator': __doc__, 'complete': True}
    manifest.write_text(json.dumps(result, indent=2)+'\n')
    return result


def load(directory):
    meta = json.loads((directory/'metadata.json').read_text())
    if not meta['complete']:
        raise ValueError('Incomplete working aperture')
    with np.load(directory/'coordinates.npz') as source:
        result = dict(source)
    result.update(moments=np.load(directory/'moments.npy', mmap_mode='r'),
        metadata_json=json.dumps(meta))
    return result


def window(capture,index,channel,first_s,last_s,*,first_range_sample=-64,last_range_sample=928):
    """Read a bounded original-row aperture, including stored negative lags.

    PROF slots 960..1023 contain lags -64..-1; slots 929..959 are unsupported.
    Reorder valid lags into one uniform range axis for symmetric interpolation.
    Decoder erasures retain their epochs and zero valid information.
    """
    capture=Path(capture);identity=index['identity'];stat=capture.stat()
    if (capture.resolve()!=Path(identity[0]).resolve() or
            [stat.st_size,stat.st_mtime_ns]!=identity[1:]):
        raise ValueError('Capture identity changed since indexing')
    if not first_s<=last_s or not -64<=first_range_sample<=last_range_sample<=928:
        raise ValueError('Invalid original-row time or supported range interval')
    logical=np.arange(first_range_sample,last_range_sample+1)
    slots=np.where(logical<0,logical+1024,logical)
    selected={'records':[r for r in index['records'] if r[1]==channel and
        r[3]+4095.5/FS>=first_s and r[2]<=last_s]}
    profiles=[];coordinates=[];information=[];segments=[]
    with capture.open('rb') as source:
        for meta in records(source,selected,channel):
            rows=np.asarray(meta['rows'],float);time=rows[:,1]+4095.5/FS
            use=(time>=first_s)&(time<=last_s)
            raw=np.frombuffer(exact(source,2*meta['count']*1024*8),'<c8').reshape(2,meta['count'],1024)
            if not use.any():continue
            profiles.append(raw[:,use,:][:,:,slots])
            coordinates.append(rows[use]);information.append(np.full(use.sum(),meta['information']))
            segments.append(meta['segment'])
    if not profiles or len(set(segments))!=1:raise ValueError('Require nonempty rows in one coherent capture segment')
    rows=np.concatenate(coordinates);time=rows[:,1]+4095.5/FS
    if np.any(np.diff(time)<=0):raise ValueError('Original row epochs are not strictly ordered')
    metadata=dict(source=str(capture.resolve()),channel=channel,first_range_sample=first_range_sample,
        last_range_sample=last_range_sample,complete=True,rows=len(time),
        operator='Original PROF C0/C1, reordered negative/positive lags; no fitted correction or averaging')
    return dict(time_s=time,information=np.concatenate(information),valid=rows[:,5]!=0,rows=rows,
        moments=np.concatenate(profiles,axis=1),metadata_json=json.dumps(metadata))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'index', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--samples', type=int, nargs=2, default=[200, 600])
    args = p.parse_args()
    index = json.loads(args.index.read_text())
    if Path(index['identity'][0]).resolve() != args.capture.resolve():
        raise ValueError('Index belongs to another capture')
    first, last = args.samples
    if not 0 <= first < last <= 928:
        raise ValueError('Invalid positive-delay crop')
    for c in range(5):
        meta = extract(args.capture, index, args.output/f'channel-{c}', c, first, last)
        print(json.dumps({k: meta[k] for k in ('channel', 'rows', 'interval_s', 'complete')}), flush=True)


if __name__ == '__main__':
    main()
