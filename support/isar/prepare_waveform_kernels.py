"""Request known waveform kernels with the saved profile's exact row identities.

Original response/noise weights and information come from PROF records. Only
the transmitted symbols are reconstructed again from CHIQ. Selection is by
the existing field bins, not target strength. A missing reconstructed row is
explicitly marked by the exporter, never replaced by a fitted response.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

from scene_dictionary import fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'fields', 'motion', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Use a new request directory')
    index = json.loads(args.index.read_text()); model = json.loads(args.motion.read_text())
    interval = model['interval_s']; capture = Path(index['identity'][0])
    grids, weights, channels = [], [], []
    with capture.open('rb') as source:
        for channel in range(5):
            path = args.fields/f'channel-{channel}.npz'
            with np.load(path) as data:
                meta = json.loads(str(data['metadata_json'])); block = meta['block_rows']
                bins = np.flatnonzero((data['time'] >= interval[0]+.03)&(data['time'] < interval[1]-.03))
                if block != 16 or not len(bins) or np.any(np.diff(bins) != 1):
                    raise ValueError('Require the existing complete 16-row field bins')
                section = slice(bins[0]*block, min((bins[-1]+1)*block, len(data['row_time'])))
                times = data['row_time'][section]; information = data['row_information'][section]
                mean_pulse = data['pulse_samples']
            records = np.full(len(times), -1, np.int32)
            for offset, c, first, last in index['records']:
                if c != channel or last+.000512 < times[0] or first > times[-1]:
                    continue
                source.seek(offset); tag, length, size = struct.unpack('<4sIQ', source.read(16))
                saved = json.loads(source.read(length))
                if tag != b'PROF' or saved['response_carriers'] != 6817 or saved['noise_carriers'] != 6817:
                    raise ValueError('Missing original profile weighting')
                source.seek(8*(saved['count']*saved['moments']*saved['size']+saved['stationary_carriers']), 1)
                h = np.frombuffer(source.read(6817*8), '<c8').astype(complex)
                noise = np.frombuffer(source.read(6817*4), '<f4').astype(float)
                if not np.isfinite(h).all() or np.any(noise <= 0):
                    raise ValueError('Invalid captured response/noise')
                weights.append(abs(h)**2/noise)
                for row in saved['rows']:
                    time = row[1]+4095.5/8e6
                    i = int(np.searchsorted(times, time))
                    if i == len(times) or abs(times[i]-time) > 2e-8:
                        if i and abs(times[i-1]-time) <= 2e-8: i -= 1
                        else: continue
                    if records[i] != -1:
                        raise ValueError('Overlapping archived row identities')
                    expected = saved['information'] if row[5] else 0.
                    if not np.isclose(information[i], expected, rtol=1e-12, atol=1e-12):
                        raise ValueError('Field weighting differs from saved capture')
                    records[i] = len(weights)-1
            if np.any(records < 0) or len(times)%block:
                raise ValueError('Requested field contains unmapped or partial blocks')
            grid = np.empty(len(times), dtype=[('time', '<f8'), ('information', '<f8'), ('record', '<u4')])
            grid['time'], grid['information'], grid['record'] = times, information, records
            grids.append(grid)
            channels.append({'channel': channel, 'rows': len(times), 'blocks': len(times)//block,
                'first_bin': int(bins[0]), 'first_last_time_s': times[[0, -1]].tolist(),
                'field': str(path), 'field_sha256': fingerprint(path),
                'mean_pulse_real_imag': np.stack((mean_pulse.real, mean_pulse.imag), axis=-1).tolist()})
    header = {'scope': __doc__, 'capture': str(capture), 'capture_size_bytes': capture.stat().st_size,
        'interval_s': interval, 'block_rows': 16, 'pulse_radius': 64,
        'weight_records': len(weights), 'channels': channels,
        'payload': 'Per-channel packed rows (f64 time, f64 information, u32 weight_record), then f64 weights[records][6817].',
        'source_sha256': {str(path): fingerprint(path) for path in (args.index, args.motion, Path(__file__))}}
    args.output.mkdir(parents=True)
    (args.output/'request.json').write_text(json.dumps(header, indent=2)+'\n')
    encoded = json.dumps(header, separators=(',', ':')).encode()
    with (args.output/'request.bin').open('xb') as output:
        output.write(b'WFKREQ01'+struct.pack('<I', len(encoded))+encoded)
        for grid in grids: output.write(grid.tobytes())
        output.write(np.asarray(weights, dtype='<f8').tobytes())
    np.savez(args.output/'rows.npz', **{f'channel_{c}': g for c, g in enumerate(grids)})
    print(json.dumps({'request': str(args.output/'request.bin'), 'channels':
        [{k:v for k,v in c.items() if k in ('channel', 'rows', 'blocks', 'first_last_time_s')} for c in channels],
        'weight_records': len(weights)}))


if __name__ == '__main__':
    main()
