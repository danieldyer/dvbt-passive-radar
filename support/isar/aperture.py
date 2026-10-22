"""Copy a bounded ship aperture from archived matched profiles, without retiming.

Both within-symbol moments, every selected row, its validity and its original
metadata are retained. This is an offline working crop; the source archive is
unchanged. No phase fitting, slow-time averaging or display normalisation is
performed here.
"""
import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'calibration'))
from phase_history import C, FS, exact, index_archive


def extract(capture, archive, channel, start_s, stop_s, first_sample, last_sample):
    if not start_s < stop_s or not 0 <= first_sample <= last_sample <= 928:
        raise ValueError('Invalid time interval or positive-delay crop')
    if archive['header']['corrected_rate_hz']/archive['header']['decimation'] != FS:
        raise ValueError('This profile format requires the DVB-T 7 MHz, 8 MS/s time grid')
    chunks, rows, information, record_index, record_metadata = [], [], [], [], []
    with capture.open('rb') as source:
        for position, ch, first, last in archive['records']:
            if ch != channel or last+4095.5/FS < start_s or first >= stop_s:
                continue
            source.seek(position)
            tag, metadata_size, payload_size = struct.unpack('<4sIQ', exact(source, 16))
            meta = json.loads(exact(source, metadata_size))
            count = meta['count']
            if tag != b'PROF' or meta['moments'] != 2 or meta['size'] != 1024:
                raise ValueError('Unsupported matched-profile layout')
            shape = (2, count, 1024)
            profile_bytes = int(np.prod(shape))*8
            if profile_bytes > payload_size or len(meta['rows']) != count:
                raise ValueError('Inconsistent matched-profile payload')
            data = np.frombuffer(exact(source, profile_bytes), '<c8').reshape(shape)
            row = np.asarray(meta['rows'], dtype=np.float64)
            time = row[:, 1]+4095.5/FS
            keep = (time >= start_s)&(time < stop_s)
            if not keep.any():
                continue
            chunks.append(data[:, keep, first_sample:last_sample+1].copy())
            rows.append(row[keep])
            information.append(np.full(keep.sum(), meta['information'], dtype=np.float64))
            record_index.append(np.full(keep.sum(), len(record_metadata), dtype=np.int32))
            record_metadata.append({k: v for k, v in meta.items() if k != 'rows'})
    if not chunks:
        raise ValueError('No profile rows in requested aperture')
    if len({meta['segment'] for meta in record_metadata}) != 1:
        raise ValueError('Split apertures at acquisition segment boundaries before coherent processing')
    row = np.concatenate(rows)
    time = row[:, 1]+4095.5/FS
    if np.any(np.diff(time) <= 0):
        raise ValueError('Non-increasing profile times: split acquisition segments explicitly')
    valid = row[:, 5] != 0
    info = np.concatenate(information)
    if np.any(~np.isfinite(info)) or np.any(info[valid] <= 0):
        raise ValueError('Invalid matched-profile information')
    meta = {'source': str(capture), 'channel': channel,
            'channel_configuration': archive['header']['applied_channels'][channel],
            'sample_rate_hz': FS, 'symbol_duration_s': 8192/FS,
            'first_range_sample': first_sample, 'last_range_sample': last_sample,
            'requested_interval_s': [start_s, stop_s],
            'row_time_reference': 'OFDM useful-symbol sample midpoint',
            'profile_phase': 'Unmodified production profile phase; no additional window-phase correction',
            'source_records': record_metadata}
    return {'moments': np.concatenate(chunks, axis=1), 'time_s': time, 'rows': row,
            'information': info, 'valid': valid, 'record_index': np.concatenate(record_index),
            'metadata_json': json.dumps(meta)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--start-s', type=float, required=True)
    parser.add_argument('--stop-s', type=float, required=True)
    parser.add_argument('--range-min-m', type=float, required=True)
    parser.add_argument('--range-max-m', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    archive = index_archive(args.capture, args.index)
    args.output.mkdir(parents=True, exist_ok=True)
    first = int(np.floor(args.range_min_m*FS/C))
    last = int(np.ceil(args.range_max_m*FS/C))
    for channel, configuration in enumerate(archive['header']['applied_channels']):
        path = args.output/f'channel-{channel}.npz'
        if path.exists():
            raise ValueError(f'Output already exists: {path}')
        data = extract(args.capture, archive, channel, args.start_s, args.stop_s, first, last)
        np.savez(path, **data)
        print(json.dumps({'file': str(path), 'rf_hz': configuration['rf_hz'],
                          'rows': len(data['time_s']), 'valid': int(data['valid'].sum()),
                          'first_time_s': float(data['time_s'][0]),
                          'last_time_s': float(data['time_s'][-1]),
                          'bytes': path.stat().st_size}), flush=True)


if __name__ == '__main__':
    main()
