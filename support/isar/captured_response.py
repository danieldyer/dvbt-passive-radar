"""Read original PROF weights and frozen cancellation uncertainty by row epoch."""
import json
from pathlib import Path
import struct

import numpy as np

from finite_reference import range_weights


def exact_read(stream, count):
    value = stream.read(count)
    if len(value) != count:
        raise ValueError('Truncated original profile record')
    return value


def captured_rows(index_path, channel, times, symbol_seconds):
    index = json.loads(Path(index_path).read_text())
    capture = Path(index['identity'][0])
    stat = capture.stat()
    if [stat.st_size, stat.st_mtime_ns] != index['identity'][1:]:
        raise ValueError('Capture identity changed since indexing')
    times = np.asarray(times)
    if times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError('Require strictly increasing requested row epochs')
    rows, records = [], []
    with capture.open('rb') as source:
        for offset, c, first, last in index['records']:
            if c != channel or last+.000512 < times[0]-2*symbol_seconds or first > times[-1]:
                continue
            source.seek(offset)
            tag, length, size = struct.unpack('<4sIQ', exact_read(source, 16))
            meta = json.loads(exact_read(source, length))
            payload_offset = source.tell()
            if tag != b'PROF' or meta['response_carriers'] != 6817 or meta['noise_carriers'] != 6817:
                raise ValueError('Missing original response/noise weighting')
            if meta.get('cancellation_variance_carriers') != 6817:
                raise ValueError('Missing saved finite cancellation uncertainty')
            profile_bytes = 8*(meta['count']*meta['moments']*meta['size']+meta['stationary_carriers'])
            if size != profile_bytes+6817*(8+4+8):
                raise ValueError('Original profile payload size disagrees with metadata')
            source.seek(profile_bytes, 1)
            h = np.frombuffer(exact_read(source, 6817*8), '<c8').astype(complex)
            noise = np.frombuffer(exact_read(source, 6817*4), '<f4').astype(float)
            beta = np.frombuffer(exact_read(source, 6817*8), '<f8').copy()
            if not np.isfinite(h).all() or not np.isfinite(noise).all() or np.any(noise <= 0):
                raise ValueError('Invalid original response/noise')
            if not np.isfinite(beta).all() or np.any(beta < 0):
                raise ValueError('Invalid cancellation uncertainty')
            records.append({'response': h, 'variance': noise, 'cancellation_variance': beta,
                'information': meta['information'], 'reference_origin': meta['reference_origin'],
                'record_offset': offset, 'segment': meta['segment'], 'capture': str(capture),
                'payload_offset': payload_offset, 'profile_count': meta['count'],
                'profile_size': meta['size'], 'profile_moments': meta['moments']})
            rows.extend((row[1]+4095.5/8e6, row, records[-1], n) for n, row in enumerate(meta['rows']))
    rows.sort(key=lambda row: row[0])
    epochs = np.array([row[0] for row in rows])
    if len(epochs) < 2 or np.any(np.diff(epochs) <= 0):
        raise ValueError('Missing or overlapping original row epochs')
    selected = np.clip(np.searchsorted(epochs, times), 1, len(epochs)-1)
    selected -= abs(epochs[selected-1]-times) < abs(epochs[selected]-times)
    if np.max(abs(epochs[selected]-times)) > 2e-8:
        raise ValueError('Requested epochs are not present in the original capture')
    output = []
    for i in selected:
        time, row, record, row_index = rows[i]
        if i == 0 or abs(time-epochs[i-1]-symbol_seconds) > 2e-7 or rows[i-1][2]['segment'] != record['segment']:
            raise ValueError('Previous source-clock row is missing or across a sample gap')
        if not row[5]:
            raise ValueError('Requested waveform probe was an original decoder erasure')
        output.append({**record, 'time_s': time, 'source_offset': row[6],
                       'previous_source_offset': rows[i-1][1][6], 'profile_row': row_index})
    return output


def observations(rows, coordinates, omega, offsets):
    """Apply the original range interpolation and moment combination to saved C0/C1."""
    if not len(rows) or any(r['capture'] != rows[0]['capture'] for r in rows):
        raise ValueError('Read a nonempty set of rows from one capture')
    if len(coordinates) != len(rows) or len(omega) != len(rows):
        raise ValueError('Observation coordinates must match the requested rows')
    result = []
    with Path(rows[0]['capture']).open('rb') as source:
        for row, coordinate, rate in zip(rows, coordinates, omega):
            if row['profile_moments'] != 2:
                raise ValueError('Require saved C0 and C1 profiles')
            centre, tap, weight = range_weights(coordinate)
            indices = centre+np.asarray(offsets)[:, None]+tap
            size = row['profile_size']
            if indices.min() < 0 or indices.max() >= size:
                raise ValueError('Original profile lacks range interpolation context')
            moments=[]
            for moment in range(2):
                source.seek(row['payload_offset']+8*size*(moment*row['profile_count']+row['profile_row']))
                profile=np.frombuffer(exact_read(source,8*size),'<c8').astype(complex)
                moments.append(profile[indices]@weight)
            result.append(moments[0]-1j*rate*moments[1])
    return np.asarray(result)
