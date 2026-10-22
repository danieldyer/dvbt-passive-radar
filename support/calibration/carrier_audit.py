"""Compare configured RF centres with archived total OFDM frequency corrections.

The receiver adds the acquired integer-carrier offset to its tracked CFO. These
samples therefore test a wrong nominal channel offset, not merely the fractional
subcarrier remainder. CFO includes receiver LO error and is not an independent
measurement of absolute transmitter frequency.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

from phase_history import C, exact, index_archive


def measure(capture, index, records_per_channel=50):
    results = []
    with capture.open('rb') as source:
        for channel, description in enumerate(index['header']['applied_channels']):
            entries = [row for row in index['records'] if row[1] == channel]
            selected = entries[::max(1, len(entries)//records_per_channel)]
            values = []
            for position, _, _, _ in selected:
                source.seek(position)
                tag, length, _ = struct.unpack('<4sIQ', exact(source, 16))
                if tag != b'PROF':
                    raise ValueError('The archive index does not point to a profile record')
                metadata = json.loads(exact(source, length))
                values.extend(row[4] for row in metadata['rows'] if row[5])
            values = np.asarray(values)
            if not len(values) or not np.isfinite(values).all():
                raise ValueError('No finite valid-row frequency metadata')
            maximum = float(abs(values).max())
            results.append({'channel': channel, 'rf_hz': description['rf_hz'],
                'sampled_profile_records': len(selected), 'valid_rows': len(values),
                'sampled_interval_s': [selected[0][2], selected[-1][3]],
                'cfo_min_p1_median_p99_max_hz': np.percentile(values, [0,1,50,99,100]).tolist(),
                'maximum_absolute_sampled_cfo_hz': maximum,
                'phase_difference_degrees_at_30km_if_full_cfo_is_carrier_error': 360*maximum*30000/C})
    return {'capture': str(capture), 'channels': results,
        'sampling': 'Approximately fifty evenly spaced profile records per channel; all valid rows in each record.',
        'phase_formula': '360 * abs(frequency difference Hz) * excess path metres / c',
        'scope': 'Diagnostic comparison of configured RF with measured total CFO, including integer acquisition. The phase figure treats all sampled CFO as RF modelling error for scale only; it is not an absolute transmitter-frequency uncertainty bound, since receiver LO error also contributes. No correction applied.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.capture, index_archive(args.capture, args.index))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Maximum sampled |CFO|:', max(c['maximum_absolute_sampled_cfo_hz'] for c in result['channels']), 'Hz')


if __name__ == '__main__':
    main()
