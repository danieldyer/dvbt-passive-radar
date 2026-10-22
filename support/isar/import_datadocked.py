"""Convert an explicit Data Docked history response to the canonical AIS CSV.

The provider does not echo the vessel identity in this response. The requested
MMSI is supplied explicitly and recorded as query provenance, not independently
verified from the response. No true heading is inferred from course over ground.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from ais import utc_seconds


def convert(source, destination, *, requested_mmsi):
    payload = json.loads(source.read_text())
    response = payload.get('response', {})
    if response.get('status') != 'success' or not isinstance(response.get('data'), list):
        raise ValueError('Expected a successful Data Docked history response')
    rows = sorted(response['data'], key=lambda row: utc_seconds(row['time']))
    if len(rows) < 2:
        raise ValueError('Too few AIS positions')
    times = [utc_seconds(row['time']) for row in rows]
    if any(b <= a for a, b in zip(times[:-1], times[1:])):
        raise ValueError('Duplicate/conflicting AIS timestamps need explicit resolution')
    with destination.open('x', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['utc', 'latitude_deg', 'longitude_deg', 'sog_knots', 'cog_deg',
                         'true_heading_deg', 'rot_deg_per_min', 'mmsi'])
        for row in rows:
            writer.writerow([row['time'], row['lat'], row['lng'], row.get('speed', ''),
                             row.get('course', ''), '', '', requested_mmsi])
    return {'provider': 'Data Docked', 'source': str(source),
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'requested_mmsi': requested_mmsi, 'identity_echoed_by_response': False,
            'positions': len(rows), 'first_utc': rows[0]['time'], 'last_utc': rows[-1]['time'],
            'canonical_csv': str(destination),
            'true_heading_available': False, 'rate_of_turn_available': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--requested-mmsi', type=int, required=True)
    args = parser.parse_args()
    result = convert(args.source, args.output, requested_mmsi=args.requested_mmsi)
    args.output.with_suffix('.provenance.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
