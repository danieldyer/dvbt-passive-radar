"""Fetch a bounded historical AIS interval using existing Data Docked credits.

Reads the key without terminal echo and sends it only in an HTTPS header.
Does not purchase credits or alter account settings. Preserves the provider's
response for inspection before conversion into a motion prior.
"""
import argparse
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ENDPOINT = 'https://datadocked.com/api/vessels_operations/get-vessel-historical-data'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mmsi', type=int, required=True)
    parser.add_argument('--from-utc', required=True)
    parser.add_argument('--to-utc', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    times = [datetime.fromisoformat(x.replace('Z', '+00:00')) for x in (args.from_utc, args.to_utc)]
    if any(x.tzinfo is None for x in times) or not 0 < (times[1]-times[0]).total_seconds() <= 90*86400:
        parser.error('Supply explicit UTC offsets and a positive interval of at most 90 days')
    if not 100000000 <= args.mmsi <= 999999999:
        parser.error('Expected a nine-digit MMSI')
    parameters = {'imo_or_mmsi': str(args.mmsi),
                  'from_date': times[0].astimezone(timezone.utc).isoformat(),
                  'to_date': times[1].astimezone(timezone.utc).isoformat()}
    key = getpass.getpass('Data Docked API key: ').strip()
    if not key:
        parser.error('No API key supplied')
    request = Request(ENDPOINT+'?'+urlencode(parameters),
                      headers={'x-api-key': key, 'Accept': 'application/json'})
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read(16*1024*1024+1)
            if len(raw) > 16*1024*1024:
                raise ValueError('Unexpectedly large history response')
    except HTTPError as error:
        detail = error.read(4096).decode('utf-8', errors='replace').replace(key, '[redacted]')
        raise SystemExit(f'Data Docked HTTP {error.code}: {detail}') from None
    data = json.loads(raw.decode().replace(key, '[redacted]'))
    result = data.get('response', {})
    if result.get('status') != 'success' or not isinstance(result.get('data'), list):
        raise SystemExit('History request did not return the documented success schema: '+json.dumps(data))
    args.output.write_text(json.dumps({'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'endpoint': ENDPOINT, 'parameters': parameters, 'response': data}, indent=2))
    print(json.dumps({'file': str(args.output), 'mmsi_requested': args.mmsi,
                      'positions': len(result['data']),
                      'fields': sorted({key for row in result['data'] for key in row})}))


if __name__ == '__main__':
    main()
