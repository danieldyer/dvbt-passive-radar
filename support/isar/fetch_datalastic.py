"""Read Datalastic quota or one UTC date of vessel history.

The credential is read with terminal echo disabled, sent only in the documented
x-api-key header, and never written to the output. A history request consumes
existing API credits; this tool neither purchases credits nor changes a plan.
"""
import argparse
from datetime import date, datetime, timezone
import getpass
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def request(endpoint, key, parameters=None):
    url = 'https://api.datalastic.com/api/v0/'+endpoint
    if parameters:
        url += '?'+urlencode(parameters)
    call = Request(url, headers={'x-api-key': key, 'Accept': 'application/json'})
    try:
        with urlopen(call, timeout=25) as response:
            raw = response.read(16*1024*1024+1)
            if len(raw) > 16*1024*1024:
                raise ValueError('Unexpectedly large API response')
    except HTTPError as error:
        detail = error.read(4096).decode('utf-8', errors='replace').replace(key, '[redacted]')
        raise RuntimeError(f'Datalastic HTTP {error.code}: {detail}') from None
    # Some API versions include a credential in metadata; redact before parsing
    # or persisting anything, not just before printing a summary.
    return json.loads(raw.decode().replace(key, '[redacted]'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--status', action='store_true')
    p.add_argument('--date', type=date.fromisoformat)
    p.add_argument('--imo', type=int, default=9357951)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.status == bool(args.date) or (args.date and args.output is None):
        p.error('Choose --status, or --date YYYY-MM-DD --output FILE')
    if args.output and args.output.exists():
        p.error('Output already exists')
    key = getpass.getpass('Datalastic API key: ').strip()
    if not key:
        p.error('No API key supplied')
    if args.status:
        data = request('stat', key)
        print(json.dumps(data, indent=2))
        return
    parameters = {'imo': args.imo, 'from': args.date.isoformat(), 'to': args.date.isoformat()}
    data = request('vessel_history', key, parameters)
    if not data.get('meta', {}).get('success'):
        raise RuntimeError('History API returned no successful result: '+json.dumps(data))
    vessel = data.get('data', {})
    if int(vessel.get('imo', 0)) != args.imo:
        raise ValueError('Vessel identity in response does not match the request')
    args.output.write_text(json.dumps({'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'endpoint': 'https://api.datalastic.com/api/v0/vessel_history',
        'parameters': parameters, 'response': data}, indent=2))
    print(json.dumps({'file': str(args.output), 'imo': vessel.get('imo'),
                      'mmsi': vessel.get('mmsi'), 'positions': len(vessel.get('positions', []))}))


if __name__ == '__main__':
    main()
