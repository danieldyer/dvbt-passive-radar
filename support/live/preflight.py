#!/usr/bin/env python3
"""Soak the production capture pipeline at explicit settings and keep a small report.

The temporary archive exercises real writes and is deleted unless --keep-data is
set. All selected range/Doppler maps are computed; native-window painting is not
included. Hardware is received only, with the metadata's fixed gain settings.
"""
import argparse
import json
import gzip
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from analyse_diagnostics import summarize as diagnostics_summary
from read_archive import summarize as archive_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, default=Path('recordings/live_mt_dandenong.json'))
    p.add_argument('--binary', type=Path, default=Path('build/release/capture_preflight'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--range-metres', type=float, default=100000)
    p.add_argument('--doppler', type=float, default=100)
    p.add_argument('--integration', type=float, default=3)
    p.add_argument('--update', type=float, default=.25)
    p.add_argument('--archive', choices=['profiles', 'channels', 'both', 'none'], default='profiles')
    p.add_argument('--map-channels', type=int, default=1, help='Simultaneous map load; default matches the one-channel native display')
    p.add_argument('--simulate', type=Path)
    p.add_argument('--simulate-once', action='store_true')
    p.add_argument('--simulate-cyclic', action='store_true')
    p.add_argument('--unpaced', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--gain-db', type=float, help='Override fixed capture gain in a temporary metadata copy')
    p.add_argument('--inject-gap-samples', type=int, default=0)
    p.add_argument('--inject-gap-at', type=float, default=2.5)
    p.add_argument('--keep-data', action='store_true')
    a = p.parse_args()
    if a.gain_db is not None and not math.isfinite(a.gain_db):
        p.error('--gain-db must be finite')
    if a.seconds <= 0:
        p.error('--seconds must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(a.metadata.read_text())
    maps = a.map_channels
    temporary = Path(tempfile.mkdtemp(prefix='payload-', dir=a.output))
    capture = temporary/'capture'
    metadata_path = a.metadata.resolve()
    if a.gain_db is not None:
        metadata['capture']['gain_db'] = a.gain_db
        metadata_path = temporary/'requested-metadata.json'
        metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    command = [str(a.binary.resolve()), '--metadata', str(metadata_path), '--output', str(capture),
               '--seconds', str(a.seconds), '--archive', a.archive, '--range-metres', str(a.range_metres),
               '--doppler', str(a.doppler), '--integration', str(a.integration), '--update', str(a.update),
               '--map-channels', str(maps), '--no-tui']
    if a.simulate:
        command += ['--simulate', str(a.simulate.resolve())]
    for flag in ['simulate_once', 'simulate_cyclic', 'unpaced', 'profile']:
        if getattr(a, flag):
            command.append('--'+flag.replace('_', '-'))
    if a.inject_gap_samples:
        command += ['--inject-gap-samples', str(a.inject_gap_samples), '--inject-gap-at', str(a.inject_gap_at)]
    environment = os.environ.copy()
    sdk = Path('build/sdk/macOS/share/vulkan/icd.d/MoltenVK_icd.json').resolve()
    if sdk.exists():
        environment.setdefault('VK_ICD_FILENAMES', str(sdk))
    try:
        with (a.output/'terminal.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=log, env=environment)
        report = dict(command=command, exit_code=result.returncode, map_channels=maps,
                      scope='Production DSP/archive plus map computation; native-window painting excluded.')
        for name in ['summary.json', 'metadata.json', 'diagnostics.jsonl']:
            if (capture/name).exists():
                shutil.copy2(capture/name, a.output/name)
        if (capture/'summary.json').exists():
            report['capture'] = json.loads((capture/'summary.json').read_text())
        if (capture/'capture.dvblive').exists():
            report['archive'] = archive_summary(capture/'capture.dvblive')
        if (capture/'diagnostics.jsonl').exists():
            report['diagnostics'] = diagnostics_summary(capture/'diagnostics.jsonl')
        s = report.get('capture', {})
        report['transport_pass'] = (result.returncode == 0 and s.get('missing_samples') == a.inject_gap_samples
            and s.get('host_discarded_samples') == s.get('unquantified_gaps') == 0)
        report['realtime_margin'] = s.get('processing_speed', 0) > 1
        report['fec_pass'] = bool(s.get('channels')) and all(c['rs_good'] > 0 and c['rs_bad'] == 0 for c in s['channels'])
        report['maps_pass'] = maps == 0 or (len(s.get('channels', [])) >= maps and all(c['maps'] > 0 for c in s['channels'][:maps]))
        report['complete'] = report.get('diagnostics', {}).get('complete', False)
        (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        # Keep diagnostic provenance without retaining an unnecessarily large
        # uncompressed sidecar from every benchmark. Production logs stay JSONL.
        diagnostic_path = a.output/'diagnostics.jsonl'
        if diagnostic_path.exists():
            with diagnostic_path.open('rb') as source, gzip.open(str(diagnostic_path)+'.gz', 'wb', compresslevel=1) as target:
                shutil.copyfileobj(source, target)
            diagnostic_path.unlink()
        print(json.dumps({k: report[k] for k in ['transport_pass', 'realtime_margin', 'fec_pass', 'maps_pass', 'complete']}, indent=2))
        print('Report:', a.output/'report.json')
        if result.returncode:
            raise SystemExit(result.returncode)
        if not all(report[key] for key in ['transport_pass', 'complete', 'maps_pass', 'fec_pass', 'realtime_margin']):
            raise SystemExit(2)
    finally:
        if not a.keep_data:
            shutil.rmtree(temporary)


if __name__ == '__main__':
    main()
