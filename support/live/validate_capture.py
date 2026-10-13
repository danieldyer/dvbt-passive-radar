#!/usr/bin/env python3
"""Validate a finite five-channel capture and exact injected transport losses."""
import argparse
import json
import os
import pathlib
import subprocess
from read_archive import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=pathlib.Path, default=pathlib.Path('build/release/dvbt_capture'))
    parser.add_argument('--metadata', type=pathlib.Path, default=pathlib.Path('recordings/live_mt_dandenong.json'))
    parser.add_argument('--fixture', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--gap-samples', type=int, default=0)
    parser.add_argument('--gap-at', type=float, default=2.5)
    parser.add_argument('--unpaced', action='store_true')
    parser.add_argument('--archive', choices=['profiles', 'channels', 'both', 'none'], default='profiles')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    command = [str(args.binary), '--metadata', str(args.metadata), '--simulate', str(args.fixture),
               '--simulate-once', '--output', str(args.output/'capture'), '--no-tui', '--archive', args.archive]
    if args.unpaced:
        command.append('--unpaced')
    if args.gap_samples:
        command += ['--inject-gap-at', str(args.gap_at), '--inject-gap-samples', str(args.gap_samples)]
    environment = os.environ.copy()
    sdk = pathlib.Path('build/sdk/macOS/share/vulkan/icd.d/MoltenVK_icd.json').resolve()
    if sdk.exists() and not environment.get('VK_ICD_FILENAMES'):
        environment['VK_ICD_FILENAMES'] = str(sdk)
    with open(args.output/'terminal.log', 'w') as log:
        subprocess.run(command, stdout=log, stderr=log, env=environment, check=True, timeout=120)
    summary = json.loads((args.output/'capture/summary.json').read_text())
    assert summary['missing_samples'] == args.gap_samples, summary
    assert summary['host_discarded_samples'] == summary['unquantified_gaps'] == summary['replay_loops'] == 0, summary
    assert len(summary['channels']) == 5
    for channel in summary['channels']:
        assert channel['rows'] > 500 and channel['rs_good'] > 1000 and channel['rs_bad'] == 0, channel
        assert channel['locked'] and channel['resets'] == bool(args.gap_samples), channel
    if args.archive != 'none':
        archive = summarize(args.output/'capture/capture.dvblive')
        assert archive['mode'] == args.archive, archive
        assert bool(archive['records'].get('CHIQ')) == (args.archive in ('channels', 'both')), archive
        if args.archive in ('profiles', 'both'):
            assert set(archive['profiles']) == {str(c) for c in range(len(summary['channels']))}, archive
            assert all(c['valid'] > 0 for c in archive['profiles'].values()), archive
        else:
            assert not archive['profiles'], archive
        assert len(archive['gaps']) == (2 if args.gap_samples else 1), archive
        (args.output/'archive.json').write_text(json.dumps(archive, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
