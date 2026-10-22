"""Validate native row identities and replay consistency of full-capture kernels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from report_measured_scatterer_phases import serial
from waveform_kernels import read_kernels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',type=Path,default=Path('build/isar-20261017/waveform-kernels-full'))
    parser.add_argument('--reference',type=Path,default=Path('build/isar-20261016/waveform-kernels-30s'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    header,records,pulses=read_kernels(args.full/'response.kernels')
    old_header,old_records,old_pulses=read_kernels(args.reference/'response.kernels')
    if header['capture']!=old_header['capture'] or header['capture_size_bytes']!=old_header['capture_size_bytes']:
        raise ValueError('Kernel capture identities differ')
    result=dict(scope=__doc__,complete=False,channels=[],old_overlap_unchanged=True)
    with np.load(args.full/'rows.npz') as grids,np.load(args.reference/'rows.npz') as old_grids:
        for c,(description,blocks,samples) in enumerate(zip(header['channels'],records,pulses)):
            grid=grids[f'channel_{c}'];old=old_grids[f'channel_{c}'];n=header['block_rows']
            if len(grid)!=n*len(blocks):raise ValueError('Partial or missing original block')
            with np.load(description['field']) as field:
                start=description['first_bin']*n
                np.testing.assert_array_equal(grid['time'],field['row_time'][start:start+len(grid)])
                np.testing.assert_array_equal(grid['information'],field['row_information'][start:start+len(grid)])
            np.testing.assert_array_equal([r['first_s'] for r in blocks],grid['time'][::n])
            np.testing.assert_array_equal([r['last_s'] for r in blocks],grid['time'][n-1::n])
            np.testing.assert_allclose([r['information'] for r in blocks],grid['information'].reshape(-1,n).sum(axis=1),rtol=2e-15)
            expected=(grid['information'].reshape(-1,n)>0).sum(axis=1)
            np.testing.assert_array_equal([r['expected_valid_rows'] for r in blocks],expected)
            valid=np.asarray([r['valid'] for r in blocks]);known=np.asarray([r['known_rows'] for r in blocks])
            np.testing.assert_array_equal(valid,(known==expected)&(np.asarray([r['information'] for r in blocks])>0))
            indices=np.searchsorted(grid['time'],old['time'])
            if np.any(indices>=len(grid)):raise ValueError('Old waveform rows outside full replay')
            np.testing.assert_array_equal(grid['time'][indices],old['time'])
            np.testing.assert_array_equal(grid['information'][indices],old['information'])
            if np.any(indices%n!=np.arange(len(indices))%n):raise ValueError('Original block boundaries changed')
            block=indices[::n]//n;np.testing.assert_array_equal(valid[block],[r['valid'] for r in old_records[c]])
            difference=samples[block]-old_pulses[c]
            scale=max(float(np.linalg.norm(old_pulses[c])),1e-300)
            error=float(np.linalg.norm(difference)/scale)
            if error>3e-6:raise ValueError(f'Full replay changes known overlap by relative norm {error}')
            centre=samples[block,64];old_centre=old_pulses[c][:,64]
            use=(abs(centre)>0)&(abs(old_centre)>0)
            centre_phase=float(np.max(abs(np.angle(centre[use]*old_centre[use].conj())),initial=0))
            if centre_phase>1e-5:raise ValueError('Known-kernel phase changed in the old aperture')
            result['channels'].append(dict(channel=c,rows=len(grid),valid_blocks=int(valid.sum()),
                blocks=len(blocks),known_rows=int(known.sum()),expected_known_rows=int(expected.sum()),
                interval_s=grid['time'][[0,-1]],invalid_block_indices=np.flatnonzero(~valid),
                old_overlap_blocks=len(block),old_overlap_relative_norm_error=error,
                old_overlap_maximum_absolute_difference=float(np.max(abs(difference))),
                old_overlap_centre_phase_difference_rad=centre_phase))
    paths=[args.full/'response.kernels',args.full/'rows.npz',args.full/'request.json',
        args.reference/'response.kernels',args.reference/'rows.npz',Path(__file__),
        Path('support/calibration/waveform_kernel_export.cpp'),Path('build/release/support/calibration/waveform_kernel_export')]
    result.update(complete=True,source_sha256={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths})
    args.output.write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    print('validated full-capture kernels',sum(c['known_rows'] for c in result['channels']),
        'known rows;',sum(c['valid_blocks'] for c in result['channels']),'/',sum(c['blocks'] for c in result['channels']),'valid blocks',flush=True)


if __name__=='__main__':main()
