"""Recorded-geometry information retention, including arbitrary scene mixtures.

Checks the actual archived spectral weights for the range crop, and the joint
64-point response space for slow-time averaging. These are conditional Gaussian
compression bounds, not a certificate of real scene or covariance correctness.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from ais import read_csv, utc_seconds
from audit_pulse_variation import archived_spectra
from compression_information import range_crop_information, scene_information
from field import Field, average
from imaging import physical_state, sites
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('support', 'alignment', 'ais', 'geometry-directory', 'spectral-audit', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    support, alignment, spectra_audit = [json.loads(v.read_text()) for v in
        (args.support, args.alignment, args.spectral_audit)]
    model_path = Path(support['source_motion'])
    model = json.loads(model_path.read_text())
    curve = SplineMotion(model)
    points = np.array(support['points_body_m'])
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    archive = json.loads(Path(spectra_audit['inputs']['index']).read_text())
    capture = Path(spectra_audit['inputs']['capture'])
    if Path(archive['identity'][0]).resolve() != capture.resolve():
        raise ValueError('Wrong archive index')
    result = {'scope': __doc__, 'channels': [], 'complete': False,
        'sources_sha256': {str(v): fingerprint(v) for v in
            (args.support, args.alignment, args.ais, args.spectral_audit, model_path)},
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('audit_scene_information.py', 'compression_information.py', 'field.py')},
        'covariance': 'Independent original rows with mean range covariance and uniform symbol energy C0/C1 moments. Range checks use each recorded response/noise spectrum exactly; waveform realizations are not substituted for that spectrum.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clock = time.perf_counter()
    for c in range(5):
        with np.load(Path(alignment['input_fields'])/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(curve.interval[0]+.02, curve.interval[1]-.02)
        d, block = field.data, field.metadata['block_rows']
        prior = physical_state(track, utc_seconds(model['nominal_start_utc'])+d['row_time'], model['attitude_prior'])
        state = curve.apply(d['row_time'], prior, tx[model['reference_channel']], rx)
        raw = np.empty((len(d['row_time']), len(field.offsets), len(points)), complex)
        for j, point in enumerate(points):
            raw[:, :, j] = field.raw_trajectory(state['position_ecef_m']+state['body_to_ecef']@point,
                state['velocity_ecef_mps']+state['frame_velocity']@point, range_bias_m=model['range_bias_m'][c])
        factor = 1+d['centre_omega']**2/12
        weights = d['row_information']
        original = np.zeros((len(points), len(points)), complex)
        for start in range(0, len(raw), 1024):
            sl = slice(start, start+1024)
            white = np.einsum('nkp,rk->nrp', raw[sl], field.whiten)*np.sqrt(8192*weights[sl]/factor[sl])[:, None, None]
            white = white.reshape(-1, len(points))
            original += white.conj().T@white
        mean, information = average(raw, weights, block)
        del raw
        noise_factor, _ = average(factor, weights, block)
        root = np.sqrt(np.divide(8192*information, noise_factor, out=np.zeros_like(information), where=noise_factor > 0))
        compressed = (np.einsum('nkp,rk->nrp', mean, field.whiten)*root[:, None, None]).reshape(-1, len(points))
        joint = scene_information(original, compressed.conj().T@compressed)
        del compressed, mean
        spectra, _ = archived_spectra(capture, archive, c, model['interval_s'])
        w = 2*np.pi*np.arange(-3408, 3409)/8192
        crop = []
        # Cover the complete ship support, including fractional lag extremes.
        for delay in np.arange(-1.5, 4.501, .25):
            measured = [range_crop_information(w, spectrum, field.offsets, delay) for spectrum in spectra]
            crop.append({'delay_samples': float(delay),
                'worst_snr_loss_db': float(max(-10*np.log10(v['matched_snr_fraction']) for v in measured)),
                'worst_delay_information_loss_db': float(max(-10*np.log10(v['delay_information_fraction']) for v in measured))})
        record = {'channel': c, 'joint_scene_temporal_information': joint,
            'recorded_spectra': len(spectra), 'range_crop': crop,
            'worst_range_snr_loss_db': max(v['worst_snr_loss_db'] for v in crop),
            'worst_range_delay_information_loss_db': max(v['worst_delay_information_loss_db'] for v in crop)}
        result['channels'].append(record)
        result.update(complete=c == 4, seconds=time.perf_counter()-clock)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'channel': c, 'scene_loss_db': joint['worst_information_loss_db'],
            'unresolved_modes': joint['unresolved_modes'], 'range_snr_loss_db': record['worst_range_snr_loss_db'],
            'range_delay_loss_db': record['worst_range_delay_information_loss_db'], 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
