"""Training-only support selection and held-out ship channel-alignment trial."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from ais import read_csv, utc_seconds
from channel_alignment import Alignment, AlignmentBand
from dense_scatterers import points as select_points
from field import Field
from imaging import geometry_from_state, physical_state, read_image, sites, write_input
from spline_motion import SplineMotion


def load_trial(path, ais, geometry_directory):
    """Recreate a completed trial's declared data, support and training split."""
    source = json.loads(Path(path).read_text())
    model = json.loads(Path(source['input_motion']).read_text())
    curve = SplineMotion(model)
    tx, rx, _ = sites(geometry_directory)
    track = read_csv(ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    epoch, bands = utc_seconds(model['nominal_start_utc']), []
    for channel in range(5):
        with np.load(Path(source['input_fields'])/f'channel-{channel}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(curve.interval[0]+.02, curve.interval[1]-.02)
        prior = physical_state(track, epoch+field.data['row_time'], model['attitude_prior'])
        state = curve.apply(field.data['row_time'], prior, tx[3], rx)
        block = source['holdout_seconds']
        selected = (np.arange(len(field.time))%2 == 0 if block == 0 else
            np.floor((field.time-curve.interval[0])/block).astype(int)%2 == 0)
        bands.append(AlignmentBand(field, state, source['points_body_m'], selected))
    return source, curve, Alignment(bands, reference=source['reference_channel'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'aperture', 'motion', 'ais', 'geometry-directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--points', type=int, default=8)
    parser.add_argument('--holdout-seconds', type=float, default=0.)
    parser.add_argument('--iterations', type=int, default=150)
    parser.add_argument('--executable', type=Path, default=Path('build/release/support/isar/isar_backprojection'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = json.loads(args.motion.read_text())
    curve = SplineMotion(model)
    if model.get('antenna_fit'):
        a = model['antenna_fit']
        if np.max(abs(np.r_[np.array(a['transmitter_offsets_enu_m']).ravel(), a['receiver_offset_enu_m']])) > 1e-8:
            raise ValueError('Use the fixed nominal-location control; no rejected coordinates enter this trial')
    if model.get('holdout_seconds') != args.holdout_seconds:
        raise ValueError('Motion and alignment must use the same held-out bins')
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    epoch = utc_seconds(model['nominal_start_utc'])
    tx, rx, _ = sites(args.geometry_directory)
    along, port = np.arange(-220, 221, 4), np.arange(-64, 65, 4)
    x, y = np.meshgrid(along, port)
    pixels = np.c_[x.ravel(), y.ravel(), np.full(x.size, 20.), np.zeros(x.size)]
    fields, states, selections, images = [], [], [], []
    started = time.perf_counter()
    for channel in range(5):
        with np.load(args.fields/f'channel-{channel}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(curve.interval[0]+.02, curve.interval[1]-.02)
        prior = physical_state(track, epoch+field.data['row_time'], model['attitude_prior'])
        state = curve.apply(field.data['row_time'], prior, tx[3], rx)
        selected = (np.arange(len(field.time))%2 == 0 if args.holdout_seconds == 0 else
            np.floor((field.time-curve.interval[0])/args.holdout_seconds).astype(int)%2 == 0)
        fields.append(field); states.append(state); selections.append(selected)
        cache = args.output/f'channel-{channel}-training-image.npz'
        if cache.exists():
            with np.load(cache) as f:
                if str(f['source_motion']) != str(args.motion) or float(f['holdout_seconds']) != args.holdout_seconds:
                    raise ValueError('Training-image cache belongs to a different experiment')
                images.append(f['power'].copy())
            continue
        with np.load(args.aperture/f'channel-{channel}.npz') as f:
            rows = np.searchsorted(f['time_s'], field.data['row_time'])
            if np.max(abs(f['time_s'][rows]-field.data['row_time'])) > 1e-8:
                raise ValueError('Original profile rows do not match the fixed observation operator')
            moments = f['moments'][:, rows]
            metadata = json.loads(str(f['metadata_json']))
        row_selected = selected[np.arange(len(rows))//field.metadata['block_rows']]
        information = field.data['row_information']*row_selected
        packed, _ = geometry_from_state(state, tx[channel], rx, field.metadata['rf_hz'],
                                         metadata['first_range_sample'], information)
        source, destination = args.output/f'channel-{channel}-training.bin', args.output/f'channel-{channel}-training-image.bin'
        write_input(source, packed, moments, pixels, field.metadata['rf_hz'])
        subprocess.run([str(args.executable), str(source), str(destination)], check=True)
        result = read_image(destination, len(pixels))
        amplitude = (result[:, 0]+1j*result[:, 1])/result[:, 2]
        power = (abs(amplitude)**2*result[:, 2]).reshape(x.shape)
        images.append(power)
        np.savez(cache, power=power, complex_image=amplitude.reshape(x.shape), along_m=along, port_m=port,
            information=result[:, 2].reshape(x.shape), source_motion=str(args.motion), holdout_seconds=args.holdout_seconds)
        print(json.dumps({'stage': 'training image', 'channel': channel, 'seconds': time.perf_counter()-started}), flush=True)
    power = np.sum(images, axis=0)
    points = select_points({'complex_image': np.sqrt(power), 'along_m': along, 'port_m': port}, args.points, 30.)
    np.savez(args.output/'training-support.npz', power=power, along_m=along, port_m=port, points_body_m=points)
    bands = [AlignmentBand(field, state, points, selected) for field, state, selected in zip(fields, states, selections)]
    alignment = Alignment(bands, reference=3)
    result = {'input_motion': str(args.motion), 'input_fields': str(args.fields),
        'interval_s': curve.interval.tolist(), 'reference_channel': 3, 'holdout_seconds': args.holdout_seconds,
        'points_body_m': points.tolist(), 'support_selection': 'Peaks of information-weighted independent channel power; training bins only, 30 m separation, fixed 20 m body-height plane.',
        'scope': __doc__, 'coordinate_or_gain_calibration_accepted': False,
        'limitation': 'Conditional on the input motion and a shared, frequency-independent complex scattering scene. Input motion used image-selected support from earlier full-aperture images; this is not a completely untouched end-to-end holdout.',
        'stages': []}
    zero = np.zeros(3*alignment.count)
    for name, parameters, diagnostic in [('Nominal', zero, {'objective': alignment.objective(zero)[0]})]:
        result['stages'].append({'name': name, 'parameters': parameters.tolist(), 'fit': diagnostic,
            'validation': alignment.validation(parameters, curve.interval)})
    phase, fit = alignment.fit(vary_range=False, iterations=args.iterations)
    result['stages'].append({'name': 'Complex channel gains', 'parameters': phase.tolist(), 'fit': fit,
        'validation': alignment.validation(phase, curve.interval)})
    (args.output/'alignment.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'stage': 'gain fit', 'seconds': time.perf_counter()-started, **{k:fit[k] for k in ('objective', 'converged', 'gain_phase_degrees')}}), flush=True)
    aligned, fit = alignment.fit(phase, iterations=args.iterations)
    result['stages'].append({'name': 'Range origins and complex gains', 'parameters': aligned.tolist(), 'fit': fit,
        'validation': alignment.validation(aligned, curve.interval)})
    result['independent_response_controls'] = {
        'nominal_ranges': alignment.validation(zero, curve.interval, independent=True),
        'aligned_ranges': alignment.validation(aligned, curve.interval, independent=True)}
    result['seconds'] = time.perf_counter()-started
    (args.output/'alignment.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'stage': 'range and gain fit', 'seconds': result['seconds'],
        **{k:fit[k] for k in ('objective', 'converged', 'range_bias_m', 'gain_phase_degrees')}}), flush=True)


if __name__ == '__main__':
    main()
