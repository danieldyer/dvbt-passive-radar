"""Expected motion information on the complete recorded-aperture operator.

All unfrozen path/transverse spline coordinates and every original row are
included. Core coefficients and selected diffuse penalties specify a synthetic
truth at the recorded geometry. This computes a conditional bound; it does not
claim that the measured residual is Gaussian or that a fitted motion reaches it.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh

from marginal_information import information
from mixed_experiment import load
from mixed_scene import profile
from physical_profile import observe_partials
from ridge_motion import point_derivatives
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve completed information audits')
    experiment = load(args.source)
    model, curve = experiment.model, SplineMotion(experiment.model)
    _, active = curve.gauge_fixed_parameters()
    active = active[(active < curve.sections[1].start)|(active >= curve.sections[1].stop)]
    started = time.monotonic()
    result = {'scope': __doc__, 'source': str(args.source), 'source_sha256': fingerprint(args.source),
        'interval_s': model['interval_s'], 'active_physical_indices': active.tolist(),
        'unknown_mean': 'All eight complex core reflectivities independently in each band; fixed stationary nuisance eliminated.',
        'known_quantities': 'Antenna coordinates, core point locations, noise variance 1, diffuse variance 1/lambda, nominal range origins and motion basis.',
        'constraint_scope': 'Data-only local bound after coordinate gauge and requested zero-yaw restriction; no acceleration or box-bound information is added.',
        'channels': [], 'complete': False}
    for c, (band, weight, core, strength) in enumerate(zip(experiment.bands, experiment.weights,
            model['core_counts'], model['fixed_penalties'])):
        fitted = profile(band, model['parameters'], np.sqrt(weight), core, strength)
        design = band.nuisance.remove(fitted['target'][band.selected])*np.sqrt(weight)
        partials = np.empty((*design.shape, len(active)), complex)
        for j, derivative in enumerate(point_derivatives(band, model['parameters'])):
            partials[:, j, :] = observe_partials(band, derivative, False)[:, active]*np.sqrt(weight[j])
        info = information(design, fitted['coefficient'], partials, core, strength)
        np.testing.assert_allclose(info['total'], info['mean']+info['covariance'], atol=1e-8)
        result['channels'].append({'channel': c, 'selected_observations': len(design),
            'original_rows': len(band.time), 'scene_points': len(band.points), 'core_points': core,
            'penalty': strength, 'core_coefficients_real_imag': np.c_[fitted['coefficient'][:core].real, fitted['coefficient'][:core].imag].tolist(),
            **{key: value.tolist() for key, value in info.items()}})
        del partials, design, fitted
        print(json.dumps({'channel': c, 'seconds': time.monotonic()-started}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    total = {key: sum(np.array(c[key]) for c in result['channels']) for key in ('mean', 'covariance', 'total')}
    eigen = np.linalg.eigvalsh(total['total'])
    if eigen.min() <= 0:
        raise ValueError('The declared data-only model has unresolved motion information nulls')
    covariance = np.linalg.inv(total['total'])
    mean_eigen = np.linalg.eigvalsh(total['mean'])
    information_gain = eigh(total['total'], total['mean'], eigvals_only=True) if mean_eigen.min()>0 else None
    epochs = np.linspace(*curve.interval, 601)
    maps = []
    for j in (0, len(curve.sections)-1):
        matrix = np.zeros((len(epochs), len(model['parameters'])))
        matrix[:, curve.sections[j]] = curve.coefficient_bases(epochs)[j]
        matrix = matrix[:, active]
        maps.append(np.sqrt(np.maximum(0., np.einsum('ni,ij,nj->n', matrix, covariance, matrix))))
    result.update(complete=True, seconds=time.monotonic()-started,
        unit_noise_information={k: v.tolist() for k, v in total.items()},
        information_eigenvalues=eigen.tolist(), information_condition=float(eigen.max()/eigen.min()),
        crlb_covariance=covariance.tolist(), covariance_information_gain_eigenvalues=None if information_gain is None else information_gain.tolist(),
        epochs_s=epochs.tolist(), path_coordinate_crlb_std_m=maps[0].tolist(), transverse_coordinate_crlb_std_m=maps[1].tolist(),
        implementation_sha256={name: fingerprint(Path(__file__).with_name(name)) for name in
            ('audit_marginal_information.py', 'marginal_information.py', 'marginal_scene.py',
             'mixed_experiment.py', 'mixed_scene.py', 'field.py', 'full_motion.py', 'ridge_motion.py', 'physical_profile.py', 'spline_motion.py')})
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'complete': True, 'parameters': len(active), 'condition': result['information_condition'],
        'maximum_path_std_m': float(maps[0].max()), 'maximum_transverse_std_m': float(maps[1].max()),
        'covariance_information_gain_range': None if information_gain is None else [float(information_gain.min()),float(information_gain.max())]}), flush=True)


if __name__ == '__main__':
    main()
