"""Reopen a completed mixed-scene experiment without changing its samples."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ais import utc_seconds
from distributed_fixed_frame import quadrature
from field import Field
from full_motion import Band
from imaging import motion_state, sites
from midpoint_model import representative
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


@dataclass
class Experiment:
    model: dict
    training: dict
    bands: list
    fast: list
    weights: list


def load(path, *, accelerated=False):
    model = json.loads(Path(path).read_text())
    if not model['complete'] or not model['fit']['converged']:
        raise ValueError('Use a completed original-row mixed-scene motion fit')
    training_path = Path(model['training'])
    if fingerprint(training_path) != model['training_sha256']:
        raise ValueError('Training has changed since the source motion fit')
    training = json.loads(training_path.read_text())
    curve = SplineMotion(model)
    if not model.get('fixed_ne_frame') or np.any(curve.yaw_coefficients):
        raise ValueError('Keep yaw exactly zero')
    tx, rx, _ = sites(Path(model['geometry_directory']))
    diffuse, area = quadrature(training['bounds_enu_m'], model['spacing_m'])
    bands, fast, weights = [], [], []
    for c in range(5):
        core = np.asarray(model['core_points_enu_m'][c])
        strength = model['fixed_penalties'][c]
        points = np.asarray(model['selected_points_body_m'][c])
        np.testing.assert_array_equal(points, core if strength is None else np.r_[core, diffuse])
        if len(core) != model['core_counts'][c]:
            raise ValueError('The explicit core changed')
        weights.append(np.ones(len(core)) if strength is None else np.r_[np.ones(len(core)), area])
        path = Path(training['fields'])/f'channel-{c}.npz'
        if fingerprint(path) != training['channels'][c]['field_sha256']:
            raise ValueError('Original field changed since penalty selection')
        with np.load(path) as d:
            field = Field(dict(d), d['pulse_samples']).windowed(curve.interval[0]+.03, curve.interval[1]-.03)
        selected = np.floor((field.time-curve.interval[0])/training['holdout_seconds']).astype(int)%2 == 0
        for f, destination in [(field, bands), *([(representative(field), fast)] if accelerated else [])]:
            prior = motion_state(None, utc_seconds(model['nominal_start_utc'])+f.data['row_time'], model)
            destination.append(Band(f, prior, tx[model['reference_channel']], rx, points, model,
                selected, range_bias_m=model['range_bias_m'][c]))
        if accelerated:
            fast[-1].nuisance, fast[-1].clean, fast[-1].energy = bands[-1].nuisance, bands[-1].clean, bands[-1].energy
    return Experiment(model, training, bands, fast, weights)
