"""Explicit long-aperture stationary nuisance model for the ship likelihood.

This support analysis profiles complex constant-delay amplitudes jointly with
moving targets. It retains coefficients and originals. Both data and target
templates receive the same projection; no moving-target Doppler bins are cut.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from field import Field, FS
from motion import C
from scatterers import normal, prediction


class Stationary:
    def __init__(self, design, selected, relative_tolerance=1e-10):
        self.design = design
        self.selected = selected
        u, s, vh = np.linalg.svd(design[selected], full_matrices=False)
        keep = s > s[0]*relative_tolerance if len(s) else np.zeros(0,bool)
        self.singular_values = s
        self.rank = int(keep.sum())
        self.q = u[:, keep]
        self.coefficients = vh[keep].conj().T/s[keep]

    def remove(self, values):
        return values-self.q@(self.q.conj().T@values)

    def solve(self, values):
        return self.coefficients@(self.q.conj().T@values)


def matched_projection(design, data, nuisance):
    """Sufficient matched statistics after eliminating stationary amplitudes.

    Data and templates receive the same orthogonal projection. Returning the
    projected template energy is essential near stationary trajectories: using
    the unprojected energy would give the wrong likelihood and image scaling.
    Inputs contain the rows used to construct nuisance.q, in the same order.
    """
    target = nuisance.remove(design)
    clean = nuisance.remove(data)
    return target.conj().T@clean, np.sum(abs(target)**2, axis=0)


def component_statistics(design,data,nuisance):
    """Exact per-component Schur scores with the other components as nuisances.

    Unlike using column energy after a joint least-squares fit, the inverse
    Gram diagonal accounts for overlapping target templates. Constant-profile
    nuisance elimination is applied to data and every template alike.
    """
    h=nuisance.remove(design);y=nuisance.remove(data);norm=np.linalg.norm(h,axis=0)
    if np.any(norm==0):raise ValueError('A scene component has no information after nuisance projection')
    u,s,vh=np.linalg.svd(h/norm,full_matrices=False)
    if len(s)!=h.shape[1] or s[-1]<=s[0]*1e-10:raise ValueError('Individual scene components are not identifiable in this fold')
    normalized=vh.conj().T@((u.conj().T@y)/s)
    coefficient=normalized/norm
    inverse_diagonal=np.sum(abs(vh.conj().T/s)**2,axis=1)
    information=norm**2/inverse_diagonal
    return information*coefficient,information,float(s[0]/s[-1])


def rank_one_projection(score,information,cross,nuisance_score,nuisance_information):
    """Additional complex nuisance, with its exact target-information loss.

    Inputs already share the same whitening and base nuisance projection.
    Nonpositive output information is left visible to the caller, not floored
    into an apparently valid scientific statistic.
    """
    norm=np.asarray(nuisance_information)
    if np.any(norm<=0) or not np.isfinite(norm).all():raise ValueError('The competing source needs positive finite projected information')
    return score-cross*nuisance_score/norm,information-abs(cross)**2/norm


def make_design(field):
    centre = field.data['centre_path_m']*FS/C
    samples = np.arange(np.floor(centre.min())+field.offsets.min()-5,
                        np.ceil(centre.max())+field.offsets.max()+6)
    design = np.column_stack([field.stationary(sample*C/FS).ravel() for sample in samples])
    return design, samples*C/FS


def joint_test(field, design, points):
    selected_time = ((field.time-90)//2)%2 == 0
    selected = np.repeat(selected_time, len(field.offsets))
    nuisance = Stationary(design, selected)
    target = np.column_stack([field.point(point).ravel() for point in points])
    y = field.y.ravel()
    projected = nuisance.remove(target[selected])
    projected_y = nuisance.remove(y[selected])
    amplitude = np.linalg.lstsq(projected, projected_y, rcond=None)[0]
    beta = nuisance.solve(y[selected]-target[selected]@amplitude)
    held_y = y[~selected]-design[~selected]@beta
    score = prediction([normal(target[~selected], held_y)], [amplitude])
    original_information = np.sum(abs(target[selected])**2, axis=0)
    retained_information = np.sum(abs(projected)**2, axis=0)
    stationary_only = nuisance.solve(y[selected])
    return {'stationary_rank': nuisance.rank, 'stationary_columns': design.shape[1],
            'stationary_singular_values': nuisance.singular_values.tolist(),
            'target_information_loss_db': (-10*np.log10(retained_information/original_information)).tolist(),
            'joint_held_out': score,
            'stationary_only_held_out_error': float(np.sum(abs(y[~selected]-design[~selected]@stationary_only)**2)),
            'raw_held_out_energy': float(np.sum(abs(y[~selected])**2)),
            'joint_held_out_error': score['prediction_error']}, beta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--channels', nargs='+', type=int, default=list(range(5)))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    points = np.array([[-50., -2., 20.], [25., 5., 20.], [90., 8., 20.], [135., 13., 20.]])
    for channel in args.channels:
        with np.load(args.fields/f'channel-{channel}.npz') as loaded:
            field = Field(dict(loaded), loaded['pulse_samples'])
        design, paths = make_design(field)
        np.save(args.output/f'channel-{channel}-stationary.npy', design)
        result, beta = joint_test(field, design, points)
        result.update(channel=channel, point_body_m=points.tolist(), stationary_paths_m=paths.tolist(),
                      scope='Joint long-aperture constant-delay nuisance and moving-point likelihood. Alternating 2-second training predicts held-out data without refitting either amplitudes or nuisance. Information loss is conditional on the declared AIS point trajectory and thermal covariance; no correction adopted.')
        np.savez(args.output/f'channel-{channel}-coefficients.npz', coefficients=beta, paths_m=paths)
        (args.output/f'channel-{channel}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
