"""Conditional information with the capture's frozen cancellation uncertainty.

For each carrier, one uncertain frozen complex coefficient is shared across
all primary rows. The known diagonal training variance is propagated through
the exact finite C0/C1 observation operator. No covariance parameter is fitted
to the ship, and no independent stationary variance is added per time block.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve_triangular

from captured_response import captured_rows
from finite_reference import FS, NFFT, coefficient_planes, covariance, echo, probes
from finite_scene_information import append_root, probe_groups, retained_information
from geographic_movie import state_at
from imaging import corrected_sites, sites
from motion import C, excess_path
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


def rank_one_whiten(values, factor):
    """Apply (I + z z^H)^(-1/2) independently to each carrier's row vector."""
    power = np.sum(abs(factor)**2, axis=0)
    root = np.sqrt(1+power)
    scale = 1/(root*(1+root))
    inner = np.einsum('rk,rkp->kp', factor.conj(), values)
    return values-factor[:, :, None]*(scale[:, None]*inner)[None, :, :]


def marginal_root(values, factor, carrier_block=128):
    root = np.empty((0, values.shape[-1]), complex)
    for first in range(0, values.shape[1], carrier_block):
        sl = slice(first, first+carrier_block)
        white = rank_one_whiten(values[:, sl], factor[:, sl])
        root = append_root(root, white.reshape(-1, values.shape[-1]))
    return root


def linear_risk(raw_root, design, covariance_root, *, optimum):
    """Actual amplitude covariance relative to the raw marginal CRLB."""
    used = solve_triangular(covariance_root, design, lower=True) if optimum else design
    u, s, vh = np.linalg.svd(used, full_matrices=False)
    if s[-1] <= s[0]*1e-12:
        raise ValueError('Scene amplitude inverse is not resolved')
    inverse = (vh.conj().T/s)@u.conj().T
    noise_map = inverse if optimum else inverse@covariance_root
    return float(np.linalg.norm(raw_root@noise_map)**2/design.shape[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'index', 'motion', 'fields', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--channels', type=int, nargs='+', default=list(range(5)))
    args = parser.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    model = json.loads(args.motion.read_text()); curve = SplineMotion(model)
    if not model['complete'] or not model['fit']['converged'] or np.any(curve.yaw_coefficients):
        raise ValueError('Use the stationary fixed-N/E physical motion control')
    nominal_tx, nominal_rx, _ = sites(Path(model['geometry_directory']))
    tx, rx = corrected_sites(nominal_tx, nominal_rx, model.get('antenna_fit'))
    selected = defaultdict(list)
    for row in probes(args.input):
        if row['channel'] in args.channels and model['interval_s'][0] <= row['time_s'] < model['interval_s'][1]:
            selected[row['channel']].append({k: v for k, v in row.items() if k not in ('measured','zero','one')})
    sources = [args.input, args.index, args.motion, Path(__file__), *[Path(__file__).with_name(n) for n in
        ('captured_response.py', 'finite_reference.py', 'finite_scene_information.py', 'geographic_movie.py', 'imaging.py', 'spline_motion.py')],
        *[args.fields/f'channel-{c}.npz' for c in args.channels]]
    result = {'complete': False, 'scope': __doc__, 'source_sha256': {str(p):fingerprint(p) for p in sources},
        'interval_s': model['interval_s'], 'channels': [],
        'limitations': 'Sparse 16-symbol blocks, fixed eight-point support and known motion. Proper-complex independent primary thermal rows and frozen diagonal training-coefficient uncertainty, conditional on fitted H. This does not certify H uncertainty, correlated interference, real-scene support or actual nonlinear focusing.'}
    args.output.mkdir(parents=True)
    offsets = np.arange(-8, 9)
    for c in sorted(selected):
        groups, rejected = probe_groups(selected[c], 16)
        rows = [row for group in groups for row in group]
        times = np.array([r['time_s'] for r in rows])
        originals = captured_rows(args.index, c, times, NFFT/FS*(1+1/rows[0]['guard']))
        beta = originals[0]['cancellation_variance']
        if len({r['segment'] for r in originals}) != 1 or any(not np.array_equal(r['cancellation_variance'], beta) for r in originals):
            raise ValueError('Frozen coefficient covariance changed within the tested aperture')
        state = state_at(None, model, curve, nominal_tx, nominal_rx, times)
        points = np.asarray(model['selected_points_body_m'][c])
        position = state['position_ecef_m'][:, None, :]+np.einsum('nij,pj->npi', state['body_to_ecef'], points)
        velocity = state['velocity_ecef_mps'][:, None, :]+np.einsum('nij,pj->npi', state['frame_velocity'], points)
        path, gradient = excess_path(position.reshape(-1,3), tx[c], rx)
        path = path.reshape(len(rows),len(points)); rate = np.sum(gradient.reshape(position.shape)*velocity,axis=-1)
        with np.load(args.fields/f'channel-{c}.npz') as field:
            epochs = field['row_time']; idx = np.clip(np.searchsorted(epochs,times),1,len(epochs)-1)
            idx -= abs(epochs[idx-1]-times)<abs(epochs[idx]-times)
            if np.max(abs(epochs[idx]-times))>2e-8:
                raise ValueError('Probe epoch differs from the saved observation')
            centre_path = field['centre_path_m'][idx]; centre_omega = field['centre_omega'][idx]
        raw = np.empty((len(rows),6817,len(points)),complex)
        factors = np.empty((len(rows),6817),complex)
        compressed, nuisance, local_q, local_b, local_n = [],[],[],[],[]
        rf = rows[0]['rf_hz']; k = 2*np.pi*rf/C; carrier = np.arange(-3408,3409)
        information = np.array([r['information'] for r in originals])
        for i,(row,saved) in enumerate(zip(rows,originals)):
            h, noise = saved['response'], saved['variance']
            origin = saved['reference_origin']-saved['source_offset']
            previous = saved['reference_origin']-saved['previous_source_offset']
            planes = coefficient_planes(row['current'],row['previous'],h,noise,origin,
                centre_path[i]*FS/C,offsets,guard=row['guard'],previous_fraction=previous-np.floor(previous))
            a = planes[0]-1j*centre_omega[i]*planes[1]
            q = np.column_stack([echo(row['current'],row['previous'],h,origin,delay,doppler,guard=row['guard'])*phase
                for delay,doppler,phase in zip((path[i]+model['range_bias_m'][c])*FS/C,-rf/C*rate[i],np.exp(-1j*k*path[i]))])
            z = h*row['current']*np.exp(-2j*np.pi*carrier*origin/NFFT)*np.sqrt(beta)
            raw[i],factors[i] = q/np.sqrt(noise)[:,None],z/np.sqrt(noise)
            local_q.append(a@q);local_b.append(a*z);local_n.append(covariance(a,noise))
            if len(local_q)==16:
                first=i-15;weight=information[first:i+1]/information[first:i+1].sum()
                coefficient=weight*np.exp(1j*k*centre_path[first:i+1])
                thermal=np.einsum('n,nij->ij',weight**2,local_n)
                lower=np.linalg.cholesky(thermal)
                compressed.append(solve_triangular(lower,np.einsum('n,nij->ij',coefficient,local_q),lower=True))
                nuisance.append(solve_triangular(lower,np.einsum('n,nij->ij',coefficient,local_b),lower=True))
                local_q,local_b,local_n=[],[],[]
        raw_root=marginal_root(raw,factors)
        thermal_root=marginal_root(raw,np.zeros_like(factors))
        del raw,factors
        design,b=np.concatenate(compressed),np.concatenate(nuisance)
        extra=b@b.conj().T
        lower=np.linalg.cholesky(np.eye(len(b))+extra)
        optimal=solve_triangular(lower,design,lower=True)
        eigen=np.linalg.eigvalsh(extra)
        file=args.output/f'channel-{c}.npz'
        np.savez(file,raw_root=raw_root,thermal_raw_root=thermal_root,design=design,
            cancellation_factor=b,covariance_cholesky=lower,times_s=times)
        record={'channel':c,'original_rows':len(rows),'blocks':len(groups),'rejected_blocks':rejected,
            'beta_variance_min_median_max':np.quantile(beta,[0,.5,1]).tolist(),
            'cancellation_covariance_relative_to_thermal':{'largest_eigenvalue':float(eigen[-1]),'trace_per_cell':float(eigen.sum()/len(eigen))},
            'compression_with_cancellation':[retained_information(raw_root,optimal,f) for f in (1e-8,1e-10,1e-12)],
            'raw_information_cost_of_cancellation_uncertainty':[retained_information(thermal_root,raw_root,f) for f in (1e-8,1e-10,1e-12)],
            'optimal_profile_mse_over_raw_crlb':linear_risk(raw_root,design,lower,optimum=True),
            'thermal_weighted_profile_mse_over_raw_crlb':linear_risk(raw_root,design,lower,optimum=False),
            'arrays':str(file),'arrays_sha256':fingerprint(file)}
        result['channels'].append(record)
        (args.output/'information.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'channel':c,**record['cancellation_covariance_relative_to_thermal'],
            'optimal_risk':record['optimal_profile_mse_over_raw_crlb'],
            'thermal_risk':record['thermal_weighted_profile_mse_over_raw_crlb'],
            'seconds':time.monotonic()-started}),flush=True)
    result.update(complete=True,seconds=time.monotonic()-started)
    (args.output/'information.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
