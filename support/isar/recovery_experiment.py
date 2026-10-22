"""Truth-controlled motion recovery through the recorded complex observation operator.

Offline validation, not an antenna calibration. Synthetic data use independently
evaluated Cartesian distances and numerical symbol quadrature. The estimator
uses Field.trajectory and its analytic derivatives. Actual row times, weights,
range covariance, known reconstructed waveform pulses and C0/C1 averaging are
retained. No phase unwrapping, amplitude splines or fitted phase screens.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares

from ais import read_csv
from cluster_phase import crop
from field import Field, FS, average
from flash_siting_report import EPOCH
from motion import C
from piecewise_ais import LocalAisReference
from report_measured_scatterer_phases import serial
from ship_site_signature import licensed_options
from waveform_kernels import KernelField, bind_subset, read_kernels
from chirp_acquisition import scores as chirp_scores

SCALE = np.array([100., 100., .5, .5, .02, .02])
NAMES = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']


def thin_field(field, stride):
    """Keep declared complete bins, with all their native symbols and identities."""
    d = field.data; block = field.metadata['block_rows']
    bins = np.arange(0, len(field.time), stride)
    bins = bins[(bins+1)*block <= len(d['row_time'])]
    rows = (bins[:, None]*block+np.arange(block)).ravel()
    data = dict(d)
    for key in ('time', 'value', 'information'):
        data[key] = d[key][bins]
    for key in ('row_time', 'row_information', 'centre_ecef_m', 'body_to_ecef',
                'centre_path_m', 'centre_velocity_ecef_mps', 'centre_omega', 'carrier_reference_path_m'):
        if key in d: data[key] = d[key][rows]
    base = Field(data, field.pulse.samples)
    if isinstance(field, KernelField):
        return KernelField(base, field.pulse.kernels, field.pulse.row_blocks[rows])
    return base


def load_fields(first, last, stride):
    root = Path('docs/data/calibration_20261016_sites')
    geo = json.loads((root/'geometry-prior.json').read_text())
    receiver, rotation = np.asarray(geo['receiver_ecef_m']), np.asarray(geo['ecef_to_enu'])
    options = licensed_options(root)
    tx = np.array([next(o for o in options if o['channel'] == c and
        o['site_id'] == ('12014' if c == 4 else '12013'))['transmitter_ecef_m'] for c in range(5)])
    kernel = Path('build/isar-20261017/waveform-kernels-full')
    _, records, samples = read_kernels(kernel/'response.kernels')
    fields = []
    with np.load(kernel/'rows.npz') as grid:
        for c in range(5):
            with np.load(f'build/isar-20261015/full-fields/channel-{c}.npz') as archive:
                f = crop(dict(archive), first, last, np.arange(-8, 9))
            f = thin_field(f, stride)
            fields.append(bind_subset(f, grid[f'channel_{c}']['time'], grid[f'channel_{c}']['information'],
                samples[c], np.array([r['valid'] for r in records[c]])))
    track = read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'), reference_height_ellipsoid_m=5., mmsi=563052200)
    mid = (first+last)/2
    reference = LocalAisReference(track, EPOCH+mid, rotation, receiver)
    state = [reference(0., i) for i in range(3)]
    return fields, rotation, receiver, tx, state, mid


class Experiment:
    """Six planar kinematic coordinates, known height and declared point support.

    Complex reflectivity is constant over an aperture and independent per band.
    A fixed complex stationary basis is eliminated by an orthogonal projection.
    This deliberately exposes the information left after unknown band phases.
    """
    def __init__(self, fields, rotation, receiver, transmitters, state, midpoint, *, offsets=None, clutter=True, training=None):
        self.fields, self.rotation = fields, np.asarray(rotation)
        self.receiver, self.transmitters = np.asarray(receiver), np.asarray(transmitters)
        self.origin = np.asarray(receiver).copy()
        self.state, self.midpoint = [np.asarray(s) for s in state], midpoint
        self.offsets = np.zeros((1, 3)) if offsets is None else np.asarray(offsets)
        self.training = ([np.arange(len(f.time)) % 4 != 3 for f in fields] if training is None
                         else [np.asarray(mask, bool) for mask in training])
        if len(self.training) != len(fields) or any(mask.shape != f.time.shape or not mask.any()
                for mask, f in zip(self.training, fields)):
            raise ValueError('Training masks must select observed epochs in every band')
        self.designs, self.q = [], []
        for f, train in zip(fields, self.training):
            paths = np.interp(midpoint, f.data['row_time'], f.data['centre_path_m'])+np.arange(-8, 9)*C/FS
            # Same declared stationary echoes in synthesis and estimation.
            design = np.column_stack([f.stationary(p).ravel() for p in paths]) if clutter else np.empty((f.y.size, 0), complex)
            use = np.repeat(train, len(f.offsets))
            u, s, _ = np.linalg.svd(design[use], full_matrices=False)
            self.designs.append(design); self.q.append(u[:, s > s[0]*1e-10] if len(s) else u)

    def kinematics(self, p, c, offset):
        f = self.fields[c]; dt = f.data['row_time']-self.midpoint
        changes = np.asarray(p)*SCALE
        state = [s.copy() for s in self.state]
        for i in range(3): state[i][:2] += changes[2*i:2*i+2]
        pos = state[0]+offset+dt[:, None]*state[1]+.5*dt[:, None]**2*state[2]
        vel = state[1]+dt[:, None]*state[2]
        position = self.origin+pos@self.rotation
        velocity = vel@self.rotation
        pj = np.zeros((len(dt), 3, 6)); vj = np.zeros_like(pj)
        for i in range(3):
            pj[:, :2, 2*i:2*i+2] = (dt**i/([1, 1, 2][i]))[:, None, None]*np.eye(2)*SCALE[2*i:2*i+2]
            if i: vj[:, :2, 2*i:2*i+2] = (dt**(i-1))[:, None, None]*np.eye(2)*SCALE[2*i:2*i+2]
        return position, velocity, np.einsum('nij,ik->nkj', pj, self.rotation), np.einsum('nij,ik->nkj', vj, self.rotation)

    def templates(self, p, c, *, derivatives=False, offsets=None):
        f = self.fields[c]; result = []; jacobian = []
        for offset in self.offsets if offsets is None else offsets:
            pos, vel, pj, vj = self.kinematics(p, c, offset)
            value = f.trajectory(pos, vel, position_jacobian=pj if derivatives else None,
                velocity_jacobian=vj if derivatives else None, transmitter=self.transmitters[c],
                receiver=self.receiver, position_offset=pos-f.data['centre_ecef_m'])
            if derivatives:
                h, j = value; result.append(h.ravel()); jacobian.append(j.reshape(-1, 6))
            else: result.append(value.ravel())
        h = np.column_stack(result)
        return (h, np.stack(jacobian, axis=1)) if derivatives else h

    def independent_signal(self, p, c, amplitude):
        """Explicit path norms + Gauss-Legendre moments, independent of Field geometry."""
        f = self.fields[c]; d = f.data; result = np.zeros_like(d['value'])
        u, w = leggauss(24); u, w = u/2, w/2
        for offset, alpha in zip(self.offsets, amplitude):
            position, velocity, _, _ = self.kinematics(p, c, offset)
            path = np.full(len(position), -np.linalg.norm(self.transmitters[c]-self.receiver))
            rate = np.zeros(len(position))
            for antenna in (self.transmitters[c], self.receiver):
                vector = position-antenna; length = np.sqrt(np.sum(vector*vector, axis=1))
                path += length; rate += np.sum(vector*velocity, axis=1)/length
            omega = -f.k*.001024*rate
            moment = np.exp(1j*omega[:, None]*u)@w-1j*d['centre_omega']*(np.exp(1j*omega[:, None]*u)@(u*w))
            lag = f.offsets[None, :]-(path-d['centre_path_m'])[:, None]*FS/C
            carrier = d.get('carrier_reference_path_m', d['centre_path_m'])
            raw = alpha*moment[:, None]*f.pulse(lag)*np.exp(-1j*f.k*(path-carrier)[:, None])
            result += average(raw, d['row_information'], f.metadata['block_rows'])[0]
        return f.apply(result).ravel()

    def profile(self, p, y, *, derivative=True):
        residuals, jacobians, coefficients, predictions = [], [], [], []
        for c, (f, train, q) in enumerate(zip(self.fields, self.training, self.q)):
            value = self.templates(p, c, derivatives=derivative)
            h, dh = value if derivative else (value, None)
            use = np.repeat(train, len(f.offsets)); data = y[c][use]
            a = h[use]-q@(q.conj().T@h[use]); clean = data-q@(q.conj().T@data)
            coefficient = np.linalg.lstsq(a, clean, rcond=1e-10)[0]
            r = clean-a@coefficient
            if derivative:
                da = dh[use]-np.einsum('ni,ijq->njq', q, np.einsum('ni,njq->ijq', q.conj(), dh[use]))
                v = np.einsum('nsq,s->nq', da, coefficient)
                inverse=np.linalg.pinv(a,rcond=1e-10)
                dc = -inverse@v
                # Residual-dependent term from differentiating the complex LS normal equations.
                dc += inverse@inverse.conj().T@np.einsum('nsq,n->sq', da.conj(), r)
                dr = -v-a@dc
                jacobians.append(np.r_[dr.real, dr.imag])
            residuals.append(np.r_[r.real, r.imag]); coefficients.append(coefficient)
            static = np.linalg.lstsq(self.designs[c][use], y[c][use]-h[use]@coefficient, rcond=1e-10)[0]
            predictions.append(h@coefficient+self.designs[c]@static)
        return np.concatenate(residuals), np.concatenate(jacobians) if derivative else None, coefficients, predictions

    def fit(self, y, seed, max_evaluations=70, free=None, *, position_halfwidth_m=500., velocity_halfwidth_mps=1.):
        free = np.arange(6) if free is None else np.asarray(free)
        if not np.isfinite(position_halfwidth_m) or position_halfwidth_m <= 0:
            raise ValueError('A positive finite position search half-width is required')
        if not np.isfinite(velocity_halfwidth_mps) or velocity_halfwidth_mps <= 0:
            raise ValueError('A positive finite velocity search half-width is required')
        limit = np.r_[position_halfwidth_m/SCALE[:2],velocity_halfwidth_mps/SCALE[2:4],2,2]
        last = [None, None]
        def evaluate(q):
            p = np.asarray(seed).copy(); p[free] = q
            if last[0] is None or not np.array_equal(last[0], q): last[:] = [q.copy(), self.profile(p, y)]
            return last[1]
        fit = least_squares(lambda q: evaluate(q)[0], np.asarray(seed)[free], jac=lambda q:evaluate(q)[1][:, free],
            x_scale='jac', max_nfev=max_evaluations, ftol=1e-11, xtol=1e-11, gtol=1e-8,
            bounds=(-limit[free], limit[free]))
        p = np.asarray(seed).copy(); p[free] = fit.x
        r, j, alpha, prediction = self.profile(p, y)
        error = [np.sum(abs(z.reshape(f.y.shape)[~train])**2) for z, f, train in
            zip([v-w for v,w in zip(y, prediction)], self.fields, self.training)]
        singular = np.linalg.svd(j[:, free], compute_uv=False)
        return dict(parameters=p, physical_correction=p*SCALE, training_rss=r@r, test_rss=error,
            gradient=fit.optimality, success=fit.success, message=fit.message, nfev=fit.nfev,
            singular_values=singular, active_bound_coordinates=free[np.asarray(fit.active_mask)!=0],
            position_halfwidth_m=position_halfwidth_m, velocity_halfwidth_mps=velocity_halfwidth_mps,
            coefficients_real_imag=[np.c_[a.real,a.imag] for a in alpha]), prediction

    def acquire(self, y, seed, count=3, *, constant_velocity=False,
                path_rate_halfwidth_mps=.6, path_acceleration_halfwidth_mps2=.015):
        """Training-only common path rate/curvature search, followed by exact geometry.

        The tone approximation supplies initial guesses only. It neither changes
        the observations nor supplies a fitted phase correction to the image.
        """
        p=np.asarray(seed).copy();p[2:]=0.
        position=self.state[0].copy();position[:2]+=p[:2]*SCALE[:2]
        tx=(self.transmitters[2]-self.origin)@self.rotation.T
        rx=(self.receiver-self.origin)@self.rotation.T
        gradient=sum((position-a)/np.linalg.norm(position-a) for a in (tx,rx))[:2]
        direction=gradient/(gradient@gradient)
        delay_grid=np.arange(-225.,225.1,C/FS/2);delay_scores=[]
        cleaned=[]
        for c,(f,train,q) in enumerate(zip(self.fields,self.training,self.q)):
            use=np.repeat(train,len(f.offsets));clean=y[c][use]-q@(q.conj().T@y[c][use])
            cleaned.append(clean)
        for delay in delay_grid:
            trial=p.copy();trial[:2]+=direction*delay/SCALE[:2];score=0.
            for c,(f,train,clean) in enumerate(zip(self.fields,self.training,cleaned)):
                h=self.templates(trial,c)[:,0].reshape(f.y.shape)[train]
                z=clean.reshape(-1,len(f.offsets))
                score+=np.sum(abs(np.sum(h.conj()*z,axis=1))**2/np.maximum(np.sum(abs(h)**2,axis=1),1e-100))
            delay_scores.append(score)
        delay_indices=[]
        for i in np.argsort(-np.asarray(delay_scores)):
            if all(abs(delay_grid[i]-delay_grid[j])>C/FS for j in delay_indices):delay_indices.append(i)
            if len(delay_indices)==3:break
        if constant_velocity and np.any(self.state[2]):
            raise ValueError('Constant-velocity acquisition requires a zero-acceleration reference')
        if path_rate_halfwidth_mps<=0 or path_acceleration_halfwidth_mps2<=0:
            raise ValueError('Positive finite chirp-search widths are required')
        acceleration_grid = np.zeros(1) if constant_velocity else np.linspace(-path_acceleration_halfwidth_mps2,path_acceleration_halfwidth_mps2,2*int(np.ceil(path_acceleration_halfwidth_mps2/.001))+1)
        rates=np.linspace(-path_rate_halfwidth_mps,path_rate_halfwidth_mps,2*int(np.ceil(path_rate_halfwidth_mps/.02))+1)
        rate, acceleration=np.meshgrid(rates,acceleration_grid,indexing='ij')
        rate,acceleration=rate.ravel(),acceleration.ravel();selected=[]
        for di in delay_indices:
            base=p.copy();base[:2]+=direction*delay_grid[di]/SCALE[:2];score=np.zeros(rate.size)
            for c,(f,train,q,clean) in enumerate(zip(self.fields,self.training,self.q,cleaned)):
                use=np.repeat(train,len(f.offsets));dt=(f.time-self.midpoint)[train]
                h=self.templates(base,c)[:,0][use].reshape(-1,len(f.offsets))
                score+=chirp_scores(h,clean.reshape(h.shape),q,dt,f.k,rate,acceleration)
            local=[]
            for i in np.argsort(-score):
                if any(abs(rate[i]-rate[j])<.06 and abs(acceleration[i]-acceleration[j])<.004 for j in local):continue
                seed=base.copy();seed[2:4]=direction*rate[i]/SCALE[2:4];seed[4:6]=direction*acceleration[i]/SCALE[4:6]
                selected.append(dict(parameters=seed,path_rate_mps=rate[i],path_acceleration_mps2=acceleration[i],
                    delay_change_m=delay_grid[di],score=score[i]));local.append(i)
                if len(local)==count:break
        return sorted(selected,key=lambda v:-v['score'])[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first', type=float, default=149.)
    parser.add_argument('--last', type=float, default=179.)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--max-evaluations', type=int, default=70)
    parser.add_argument('--image-side', type=int, default=31)
    parser.add_argument('--acquire', action='store_true')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); out = args.output
    fields, rotation, rx, tx, state, mid = load_fields(args.first, args.last, args.stride)
    experiment = Experiment(fields, rotation, rx, tx, state, mid)
    truth = np.array([25., -40., .07, -.06, .002, -.001])/SCALE
    gains = np.array([1., .8*np.exp(.4j), 1.1*np.exp(-.8j), .9*np.exp(1.2j), .7*np.exp(-.3j)])
    # Fixed signal level: 10 dB mean matched information per retained epoch in each band.
    gains *= [np.sqrt(10*len(f.time)/np.vdot(experiment.templates(truth,c), experiment.templates(truth,c)).real)
              for c,f in enumerate(fields)]
    rng = np.random.default_rng(20261017)
    clean = [experiment.independent_signal(truth,c,[a]) for c,a in enumerate(gains)]
    agreement = [np.linalg.norm(z-experiment.templates(truth,c)[:,0]*gains[c])/np.linalg.norm(z) for c,z in enumerate(clean)]
    if max(agreement)>2e-6: raise ValueError(f'Independent forward validation failed: {agreement}')
    static = [d@(rng.normal(size=d.shape[1])+1j*rng.normal(size=d.shape[1]))*.0002 for d in experiment.designs]
    noise = [(rng.normal(size=z.size)+1j*rng.normal(size=z.size))/np.sqrt(2) for z in clean]
    result = dict(scope=__doc__, complete=False, production_changed=False, accepted_calibration=False,
        interval_s=[args.first,args.last], stride=args.stride, retained_epochs=[len(f.time) for f in fields],
        native_rows=[len(f.data['row_time']) for f in fields], truth_parameters=truth,
        truth_physical_correction=truth*SCALE, independent_forward_relative_error=agreement,
        limitations=['Planar quadratic translation; fixed known height; constant per-band point reflectivity.',
            'Native rows and waveform kernels are real; only selected complete time bins enter this diagnostic.',
            'Noise follows the existing conditional range covariance, not an asserted full interference model.',
            'Transmitter assignment 00002 is a test hypothesis, not a calibrated result.'], fits=[])
    def save():
        result['seconds']=time.monotonic()-started
        (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    save(); print('independent forward relative errors', agreement, flush=True)
    for name, data in [('noiseless', [z+s for z,s in zip(clean,static)]),
                       ('noisy', [z+s+n for z,s,n in zip(clean,static,noise)]),
                       ('recorded', [f.y.ravel().copy() for f in fields])]:
        for start, delta in enumerate((np.array([40.,-50.,.12,-.1,.003,-.002]),np.array([-60.,40.,-.1,.12,-.002,.003]))):
            seed = truth+delta/SCALE if name!='recorded' else delta/SCALE
            if args.acquire:
                acquisition=experiment.acquire(data,seed,count=1)
                seed=acquisition[0]['parameters'];print(name,start,'acquisition',serial(acquisition),flush=True)
            fit, pred = experiment.fit(data, seed, args.max_evaluations)
            fit.update(case=name,start=start,seed=seed)
            if name!='recorded':
                fit['physical_error']=(fit['parameters']-truth)*SCALE
                fit['signal_prediction_relative_error'] = [np.linalg.norm(p-z-s)/np.linalg.norm(z) for p,z,s in zip(pred,clean,static)]
            result['fits'].append(fit);save()
            print(name, start, 'RSS',fit['training_rss'],'gradient',fit['gradient'],'physical',fit['physical_correction'],flush=True)
    # Data-only information and an explicit near-null perturbation, no antenna priors.
    noiseless = [z+s for z,s in zip(clean,static)]
    _, j, _, _ = experiment.profile(truth, noiseless)
    _, singular, vh = np.linalg.svd(j, full_matrices=False)
    result['truth_information'] = dict(singular_values=singular, right_singular_vectors=vh,
        local_standard_deviation_physical=np.sqrt(np.diag(np.linalg.pinv(2*j.T@j,rcond=1e-16)))*SCALE)
    result['complete']=True;save()
    np.savez(out/'observations.npz', **{f'{kind}_{c}':z for kind,seq in [('clean',clean),('static',static),('noise',noise)] for c,z in enumerate(seq)})
    print('complete',result['seconds'],flush=True)


if __name__ == '__main__': main()
