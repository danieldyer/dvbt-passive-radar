"""Joint point-scatterer calibration with channel phases shared across targets.

Each target retains its own complex scattering coefficient and positive
per-channel magnitude. Only channel phase offsets transfer between targets.
This is a testable non-dispersive point-scatterer hypothesis, not an assumption
that arbitrary distributed targets have frequency-independent scattering.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_array

from complex_likelihood import projected_profile, profile
from fit_geometry import observations
from phase_fit import load_histories
from polar_fit import Problem

OTHER = np.array([0, 2, 3, 4])


def constant_path_direction(problem,state):
    """Local trajectory change for a constant one-metre excess-path shift.

    Polar radial controls already parameterize excess path. Road controls
    parameterize distance along the road, whose bistatic derivative varies.
    Solve that derivative before making a phase-ambiguity seed.
    """
    channel=problem.reference_channel
    d=problem.channels[channel]
    _,j=problem.model.paths(state,d['time'],np.full(len(d['time']),channel),problem.sites,True)
    return np.linalg.lstsq(j[:,:problem.model.radial_count],np.ones(len(d['time'])),rcond=None)[0]


def transport_path_shift(problem,state,shift):
    """Refine a finite ambiguity shift against exact geometric path lengths."""
    channel=problem.reference_channel;time=problem.channels[channel]['time']
    channels=np.full(len(time),channel);model=problem.model;n=model.radial_count
    base,_=model.paths(state,time,channels,problem.sites,True)
    result=state.copy();result[:n]+=constant_path_direction(problem,state)*shift
    low,high=model.bounds(True)
    result=np.clip(result,low+1e-8,high-1e-8)
    for _ in range(6):
        path,j=model.paths(result,time,channels,problem.sites,True)
        difference=base+shift-path
        if np.max(abs(difference))<1e-4:break
        result[:n]+=np.linalg.lstsq(j[:,:n],difference,rcond=None)[0]
        result[:n]=np.clip(result[:n],low[:n]+1e-8,high[:n]-1e-8)
    error=model.paths(result,time,channels,problem.sites,True)[0]-base-shift
    return result,float(np.max(abs(error)))


class CoherentProblem:
    def __init__(self, problems,range_bias=False,site_prior_sigma=None):
        self.problems = problems
        self.n = problems[0].model.count
        self.m = len(problems)
        self.beta_offset = 9+self.n*self.m
        self.magnitude_offset = self.beta_offset+4
        self.bias_offset=self.magnitude_offset+4*self.m
        self.range_bias=range_bias
        self.prior_weights=np.ones(9) if site_prior_sigma is None else np.tile([30,30,75],3)/np.broadcast_to(site_prior_sigma,(3,3)).ravel()
        self.count = self.bias_offset+(4 if range_bias else 0)
        self.data = [np.concatenate([pr.channel_data(ch) for ch in range(5)]) for pr in problems]

    def initial(self, states, corrections=None):
        p = np.zeros(self.count)
        if corrections is not None:
            p[:9] = corrections
        p[9:self.beta_offset] = np.concatenate(states)
        amps = []
        for i,pr in enumerate(self.problems):
            state = np.r_[states[i],p[:9]]
            a = []
            for ch,d in enumerate(pr.channels):
                u,_ = pr.channel_model(state,ch,True)
                _,amp = projected_profile(pr.channel_data(ch),u)
                a.append(amp)
            a = np.asarray(a)
            amps.append(a)
            p[self.magnitude_offset+4*i:self.magnitude_offset+4*(i+1)] = np.log(abs(a[OTHER]/a[1]))
        p[self.beta_offset:self.magnitude_offset] = np.angle(amps[0][OTHER]/amps[0][1])
        return p

    def bounds(self):
        bounds=[problem.model.bounds() for problem in self.problems]
        lo=np.concatenate([b[0] for b in bounds]);hi=np.concatenate([b[1] for b in bounds])
        return (np.r_[-3/self.prior_weights,lo,np.full(4,-100.),np.full(4*self.m,-5.),np.full(4 if self.range_bias else 0,-75.)],
                np.r_[3/self.prior_weights,hi,np.full(4,100.),np.full(4*self.m,5.),np.full(4 if self.range_bias else 0,75.)])

    def residual(self, p, jacobian=False, include_motion=True, include_site_prior=True):
        pieces = []
        for i,pr in enumerate(self.problems):
            offset = 9+i*self.n
            state = np.r_[p[offset:offset+self.n],p[:9]]
            magnitudes = np.ones(5)
            magnitudes[OTHER] = np.exp(p[self.magnitude_offset+4*i:self.magnitude_offset+4*(i+1)])
            beta = np.zeros(5); beta[OTHER] = p[self.beta_offset:self.magnitude_offset]
            biases=np.zeros(5)
            if self.range_bias:
                biases[OTHER]=p[self.bias_offset:]
            templates,derivatives = [],[]
            for ch,d in enumerate(pr.channels):
                base,local = pr.channel_model(state,ch,True,jacobian,biases[ch],self.range_bias)
                gain=magnitudes[ch]*np.exp(1j*beta[ch]);u=gain*base
                templates.append(u)
                if jacobian:
                    local=gain*local
                    du = np.zeros((len(u),self.count),complex)
                    du[:,offset:offset+self.n] = local[:,:self.n]
                    du[:,:9] = local[:,self.n:self.n+9]
                    if ch!=1:
                        c = int(np.flatnonzero(OTHER==ch)[0])
                        du[:,self.beta_offset+c] = 1j*u
                        du[:,self.magnitude_offset+4*i+c] = u
                        if self.range_bias:
                            du[:,self.bias_offset+c]=local[:,-1]
                    derivatives.append(du)
            result = projected_profile(self.data[i],np.concatenate(templates),
                                       np.concatenate(derivatives) if jacobian else None)
            r = result[1] if jacobian else result[0]
            pieces.append(np.r_[r.real,r.imag])
            for local in pr.auxiliary(state,True,jacobian,with_motion=include_motion)[:-1]:
                if jacobian:
                    out=np.zeros((len(local),self.count))
                    out[:,offset:offset+self.n]=local[:,:self.n];out[:,:9]=local[:,self.n:]
                    pieces.append(out)
                else:
                    pieces.append(local)
        if include_site_prior:
            pieces.append(np.c_[np.diag(self.prior_weights),np.zeros((9,self.count-9))] if jacobian else p[:9]*self.prior_weights)
        return np.concatenate(pieces)

    def fit(self, start, max_nfev=500, sparse=False, callback=None):
        def derivative(p):
            j=self.residual(p,True)
            return csr_array(j) if sparse else j
        return least_squares(self.residual,start,jac=derivative,
            bounds=self.bounds(),x_scale='jac',max_nfev=max_nfev,ftol=1e-8,xtol=1e-9,gtol=1e-5,
            tr_solver='lsmr' if sparse else 'exact',
            tr_options={'atol':1e-9,'btol':1e-9,'maxiter':1000} if sparse else {},callback=callback)

    def fit_columns(self,start,active,iterations):
        """Fit selected nuisance coordinates while calibration stays fixed."""
        from calibration_solver import solve
        def residual(p,jacobian=False):
            state=start.copy();state[active]=p
            result=self.residual(state,jacobian)
            return result[:,active] if jacobian else result
        lo,hi=self.bounds()
        fit=solve(residual,start[active],(lo[active],hi[active]),iterations)
        state=start.copy();state[active]=fit.x
        return fit,state

    def fixed_phase_starts(self,initial,shifts):
        """Exact-likelihood road/path proposals for one target at fixed phases."""
        if self.m!=1:raise ValueError('Fixed-phase path proposals require one target')
        local=np.r_[initial[9:self.beta_offset],initial[:9]]
        starts=[]
        for shift in shifts:
            moved,_=transport_path_shift(self.problems[0],local,float(shift))
            q=initial.copy();q[9:self.beta_offset]=moved[:self.n]
            r=self.residual(q);starts.append((float(r@r/2),q))
        return sorted(starts,key=lambda q:q[0])

    def fitted_observations(self, p):
        """Expected complex data, retaining each fitted nuisance amplitude."""
        values = []
        for i, pr in enumerate(self.problems):
            state = np.r_[p[9+i*self.n:9+(i+1)*self.n], p[:9]]
            magnitudes = np.ones(5)
            magnitudes[OTHER] = np.exp(p[self.magnitude_offset+4*i:self.magnitude_offset+4*(i+1)])
            beta = np.zeros(5); beta[OTHER] = p[self.beta_offset:self.magnitude_offset]
            biases = np.zeros(5)
            if self.range_bias:
                biases[OTHER] = p[self.bias_offset:]
            u = np.concatenate([magnitudes[ch]*np.exp(1j*beta[ch])*
                pr.channel_model(state,ch,True,range_bias=biases[ch])[0] for ch in range(5)])
            _, a = projected_profile(self.data[i], u)
            values.append(a*u)
        return values

    def constant_phase_starts(self, initial, count=32):
        """Cheap ambiguity search using rigid path shifts and shared RF phases.

        Shift the physical excess path, including when trajectory controls
        represent along-road distance. The full likelihood decides acceptance.
        """
        mapping = np.zeros((self.count,4+self.m))
        mapping[self.beta_offset:self.magnitude_offset,:4] = np.eye(4)
        directions=[];shift_bounds=[]
        for i,pr in enumerate(self.problems):
            start=9+i*self.n;n=pr.model.radial_count
            state=np.r_[initial[start:start+self.n],initial[:9]]
            direction=constant_path_direction(pr,state);directions.append(direction)
            mapping[start:start+n,4+i] = direction
            lo,hi=pr.model.bounds();active=abs(direction)>1e-10
            a=(lo[:n][active]-initial[start:start+n][active])/direction[active]
            b=(hi[:n][active]-initial[start:start+n][active])/direction[active]
            shift_bounds.append((max(-25.,float(np.minimum(a,b).max())),
                                 min(25.,float(np.maximum(a,b).min()))))
        # Cache the local path linearisation for this initial ambiguity search.
        entries = []
        for i,pr in enumerate(self.problems):
            state = np.r_[initial[9+i*self.n:9+(i+1)*self.n],initial[:9]]
            channel_entries = [];observed=[]
            for ch,d in enumerate(pr.channels):
                path,j = pr.model.paths(state,d['time'],np.full(len(d['time']),ch),pr.sites,True)
                u0,_ = pr.channel_model(state,ch,True)
                # Only the seed stage approximates a path shift by its mean.
                slope = np.mean(j[:,:pr.model.radial_count]@directions[i])
                c = None if ch==1 else int(np.flatnonzero(OTHER==ch)[0])
                magnitude = 1. if c is None else np.exp(initial[self.magnitude_offset+4*i+c])
                base_phase = 0. if c is None else initial[self.beta_offset+c]
                base=magnitude*u0;norm=np.sqrt(np.vdot(base,base).real)
                observed.append(np.vdot(base,pr.channel_data(ch))/norm)
                channel_entries.append((norm,slope*d['k'],c,base_phase))
            # For a constant phase rotation, each channel's projection onto
            # its fixed template is sufficient. The orthogonal data energy is
            # constant; avoid carrying thousands of unrelated cells into the
            # small ambiguity search. Final ranking uses the full likelihood.
            entries.append((channel_entries,np.asarray(observed)))
        def fun(x,jacobian=False):
            result = []
            for i,(channels,observed) in enumerate(entries):
                u,du = [],[]
                for base,k,c,beta in channels:
                    value = base*np.exp(1j*(beta+(0 if c is None else x[c])-k*x[4+i]))
                    u.append(value)
                    if jacobian:
                        j = np.zeros(len(x),complex)
                        if c is not None:
                            j[c] = 1j*value
                        j[4+i] = -1j*k*value
                        du.append(j)
                fit = projected_profile(observed,np.asarray(u),np.asarray(du) if jacobian else None)
                r = fit[1] if jacobian else fit[0]
                result.append(np.r_[r.real,r.imag])
            # Wideband envelope already constrains the allowed ambiguity shifts.
            result.append(np.c_[np.zeros((self.m,4)),np.eye(self.m)/10] if jacobian else x[4:]/10)
            return np.concatenate(result)
        rng = np.random.default_rng(4713)
        residual=self.residual(initial)
        trials = [(float(residual@residual/2),initial.copy())]
        low=np.array([v[0] for v in shift_bounds])+1e-7
        high=np.array([v[1] for v in shift_bounds])-1e-7
        if np.any(low>=high):raise ValueError('No interior constant-path ambiguity search interval')
        for i in range(count):
            start = np.r_[np.zeros(4),np.clip(np.zeros(self.m),low,high)] if i==0 else np.r_[rng.uniform(-np.pi,np.pi,4),rng.uniform(low,high)]
            fit = least_squares(fun,start,jac=lambda p:fun(p,True),
                bounds=(np.r_[np.full(4,-30.),low],np.r_[np.full(4,30.),high]),
                max_nfev=150,ftol=1e-7)
            proposed=initial+mapping@fit.x
            for target,pr in enumerate(self.problems):
                first=9+target*self.n
                local=np.r_[initial[first:first+self.n],initial[:9]]
                transported,_=transport_path_shift(pr,local,fit.x[4+target])
                proposed[first:first+self.n]=transported[:self.n]
            # The Fourier phase approximation only generates proposals. Rank
            # them with the complete range/phase likelihood before refinement.
            # A curved mapped road need not admit a perfectly constant path
            # translation. Rejecting an approximate proposal on that basis
            # loses legitimate phase branches; its exact likelihood already
            # accounts for every resulting path error.
            residual=self.residual(proposed)
            trials.append((float(residual@residual/2),proposed))
        distinct = []
        for item in sorted(trials,key=lambda item:item[0]):
            if not distinct or abs(item[0]-distinct[-1][0])>.01:
                distinct.append(item)
        return distinct


def information(problem, state):
    # At nonzero residual, the derivative of variable projection contains an
    # extra term involving the unfitted data. That is an objective derivative,
    # not expected Fisher information. Substitute the fitted mean before the
    # Jacobian/Schur calculation; noise weights and fitted amplitudes stay fixed.
    expected = copy.copy(problem)
    expected.data = problem.fitted_observations(state)
    result = {'construction':'Expected signal Jacobian with fitted gains and empirical covariance; nuisance parameters eliminated by orthogonal projection. Motion rows are reported separately.'}
    for include_motion,name in [(False,'data'),(True,'data_and_motion')]:
        j = expected.residual(state,True,include_motion,False)
        nuisance = j[:,9:]
        u,s,_ = np.linalg.svd(nuisance,full_matrices=False)
        u = u[:,s>s[0]*1e-10]
        projected = j[:,:9]-u@(u.T@j[:,:9])
        singular = np.linalg.svd(projected,compute_uv=False)
        result[name+'_normalized_singular_values'] = singular.tolist()
        result[name+'_posterior_covariance_normalized'] = np.linalg.inv(projected.T@projected+np.diag(problem.prior_weights**2)).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--histories',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--labels',nargs='+',default=['A','B','C','E'])
    parser.add_argument('--start',default='joint-polar-primary.json')
    parser.add_argument('--motion',nargs=3,type=float,default=[60.,.5,.2])
    parser.add_argument('--starts',type=int,default=40)
    parser.add_argument('--refine',type=int,default=4)
    parser.add_argument('--name',default='joint-coherent')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--patches',type=Path)
    parser.add_argument('--templates',type=Path)
    parser.add_argument('--range-bias',action='store_true',help='Test a shared per-channel range-envelope offset; SBS fixes the delay gauge')
    args = parser.parse_args()
    sites = np.array(json.loads((args.output/'geometry-prior.json').read_text())['enu_transmitters_m'])
    coarse = observations(args.histories,args.labels)
    problems=[]
    if args.patches:
        from range_fit import RangeProblem,load_fields
        from geographic_io import read_profiles
        if args.templates is None:
            raise ValueError('Range patches require a template-autocorrelation export')
        _,records=read_profiles(args.templates)
    for label,obs in zip(args.labels,coarse):
        channels,description=load_histories(args.histories,label,1.)
        if args.patches:
            problems.append(RangeProblem(channels,obs,sites,load_fields(args.patches,label,description,records),motion=args.motion))
        else:
            problems.append(Problem(channels,obs,sites,motion=args.motion))
    if args.range_bias and not args.patches:
        raise ValueError('Range biases are identifiable only with the range aperture')
    problem = CoherentProblem(problems,args.range_bias)
    previous = json.loads((args.output/args.start).read_text())
    old = np.array(previous['state']); old_labels = previous['labels']
    states = [old[9+old_labels.index(label)*problem.n:9+(old_labels.index(label)+1)*problem.n] for label in args.labels]
    start = problem.initial(states,old[:9])
    if args.resume:
        saved = json.loads(args.resume.read_text())
        previous_state=np.array(saved['best']['state'])
        if args.range_bias and len(previous_state)==problem.bias_offset:
            previous_state=np.r_[previous_state,np.zeros(4)]
        if len(previous_state)!=problem.count:
            raise ValueError('Resume model dimensions disagree')
        candidates = [(saved['best']['cost'],previous_state)]
    else:
        candidates = problem.constant_phase_starts(start,args.starts)
    print('Ambiguity seed costs',[x[0] for x in candidates[:10]],flush=True)
    trials = []
    for index,(_,initial) in enumerate(candidates[:args.refine]):
        fit = problem.fit(initial)
        trial = {'cost':fit.cost,'state':fit.x.tolist(),'converged':bool(fit.success),
                 'evaluations':fit.nfev,'termination':fit.message,
                 'transmitter_corrections_enu_m':(fit.x[:9].reshape(3,3)*[30,30,75]).tolist()}
        trials.append(trial)
        print('Fit',index,'cost',fit.cost,'converged',fit.success,'TX',trial['transmitter_corrections_enu_m'],flush=True)
        (args.output/(args.name+'.json')).write_text(json.dumps({'labels':args.labels,'trials':trials},indent=2))
    best = min(trials,key=lambda v:v['cost'])
    state = np.array(best['state'])
    result = {'labels':args.labels,'best':best,'trials':trials,'information':information(problem,state),
              'range_envelope_bias_m':(state[problem.bias_offset:].tolist() if args.range_bias else None),
              'range_envelope_bias_channels':OTHER.tolist(),
              'motion_rms_penalty_scales':args.motion,'status':'hypothesis test; not accepted calibration',
              'model':'shared channel phase offsets; target complex coefficients and independent positive channel magnitudes',
              'targets':[]}
    for i,(label,pr) in enumerate(zip(args.labels,problems)):
        local = np.r_[state[9+i*problem.n:9+(i+1)*problem.n],state[:9]]
        result['targets'].append({'label':label,'curve':pr.describe(local,True)})
    (args.output/(args.name+'.json')).write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
