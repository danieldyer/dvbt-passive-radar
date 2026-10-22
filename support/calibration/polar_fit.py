"""Phase-centre trial using continuous excess-path and angular target motion."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from complex_likelihood import profile, operator_for, projected_profile
from fit_geometry import observations
from geometry import SITES, unit, observables
from phase_fit import load_histories
from polar_model import PolarTrajectory


class Problem:
    def __init__(self, channels, coarse, sites, reference_channel=1, motion=(60.,.5,.2), train=None):
        self.channels,self.coarse,self.sites = channels,coarse,sites
        self.reference_channel = reference_channel
        self.model = PolarTrajectory(channels[reference_channel]['seed'],reference_channel)
        self.motion = np.array(motion)
        self.operators = [operator_for(d,None if train is None else train[c]) for c,d in enumerate(channels)]
        self.coarse_time = coarse['t']+channels[0]['seed']['epoch']
        self.coarse_train = np.ones(len(self.coarse_time),dtype=bool)
        if train is not None:
            self.coarse_train = np.array([np.interp(t,channels[c]['time'],train[c].astype(float))>.5
                                         for t,c in zip(self.coarse_time,coarse['channel'])])

    def residual(self, p, free_sites=False, jacobian=False, columns=None, with_motion=True):
        pieces = []
        for c in range(len(self.channels)):
            u,j = self.channel_model(p,c,free_sites,jacobian)
            if columns is not None and j is not None:
                j = j[:,columns]
            result=projected_profile(self.channel_data(c),u,j)
            r=result[1] if jacobian else result[0]
            pieces.append(np.r_[r.real,r.imag])
        pieces.extend(self.auxiliary(p,free_sites,jacobian,columns,with_motion))
        return np.concatenate(pieces)

    def channel_data(self,channel):
        return self.operators[channel]@self.channels[channel]['value']

    def channel_model(self,p,channel,free_sites=False,jacobian=False,range_bias=0.,bias_derivative=False):
        if range_bias or bias_derivative:
            raise ValueError('A range-envelope bias requires range-aperture observations')
        d=self.channels[channel];q=self.operators[channel]
        path,j=self.model.paths(p,d['time'],np.full(len(d['time']),channel),self.sites,free_sites)
        u=np.exp(-1j*d['k']*(path-d['seed_delta']))
        return q@u,(q@(-1j*d['k']*u[:,None]*j) if jacobian else None)

    def auxiliary(self,p,free_sites=False,jacobian=False,columns=None,with_motion=True,with_coarse=True):
        pieces = []
        path,j = self.model.paths(p,self.coarse_time,self.coarse['channel'],self.sites,free_sites)
        if columns is not None:
            j = j[:,columns]
        if with_coarse:
            pieces.append((j/self.coarse['sigma_l'][:,None])[self.coarse_train] if jacobian else
                          ((path-self.coarse['length'])/self.coarse['sigma_l'])[self.coarse_train])
        if with_motion:
            values,derivatives = self.model.kinematics(p,self.sites,free_sites)
            for value,j,scale in zip(values,derivatives,self.motion):
                if columns is not None:
                    j = j[:,:,columns]
                pieces.append(j.reshape(-1,j.shape[-1])/(scale*np.sqrt(len(value))) if jacobian else
                              value.ravel()/(scale*np.sqrt(len(value))))
        if free_sites:
            j = np.c_[np.zeros((9,self.model.count)),np.eye(9)]
            pieces.append((j if columns is None else j[:,columns]) if jacobian else p[-9:])
        return pieces

    def initial_radial(self):
        d = self.channels[self.reference_channel]
        q = d['operator']; basis = d['basis']
        field = basis@np.linalg.lstsq(q@basis,q@d['value'],rcond=None)[0]
        phase = np.unwrap(np.angle(field))
        phase -= np.interp(self.model.seed['epoch'],d['time'],phase)
        _,spline = self.model.length_basis(d['time'])
        w = abs(field)/np.sqrt(d['variance'])
        return np.linalg.lstsq(spline*w[:,None],-phase/d['k']*w,rcond=None)[0]

    def multistart(self, max_nfev=50):
        count = self.model.count
        radial = self.initial_radial()
        columns = np.arange(self.model.radial_count,count)
        low,high = self.model.bounds()
        tx_angle = np.arctan2(self.sites[:,1].mean(),self.sites[:,0].mean())
        azimuths = np.unique(np.r_[np.linspace(-np.pi,np.pi,13)[:-1],tx_angle+np.array([-.3,-.1,.1,.3])])
        trials = []
        for azimuth in azimuths:
            for elevation in [-.03,.15,.4]:
                state = np.r_[radial,np.zeros(count-len(radial))]
                state[len(radial)] = azimuth
                state[len(radial)+self.model.angular_degree+1] = elevation
                def fun(angles,jacobian=False):
                    p = state.copy();p[columns] = angles
                    return self.residual(p,jacobian=jacobian,columns=columns)
                fit = least_squares(fun,state[columns],jac=lambda p:fun(p,True),
                                    bounds=(low[columns],high[columns]),x_scale='jac',max_nfev=max_nfev,
                                    ftol=1e-7,xtol=1e-8,gtol=1e-5)
                state[columns] = fit.x
                trials.append((fit.cost,state))
        trials.sort(key=lambda x:x[0])
        return trials

    def fit(self, start, free_sites=False, max_nfev=500):
        if free_sites and len(start)==self.model.count:
            start = np.r_[start,np.zeros(9)]
        return least_squares(lambda p:self.residual(p,free_sites),start,
            jac=lambda p:self.residual(p,free_sites,True),bounds=self.model.bounds(free_sites),
            x_scale='jac',max_nfev=max_nfev,ftol=1e-8,xtol=1e-9,gtol=1e-5)

    def describe(self, state, free_sites=False):
        time = self.model.seed['epoch']+np.linspace(-15,15,151)
        x,_,tx = self.model.positions(state,time,self.sites,free_sites)
        v = (self.model.positions(state,time+.001,self.sites,free_sites)[0]-
             self.model.positions(state,time-.001,self.sites,free_sites)[0])/.002
        value,_ = self.model.kinematics(state,self.sites,free_sites)
        channels = []
        for c,d in enumerate(self.channels):
            path,_ = self.model.paths(state,d['time'],np.full(len(d['time']),c),self.sites,free_sites)
            r,amp,u = profile(d['value'],d['operator'],-d['k']*(path-d['seed_delta']))
            snr = abs(amp)**2*np.linalg.norm(d['operator']@u)**2
            channels.append({'coherent_snr_db':float(10*np.log10(snr)),
                             'weighted_error':float(np.linalg.norm(r)**2/2),
                             'amplitude':[amp.real,amp.imag]})
        ct = self.coarse_time
        cx,_,_ = self.model.positions(state,ct,self.sites,free_sites)
        cv = (self.model.positions(state,ct+.001,self.sites,free_sites)[0]-
              self.model.positions(state,ct-.001,self.sites,free_sites)[0])/.002
        transmitter = tx[self.model.channel_sites[self.coarse['channel']]]+self.model.channel_offsets[self.coarse['channel']]
        length,rate = observables(cx,cv,transmitter,np.zeros(3))
        look = unit(x-tx.mean(axis=0))
        return {'position_enu_m':x.tolist(),'time':time.tolist(),
            'max_speed_mps':float(np.max(np.linalg.norm(v,axis=1))),
            'rms_acceleration_mps2':float(np.sqrt(np.mean(np.sum(value[1]**2,axis=1)))),
            'rms_jerk_mps3':float(np.sqrt(np.mean(np.sum(value[2]**2,axis=1)))),
            'range_rms_m':float(np.sqrt(np.mean((length-self.coarse['length'])**2))),
            'path_rate_rms_mps':float(np.sqrt(np.mean((rate-self.coarse['rate'])**2))),
            'measured_time':ct.tolist(),'measured_channel':self.coarse['channel'].tolist(),
            'measured_length_m':self.coarse['length'].tolist(),'predicted_length_m':length.tolist(),
            'measured_path_rate_mps':self.coarse['rate'].tolist(),'predicted_path_rate_mps':rate.tolist(),
            'midpoint_look_from_transmitter':look[len(look)//2].tolist(),'channels':channels}


def joint(problems, starts, max_nfev=700):
    n = problems[0].model.count
    p = np.r_[np.zeros(9),np.concatenate(starts)]
    def fun(p,jacobian=False):
        pieces = []
        for i,problem in enumerate(problems):
            state = np.r_[p[9+i*n:9+(i+1)*n],p[:9]]
            if jacobian:
                local = problem.residual(state,True,True)[:-9]
                j = np.zeros((len(local),len(p)))
                j[:,:9] = local[:,n:];j[:,9+i*n:9+(i+1)*n] = local[:,:n]
                pieces.append(j)
            else:
                pieces.append(problem.residual(state,True)[:-9])
        pieces.append(np.c_[np.eye(9),np.zeros((9,n*len(problems)))] if jacobian else p[:9])
        return np.concatenate(pieces)
    lo,hi = problems[0].model.bounds()
    lo,hi = np.r_[np.full(9,-3.),np.tile(lo,len(problems))],np.r_[np.full(9,3.),np.tile(hi,len(problems))]
    return least_squares(fun,p,jac=lambda p:fun(p,True),bounds=(lo,hi),x_scale='jac',
                         max_nfev=max_nfev,ftol=1e-8,xtol=1e-9,gtol=1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--histories',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--labels',nargs='+',default=['A','B','C','E','F'])
    parser.add_argument('--joint-only',action='store_true')
    parser.add_argument('--reuse',action='store_true',help='Reuse existing individual fits with the same motion scales')
    parser.add_argument('--motion',nargs=3,type=float,default=[60.,.5,.2])
    parser.add_argument('--name',default='joint-polar')
    parser.add_argument('--bandwidth-hz',type=float,default=1.)
    args = parser.parse_args()
    sites = np.array(json.loads((args.output/'geometry-prior.json').read_text())['enu_transmitters_m'])
    data = observations(args.histories,args.labels)
    problems,starts = [],[]
    for label,coarse in zip(args.labels,data):
        channels,_ = load_histories(args.histories,label,args.bandwidth_hz)
        problem = Problem(channels,coarse,sites,motion=args.motion)
        problems.append(problem)
        path = args.output/f'{label}-polar.json'
        if args.joint_only or (args.reuse and path.exists()):
            saved = json.loads(path.read_text())
            if not args.joint_only and saved['motion_rms_penalty_scales']!=args.motion:
                raise ValueError('Existing fit uses different motion scales')
            starts.append(np.array(saved['fixed_state']));continue
        trials = problem.multistart()
        fits = [problem.fit(state) for _,state in trials[:3]]
        fixed = min(fits,key=lambda f:f.cost)
        free = problem.fit(fixed.x,True)
        starts.append(fixed.x)
        result = {'label':label,'fixed_state':fixed.x.tolist(),'free_state':free.x.tolist(),
            'fixed_cost':fixed.cost,'free_cost_including_prior':free.cost,
            'fixed_converged':bool(fixed.success),'free_converged':bool(free.success),
            'fixed':problem.describe(fixed.x),'free':problem.describe(free.x,True),
            'transmitter_corrections_enu_m':(free.x[-9:].reshape(3,3)*[30,30,75]).tolist(),
            'model':'cubic excess-length spline, 1.5 s knots; quadratic receive azimuth/elevation; exact bistatic geometry',
            'motion_rms_penalty_scales':args.motion,
            'selected_doppler_half_width_hz':args.bandwidth_hz,
            'multistarts':[{'cost':cost,'state':state.tolist()} for cost,state in trials]}
        path.write_text(json.dumps(result,indent=2))
        print(label,'fixed/free cost',fixed.cost,free.cost,'fixed range/rate',result['fixed']['range_rms_m'],
              result['fixed']['path_rate_rms_mps'],'speed/a/j',result['fixed']['max_speed_mps'],
              result['fixed']['rms_acceleration_mps2'],result['fixed']['rms_jerk_mps3'],flush=True)
    fit = joint(problems,starts)
    n = problems[0].model.count
    result = {'labels':args.labels,'state':fit.x.tolist(),'converged':bool(fit.success),'cost':fit.cost,
        'selected_doppler_half_width_hz':args.bandwidth_hz,
        'termination':fit.message,'evaluations':fit.nfev,'motion_rms_penalty_scales':args.motion,
        'transmitter_corrections_enu_m':(fit.x[:9].reshape(3,3)*[30,30,75]).tolist(),
        'targets':[],'status':'diagnostic fit; no production metadata update'}
    for i,(label,problem) in enumerate(zip(args.labels,problems)):
        state = np.r_[fit.x[9+i*n:9+(i+1)*n],fit.x[:9]]
        result['targets'].append({'label':label,'curve':problem.describe(state,True)})
    (args.output/(args.name+'.json')).write_text(json.dumps(result,indent=2))
    print(args.name,'cost',fit.cost,'success',fit.success,'TX',result['transmitter_corrections_enu_m'],flush=True)


if __name__=='__main__':
    main()
