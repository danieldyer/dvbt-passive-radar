"""Joint receiver-position nuisance fit for offline road/antenna calibration.

Road coordinates remain fixed in the original Earth-fixed local frame. The
receiver moves in that frame, including in the direct-path reference distance;
this is not an independent per-channel delay or phase adjustment.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from calibration_solver import solve
from coherent_fit import OTHER
from complex_likelihood import projected_profile
from road_fit import load_trial
from profiled_bands import ProfiledBandProblem


class ProfiledReceiverProblem(ProfiledBandProblem):
    """Physical receiver uncertainty with the same exact band-gain likelihood."""
    def __init__(self, joint, sigma, channel_sigma=None):
        if joint.range_bias:
            raise ValueError('Use physical receiver delays without independent range biases')
        copied=copy.copy(joint);copied.problems=[]
        for source in joint.problems:
            problem=copy.copy(source);problem.model=copy.copy(source.model)
            copied.problems.append(problem)
        super().__init__(copied)
        self.offset=self.count;self.sigma=np.asarray(sigma)
        self.channel_sigma=None if channel_sigma is None else np.asarray(channel_sigma)
        for width in (self.sigma,self.channel_sigma):
            if width is not None and (width.shape!=(3,) or not np.all(np.isfinite(width)&(width>0))):
                raise ValueError('Receiver coordinate prior widths must be three finite positive values')
        self.receiver_count=3 if self.channel_sigma is None else 15
        self.count+=self.receiver_count

    def coordinates(self,p):
        common=p[self.offset:self.offset+3]*self.sigma
        offsets=np.zeros((5,3))
        if self.channel_sigma is not None:
            offsets[OTHER]=p[self.offset+3:].reshape(4,3)*self.channel_sigma
        return common,offsets

    def initial(self,state,common=None,offsets=None):
        base=super().initial(state)
        common=np.zeros(3) if common is None else np.asarray(common)
        result=np.r_[base,common/self.sigma]
        if self.channel_sigma is not None:
            offsets=np.zeros((5,3)) if offsets is None else np.asarray(offsets)
            if offsets.shape!=(5,3) or np.any(offsets[1]!=0):
                raise ValueError('Receiver offsets use SBS as their fixed spatial reference')
            result=np.r_[result,(offsets[OTHER]/self.channel_sigma).ravel()]
        return result

    def channel_model(self,p,index,channel,jacobian=False,field=None):
        problem=self.joint.problems[index]
        common,offsets=self.coordinates(p)
        problem.model.receiver_enu_m=common
        problem.model.receiver_channel_offsets_enu_m=offsets
        u,j=super().channel_model(p,index,channel,jacobian,field)
        if jacobian:
            first=9+index*self.joint.n
            local=np.r_[p[first:first+self.joint.n],p[:9]]
            f=problem.fields[channel] if field is None else field
            channels=np.full(len(f.time),channel)
            path,_=problem.model.paths(local,f.time,channels,problem.sites,True)
            derivative=problem.model.receiver_derivative(local,f.time,channels,problem.sites,True)
            _,columns=f.template(path,derivative)
            beta=0 if channel==1 else p[self.joint.beta_offset+int(np.flatnonzero(OTHER==channel)[0])]
            columns*=np.exp(1j*beta)
            j[:,self.offset:self.offset+3]=columns*self.sigma
            if self.channel_sigma is not None and channel!=1:
                first=self.offset+3+3*int(np.flatnonzero(OTHER==channel)[0])
                j[:,first:first+3]=columns*self.channel_sigma
        return u,j

    def residual(self,p,jacobian=False,include_motion=True,include_site_prior=True):
        result=super().residual(p,jacobian,include_motion,include_site_prior)
        prior=np.c_[np.zeros((self.receiver_count,self.offset)),np.eye(self.receiver_count)] if jacobian else p[self.offset:]
        return np.r_[result,prior]

    def bounds(self):
        low,high=super().bounds()
        return np.r_[low,np.full(self.receiver_count,-3.)],np.r_[high,np.full(self.receiver_count,3.)]

    def information(self,p):
        expected=copy.copy(self);expected.joint=copy.copy(self.joint);expected.joint.problems=[]
        for index,source in enumerate(self.joint.problems):
            problem=copy.copy(source);problem.model=copy.copy(source.model);problem.fields=[]
            _,amplitudes=self.target(p,index)
            for channel,source_field in enumerate(source.fields):
                field=copy.copy(source_field)
                field.y=amplitudes[channel]*self.channel_model(p,index,channel)[0]
                problem.fields.append(field)
            expected.joint.problems.append(problem)
        selected=np.r_[np.arange(9),np.arange(self.joint.beta_offset,self.offset),np.arange(self.offset,self.count)]
        nuisance=np.arange(9,self.joint.beta_offset)
        result={'construction':'Expected signal Jacobian with exact nonnegative band gains profiled. Conditional on roads, scattering model and declared receiver prior.',
                'order':'nine normalized mast corrections, four band phases, three normalized common receiver corrections'+
                    (', then twelve normalized receive-centre offsets relative to SBS in Seven/Nine/Ten/ABC order' if self.channel_sigma is not None else '')}
        for with_motion,name in ((False,'data_receiver_prior'),(True,'data_motion_receiver_prior')):
            j=expected.residual(p,True,with_motion,False)
            u,s,_=np.linalg.svd(j[:,nuisance],full_matrices=False);u=u[:,s>s[0]*1e-10]
            reduced=j[:,selected]-u@(u.T@j[:,selected])
            information=reduced.T@reduced
            result[name+'_information_eigenvalues']=np.linalg.eigvalsh(information).tolist()
            information[:9,:9]+=np.diag(self.joint.prior_weights**2)
            eigenvalues=np.linalg.eigvalsh(information)
            rank=int(np.count_nonzero(eigenvalues>eigenvalues[-1]*1e-12))
            result[name+'_posterior_rank']=rank
            result[name+'_posterior_covariance_normalized']=(np.linalg.inv(information).tolist()
                if rank==len(information) else None)
        return result


class ReceiverProblem:
    def __init__(self,joint,sigma):
        if joint.range_bias:raise ValueError('Receiver calibration uses physical delays, without extra band-delay biases')
        self.joint=copy.copy(joint);self.joint.problems=[]
        for source in joint.problems:
            problem=copy.copy(source);problem.model=copy.copy(source.model)
            self.joint.problems.append(problem)
        self.sigma=np.asarray(sigma)
        self.count=joint.count+3

    def apply(self,state):
        for problem in self.joint.problems:
            problem.model.receiver_enu_m=state[-3:]*self.sigma

    def receiver_columns(self,state,include_motion,include_site_prior):
        p=state[:-3];pieces=[]
        for i,problem in enumerate(self.joint.problems):
            local=np.r_[p[9+i*self.joint.n:9+(i+1)*self.joint.n],p[:9]]
            gain=np.ones(5,dtype=complex)
            gain[OTHER]=np.exp(p[self.joint.magnitude_offset+4*i:self.joint.magnitude_offset+4*(i+1)]+
                1j*p[self.joint.beta_offset:self.joint.magnitude_offset])
            templates=[];derivatives=[]
            for ch,field in enumerate(problem.fields):
                channels=np.full(len(field.time),ch)
                path,_=problem.model.paths(local,field.time,channels,problem.sites,True)
                derivative=problem.model.receiver_derivative(local,field.time,channels,problem.sites,True)*self.sigma
                u,j=field.template(path,derivative)
                templates.append(gain[ch]*u);derivatives.append(gain[ch]*j)
            _,j,_=projected_profile(self.joint.data[i],np.concatenate(templates),np.concatenate(derivatives))
            pieces.append(np.r_[j.real,j.imag])
            for rows in problem.auxiliary(local,True,with_motion=include_motion)[:-1]:
                pieces.append(np.zeros((len(rows),3)))
        if include_site_prior:pieces.append(np.zeros((9,3)))
        return np.concatenate(pieces)

    def residual(self,state,jacobian=False,include_motion=True,include_site_prior=True):
        self.apply(state)
        base=self.joint.residual(state[:-3],jacobian,include_motion,include_site_prior)
        if jacobian:
            combined=np.c_[base,self.receiver_columns(state,include_motion,include_site_prior)]
            return np.r_[combined,np.c_[np.zeros((3,self.joint.count)),np.eye(3)]]
        return np.r_[base,state[-3:]]

    def bounds(self):
        low,high=self.joint.bounds()
        return np.r_[low,np.full(3,-3.)],np.r_[high,np.full(3,3.)]

    def information(self,state):
        self.apply(state)
        expected=copy.copy(self);expected.joint=copy.copy(self.joint)
        expected.joint.data=self.joint.fitted_observations(state[:-3])
        result={'construction':'Expected complex signal Jacobian with shared receiver uncertainty marginalized alongside trajectories, gains and band phases. The stated receiver prior is included in both cases; the site prior is added after nuisance projection.'}
        for motion,name in ((False,'data'),(True,'data_and_motion')):
            j=expected.residual(state,True,motion,False)
            u,s,_=np.linalg.svd(j[:,9:],full_matrices=False);u=u[:,s>s[0]*1e-10]
            projected=j[:,:9]-u@(u.T@j[:,:9])
            result[name+'_normalized_singular_values']=np.linalg.svd(projected,compute_uv=False).tolist()
            result[name+'_posterior_covariance_normalized']=np.linalg.inv(projected.T@projected+
                np.diag(self.joint.prior_weights**2)).tolist()
        j=expected.residual(state,True,True,True)
        selected=np.r_[np.arange(9),np.arange(self.joint.beta_offset,self.joint.magnitude_offset),
                       np.arange(self.joint.count,self.count)]
        u,s,_=np.linalg.svd(np.delete(j,selected,axis=1),full_matrices=False);u=u[:,s>s[0]*1e-10]
        projected=j[:,selected]-u@(u.T@j[:,selected])
        result['joint_calibration_covariance']=np.linalg.inv(projected.T@projected).tolist()
        result['joint_covariance_order']='nine normalized transmitter corrections, four phase offsets relative to SBS, three normalized receiver corrections'
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fit','histories','patches','templates','routes','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--receiver-sigma',nargs=3,type=float,default=[15.,15.,7.5])
    parser.add_argument('--channel-sigma',nargs=3,type=float,
                        help='Test fixed per-band receive phase-centre offsets, prior standard deviations in metres relative to SBS')
    parser.add_argument('--iterations',type=int,default=400)
    args=parser.parse_args();saved=json.loads(args.fit.read_text())
    joint=load_trial(saved,args.routes,args.histories,args.patches,args.templates)
    profiled=saved.get('profiled_band_gains',False)
    channel_sigma=args.channel_sigma or saved.get('receiver_channel_prior_sigma_m')
    if channel_sigma is not None and not profiled:
        raise ValueError('Frequency-dependent receiver centres require the exact profiled-gain likelihood')
    problem=(ProfiledReceiverProblem(joint,args.receiver_sigma,channel_sigma) if profiled else ReceiverProblem(joint,args.receiver_sigma))
    common=np.array(saved.get('receiver_correction_enu_m',[0.,0.,0.]))
    offsets=np.array(saved.get('receiver_channel_offsets_enu_m',np.zeros((5,3))))
    initial=(problem.initial(saved['state'],common,offsets) if profiled else np.r_[saved['state'],common/problem.sigma])
    def progress(fit):
        if fit.nfev%10==0:
            common=problem.coordinates(fit.x)[0] if profiled else fit.x[-3:]*problem.sigma
            print('receiver fit',fit.nfev,round(fit.cost,3),common.tolist(),flush=True)
    fit=solve(problem.residual,initial,problem.bounds(),args.iterations,progress)
    state,absent=problem.expanded_state(fit.x) if profiled else (fit.x[:-3],saved.get('zero_amplitude_channels_by_target'))
    common,offsets=problem.coordinates(fit.x) if profiled else (fit.x[-3:]*problem.sigma,offsets)
    result={**saved,'state':state.tolist(),'source_fit':str(args.fit),
        'receiver_reduced_state':fit.x.tolist(),'zero_amplitude_channels_by_target':absent,
        'receiver_prior_sigma_m':problem.sigma.tolist(),'receiver_correction_enu_m':common.tolist(),
        'site_corrections_enu_m':(fit.x[:9].reshape(3,3)*[30,30,75]).tolist(),
        'cost':float(fit.cost),'converged':bool(fit.success),'evaluations':fit.nfev,
        'optimality':float(fit.optimality),'termination':str(fit.message),'information':problem.information(fit.x),
        'status':'shared receiver and antenna calibration trial; held-out and alternative-branch validation required'}
    if channel_sigma is not None:
        result.update(receiver_channel_prior_sigma_m=list(channel_sigma),
            receiver_channel_offsets_enu_m=offsets.tolist(),receiver_reference_channel=1,
            receiver_channel_prior_scope='Independent Gaussian offsets relative to SBS, shared across every target. Declared sensitivity prior, not a measured antenna model. Exact echo and direct-reference paths use each band receiver centre.')
    args.output.write_text(json.dumps(result,indent=2))
    print('complete',fit.cost,fit.success,result['receiver_correction_enu_m'],flush=True)


if __name__=='__main__':main()
