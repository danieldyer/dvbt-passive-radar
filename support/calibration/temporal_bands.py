"""Shared channel calibration with finite-duration target scattering coefficients.

Geometry, kinematic continuity and channel calibration span the entire track.
Each non-overlapping observation block has one complex target reflectivity and
positive band magnitudes. Thus the target can fade without introducing an
independent phase correction in any channel. A constant-reflectivity control
uses exactly the same block observation operators and covariance.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from calibration_solver import solve
from coherent_fit import OTHER
from profiled_bands import ProfiledBandProblem,profile_bands
from road_fit import load_trial


class TemporalBandProblem(ProfiledBandProblem):
    def __init__(self,joint,seconds=3.,variable=True):
        super().__init__(joint)
        self.variable=variable;self.seconds=seconds;self.blocks=[];self.boundaries=[]
        if seconds<=0:raise ValueError('Positive scattering-block duration required')
        for source in joint.problems:
            start=source.model.seed['epoch']-source.model.half
            end=start+2*source.model.half
            # Equal blocks cover the complete interval, including any short
            # tail, without introducing a different boundary in each channel.
            boundaries=np.linspace(start,end,max(1,round((end-start)/seconds))+1)
            self.boundaries.append(boundaries)
            self.blocks.append([[field.windowed(lo,hi) for field in source.fields]
                                for lo,hi in zip(boundaries[:-1],boundaries[1:])])

    def block_models(self,p,index,jacobian=False):
        result=[]
        for fields in self.blocks[index]:
            models=[self.channel_model(p,index,ch,jacobian,field) for ch,field in enumerate(fields)]
            result.append(([f.y for f in fields],[u for u,_ in models],
                           [du for _,du in models] if jacobian else None))
        return result

    def target(self,p,index,jacobian=False):
        blocks=self.block_models(p,index,jacobian)
        if not self.variable:
            data=[np.concatenate([block[0][ch] for block in blocks]) for ch in range(5)]
            templates=[np.concatenate([block[1][ch] for block in blocks]) for ch in range(5)]
            derivatives=[np.concatenate([block[2][ch] for block in blocks]) for ch in range(5)] if jacobian else None
            return profile_bands(data,templates,derivatives)
        results=[profile_bands(*block) for block in blocks]
        if jacobian:
            return (np.concatenate([r[0] for r in results]),np.concatenate([r[1] for r in results]),
                    np.array([r[2] for r in results]))
        return np.concatenate([r[0] for r in results]),np.array([r[1] for r in results])

    def expanded_state(self,p):
        raise ValueError('Time-dependent gains cannot be represented as one complex coefficient per target')

    def geometry_state(self,p):
        biases=p[self.bias_offset:] if self.joint.range_bias else []
        return np.r_[p[:self.bias_offset],np.zeros(4*self.joint.m),biases]

    def diagnostics(self,p):
        beta=np.zeros(5);beta[OTHER]=p[self.joint.beta_offset:self.joint.magnitude_offset]
        targets=[]
        for index in range(self.joint.m):
            blocks=self.block_models(p,index)
            _,amplitudes=self.target(p,index)
            if not self.variable:amplitudes=np.tile(amplitudes,(len(blocks),1))
            entries=[]
            for block,(data,templates,_) in enumerate(blocks):
                norms=np.array([np.vdot(u,u).real for u in templates])
                independent=np.array([np.vdot(u,y)/n for u,y,n in zip(templates,data,norms)])
                # These templates already contain the shared channel phase.
                contrast=np.angle(independent*np.conj(independent[1]))
                energy=abs(independent)**2*norms
                entries.append({'start_s':float(self.boundaries[index][block]),
                    'end_s':float(self.boundaries[index][block+1]),
                    'phase_contrast_degrees':np.degrees(contrast).tolist(),
                    'independent_focused_energy':energy.tolist(),
                    'retained_amplitude_ratio':(abs(amplitudes[block])/np.maximum(abs(independent),1e-30)).tolist(),
                    'amplitudes':np.c_[amplitudes[block].real,amplitudes[block].imag].tolist()})
            targets.append(entries)
        return targets

    def expected_signal(self,p,all_band_powers=False):
        expected=copy.copy(self);expected.blocks=[]
        for index,blocks in enumerate(self.blocks):
            copied=[];_,amplitudes=self.target(p,index)
            if not self.variable:amplitudes=np.tile(amplitudes,(len(blocks),1))
            for block,fields in enumerate(blocks):
                values=[]
                if all_band_powers:
                    templates=[self.channel_model(p,index,ch,field=field)[0] for ch,field in enumerate(fields)]
                    independent=np.array([np.vdot(u,f.y)/np.vdot(u,u).real for u,f in zip(templates,fields)])
                    amplitudes[block]=abs(independent)*np.exp(1j*np.angle(independent[1]))
                for ch,source in enumerate(fields):
                    field=copy.copy(source)
                    u,_=self.channel_model(p,index,ch,field=field)
                    field.y=amplitudes[block,ch]*u;values.append(field)
                copied.append(values)
            expected.blocks.append(copied)
        return expected

    def information(self,p):
        expected=self.expected_signal(p)
        selected=np.r_[np.arange(9),np.arange(self.joint.beta_offset,self.count)]
        nuisance=np.arange(9,self.joint.beta_offset)
        result={'scope':'Expected signal Jacobian, with block scattering coefficients and target trajectories eliminated; conditional on the selected roads, empirical covariance and block model.',
                'order':'nine normalized transmitter coordinates, four shared phases, then four range origins if present'}
        for motion,prior,name in ((False,False,'data'),(True,True,'data_motion_prior')):
            j=expected.residual(p,True,motion,prior)
            u,s,_=np.linalg.svd(j[:,nuisance],full_matrices=False)
            u=u[:,s>s[0]*1e-10]
            projected=j[:,selected]-u@(u.T@j[:,selected])
            values,vectors=np.linalg.eigh(projected.T@projected)
            result[name+'_information_eigenvalues']=values.tolist()
            if prior and values[0]>values[-1]*1e-12:
                result['posterior_covariance_normalized']=((vectors/values)@vectors.T).tolist()
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fit','histories','patches','templates','routes','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=3.)
    parser.add_argument('--iterations',type=int,default=400)
    args=parser.parse_args();saved=json.loads(args.fit.read_text())
    joint=load_trial(saved,args.routes,args.histories,args.patches,args.templates)
    clock=time.monotonic()
    constant=TemporalBandProblem(joint,args.seconds,False)
    initial=constant.initial(np.array(saved['state']))
    def progress(fit):
        if fit.nfev%20==0:print('eval',fit.nfev,'cost',round(fit.cost,3),'elapsed',round(time.monotonic()-clock,1),flush=True)
    fitted_constant=solve(constant.residual,initial,constant.bounds(),args.iterations,progress)
    print('constant',fitted_constant.cost,fitted_constant.success,flush=True)
    variable=TemporalBandProblem(joint,args.seconds,True)
    fitted=solve(variable.residual,fitted_constant.x,variable.bounds(),args.iterations,progress)
    print('variable',fitted.cost,fitted.success,flush=True)
    result={**saved,'source_fit':str(args.fit),'amplitude_block_seconds':args.seconds,
        'state':variable.geometry_state(fitted.x).tolist(),'reduced_state':fitted.x.tolist(),
        'cost':float(fitted.cost),'converged':bool(fitted.success),'evaluations':fitted.nfev,
        'optimality':float(fitted.optimality),'termination':str(fitted.message),
        'site_corrections_enu_m':(fitted.x[:9].reshape(3,3)*[30,30,75]).tolist(),
        'range_bias_m':fitted.x[variable.bias_offset:].tolist(),
        'state_purpose':'Geometry, shared phase and range origins only. Magnitude slots are placeholders; the block amplitudes specify the scattering model.',
        'constant_control':{'cost':float(fitted_constant.cost),'converged':bool(fitted_constant.success),
                            'state':fitted_constant.x.tolist()},
        'block_diagnostics':variable.diagnostics(fitted.x),'information':variable.information(fitted.x),
        'status':'Offline scattering-model trial. Same block data/covariance for constant and varying controls; extra nuisance coefficients require held-out prediction tests. No calibration adopted.'}
    result.pop('zero_amplitude_channels_by_target',None)
    args.output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
