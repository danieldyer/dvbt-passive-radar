"""Withheld-vehicle and withheld-time predictions for block scattering models."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from calibration_solver import solve
from coherent_fit import OTHER,transport_path_shift
from complex_likelihood import projected_profile
from road_fit import load_trial
from road_validation import related_intervals,split_problem,subset
from temporal_bands import TemporalBandProblem


class IndependentBlocks(TemporalBandProblem):
    def target(self,p,index,jacobian=False):
        results=[];amplitudes=[]
        for data,templates,derivatives in self.block_models(p,index,jacobian):
            block=[]
            for ch,(y,u) in enumerate(zip(data,templates)):
                result=projected_profile(y,u,derivatives[ch] if jacobian else None)
                results.append(result);block.append(result[-1])
            amplitudes.append(block)
        if jacobian:
            return np.concatenate([r[0] for r in results]),np.concatenate([r[1] for r in results]),np.array(amplitudes)
        return np.concatenate([r[0] for r in results]),np.array(amplitudes)


def predict(train,test,state):
    _,amplitudes=train.target(state,0)
    prediction=0.;focused=0.;blocks=[]
    for block,(data,templates,_) in enumerate(test.block_models(state,0)):
        model=np.concatenate([a*u for a,u in zip(amplitudes[block],templates)])
        observed=np.concatenate(data)
        prediction+=float(np.vdot(model-observed,model-observed).real)
        if np.vdot(model,model).real>1e-20:
            _,a=projected_profile(observed,model)
            focused+=float(abs(a)**2*np.vdot(model,model).real)
        energy=[];coefficients=[]
        for y,u in zip(data,templates):
            _,a=projected_profile(y,u)
            energy.append(float(abs(a)**2*np.vdot(u,u).real));coefficients.append(a)
        coefficients=np.array(coefficients)
        blocks.append({'phase_contrast_degrees':np.degrees(np.angle(coefficients*np.conj(coefficients[1]))).tolist(),
                       'independent_band_energy':energy,
                       'training_amplitudes':np.c_[amplitudes[block].real,amplitudes[block].imag].tolist()})
    return {'prediction_error':prediction,'focused_energy':focused,'blocks':blocks}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fit','histories','patches','templates','routes','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--labels',nargs='+')
    parser.add_argument('--iterations',type=int,default=350)
    args=parser.parse_args();saved=json.loads(args.fit.read_text())
    seconds=saved['amplitude_block_seconds']
    joint=load_trial(saved,args.routes,args.histories,args.patches,args.templates,allow_time_dependent=True)
    state=np.array(saved['state'])
    result={'source_fit':str(args.fit),'block_seconds':seconds,'targets':[],
        'scope':'A whole vehicle is excluded from calibration (possible E/O fragments together). Its path and block coefficients fit alternating one-second training intervals. Prediction errors use those same training coefficients on the withheld samples. Focused energies instead profile one common amplitude per held-out block; independent control profiles five. Road discovery and initialization still condition on the complete capture. No calibration adopted.'}
    if args.output.exists():
        result=json.loads(args.output.read_text())
        if result['source_fit']!=str(args.fit):raise ValueError('Validation output belongs to a different trial')
    completed={q['label'] for q in result['targets']};clock=time.monotonic()
    for index,label in enumerate(saved['labels']):
        if label in completed or (args.labels and label not in args.labels):continue
        excluded=[i for i in range(joint.m) if i==index or related_intervals(joint.problems[index],joint.problems[i])]
        others,initial=subset(joint,state,[i for i in range(joint.m) if i not in excluded])
        training=TemporalBandProblem(others,seconds)
        fit=solve(training.residual,training.initial(initial),training.bounds(),args.iterations)
        calibrated=training.geometry_state(fit.x)
        one,initial=subset(joint,state,[index])
        initial[:9]=calibrated[:9]
        initial[one.beta_offset:one.magnitude_offset]=calibrated[others.beta_offset:others.magnitude_offset]
        if one.range_bias:initial[one.bias_offset:]=calibrated[others.bias_offset:]
        # Start with the selected independent road trajectory, not a path
        # trained on the withheld samples under the shared calibration.
        initial[9:one.beta_offset]=saved['selected_roads'][index]['state']
        models=[]
        for parity in (0,1):
            copy_joint=copy.copy(one)
            copy_joint.problems=[split_problem(one.problems[0],parity)]
            models.append(TemporalBandProblem(copy_joint,seconds))
        train,test=models
        reduced=train.initial(initial);active=np.arange(9,one.beta_offset)
        starts=[]
        for shift in np.linspace(-20,20,81):
            local=np.r_[initial[9:one.beta_offset],initial[:9]]
            moved,_=transport_path_shift(one.problems[0],local,shift)
            p=reduced.copy();p[9:one.beta_offset]=moved[:one.n]
            r=train.residual(p);starts.append((float(r@r/2),p))
        starts.sort(key=lambda q:q[0])
        fitted=[train.fit_columns(p,active,args.iterations) for _,p in starts[:3]]
        target_fit,target_state=min(fitted,key=lambda q:q[0].cost)
        independent_train=IndependentBlocks(train.joint,seconds)
        independent_test=IndependentBlocks(test.joint,seconds)
        candidates=[independent_train.fit_columns(p,active,args.iterations) for p in (target_state,reduced)]
        independent_fit,independent_state=min(candidates,key=lambda q:q[0].cost)
        shared=predict(train,test,target_state)
        independent=predict(independent_train,independent_test,independent_state)
        # Independent complex amplitudes are separately profiled in each
        # withheld band, rather than reusing the shared coefficient statistic.
        independent_energy=sum(sum(b['independent_band_energy']) for b in independent['blocks'])
        entry={'label':label,'excluded_labels':[saved['labels'][i] for i in excluded],
            'calibration_converged':bool(fit.success),'target_converged':bool(target_fit.success),
            'independent_converged':bool(independent_fit.success),
            'calibration_state':fit.x.tolist(),'target_state':target_state.tolist(),
            'independent_state':independent_state.tolist(),'shared':shared,'independent':independent,
            'heldout_loss_db':float(10*np.log10(independent_energy/shared['focused_energy'])),
            'prediction_error_reduction_fraction':float(1-shared['prediction_error']/independent['prediction_error'])}
        result['targets'].append(entry)
        args.output.write_text(json.dumps(result,indent=2))
        print(label,'loss',round(entry['heldout_loss_db'],3),'prediction change',round(entry['prediction_error_reduction_fraction'],4),
              'converged',fit.success,target_fit.success,independent_fit.success,'elapsed',round(time.monotonic()-clock,1),flush=True)


if __name__=='__main__':main()
