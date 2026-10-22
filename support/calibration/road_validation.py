"""Withhold a complete vehicle from calibration, then withhold its time blocks.

The selected road and initial phase branch are conditional on discovery in the
full capture. Shared antenna/channel parameters are refitted without the test
vehicle. Its trajectory and magnitudes use even one-second blocks; odd blocks
test their prediction. Independent complex band responses are the control.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from calibration_solver import solve
from coherent_fit import CoherentProblem,OTHER
from complex_likelihood import projected_profile
from range_fit import Field
from road_fit import load_trial
from receiver_fit import ReceiverProblem,ProfiledReceiverProblem
from profiled_bands import ProfiledBandProblem
from antenna_offsets import AntennaProblem


def related_intervals(first,second):
    """Conservatively group overlapping detections inside one range/Doppler cell.

    This only prevents validation leakage between possible fragments of one
    vehicle. It does not declare a physical association or change its samples.
    """
    a,b=first.model.seed,second.model.seed
    start=max(a['epoch']-first.model.half,b['epoch']-second.model.half)
    end=min(a['epoch']+first.model.half,b['epoch']+second.model.half)
    if end<=start:return False
    time=np.linspace(start,end,21)
    paths=[];dopplers=[]
    for seed in (a,b):
        u=(time-seed['epoch'])/seed.get('scale',15.)
        paths.append(np.polynomial.polynomial.polyval(u,seed['path']))
        dopplers.append(-seed['rf_hz']/299792458/seed.get('scale',15.)*
            np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyder(seed['path'])))
    return bool(np.mean((abs(paths[0]-paths[1])<2*299792458/8e6)&
                       (abs(dopplers[0]-dopplers[1])<2.))>.5)


def subset(problem,state,indices):
    result=CoherentProblem([problem.problems[i] for i in indices],
        range_bias=problem.range_bias,
        site_prior_sigma=(np.tile([30,30,75],3)/problem.prior_weights).reshape(3,3))
    p=np.zeros(result.count);p[:9]=state[:9]
    for new,old in enumerate(indices):
        p[9+new*result.n:9+(new+1)*result.n]=state[9+old*problem.n:9+(old+1)*problem.n]
        p[result.magnitude_offset+4*new:result.magnitude_offset+4*(new+1)]=state[problem.magnitude_offset+4*old:problem.magnitude_offset+4*(old+1)]
    p[result.beta_offset:result.magnitude_offset]=state[problem.beta_offset:problem.magnitude_offset]
    if problem.range_bias:p[result.bias_offset:]=state[problem.bias_offset:]
    return result,p


def split_problem(problem,parity):
    result=copy.copy(problem);result.fields=[]
    start=problem.model.seed['epoch']-problem.model.half
    for field in problem.fields:
        selected=np.floor(field.time-start).astype(int)%2==parity
        result.fields.append(Field(field.data,field.pulse,field.seed,selected=selected))
    return result


def templates(problem,state):
    local=np.r_[state[9:problem.beta_offset],state[:9]]
    gains=np.ones(5,dtype=complex)
    gains[OTHER]=np.exp(state[problem.magnitude_offset:problem.bias_offset]+1j*state[problem.beta_offset:problem.magnitude_offset])
    biases=np.zeros(5)
    if problem.range_bias:biases[OTHER]=state[problem.bias_offset:]
    values=[problem.problems[0].channel_model(local,ch,True,range_bias=biases[ch])[0] for ch in range(5)]
    return values,gains


def evaluate(train,test,state,independent_train,independent_test,independent_state):
    u,gain=templates(train,state);v,_=templates(test,state)
    training=np.concatenate([a*b for a,b in zip(u,gain)])
    testing=np.concatenate([a*b for a,b in zip(v,gain)])
    _,amplitude=projected_profile(train.data[0],training)
    _,test_amplitude=projected_profile(test.data[0],testing)
    coherent=float(abs(test_amplitude)**2*np.vdot(testing,testing).real)
    prediction=float(np.vdot(amplitude*testing-test.data[0],amplitude*testing-test.data[0]).real)
    independent=[];same_path=[];same_path_amplitudes=[]
    for ch in range(5):
        it,_=independent_train.channel_model(independent_state,ch)
        iv,_=independent_test.channel_model(independent_state,ch)
        _,a=projected_profile(independent_train.channel_data(ch),it)
        _,b=projected_profile(independent_test.channel_data(ch),iv)
        error=a*iv-independent_test.channel_data(ch)
        independent.append({'focused_energy':float(abs(b)**2*np.vdot(iv,iv).real),
            'prediction_error':float(np.vdot(error,error).real)})
        _,b=projected_profile(test.problems[0].channel_data(ch),v[ch])
        same_path.append(float(abs(b)**2*np.vdot(v[ch],v[ch]).real))
        same_path_amplitudes.append(b)
    independent_energy=sum(x['focused_energy'] for x in independent)
    amps=np.asarray(same_path_amplitudes)
    beta=np.zeros(5);beta[OTHER]=state[test.beta_offset:test.magnitude_offset]
    contrast=np.degrees(np.angle(amps*np.conj(amps[1])*np.exp(-1j*beta)))
    phase_sigma=np.degrees(np.sqrt(.5/np.maximum(same_path,1e-30)+.5/max(same_path[1],1e-30)))
    phase_sigma[1]=0
    return {'shared_focused_energy':coherent,'independent_focused_energy':independent_energy,
        'heldout_loss_db':float(10*np.log10(independent_energy/coherent)),
        'shared_prediction_error':prediction,'independent_prediction_error':sum(x['prediction_error'] for x in independent),
        'phase_only_loss_db_same_trajectory':float(10*np.log10(sum(same_path)/coherent)),
        'independent_channels':independent,
        'same_path_independent_band_energy':same_path,
        'phase_contrast_degrees':contrast.tolist(),
        'phase_sigma_degrees_conditional':phase_sigma.tolist(),
        'shared_to_independent_amplitude_ratio':(abs(test_amplitude*gain)/np.maximum(abs(amps),1e-30)).tolist(),
        'band_diagnostic_scope':'All five withheld bands remain visible. A small combined loss is not acceptance if a band with measurable signal is assigned negligible amplitude or inconsistent phase.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('fit','histories','patches','templates','routes','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--labels',nargs='+')
    p.add_argument('--iterations',type=int,default=200)
    p.add_argument('--continue-incomplete',action='store_true',
                   help='Warm-start unfinished folds from their own saved calibration, with unchanged inputs')
    args=p.parse_args();saved=json.loads(args.fit.read_text())
    joint=load_trial(saved,args.routes,args.histories,args.patches,args.templates)
    state=np.array(saved['state']);result={'source_fit':str(args.fit),'assignment':saved['assignment'],
        'input_histories':str(args.histories),'input_patches':str(args.patches),
        'input_templates':str(args.templates),'input_routes':str(args.routes),
        'status':'conditional leave-one-vehicle and alternating-time-block validation',
        'limitations':'Road association and discovery interval use the full capture. Calibration refits start in the discovered local branch; this is not an independent global association test. Terrain is fixed. Receiver uncertainty is refitted only when explicitly present in the source model. Reported held-out energy profiles a common complex amplitude (shared model) or five amplitudes (independent control) on test blocks; prediction-error statistics instead use amplitudes learned on training blocks.',
        'targets':[]}
    if args.output.exists():
        cached=json.loads(args.output.read_text())
        for key in ('source_fit','input_histories','input_patches','input_templates','input_routes'):
            if cached.get(key)!=result[key]:
                raise ValueError('Saved validation has different or undocumented inputs; select a new output file')
        result=cached
    previous={t['label']:t for t in result['targets']}
    completed={label for label,target in previous.items() if not args.continue_incomplete or
               all(target[k] for k in ('calibration_converged','target_converged','independent_converged'))}
    clock=time.monotonic()
    for index,label in enumerate(saved['labels']):
        if label in completed or (args.labels and label not in args.labels):continue
        excluded=[i for i in range(joint.m) if i==index or related_intervals(joint.problems[index],joint.problems[i])]
        training,initial=subset(joint,state,[i for i in range(joint.m) if i not in excluded])
        receiver=joint.problems[index].model.receiver_enu_m.copy()
        receiver_offsets=joint.problems[index].model.receiver_channel_offsets_enu_m.copy()
        earlier=previous.get(label)
        if earlier is not None:
            if earlier['excluded_calibration_intervals']!=[saved['labels'][i] for i in excluded]:
                raise ValueError('The held-out grouping changed; select a new output file')
            initial=np.asarray(earlier['calibration_state'])
            if initial.shape!=(training.count,):
                raise ValueError('Saved fold calibration has a different parameter layout')
            receiver=np.asarray(earlier['receiver_correction_enu_m'])
            receiver_offsets=np.asarray(earlier.get('receiver_channel_offsets_enu_m',receiver_offsets))
        antenna_offsets=None
        antenna_diagnostics={}
        if 'antenna_state' in saved:
            if 'nominal_channel_offsets_enu_m' not in saved:
                raise ValueError('Five-centre holdout requires the original licensed offsets, not fitted offsets as a new prior')
            uncertain=AntennaProblem(training,saved['antenna_offset_prior_sigma_m'],
                                     saved['site_position_prior_sigma_m'],
                                     nominal_offsets=saved['nominal_channel_offsets_enu_m'])
            start=uncertain.initial(initial)
            start[uncertain.offset:]=np.asarray((earlier or saved)['channel_position_corrections_enu_m']).ravel()/uncertain.scale
            fit,fitted=uncertain.fit_columns(start,np.arange(9,uncertain.count),args.iterations)
            calibrated,_=uncertain.expanded_state(fitted)
            corrections=fitted[uncertain.offset:].reshape(5,3)*uncertain.scale
            antenna_offsets=uncertain.original_offsets+corrections
            antenna_diagnostics={'channel_position_corrections_enu_m':corrections.tolist(),
                'calibration_antenna_state':fitted.tolist(),
                'nominal_channel_offsets_enu_m':uncertain.original_offsets.tolist(),
                'antenna_holdout_scope':'All fifteen antenna coordinates refitted without the target, with priors centred on the original licensed offsets. Full-data offsets are only initial guesses.'}
        elif 'receiver_prior_sigma_m' in saved:
            profiled=saved.get('profiled_band_gains',False)
            uncertain=(ProfiledReceiverProblem(training,saved['receiver_prior_sigma_m'],
                                               saved.get('receiver_channel_prior_sigma_m')) if profiled
                       else ReceiverProblem(training,saved['receiver_prior_sigma_m']))
            start=(uncertain.initial(initial,receiver,receiver_offsets) if profiled
                   else np.r_[initial,receiver/uncertain.sigma])
            fit=solve(uncertain.residual,start,uncertain.bounds(),args.iterations)
            calibrated=uncertain.expanded_state(fit.x)[0] if profiled else fit.x[:-3]
            if profiled:receiver,receiver_offsets=uncertain.coordinates(fit.x)
            else:receiver=fit.x[-3:]*uncertain.sigma
            antenna_diagnostics={'calibration_receiver_state':fit.x.tolist(),
                'receiver_prior_sigma_m':uncertain.sigma.tolist(),
                'receiver_channel_offsets_enu_m':receiver_offsets.tolist(),
                'receiver_channel_prior_sigma_m':saved.get('receiver_channel_prior_sigma_m'),
                'receiver_holdout_scope':'The physical receiver position is refitted without the withheld target, with the original roof-position prior. Its full-data value is only an initial guess. The exact source band-gain likelihood is retained.'}
        else:
            if saved.get('profiled_band_gains'):
                reduced=ProfiledBandProblem(training)
                fit=solve(reduced.residual,reduced.initial(initial),reduced.bounds(),args.iterations)
                calibrated,_=reduced.expanded_state(fit.x)
            else:
                fit=solve(training.residual,initial,training.bounds(),args.iterations)
                calibrated=fit.x
        one,initial=subset(joint,state,[index]);initial[:9]=calibrated[:9]
        one.problems[0]=copy.copy(one.problems[0]);one.problems[0].model=copy.copy(one.problems[0].model)
        one.problems[0].model.receiver_enu_m=receiver
        one.problems[0].model.receiver_channel_offsets_enu_m=receiver_offsets.copy()
        if antenna_offsets is not None:
            one.problems[0].model.channel_offsets=antenna_offsets.copy()
        initial[one.beta_offset:one.magnitude_offset]=calibrated[training.beta_offset:training.magnitude_offset]
        if joint.range_bias:initial[one.bias_offset:]=calibrated[training.bias_offset:]
        # The withheld target starts from its independently fitted road branch.
        initial[9:one.beta_offset]=saved['selected_roads'][index]['state']
        train=CoherentProblem([split_problem(one.problems[0],0)],range_bias=joint.range_bias)
        test=CoherentProblem([split_problem(one.problems[0],1)],range_bias=joint.range_bias)
        active=np.r_[np.arange(9,one.beta_offset),np.arange(one.magnitude_offset,one.bias_offset)]
        low,high=train.bounds();initial=np.clip(initial,low+1e-7,high-1e-7)
        starts=train.fixed_phase_starts(initial,np.linspace(-20,20,41))
        if saved.get('profiled_band_gains'):
            reduced=ProfiledBandProblem(train)
            active=np.arange(9,one.beta_offset)
            fits=[]
            for _,q in starts[:3]:
                fixed=reduced.initial(q)
                fitted,fixed=reduced.fit_columns(fixed,active,args.iterations)
                expanded,_=reduced.expanded_state(fixed)
                fits.append((fitted,expanded))
            target_fit,target_state=min(fits,key=lambda q:q[0].cost)
        else:
            fits=[train.fit_columns(q,active,args.iterations) for _,q in starts[:3]]
            target_fit,target_state=min(fits,key=lambda q:q[0].cost)
        independent_train=copy.copy(train.problems[0]);independent_test=copy.copy(test.problems[0])
        for pr in (independent_train,independent_test):
            pr.sites=pr.sites+calibrated[:9].reshape(3,3)*[30,30,75]
            if joint.range_bias:
                pr.range_origins=np.zeros(5)
                pr.range_origins[OTHER]=calibrated[training.bias_offset:]
        independent_fits=[independent_train.fit(q,max_nfev=args.iterations) for q in
            (target_state[9:one.beta_offset],np.array(saved['selected_roads'][index]['state']))]
        independent=min(independent_fits,key=lambda f:f.cost)
        metrics=evaluate(train,test,target_state,independent_train,independent_test,independent.x)
        entry={'label':label,'calibration_converged':bool(fit.success),'calibration_cost':float(fit.cost),
            'calibration_state':calibrated.tolist(),'calibration_termination':str(fit.message),
            'excluded_calibration_intervals':[saved['labels'][i] for i in excluded],
            'calibration_evaluations':fit.nfev,'calibration_optimality':float(fit.optimality),
            'transmitter_corrections_enu_m':(calibrated[:9].reshape(3,3)*[30,30,75]).tolist(),
            'receiver_correction_enu_m':receiver.tolist(),
            'channel_phases_radians':calibrated[training.beta_offset:training.magnitude_offset].tolist(),
            'range_bias_m':calibrated[training.bias_offset:].tolist(),
            'target_converged':bool(target_fit.success),'independent_converged':bool(independent.success),
            'target_state':target_state.tolist(),'independent_state':independent.x.tolist(),**antenna_diagnostics,**metrics}
        if earlier is not None:
            entry['continued_from']={'cost':earlier['calibration_cost'],
                'evaluations':earlier['calibration_evaluations'],'optimality':earlier['calibration_optimality']}
            result['targets'][next(i for i,r in enumerate(result['targets']) if r['label']==label)]=entry
        else:
            result['targets'].append(entry)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2))
        print(label,'held-out loss',round(entry['heldout_loss_db'],3),'same-path phase loss',round(entry['phase_only_loss_db_same_trajectory'],3),
            'fit',fit.success,target_fit.success,independent.success,'elapsed',round(time.monotonic()-clock,2),flush=True)


if __name__=='__main__':main()
