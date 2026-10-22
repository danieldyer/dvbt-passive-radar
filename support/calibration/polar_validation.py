"""Held-out and bookkeeping checks for candidate polar calibration fits."""
import argparse
import json
from pathlib import Path

import numpy as np

from complex_likelihood import operator_for, profile
from fit_geometry import observations
from phase_fit import load_histories
from polar_fit import Problem


def shared_response(problem, state, free_sites=False, supplied_beta=None):
    """Compare shared versus independent time responses with projected noise.

    Geometry is held fixed. Gains learn on alternating 1-second time blocks,
    then transfer to the other blocks; each test block has a nuisance amplitude.
    Both models use identical linear observations and projected covariance.
    """
    train_matrices,test_matrices,train_values,test_values = [],[],[],[]
    for ch,d in enumerate(problem.channels):
        path,_ = problem.model.paths(state,d['time'],np.full(len(d['time']),ch),problem.sites,free_sites)
        u = np.exp(-1j*d['k']*(path-d['seed_delta']))
        bins = np.clip(np.floor(d['time']-(problem.model.seed['epoch']-15)).astype(int),0,29)
        basis = np.eye(30)[bins]
        train = bins%2==0
        qt,qe = operator_for(d,train),operator_for(d,~train)
        train_matrices.append(qt@(u[:,None]*basis[:,::2]))
        test_matrices.append(qe@(u[:,None]*basis[:,1::2]))
        train_values.append(qt@d['value']);test_values.append(qe@d['value'])
    if supplied_beta is None:
        coefficients = np.array([np.linalg.lstsq(m,y,rcond=None)[0] for m,y in zip(train_matrices,train_values)])
        beta = np.linalg.svd(coefficients,full_matrices=False)[0][:,0]
        for _ in range(100):
            m = np.concatenate([b*q for b,q in zip(beta,train_matrices)])
            a = np.linalg.lstsq(m,np.concatenate(train_values),rcond=None)[0]
            beta = np.array([np.vdot(q@a,y)/np.linalg.norm(q@a)**2 for q,y in zip(train_matrices,train_values)])
            beta /= np.linalg.norm(beta)
    else:
        beta = np.asarray(supplied_beta)
    matrix = np.concatenate([b*m for b,m in zip(beta,test_matrices)])
    y = np.concatenate(test_values)
    a = np.linalg.lstsq(matrix,y,rcond=None)[0]
    pc = np.linalg.norm(matrix@a)**2-np.linalg.matrix_rank(matrix)
    pi = 0.; individual = []
    for m,y in zip(test_matrices,test_values):
        ai = np.linalg.lstsq(m,y,rcond=None)[0]
        pi += np.linalg.norm(m@ai)**2-np.linalg.matrix_rank(m)
        individual.append(ai)
    return {'heldout_common_response_loss_db':float(10*np.log10(pi/pc)) if min(pi,pc)>0 else None,
        'shared_signal_score':float(pc),'independent_signal_score':float(pi),
        'beta_real':beta.real.tolist(),'beta_imag':beta.imag.tolist(),
        'independent_test_real':np.array(individual).real.tolist(),
        'independent_test_imag':np.array(individual).imag.tolist(),
        'assumptions':'fixed fitted geometry; empirical diagonal pre-projection noise; common response constant gains across time',
        'scope':'a held-out gain-model check, not an independent target-detection significance test'}


def heldout_score(training, testing, state, free_sites):
    result = []
    for c,d in enumerate(training.channels):
        path,_ = training.model.paths(state,d['time'],np.full(len(d['time']),c),training.sites,free_sites)
        phase = -d['k']*(path-d['seed_delta'])
        _,a,u = profile(d['value'],training.operators[c],phase)
        r = testing.operators[c]@(a*u-d['value'])
        signal = testing.operators[c]@u;y = testing.operators[c]@d['value']
        score = abs(np.vdot(signal,y))**2/np.linalg.norm(signal)**2
        result.append({'prediction_error':float(np.linalg.norm(r)**2),'focused_score':float(score)})
    return result


def binning_check(problem,state,directory,label):
    checks = []
    for ch,d in enumerate(problem.channels):
        with np.load(directory/f'{label}-{ch}.npz') as source:
            t,w = source['row_time'],source['row_information']
        path,_ = problem.model.paths(state,t,np.full(len(t),ch),problem.sites)
        seed = d['seed']
        delta = np.polynomial.polynomial.polyval((t-seed['epoch'])/15,seed['path'])-seed['path'][0]
        u = np.exp(-1j*d['k']*(path-delta))
        bins = np.floor((t-(seed['epoch']-15))/.02).astype(int)
        norm = np.bincount(bins,weights=w,minlength=1500)
        mean = (np.bincount(bins,weights=w*u.real,minlength=1500)+
                1j*np.bincount(bins,weights=w*u.imag,minlength=1500))/norm
        centre = np.bincount(bins,weights=w*t,minlength=1500)/norm
        path,_ = problem.model.paths(state,centre,np.full(len(centre),ch),problem.sites)
        delta = np.polynomial.polynomial.polyval((centre-seed['epoch'])/15,seed['path'])-seed['path'][0]
        instantaneous = np.exp(-1j*d['k']*(path-delta))
        good = norm>0
        error = np.angle(mean[good]/instantaneous[good])
        retention = np.average(abs(mean[good])**2,weights=norm[good])
        path,_ = problem.model.paths(state,d['time'],np.full(len(d['time']),ch),problem.sites)
        template = np.exp(-1j*d['k']*(path-d['seed_delta']))
        retained = np.linalg.norm(d['operator']@template)**2/np.sum(1/d['variance'])
        checks.append({'channel':ch,'max_bin_phase_error_deg':float(np.max(abs(error))*180/np.pi),
            'mean_bin_power_loss_db':float(-10*np.log10(retention)),
            'selected_doppler_subspace_template_loss_db':float(-10*np.log10(retained))})
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--histories',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--labels',nargs='+',default=['A','B','C','E','F'])
    p.add_argument('--skip-splits',action='store_true')
    p.add_argument('--name',default='polar-validation')
    args = p.parse_args()
    sites = np.array(json.loads((args.output/'geometry-prior.json').read_text())['enu_transmitters_m'])
    coarse = observations(args.histories,args.labels)
    result = {'response_checks':{},'A_split_checks':[]}
    beta = None
    for label,obs in zip(args.labels,coarse):
        channels,_ = load_histories(args.histories,label,1.)
        saved = json.loads((args.output/f'{label}-polar.json').read_text())
        state = np.array(saved['fixed_state'])
        problem = Problem(channels,obs,sites)
        check = shared_response(problem,state)
        if beta is None:
            beta = np.array(check['beta_real'])+1j*np.array(check['beta_imag'])
        else:
            check['gain_transfer_from_A'] = shared_response(problem,state,supplied_beta=beta)
        result['response_checks'][label] = check
        print(label,'common response heldout loss',check['heldout_common_response_loss_db'],flush=True)
        if label=='A':
            result['A_binning_checks'] = binning_check(problem,state,args.histories,label)
            if not args.skip_splits:
                masks = [('even 1-second blocks',[np.floor(d['time']-35).astype(int)%2==0 for d in channels]),
                         ('first half',[d['time']<50 for d in channels]),
                         ('second half',[d['time']>=50 for d in channels])]
                for name,train in masks:
                    training = Problem(channels,obs,sites,train=train)
                    testing = Problem(channels,obs,sites,train=[~x for x in train])
                    fixed = training.fit(state)
                    free = training.fit(fixed.x,True)
                    a,b = heldout_score(training,testing,fixed.x,False),heldout_score(training,testing,free.x,True)
                    item = {'training':name,'transmitter_corrections_enu_m':(free.x[-9:].reshape(3,3)*[30,30,75]).tolist(),
                        'fixed_converged':bool(fixed.success),'free_converged':bool(free.success),
                        'fixed_test':a,'refined_test':b,
                        'heldout_focus_gain_db':float(10*np.log10(sum(v['focused_score'] for v in b)/sum(v['focused_score'] for v in a))),
                        'fixed_test_prediction_error':sum(v['prediction_error'] for v in a),
                        'refined_test_prediction_error':sum(v['prediction_error'] for v in b)}
                    result['A_split_checks'].append(item)
                    print('A',name,'heldout gain',item['heldout_focus_gain_db'],
                          'TX',item['transmitter_corrections_enu_m'],flush=True)
        (args.output/(args.name+'.json')).write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
