"""Audit shared inter-band phase against independent complex target responses."""
import argparse
import json
from pathlib import Path

import numpy as np

from complex_likelihood import projected_profile
from fit_geometry import observations
from geographic_io import read_profiles
from phase_fit import load_histories
from range_fit import load_fields,Field
from road_geometry import RoadCurve
from road_model import RoadTrajectory,RoadProblem
from coherent_fit import CoherentProblem,OTHER


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('fit','histories','patches','templates','routes','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--jackknife-seconds',type=float,default=0.,help='Delete complete slow-time blocks to measure empirical phase stability')
    args=p.parse_args();saved=json.loads(args.fit.read_text())
    sites=np.array(saved['site_positions_enu_m']);offsets=np.array(saved['channel_offsets_enu_m'])
    roads=json.loads(args.routes.read_text())['routes'];_,records=read_profiles(args.templates)
    problems=[]
    for label,coarse,selected in zip(saved['labels'],observations(args.histories,saved['labels']),saved['selected_roads']):
        channels,description=load_histories(args.histories,label,1.)
        model=RoadTrajectory(channels[1]['seed'],RoadCurve(roads[selected['route']]),selected['coarse_state'],saved['assignment'],offsets)
        model.receiver_enu_m=np.array(saved.get('receiver_correction_enu_m',[0.,0.,0.]))
        model.receiver_channel_offsets_enu_m=np.array(saved.get('receiver_channel_offsets_enu_m',np.zeros((5,3))))
        problems.append(RoadProblem(channels,coarse,sites,load_fields(args.patches,label,description,records),model))
    joint=CoherentProblem(problems,range_bias=saved.get('range_bias',False),site_prior_sigma=np.array(saved['site_prior_sigma_m']))
    state=np.array(saved['state']);beta=np.zeros(5);beta[OTHER]=state[joint.beta_offset:joint.magnitude_offset]
    biases=np.zeros(5)
    if joint.range_bias:biases[OTHER]=state[joint.bias_offset:]
    targets=[]
    for i,pr in enumerate(problems):
        local=np.r_[state[9+i*joint.n:9+(i+1)*joint.n],state[:9]]
        amps=[];energies=[];norms=[];templates=[]
        for ch in range(5):
            u,_=pr.channel_model(local,ch,True,range_bias=biases[ch])
            _,a=projected_profile(pr.channel_data(ch),u)
            amps.append(a);norms.append(float(np.vdot(u,u).real));energies.append(float(abs(a)**2*norms[-1]));templates.append(u)
        amps=np.array(amps);energies=np.array(energies)
        phases=np.angle(amps*np.conj(amps[1])*np.exp(-1j*beta))
        magnitudes=np.ones(5);magnitudes[OTHER]=np.exp(state[joint.magnitude_offset+4*i:joint.magnitude_offset+4*(i+1)])
        u=np.concatenate([templates[ch]*magnitudes[ch]*np.exp(1j*beta[ch]) for ch in range(5)])
        _,a=projected_profile(joint.data[i],u)
        energy=float(abs(a)**2*np.vdot(u,u).real)
        x=pr.model.positions(local,np.array([pr.model.seed['epoch']]),pr.sites,True)[0][0]
        x-=pr.model.receiver_enu_m
        phase_sigma=np.degrees(np.sqrt(.5/np.maximum(energies,1e-30)+.5/max(energies[1],1e-30)))
        phase_sigma[1]=0.
        targets.append({'label':saved['labels'][i],'receive_bearing_deg':float(np.degrees(np.arctan2(x[0],x[1]))),
            'phase_contrast_degrees':np.degrees(phases).tolist(),'independent_focused_energy':energies.tolist(),
            'phase_standard_deviation_degrees_conditional':phase_sigma.tolist(),
            'shared_to_independent_amplitude_ratio':(abs(a)*magnitudes/np.maximum(abs(amps),1e-30)).tolist(),
            'shared_phase_energy':energy,'independent_phase_energy':float(energies.sum()),
            'shared_response_loss_db':float(10*np.log10(energies.sum()/energy))})
        if args.jackknife_seconds:
            start=pr.model.seed['epoch']-pr.model.half
            count=int(np.ceil(2*pr.model.half/args.jackknife_seconds))
            paths=[pr.model.paths(local,f.time,np.full(len(f.time),ch),pr.sites,True)[0]
                   for ch,f in enumerate(pr.fields)]
            estimates=[]
            for block in range(count):
                deleted=[]
                for ch,field in enumerate(pr.fields):
                    selected=np.floor((field.time-start)/args.jackknife_seconds).astype(int)!=block
                    trial=Field(field.data,field.pulse,field.seed,selected=selected)
                    u,_=trial.template(paths[ch],range_bias=biases[ch]);_,amp=projected_profile(trial.y,u)
                    deleted.append(amp)
                deleted=np.asarray(deleted)
                estimates.append(np.angle(deleted*np.conj(deleted[1])*np.exp(-1j*beta)))
            difference=np.angle(np.exp(1j*(np.array(estimates)-phases)))
            centred=difference-difference.mean(axis=0)
            targets[-1]['delete_block_phase_sigma_degrees']=np.degrees(np.sqrt((count-1)/count*np.sum(centred**2,axis=0))).tolist()
            targets[-1]['delete_block_max_phase_change_degrees']=np.degrees(np.max(abs(difference),axis=0)).tolist()
    result={'assignment':saved['assignment'],'source_fit':str(args.fit),'targets':targets,
        'status':'in-sample model diagnostic, not held-out calibration acceptance',
        'scope':'Same fitted geometry and paths for both statistics; independent per-band complex gains versus the fitted shared-phase response. Thermal phase error bars are conditional on the empirical covariance.',
        'jackknife_block_seconds':args.jackknife_seconds,
        'jackknife_scope':'If requested, delete whole time blocks and recompute the same gated likelihood with the fitted path held fixed. The resulting phase stability is empirical and conditional, not a full confidence interval for position, site choice or scatterer structure.',
        'receiver_pattern_caveat':'Direct-reference correlation removes direction-independent electronic phase; it does not by itself measure a receiving antenna phase pattern toward other directions. A directional/frequency-dependent antenna response is one possible explanation of systematic phase contrasts, not an established diagnosis here.',
        'antenna_phase_reference':'https://doi.org/10.1587/transfun.E102.A.1383'}
    args.output.write_text(json.dumps(result,indent=2))
    for q in targets:print(q['label'],round(q['shared_response_loss_db'],2),np.round(q['phase_contrast_degrees'],1),flush=True)


if __name__=='__main__':main()
