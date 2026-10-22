"""Conditional licensed-site comparison using an extended coherent ship scene.

Physical trajectories, observations and common source locations remain fixed.
Each allowed site changes its actual envelope delay, RF phase and Doppler.
One constant band phase and one phase per point are fitted on training data;
nonnegative band/point brightnesses are nuisance parameters. The coefficients
are frozen before excluded-data evaluation. This screen alone cannot establish
unique sites: motion and source locations must subsequently be refitted.
"""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from full_ship_field import load
from full_path_point import carrier_bin
from local_ship_motion import load_motion
from motion import excess_path
from rigid_patch import profile_static
from ship_site_signature import licensed_options
from shared_scene import phase_coherent
from report_measured_scatterer_phases import serial
from sparse_positive_envelope import Normal
from full_path_scene import solve_normal_phases
from audit_ship_scattering_envelopes import linear_basis
from scipy.sparse import csr_matrix


def score(trials):
    gram=np.asarray([r['gram'] for r in trials]);rhs=np.asarray([r['rhs'] for r in trials]);energy=np.asarray([r['energy'] for r in trials])
    fit=phase_coherent(gram,rhs,energy);cost=0.
    for r,z in zip(trials,fit['coefficient']):cost+=r['test_energy']-2*np.vdot(z,r['test_rhs']).real+np.vdot(z,r['test_gram']@z).real
    total=sum(r['test_energy'] for r in trials)
    return dict(training_explained_fraction=1-fit['cost']/energy.sum(),unused_explained_fraction=1-cost/total,
        unused_rss=cost,unused_energy=total,phase_gradient=fit['maximum_gradient'],phase_converged=fit['converged'],
        gain_real_imag=np.c_[fit['gain'].real,fit['gain'].imag],point_phase_rad=fit['point_phase_rad'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--geometry',type=Path,default=Path('docs/data/calibration_20261016_sites'))
    p.add_argument('--positive-envelope-spacing-s',type=float,help='Use tied-phase nonnegative linear brightness envelopes instead of constant brightness')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);d=json.loads(a.motion.read_text());motion=load_motion(a.motion)
    points=np.asarray(d['scene_points_body_m'])
    if points.ndim!=2:raise ValueError('Use one isolated extended-scene aperture for this site screen')
    fields,coordinates,_=load(Path(d['field']));options=licensed_options(a.geometry);codes={'12013':'0','12015':'1','12014':'2'};statistics=[]
    for c,(full,coordinate) in enumerate(zip(fields,coordinates)):
        f=full.windowed(*d['interval_s']);indices=np.searchsorted(full.time,f.time);train=coordinate['training'][indices]
        native_time=coordinate['native_time_s'].reshape(-1,16)[indices].ravel();native_information=coordinate['native_information'].reshape(-1,16)[indices].ravel()
        groups=coordinate.get('stationary_groups',np.zeros(len(full.time),int))[indices]
        state=motion.state(f.time)
        position=state['position_ecef_m'][:,None,:]+np.einsum('tij,pj->tpi',state['body_to_ecef'],points)
        velocity=state['velocity_ecef_mps'][:,None,:]+np.einsum('tij,pj->tpi',state['frame_velocity'],points)
        candidates={codes[o['site_id']]:np.asarray(o['transmitter_ecef_m']) for o in options if o['channel']==c}
        candidates['incumbent']=motion.tx[c];band={}
        for code,tx in candidates.items():
            columns=[]
            for point in range(len(points)):
                pos,vel=position[:,point],velocity[:,point];_,g=excess_path(pos,tx,motion.rx)
                factor,_=carrier_bin(f.time,np.sum(g*vel,axis=1),native_time,native_information,f.k)
                raw=f.raw_trajectory(pos,vel,transmitter=tx,receiver=motion.rx,position_offset=pos-f.data['centre_ecef_m'])
                columns.append(f.apply(raw*factor[:,None]))
            h=np.stack(columns,axis=-1);stats={}
            for prefix,mask in [('',train),('test_',~train)]:
                x=profile_static(h,f.root_information,mask,groups)[mask].reshape(-1,len(points));y=profile_static(f.y,f.root_information,mask,groups)[mask].ravel()
                stats[prefix+'gram']=x.conj().T@x;stats[prefix+'rhs']=x.conj().T@y;stats[prefix+'energy']=float(np.vdot(y,y).real)
            stats['transmitter_ecef_m']=tx;band[code]=stats
            if a.positive_envelope_spacing_s is not None:
                basis,_=linear_basis(f.time,d['interval_s'],a.positive_envelope_spacing_s)
                basis=csr_matrix(basis)
                stats['normal']=Normal(h,f.y,f.root_information,train,basis,groups)
                stats['test_normal']=Normal(h,f.y,f.root_information,~train,basis,groups)
                pivot=len(f.time)//2
                new_path=excess_path(position[pivot:pivot+1,0],tx,motion.rx)[0][0]
                old_path=excess_path(position[pivot:pivot+1,0],motion.tx[c],motion.rx)[0][0]
                stats['constant_phase_initializer_change']=f.k*(new_path-old_path)
                stats['phase_groups']=np.tile(np.arange(len(points)),basis.shape[1])
        statistics.append(band);print(json.dumps(dict(stage='templates',channel=c,options=list(band))),flush=True)
    reports=[]
    for assignment in [tuple(['incumbent']*5),*product(*[[k for k in b if k!='incumbent'] for b in statistics])]:
        trials=[b[k] for b,k in zip(statistics,assignment)]
        if a.positive_envelope_spacing_s is None:report=score(trials)
        else:
            initial=np.r_[np.angle([complex(*z) for z in d['physical_gains_real_imag']])+[r['constant_phase_initializer_change'] for r in trials],np.asarray(d['scene_point_phase_radians'])[1:]]
            phase,solutions,diagnostic=solve_normal_phases([r['normal'] for r in trials],initial,[r['phase_groups'] for r in trials],200)
            testing=[]
            for r,solution in zip(trials,solutions):
                normal=r['test_normal'];z=solution['coefficient']
                cost=float(normal.energy-2*np.vdot(z,normal.rhs).real+np.vdot(z,normal.multiply(z)).real)
                testing.append(dict(cost=cost,energy=normal.energy,explained_fraction=1-cost/normal.energy))
            total=sum(t['energy'] for t in testing);cost=sum(t['cost'] for t in testing)
            report=dict(training_explained_fraction=1-diagnostic['objective'],unused_explained_fraction=1-cost/total,unused_rss=cost,unused_energy=total,
                phase_gradient=diagnostic['maximum_gradient'],phase_converged=diagnostic['numerical_convergence'],phase_optimization=diagnostic,unused_by_band=testing,
                band_phase_radians=phase[:5],point_phase_rad=np.r_[0.,phase[5:]])
        report['assignment']='incumbent' if assignment[0]=='incumbent' else ''.join(assignment)
        reports.append(report);print(json.dumps(serial(report)),flush=True)
        (a.output/'analysis.json').write_text(json.dumps(serial(dict(complete=False,scope=__doc__,motion=str(a.motion),geometry=str(a.geometry),trials=reports)),indent=2)+'\n')
    ranked=sorted(reports,key=lambda r:-r['training_explained_fraction'])
    report=dict(complete=True,scope=__doc__,motion=str(a.motion),geometry=str(a.geometry),trials=reports,
        training_ranked_assignments=[r['assignment'] for r in ranked],unique_sites_proven=False,
        accepted_as_calibration=False,points=len(points),interval_s=d['interval_s'])
    report['positive_envelope_spacing_s']=a.positive_envelope_spacing_s
    (a.output/'analysis.json').write_text(json.dumps(serial(report),indent=2)+'\n')
    from matplotlib import pyplot as plt
    fig,ax=plt.subplots(figsize=(15,6),layout='constrained');x=np.arange(len(ranked))
    ax.plot(x,[100*r['training_explained_fraction'] for r in ranked],'.-',label='Training')
    ax.plot(x,[100*r['unused_explained_fraction'] for r in ranked],'.-',label='Excluded')
    ax.set(xticks=x,xticklabels=[r['assignment'] for r in ranked],ylabel='Full-crop energy predicted (%)',xlabel='Site assignment: 0 Ornata / 1 Eyre / 2 Ridge; channels Seven, SBS, Nine, Ten, ABC')
    ax.tick_params(axis='x',rotation=65);ax.legend();fig.suptitle('Literal approved tower priors: conditional 32-assignment phase-coherent scene screen\nFixed physical motion/source positions; one constant phase per band; no assignment adopted without joint refinement')
    fig.savefig(a.output/'licensed-site-screen.png',dpi=145);plt.close(fig)


if __name__=='__main__':main()
