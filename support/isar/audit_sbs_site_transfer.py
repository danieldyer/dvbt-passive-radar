"""Conditional whole-record SBS site test at a fixed common physical scene.

Compare the two already fitted physical site positions, Ornata and Ridge.
The ship path, every shared source phase and receiver position stay fixed.
Only one constant SBS phase and positive brightness are profiled per case.
This conditional comparison cannot certify antenna positions or site choice.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from full_scene_context import open_scene
from full_path_point import carrier_bin
from bistatic_path_lift import lift
from sparse_positive_envelope import Normal
from rigid_patch import profile_static


def site_design(band,channel,transmitter):
    b=band.point
    state=lift(b.anchor,b.velocity,b.direction,b.direction_rate,b.base_path,b.base_rate,b.motion.tx[3],b.motion.rx)
    points=band.points[band.window_index] if band.points.ndim==3 else np.broadcast_to(band.points,(len(b.t),*band.points.shape))
    output=np.zeros((*b.field.y.shape,band.points_per_window),complex)
    for p in range(band.points_per_window):
        enabled=band.active_points[band.window_index,p];edges=np.diff(np.r_[False,enabled,False].astype(int))
        for first,last in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            view=b.waveform_window(first,last);f=view.field;sl=slice(first,last);offset=points[sl,p]-b.reference_point
            position=state['position_ecef_m'][sl]+np.einsum('tij,tj->ti',view.frame,offset)
            velocity=state['velocity_ecef_mps'][sl]+np.einsum('tij,tj->ti',view.frame_rate,offset)
            delta,omega,gradient=f.path_kinematics(position,velocity,transmitter=transmitter,receiver=b.motion.rx,
                position_offset=position-f.data['centre_ecef_m'])
            speed=np.sum(gradient*velocity,axis=1)
            factor,_=carrier_bin(view.t,speed,view.coordinate['native_time_s'],view.coordinate['native_information'],f.k)
            output[sl,:,p]=f.apply(f.path_response(delta,omega)*factor[:,None])
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();fit,motion,bands,saved=open_scene(a.motion);band=bands[1];b=band.point
    phi=np.asarray(fit['scene_point_phase_radians']);initial=np.angle(complex(*fit['physical_gains_real_imag'][1]))
    records=[];curves=[];reference=band.design(np.zeros(b.b.shape[1]),1,False)
    for name,tx in [('Ornata incumbent',motion.tx[1]),('Ridge incumbent phase centre',motion.tx[4])]:
        h=site_design(band,1,tx)
        if name=='Ornata incumbent':
            error=float(np.linalg.norm(h-reference)/np.linalg.norm(reference))
            if error>1e-8:raise ValueError(f'Independent site operator changed the incumbent waveform: {error}')
        normal=Normal(h,b.field.y,b.field.root_information,b.training,b.amplitude,b.groups);warm=None
        def objective(beta):
            nonlocal warm
            solution=normal.solve(phi+beta,warm,band.phase_groups)
            if not solution['numerical_convergence']:raise ValueError(f"Site brightness solve failed KKT {solution['maximum_scaled_kkt']}")
            warm=solution['amplitude'];return solution['cost']/normal.energy
        angles=np.unique(np.r_[np.arange(32)*2*np.pi/32,initial]);values=np.array([objective(x) for x in angles])
        candidates=[]
        for i in np.argsort(values)[:2]:
            result=minimize_scalar(objective,bounds=(angles[i]-2*np.pi/32,angles[i]+2*np.pi/32),method='bounded',options=dict(xatol=1e-9))
            candidates.append((result.fun,result.x))
        value,beta=min(candidates);solution=normal.solve(phi+beta,warm,band.phase_groups)
        coefficient=solution['amplitude'].reshape(-1,band.points_per_window)
        model=np.sum(h*((b.amplitude@coefficient)*np.exp(1j*band.phase_at_time(phi+beta)))[:,None,:],axis=2)
        excluded=profile_static(model,b.field.root_information,~b.training,b.groups)*(~b.training)[:,None]-b.data[False]
        test_cost=float(np.vdot(excluded,excluded).real)
        record=dict(site=name,transmitter_ecef_m=tx.tolist(),constant_phase_radians=float(beta),training_rss=solution['cost'],
            training_energy=normal.energy,training_explained_fraction=1-value,excluded_rss=test_cost,excluded_energy=b.energy[False],
            excluded_explained_fraction=1-test_cost/b.energy[False],amplitude_kkt=solution['maximum_scaled_kkt'],seconds=time.monotonic()-started)
        if name=='Ornata incumbent':
            record['independent_original_operator_relative_error']=error
            record['upstream_training_cost_at_initial_phase_relative_error']=abs(objective(initial)-fit['training'][1]['cost']/normal.energy)
            if record['upstream_training_cost_at_initial_phase_relative_error']>1e-8:raise ValueError('Incumbent site likelihood does not reproduce its upstream original-data fit')
        records.append(record);curves.append((angles,values));print(json.dumps(record),flush=True)
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for record,(angles,values) in zip(records,curves):
        axes[0].plot(np.degrees(angles),100*(1-values),'.-',label=record['site'])
    axes[0].set(xlabel='One constant SBS phase (degrees)',ylabel='SBS training crop energy predicted (%)');axes[0].legend(fontsize=8)
    for j,record in enumerate(records):axes[1].bar(np.arange(2)+(j-.5)*.35,[100*record['training_explained_fraction'],100*record['excluded_explained_fraction']],.35,label=record['site'])
    axes[1].set(xticks=[0,1],xticklabels=['Training','Excluded'],ylabel='SBS crop energy predicted (%)');axes[1].legend(fontsize=8)
    fig.suptitle('Whole-record conditional site test: same ship path, shared source phases and receiver\nOne constant SBS phase; positive brightness; no per-window band calibration or geometry adoption')
    fig.savefig(a.output/'sbs-site-transfer.png',dpi=145);plt.close(fig)
    (a.output/'analysis.json').write_text(json.dumps(dict(complete=True,scope=__doc__,motion=str(a.motion),channels=records,
        geometry_adopted=False,seconds=time.monotonic()-started),indent=2)+'\n')


if __name__=='__main__':main()
