"""Test a separable scattering-phase assumption on excluded complex samples.

The control frees only a constant phase for each fixed physical scatterer in
each band. Motion, locations, original epochs, positive brightness basis and
stationary nuisance projection are unchanged. It never applies a per-band
time-varying correction and is not an interband imaging calibration.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from full_scene_context import open_scene
from sparse_positive_envelope import Normal
from rigid_patch import profile_static


def fit_phases(normal, initial, groups, iterations):
    warm=None
    def objective(phase):
        nonlocal warm
        value=normal.solve(phase,warm,groups)
        if not value['numerical_convergence']:
            raise ValueError(f"Original amplitude solve failed KKT: {value['maximum_scaled_kkt']}")
        warm=value['amplitude']
        return value['cost']/normal.energy,value['phase_gradient']/normal.energy
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',
        options=dict(maxiter=iterations,gtol=1e-8,ftol=1e-12,maxcor=30))
    value=normal.solve(result.x,warm,groups)
    return result.x,value,dict(iterations=result.nit,message=str(result.message),
        maximum_normalized_phase_gradient=float(max(abs(value['phase_gradient']))/normal.energy),
        amplitude_kkt=value['maximum_scaled_kkt'],optimizer_success=bool(result.success))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=300)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    fit,motion,bands,saved=open_scene(a.motion)
    if bands[0].points.ndim!=2:
        raise ValueError('This diagnostic requires a single persistent source support')
    phi=np.asarray(fit['scene_point_phase_radians']);records=[];phase_delta=[];powers=[];arrays={}
    for c,band in enumerate(bands):
        b=band.point;h=band.design(np.zeros(b.b.shape[1]),c,False)
        beta=np.angle(complex(*fit['physical_gains_real_imag'][c]));initial=phi+beta
        normal=Normal(h,b.field.y,b.field.root_information,b.training,b.amplitude,b.groups)
        phase,solution,report=fit_phases(normal,initial,band.phase_groups,a.iterations)
        coefficient=solution['amplitude'].reshape(-1,band.points_per_window)
        raw=np.sum(h*((b.amplitude@coefficient)*np.exp(1j*phase))[:,None,:],axis=2)
        baseline=np.sum(h*(saved[f'amplitude_{c}']*np.exp(1j*initial))[:,None,:],axis=2)
        record=dict(channel=c,phase_fit=report)
        for label,mask in [('training',b.training),('excluded',~b.training)]:
            record[label+'_energy']=b.energy[label=='training']
            for name,model in [('separable',baseline),('independent_constant_source_phase',raw)]:
                residual=profile_static(model,b.field.root_information,mask,b.groups)*mask[:,None]-b.data[label=='training']
                record[label+'_'+name+'_rss']=float(np.vdot(residual,residual).real)
        delta=np.angle(np.exp(1j*(phase-initial)));phase_delta.append(delta)
        powers.append(np.sum(abs(h)**2*(saved[f'amplitude_{c}'][:,None,:]**2),axis=(0,1)))
        arrays[f'phase_radians_{c}']=phase;arrays[f'coefficient_{c}']=coefficient
        records.append(record);print(json.dumps(record),flush=True)
    totals={}
    for label in ['training','excluded']:
        energy=sum(r[label+'_energy'] for r in records)
        totals[label]={key:1-sum(r[label+'_'+key+'_rss'] for r in records)/energy
            for key in ['separable','independent_constant_source_phase']}
    np.savez_compressed(a.output/'constant-source-phases.npz',**arrays)
    from matplotlib import pyplot as plt
    phase_delta=np.asarray(phase_delta);powers=np.asarray(powers)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    ax=axes[0]
    for c,name in enumerate(['Seven','SBS','Nine','Ten','ABC']):
        ax.scatter(np.arange(len(phi))+(c-2)*.12,np.degrees(phase_delta[c]),
            s=8+90*powers[c]/max(powers[c].max(),1e-30),label=name,alpha=.7)
    ax.set(xlabel='Fixed physical source index',ylabel='Change from shared-source + band phase (degrees)',ylim=[-185,185],
        title='Only constant source phases are freed; motion unchanged')
    ax.legend(ncol=3,fontsize=8)
    ax=axes[1];x=np.arange(2);width=.34
    for j,(key,label) in enumerate([('separable','Common source phase + band phase'),('independent_constant_source_phase','Independent constant source/band phase')]):
        ax.bar(x+(j-.5)*width,[100*totals[s][key] for s in ['training','excluded']],width,label=label)
    ax.set(xticks=x,xticklabels=['Training','Excluded'],ylabel='Explained original crop energy (%)',
        title='Same positive brightness basis, samples and motion')
    ax.legend(fontsize=8)
    fig.suptitle('Diagnostic of a scattering assumption; no independent band phase histories or production correction')
    fig.savefig(a.output/'constant-source-phase-control.png',dpi=145);plt.close(fig)
    result=dict(complete=True,scope=__doc__,motion=str(a.motion),totals=totals,channels=records,
        phase_changes_degrees=np.degrees(phase_delta).tolist(),adopted_as_imaging_calibration=False,seconds=time.monotonic()-started)
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(totals=totals,seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
