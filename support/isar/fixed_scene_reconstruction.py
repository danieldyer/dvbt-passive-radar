"""Fixed-motion coherent scene fit with positive scattering envelopes.

Only one common phase per spatial source is fitted. Relative band phases are
held at their upstream calibration. Each band's positive visibility is real;
there are no independent time-varying phase screens. All scoring uses the
original complex rows and explicit stationary-nuisance projection.
"""
from dataclasses import dataclass
import numpy as np
from sparse_positive_envelope import Normal
from full_path_scene import solve_normal_phases
from rigid_patch import profile_static


@dataclass
class SceneFit:
    band_phase: np.ndarray
    source_phase: np.ndarray
    amplitude: list
    model: list
    cost: float
    energy: float
    optimization: dict


def fit_scene(design,bands,selected,band_phase,source_phase,*,iterations=300):
    if len(design)!=5 or len(bands)!=5 or len(selected)!=5:
        raise ValueError('Use all five explicitly calibrated band observations')
    count=design[0].shape[2]
    if any(h.shape[2]!=count for h in design):raise ValueError('A shared spatial support is required')
    phase=np.asarray(source_phase,float);beta=np.asarray(band_phase,float)
    if phase.shape!=(count,) or beta.shape!=(5,):raise ValueError('Inconsistent source/band phase identities')
    initial=np.r_[beta+phase[0],phase[1:]-phase[0]];offset=beta-beta[3]
    normals=[Normal(h,b.field.y,b.field.root_information,mask,b.amplitude,b.groups)
        for h,b,mask in zip(design,bands,selected)]
    groups=[np.tile(np.arange(count),b.amplitude.shape[1]) for b in bands]
    parameters,solutions,report=solve_normal_phases(normals,initial,groups,iterations,fixed_band_offsets=offset)
    if not report['numerical_convergence']:
        # A new L-BFGS history is permissible; the physical objective and
        # original KKT/phase acceptance thresholds are unchanged.
        parameters,solutions,report=solve_normal_phases(normals,parameters,groups,iterations,fixed_band_offsets=offset)
    if not report['numerical_convergence']:raise ValueError(f'Constant-phase scene fit did not converge: {report}')
    phi=np.r_[0.,parameters[5:]];beta=parameters[:5]
    amplitude=[r['amplitude'].reshape(-1,count) for r in solutions]
    model=[np.sum(h*((b.amplitude@a)*np.exp(1j*(phi+beta[c])))[:,None,:],axis=2)
        for c,(h,b,a) in enumerate(zip(design,bands,amplitude))]
    return SceneFit(beta,phi,amplitude,model,sum(r['cost'] for r in solutions),sum(n.energy for n in normals),report)


def score_scene(fit,bands,selected):
    records=[]
    for c,(b,mask,model) in enumerate(zip(bands,selected,fit.model)):
        projected=profile_static(b.field.y,b.field.root_information,mask,b.groups)*mask[:,None]
        residual=profile_static(model,b.field.root_information,mask,b.groups)*mask[:,None]-projected
        records.append(dict(channel=c,cost=float(np.vdot(residual,residual).real),energy=float(np.vdot(projected,projected).real)))
    cost=sum(r['cost'] for r in records);energy=sum(r['energy'] for r in records)
    if energy<=0:raise ValueError('No original-data energy in this scoring fold')
    return dict(cost=cost,energy=energy,explained_fraction=1-cost/energy,channels=records)


def component_diagnostics(fit,design,bands,selected):
    power=[];ratios=[]
    for c,(h,b,mask,a) in enumerate(zip(design,bands,selected,fit.amplitude)):
        components=h*((b.amplitude@a)*np.exp(1j*(fit.source_phase+fit.band_phase[c])))[:,None,:]
        components=profile_static(components,b.field.root_information,mask,b.groups)*mask[:,None,None]
        component=np.sum(abs(components)**2,axis=(0,1));total=np.sum(abs(components.sum(axis=2))**2)
        power.append(component);ratios.append(float(10*np.log10(component.sum()/total)) if total>0 else None)
    return dict(component_predicted_energy=power,component_sum_to_combined_power_db=ratios)
