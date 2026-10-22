"""Residual-driven sparse scene support at fixed physical motion.

Vulkan proposes locations using the representative-epoch adjoint. Every
accepted prefix is fitted/scored with the original-bin complex likelihood.
Only inner blocked prediction chooses the support size; the original outer
excluded fold does not enter source selection. This is greedy sparse model
selection, not a globally optimal image or unique physical point solution.
"""
import copy
import json
from pathlib import Path
import time
import numpy as np
from coherent_amplitude_glrt import statistic
from physical_scene_adjoint import PhysicalSceneAdjoint
from fixed_scene_reconstruction import fit_scene,score_scene,component_diagnostics
from rigid_patch import profile_static


def empty_score(bands,masks):
    energy=0.
    for b,mask in zip(bands,masks):
        y=profile_static(b.field.y,b.field.root_information,mask,b.groups)*mask[:,None]
        energy+=float(np.vdot(y,y).real)
    return dict(cost=energy,energy=energy,explained_fraction=0.)


def reconstruct(bands,pixels,band_phase,directory,*,maximum_sources=20,phase_iterations=300,proposal_candidates=8):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    pixels=np.asarray(pixels,float);beta=np.asarray(band_phase,float);offset=beta-beta[3]
    original=[b.training.copy() for b in bands]
    first=min(b.t[0] for b in bands)
    choose=[np.floor((b.t-first)/2).astype(int)%2!=0 for b in bands]
    training=[old&~fold for old,fold in zip(original,choose)]
    validation=[old&fold for old,fold in zip(original,choose)]
    outer=[~old for old in original]
    if any(np.count_nonzero(m)<20 for masks in [training,validation,outer] for m in masks):
        raise ValueError('Insufficient original samples for the declared three-way blocked split')
    views=[];information=[]
    for b,mask in zip(bands,training):
        view=copy.copy(b);view.training=mask;views.append(view)
        h=b.waveform(np.zeros(b.b.shape[1]),len(views)-1,False,np.zeros(3))
        h=profile_static(h,b.field.root_information,mask,b.groups)*mask[:,None]
        information.append(float(np.vdot(h,h).real))
    if min(information)<=0:raise ValueError('A proposal band has no known-reference information')
    information=np.asarray(information);trace=[];selected=[];cache={};fit=None;best=None
    best_validation=empty_score(bands,validation);best_count=0
    operator=PhysicalSceneAdjoint(views,pixels,directory/'proposal-worker')
    try:
        for step in range(maximum_sources):
            residual=[b.field.y if fit is None else b.field.y-fit.model[c] for c,b in enumerate(bands)]
            z=operator.evaluate(residual)
            z*=np.exp(-1j*offset)[:,None]/np.sqrt(information[:,None])
            score,phase=statistic(z);score[selected]=-np.inf
            ranking=score.copy();candidates=[]
            for _ in range(proposal_candidates):
                index=int(np.argmax(ranking))
                if not np.isfinite(ranking[index]):break
                candidates.append(index)
                # Proposal diversity only. No nearby location is forbidden
                # in later residual passes or in the final likelihood.
                ranking[np.linalg.norm(pixels[:,:2]-pixels[index,:2],axis=1)<5.]=-np.inf
            if not candidates:raise ValueError('No unused proposal location remains')
            best_trial=None
            for index in candidates:
                cache[index]=[b.waveform(np.zeros(b.b.shape[1]),c,False,pixels[index]-b.reference_point) for c,b in enumerate(bands)]
                support=selected+[index]
                design=[np.stack([cache[j][c] for j in support],axis=2) for c in range(5)]
                if fit is None:initial_beta=offset+phase[index];initial_phi=np.zeros(1)
                else:initial_beta=fit.band_phase;initial_phi=np.r_[fit.source_phase,phase[index]-fit.band_phase[3]]
                candidate_fit=fit_scene(design,bands,training,initial_beta,initial_phi,iterations=phase_iterations)
                if best_trial is None or candidate_fit.cost<best_trial[0].cost:best_trial=(candidate_fit,index)
            previous=None if fit is None else fit.cost
            fit,index=best_trial;selected.append(index);cache={j:cache[j] for j in selected}
            if previous is not None and fit.cost>previous+1e-8*fit.energy:
                raise ValueError('Adding a zero-feasible source increased the original training likelihood')
            test=score_scene(fit,bands,validation)
            record=dict(sources=len(selected),candidate_grid_index=index,point_body_m=pixels[index].tolist(),
                training_explained_fraction=1-fit.cost/fit.energy,inner_validation_explained_fraction=test['explained_fraction'],
                maximum_phase_gradient=fit.optimization['maximum_gradient'],seconds=time.monotonic()-started)
            trace.append(record)
            if test['cost']<best_validation['cost']:
                best_validation=test;best=fit;best_count=len(selected)
            (directory/'trace.json').write_text(json.dumps(dict(complete=False,scope=__doc__,trace=trace,
                selected_grid_indices=selected,best_prefix_sources=best_count),indent=2)+'\n')
            print(json.dumps(record),flush=True)
    finally:operator.close()
    selected=selected[:best_count]
    result=dict(complete=True,scope=__doc__,trace=trace,selected_grid_indices=selected,points_body_m=pixels[selected].tolist(),
        selected_sources=best_count,maximum_sources=maximum_sources,proposal_candidates=proposal_candidates,
        selection='Minimum inner blocked complex prediction error among zero through the declared maximum sources',
        inner_validation=best_validation,outer_fold_used_for_selection=False,relative_band_phases_fixed=True,
        unique_scatterer_positions_proven=False,seconds=time.monotonic()-started)
    if best is None:
        result['outer']=empty_score(bands,outer)
    else:
        design=[np.stack([cache[j][c] for j in selected],axis=2) for c in range(5)]
        final=fit_scene(design,bands,original,best.band_phase,best.source_phase,iterations=phase_iterations)
        result.update(outer=score_scene(final,bands,outer),training=score_scene(final,bands,original),optimization=final.optimization)
        diagnostics=component_diagnostics(final,design,bands,outer)
        result['component_sum_to_combined_power_db']=diagnostics['component_sum_to_combined_power_db']
        arrays=dict(points_body_m=pixels[selected],band_phase_radians=final.band_phase,source_phase_radians=final.source_phase,
            component_excluded_predicted_energy=np.asarray(diagnostics['component_predicted_energy']))
        for c,(b,amplitude) in enumerate(zip(bands,final.amplitude)):
            arrays[f'amplitude_coefficients_{c}']=amplitude;arrays[f'amplitude_{c}']=b.amplitude@amplitude
            arrays[f'time_s_{c}']=b.t
        np.savez_compressed(directory/'scene.npz',**arrays)
    (directory/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
