"""Acquire additional phase/range features near the main flash return.

Use training rows only and the corrected independent four-return model.
Remove its static/active-brightness and local trajectory nuisance from both
the residual and trial echoes. Distinct points may share the same range cell.
The midpoint tone/chirp approximation is acquisition only: every selected
hypothesis needs an original-row nonlinear fit and unused-data validation.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter

from audit_joint_kernel_phase import known_band
from cluster_phase import SplinePath
from compare_ais_phase import NAMES
from efficient_echo_score import scores
from fit_fixed_grid_phases import prepare_fixed
from motion import C
from report_measured_scatterer_phases import load_fields,serial
from smooth_echo_cluster import SmoothEchoCluster
from waveform_kernels import read_kernels


def fitted_nuisance(cluster,parameters,free):
    """Prescale before nuisance elimination; don't promote numerical nulls."""
    b=cluster.base;v=cluster.evaluate(parameters);active=v['amplitude']>0
    original=v['design'][b.selected][:,active];projected=b.nuisance.remove(original)
    a=np.r_[projected.real,projected.imag];norm=np.linalg.norm(original,axis=0)
    raw=[];k=len(b.knots)
    for (_,derivative),amplitude in zip(cluster.response_blocks(parameters),v['amplitude'].reshape(cluster.components,k)):
        raw.append(b.reduced(derivative*(b.amplitude_basis@amplitude)[:,None,None])[b.selected])
    original_derivative=np.concatenate(raw,axis=1)
    columns=np.c_[a,v['mean_derivative'][:,free]]
    scale=np.r_[norm,np.linalg.norm(original_derivative[:,free],axis=0)]
    columns=columns/np.where(scale>0,scale,1.)
    q,s,_=np.linalg.svd(columns,full_matrices=False)
    keep=s>1e-8*max(1.,s[0]);q=q[:,keep]
    residual=b.clean-b.nuisance.remove(v['design'][b.selected])@v['amplitude']
    return residual,q,dict(rank=int(keep.sum()),scaled_singular_values=s,
        rank_policy='Normalize by pre-projection derivative/design norms, then absolute-relative 1e-8 numerical rank. No model priors counted as measured information.')


def main():
    root=Path('docs/data/isar_20261017');out=root/'main-return-residual-acquisition'
    if out.exists():raise ValueError('Use a fresh output directory')
    source=root/'longer-known-phases-6s/fits.json';saved=json.loads(source.read_text())
    if not saved['complete']:raise ValueError('Wait for independent source fits')
    baseline_path=Path(saved['source']);baseline=json.loads(baseline_path.read_text())
    std=saved['validation_selected_jerk_std'];group=next(g for g in saved['groups'] if g['jerk_std']==std)
    geometry=Path('docs/data/calibration_20261016_sites');first,last=saved['interval_s']
    mean,parts,_=prepare_fixed(load_fields(Path('build/isar-20261015/full-fields'),geometry),geometry,first,last,stationary=True)
    kernel=Path('build/isar-20261016/waveform-kernels-30s');_,records,pulses=read_kernels(kernel/'response.kernels')
    origins=np.arange(-2.5,2.501,.25);rates=np.arange(-2.5,.501,.05);accelerations=np.array([-.2,-.1,0.,.1,.2])
    rr,aa=np.meshgrid(rates,accelerations);rate=rr.ravel();acceleration=aa.ravel()
    maps=[];retentions=[];diagnostics=[];errors=[]
    curve=SplinePath(first,last,saved['knot_spacing_s'])
    with np.load(kernel/'rows.npz') as grids:
        bands=[known_band(b,grids[f'channel_{c}'],records[c],pulses[c]) for c,b in enumerate(mean)]
    for c,b in enumerate(bands):
        model=SmoothEchoCluster([b],baseline['selected_origins_samples'],(1,3),curve,shared=False,jerk_std=std)
        fit=group['independent'][c]['trials'][group['independent'][c]['best_index']]
        p=np.array(fit['parameters']);r,_=model.evaluate(p);error=abs(r@r*model.energy-fit['cost']);errors.append(error)
        if error>1e-7:raise ValueError('Original fitted objective changed')
        cluster=model.clusters[0];local=p[model.indices[0]];free=np.arange(len(local))
        if c==3:
            block=cluster.slices[2];free=np.setdiff1d(free,np.arange(block.start+2,block.stop))
        residual,q,detail=fitted_nuisance(cluster,local,free)
        dt=b.field.time-b.epoch
        tone=np.exp(-1j*b.field.k*(dt[:,None]*rate[None,:]+.5*dt[:,None]**2*acceleration[None,:]))
        power=np.zeros((len(accelerations),len(rates),len(origins)));retained=power.copy()
        for j,origin in enumerate(origins):
            raw,_=b.response(np.array([origin,0.,0.,0.]))
            unit=b.reduced(raw[:,:,None])[:,0].reshape(b.field.y.shape)
            h=(unit[:,:,None]*tone[:,None,:]).reshape(b.field.y.size,len(rate))[b.selected]
            original_norm=np.sum(abs(h)**2,axis=0);projected=b.nuisance.remove(h)
            answer=scores(projected,residual,q)
            power[:,:,j]=answer['score'].reshape(len(accelerations),len(rates))
            retained[:,:,j]=(answer['retained_information']*np.sum(abs(projected)**2,axis=0)/np.maximum(original_norm,1e-300)).reshape(len(accelerations),len(rates))
        maps.append(power);retentions.append(retained);diagnostics.append(detail)
        print(NAMES[c],'maximum residual score',power.max(),'nuisance rank',detail['rank'],flush=True)
    joint=np.min(maps,axis=0);best_acc=np.argmax(joint,axis=0);score=np.max(joint,axis=0)
    peaks=(score==maximum_filter(score,size=3,mode='constant',cval=-np.inf));peaks[[0,-1]]=False;peaks[:,[0,-1]]=False
    locations=sorted(zip(*np.nonzero(peaks)),key=lambda p:-score[p])[:12]
    candidates=[dict(origin_samples=origins[j],origin_m=origins[j]*C/8e6,reference_path_rate_mps=rates[i],
        acceleration_mps2=accelerations[best_acc[i,j]],worst_band_training_score=score[i,j],
        band_scores=[m[best_acc[i,j],i,j] for m in maps],
        band_information_retained=[m[best_acc[i,j],i,j] for m in retentions]) for i,j in locations]
    out.mkdir();np.savez(out/'score-grids.npz',scores=np.array(maps),retained=np.array(retentions),origins=origins,rates=rates,accelerations=accelerations)
    fig,axes=plt.subplots(1,5,figsize=(20,5.4),layout='constrained',sharex=True,sharey=True)
    upper=np.ceil(10*np.log10(max(m.max() for m in maps)))
    for c,ax in enumerate(axes):
        # Keep the same common acceleration hypothesis in each band/pixel;
        # each channel does not independently maximize a different chirp.
        image_data=np.take_along_axis(maps[c],best_acc[None,:,:],axis=0)[0]
        image=ax.imshow(10*np.log10(np.maximum(image_data,1e-8)),origin='lower',aspect='auto',cmap='magma',
            extent=[origins[0]*C/8e6,origins[-1]*C/8e6,rates[0],rates[-1]],vmin=0,vmax=upper,interpolation='nearest')
        for j,p in enumerate(candidates[:6]):
            ax.plot(p['origin_m'],p['reference_path_rate_mps'],'s',mfc='none',mec='#00e5ff',ms=8)
            ax.annotate(str(j+1),(p['origin_m'],p['reference_path_rate_mps']),xytext=(5,5),textcoords='offset points',color='#00e5ff',fontsize=8)
        ax.set(title=NAMES[c],xlabel='Offset from fixed range reference (m)')
    axes[0].set_ylabel('Reference-band path rate (m/s)');fig.colorbar(image,ax=axes,label='Training score in original conditional noise units (dB)',shrink=.7)
    fig.suptitle('Search near the main ship return after the fitted returns and their local nuisance directions\n'
        'No minimum range separation: points may differ mainly in Doppler. Acquisition hypotheses only, not accepted scatterers.',fontsize=12)
    fig.savefig(out/'01-residual-candidates.png',dpi=150);plt.close(fig)
    result=dict(scope=__doc__,complete=True,accepted_calibration=False,production_changed=False,
        interval_s=saved['interval_s'],candidate_hypotheses=candidates,nuisance=diagnostics,
        original_objectives_reproduced=len(errors),maximum_absolute_cost_error=max(errors),
        acquisition_approximation='Use exact original-weight midpoint-range/C0/C1 response times retained-bin carrier chirps. New trial range migration and Doppler changes inside 16-row means are omitted only for acquisition; final fits must use the full original-row operator. All scores use training rows only; no score is a calibrated detection claim after scanning.',
        validation_required='Fit each proposed extra point jointly on original rows; compare unused complex prediction, nonlinear nuisance-marginalized phase precision and component cancellation. Do not assign a hull point from a score peak alone.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (source,baseline_path,Path(__file__),Path('support/isar/efficient_echo_score.py'),Path('support/isar/test_efficient_echo_score.py'))})
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    print(out/'01-residual-candidates.png',flush=True)


if __name__=='__main__':main()
