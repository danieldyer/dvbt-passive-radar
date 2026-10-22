"""Interpret finite geometry perturbations under the declared Gaussian model."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr


def pair_error(distance):
    """Exact equal-prior error for CN(mu0,I) versus CN(mu1,I).

    distance is ||mu1-mu0||^2, also their KL divergence. The real log likelihood
    ratio has variance 2*distance and mean +/-distance. This probability concerns
    two specified hypothetical means, not model selection over fitted captures.
    """
    distance=np.asarray(distance,dtype=float)
    if not np.all(np.isfinite(distance)&(distance>=0)):
        raise ValueError('Squared whitened mean distance must be finite and nonnegative')
    return ndtr(-np.sqrt(distance/2))


def render(directory):
    path=directory/'shared-seed-finite-site-sensitivity.json'
    if not path.exists():return ''
    data=json.loads(path.read_text())
    records=[r for r in data['records'] if r['displacement_m']==10]
    names=['Ornata','Eyre','Ridge']
    directions={'towards_receiver':'towards receiver','east':'east','north':'north','up':'up'}
    keys=['unknown_common_target_reflectivity_squared_distance',
          'unknown_common_target_and_channel_phase_squared_distance',
          'unknown_per_band_target_reflectivity_squared_distance']
    labels=['Common target response; band phases fixed',
            'Common target response; shared band phases free',
            'Independent complex target response in each band']
    values=np.array([[r[key] for key in keys] for r in records])
    figure,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    y=np.arange(len(records))
    for i,(label,colour) in enumerate(zip(labels,['#9cb8cd','#2f779f','#b97636'])):
        axes[0].barh(y+(i-1)*.23,values[:,i],height=.21,label=label,color=colour)
        axes[1].plot(100*pair_error(values[:,i]),y+(i-1)*.23,'o',ms=4,color=colour)
    text=[names[r['site']]+' · '+directions[r['direction']] for r in records]
    axes[0].set(yticks=y,yticklabels=text,xscale='log',xlabel='Squared whitened mean distance (KL)',
                title='Change caused by a +10 m site displacement',ylim=(len(records)-.5,-.5))
    figure.legend(*axes[0].get_legend_handles_labels(),fontsize=8,loc='outside lower center',ncols=3)
    axes[1].set(yticks=y,yticklabels=[],xlabel='Optimal equal-prior binary error (%)',
                title='Conditional Gaussian discrimination',xlim=(-1,51),ylim=(len(records)-.5,-.5))
    figure.suptitle('Optimistic fixed-known-trajectory experiment on the revised observations\n'
                   'No coordinate prior; hypothetical ideal point responses at measured band powers')
    figure.savefig(directory/'shared-seed-geometry-discrimination.png',dpi=150)
    plt.close(figure)
    rows=[];summary=[]
    for record,value in zip(records,values):
        probability=pair_error(value)
        rows.append('<tr><td>'+names[record['site']]+'</td><td>'+directions[record['direction']]+
                    '</td>'+''.join(f'<td>{v:.4g}</td><td>{100*p:.2f}%</td>' for v,p in zip(value[1:],probability[1:]))+'</tr>')
        summary.append({'site':record['site'],'direction':record['direction'],
                        'displacement_m':10,'squared_mean_distance':dict(zip(keys,value.tolist())),
                        'equal_prior_binary_error':dict(zip(keys,probability.tolist()))})
    result={'source':str(path),'records':summary,
        'assumptions':data['scope'],
        'formula':'For proper complex unit-variance Gaussian noise, KL=||mu1-mu0||^2 and optimal equal-prior error=Phi(-sqrt(KL/2)).',
        'scope':'Two specified hypothetical mean signals within the retained calibration observations. Optimized nuisance values exhibit alternative means; no claim of globally optimal nuisance fitting is needed to exhibit such a pair. Actual noise/interference, trajectories and scattering remain conditional assumptions. This is not an experimental CRLB or proof that every other return in the recording is unusable.'}
    (directory/'shared-seed-geometry-discrimination.json').write_text(json.dumps(result,indent=2)+'\n')
    return '''<h2 id="geometry-discrimination">What a transmitter displacement actually changes</h2>
<p>This is a controlled forward-model experiment, not another attempt to minimise
the real-data fitting error. The ten candidate trajectories and receiver position
are held exactly known. Ideal point echoes have each band's independently measured
power, including bands rejected by the common-phase fit. Exact path lengths and
range responses are reevaluated after moving one transmitter site.</p>
<img src="shared-seed-geometry-discrimination.png" alt="Whitened differences and binary discrimination probabilities for ten metre transmitter changes">
<p>For proper complex Gaussian noise with covariance I, the squared mean difference
D = ||μ₁−μ₀||² is the KL divergence. The optimal error probability for distinguishing
these two specified means with equal prior probability is Φ(−√(D/2)). This follows
from the Gaussian log likelihood ratio: its mean is ±D and variance 2D. It provides
a conditional scale for the forward-model differences, not a probability that a
particular licensed site assignment is correct.</p>
<table><tr><th>Site</th><th>+10 m direction</th><th>D: common target / free band phases</th>
<th>Binary error</th><th>D: independent band responses</th><th>Binary error</th></tr>'''+''.join(rows)+'''
</table><p>The information depends strongly on the scattering assumptions. Some
directions are weak even when every trajectory is treated as perfectly known.
Allowing independent complex band responses makes many site changes almost
indistinguishable in these selected observations. Freeing unknown trajectories
cannot add information. These calculations use the existing empirical covariance
and retained observation subspace; they do not certify the full recording's CRLB.</p>
<p><a href="shared-seed-geometry-discrimination.json">Values and assumptions</a> ·
<a href="shared-seed-finite-site-sensitivity.json">All ±10, ±30 and ±100 m perturbations</a>.</p>'''
