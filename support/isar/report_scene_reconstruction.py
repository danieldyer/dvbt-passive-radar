"""Review conditional support recovery and blocked complex-data prediction."""
import argparse,json
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    read=lambda name:json.loads((a.root/name/'analysis.json').read_text())
    hull=read('known-hull-blind-support-reconstruction');real=read('recorded-flash-blind-support-reconstruction')
    constant=read('recorded-flash-constant-visibility-reconstruction');fixed=read('recorded-flash-amplitude-predictability')
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    truth=np.asarray(hull['truth']['points_body_m']);selected=np.asarray(hull['points_body_m']);ax=axes[0,0]
    ax.plot(truth[:,0],truth[:,1],'+',color='#00a6b5',markersize=12,label='Synthetic locations, withheld from pursuit')
    ax.scatter(selected[:,0],selected[:,1],s=65,facecolors='none',edgecolors='black',label='Selected inverse support')
    ax.set(aspect='equal',xlabel='Body x (m)',ylabel='Body y (m)',title='Given motion/calibration: blind spatial support')
    ax.legend(fontsize=8)
    ax=axes[0,1]
    for d,label in [(real,'5 s positive brightness'),(constant,'Constant brightness')]:
        x=[r['sources'] for r in d['trace']]
        ax.plot(x,[100*r['training_explained_fraction'] for r in d['trace']],':',label=label+' training')
        ax.plot(x,[100*r['inner_validation_explained_fraction'] for r in d['trace']],label=label+' 2 s gaps')
    ax.axhline(0,color='black',linewidth=.7);ax.set(xlabel='Greedy sources',ylabel='Crop energy predicted (%)',title='Recorded flash: blind sparse acquisition fails transfer')
    ax.legend(fontsize=8)
    ax=axes[1,0];models=[d for d in fixed['models'] if d['name']=='inner-training']
    positions=np.arange(len(models));labels=['constant' if r['amplitude_spacing_s']==0 else f"{r['amplitude_spacing_s']:g} s" for r in models]
    ax.bar(positions-.16,[100*m['training']['explained_fraction'] for m in models],.32,label='Fit observations')
    ax.bar(positions+.16,[100*m['inner_validation']['explained_fraction'] for m in models],.32,label='2 s omitted blocks')
    ax.axhline(0,color='black',linewidth=.7);ax.set(xticks=positions,xticklabels=labels,xlabel='Visibility basis, same eighteen points',ylabel='Crop energy predicted (%)',title='Flexible brightness does not predict longer gaps')
    ax.legend(fontsize=8)
    ax=axes[1,1]
    for name,label in [('recorded-flash-distributed-constant-scene-v2','Recorded distributed scene'),('known-hull-distributed-constant-scene-v2','Synthetic distributed scene')]:
        path=a.root/name/'analysis.json'
        if path.exists():
            d=json.loads(path.read_text());ax.semilogx([v['relative_strength'] for v in d['trials']],
                [100*v['inner_validation']['explained_fraction'] for v in d['trials']],'.-',label=label)
    ax.axhline(0,color='black',linewidth=.7);ax.set(xlabel='Ridge / largest initial Gram eigenvalue',ylabel='Inner block energy predicted (%)',title='Regularization is selected by prediction')
    if ax.lines:ax.legend(fontsize=8)
    fig.suptitle('Exact retained-waveform likelihood, five bands; no independent time-varying phase corrections\nPrediction fractions include other returns/interference and are not target SNR efficiencies')
    fig.savefig(a.output/'support-and-prediction.png',dpi=150);plt.close(fig)
    maximum=max(hull['nearest_true_point_distance_m'])
    html=f'''<!doctype html><meta charset="utf-8"><title>Ship scene recovery and blocked prediction</title>
<style>body{{font:17px/1.5 system-ui;max-width:1200px;margin:40px auto;padding:20px}}img{{width:100%}}.status{{background:#fff1d6;padding:18px}}</style>
<h1>Blind scene recovery and the recorded-data model mismatch</h1>
<p class="status">The full-capture focus goal remains active. These are conditional diagnostic experiments, not accepted hull reconstructions.</p>
<p>The synthetic point is recovered exactly on the 0.75 m grid, without supplying its location. Eleven synthetic scatterers are all recovered within {maximum:.2f} m; thirteen sources are selected. Motion, relative band calibration and original waveform epochs are given. Truth is grid-aligned, and synthesis uses the same tested physical response family. This does not establish blind motion or calibration recovery.</p>
<img src="support-and-prediction.png">
<p>Both the 5-second and constant-brightness blind sparse fits select the null model on the recorded flash when judged on alternating two-second omitted blocks. Adding sources improves training fit but worsens prediction. Their selected coefficient maps therefore must not be presented as focused images of the recording.</p>
<p>Even the previously selected eighteen-point support fails this longer-gap test with flexible brightness. Constant brightness predicts a small positive fraction across the same gaps. This is a predictive-model limitation, distinct from the separately validated phase of the GPU adjoint. The distributed-scene test uses a Gaussian ridge prior with complex coefficients common to all bands and five positive constant spectrum magnitudes. Its prior strength is selected on inner blocks; the original outer fold is scored after that selection.</p>
<p>The upstream motion/support used the full original training fold, including inner validation observations. These are exploratory conditional comparisons. No inner score is represented as a blinded end-to-end test. The source covariance, physical uniqueness and vessel outline are not proved.</p>
<p><a href="../physical-scene-adjoint-validation.json">Synthetic GPU/CPU operator check</a> · <a href="../recorded-scene-adjoint-validation.json">Recorded-location check</a> · <a href="../regularized-scene-validation.log">Ridge/spectrum solver tests</a> · <a href="../index.html">Main report</a></p>'''
    (a.output/'index.html').write_text(html)


if __name__=='__main__':main()
