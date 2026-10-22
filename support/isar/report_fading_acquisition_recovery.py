"""Compare acquisition basins using known material-point coherent power."""
import argparse
import json
from pathlib import Path

import numpy as np
from scene_dictionary import fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('docs/data/isar_20261018'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sources=[('Direct local fit','known-fading-physical-motion-recovery-noise-objective'),
        ('30 s constant brightness acquisition','known-fading-coarse-polynomial-recovery'),
        ('30 s matched-power brightness acquisition','known-fading-coarse-magnitude-recovery'),
        ('5 s strongest-point acquisition','known-fading-short-point-acquisition-recovery'),
        ('5 s joint acquisition','known-fading-short-joint-acquisition-recovery'),
        ('Near-truth initialization control','known-fading-near-truth-recovery'),
        ('Known-brightness acquisition control','known-fading-coarse-truth-brightness-recovery')]
    records=[]
    for label,name in sources:
        path=a.root/name/'analysis.json';d=json.loads(path.read_text())
        if not d.get('complete'):raise ValueError('Use completed controls')
        case=next(q for q in d['cases'] if not q['competing_return'])
        records.append(dict(label=label,source=str(path),sha256=fingerprint(path),case=case))
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(15,11),layout='constrained',sharex=True)
    fraction=np.array([r['case']['phase_only_five_band_coherent_fraction_by_point'] for r in records]);x=np.arange(len(records))
    for point in range(fraction.shape[1]):axes[0].plot(x,1-fraction[:,point],'.-',label=f'Material point {point+1}',alpha=.7)
    weighted=np.array([r['case']['phase_only_five_band_coherent_fraction_information_weighted'] for r in records])
    axes[0].plot(x,1-weighted,'ko-',lw=2,label='Information-weighted mean')
    axes[0].set(yscale='log',ylabel='Coherent-power deficit (1 − fraction)',ylim=[1e-5,1.05]);axes[0].legend(ncol=4,fontsize=8)
    gap=[r['case']['objective_gap_from_known_truth'] for r in records]
    axes[1].plot(x,gap,'o-');axes[1].axhline(0,color='grey',ls='--');axes[1].set(yscale='symlog',ylabel='Training profile cost fraction\nminus known-truth cost',xticks=x,xticklabels=[r['label'].replace(' acquisition','\nacquisition').replace(' initialization','\ninitialization') for r in records]);axes[1].tick_params(axis='x',labelrotation=25)
    axes[1].set_yscale('symlog',linthresh=1e-5)
    fig.suptitle('Known fading ship: the acquisition basin matters\nActual waveform kernels, epochs and five RFs; independent Cartesian synthesis; point support and attitude given')
    fig.savefig(a.output/'acquisition-basin-and-coherent-power.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    u=np.linspace(-1,1,2001);true_poly=np.linalg.lstsq(np.c_[u,u*u,u*u*u],-.7*np.sin(np.pi*u),rcond=None)[0]
    for ax,(label,name) in zip(axes,[sources[1],sources[4]]):
        d=json.loads((a.root/name/'chirp-initializer-0.json').read_text());grid=np.array(d['coarse_grid_coefficients_m']);score=np.array(d['coarse_score_fraction'])
        v=np.unique(grid[:,0]);j=np.unique(grid[:,2]);surface=np.array([[score[(grid[:,0]==b1)&(grid[:,2]==b3)].max() for b1 in v] for b3 in j])
        im=ax.imshow(surface,extent=[v[0]-.25,v[-1]+.25,j[0]-.25,j[-1]+.25],origin='lower',cmap='viridis',aspect='equal')
        best=d['selected']['coefficients_m'];ax.plot(true_poly[0],true_poly[2],'x',color='cyan',ms=12,label='Cubic approximation to true correction')
        ax.plot(best[0],best[2],'+',color='red',ms=12,label='Acquired correction')
        ax.set(title=label+'\nQuadratic term profiled; scores have different nuisance dimensions',xlabel='Linear normalized-time path coefficient (m)',ylabel='Cubic normalized-time path coefficient (m)');ax.legend(fontsize=8)
        fig.colorbar(im,ax=ax,shrink=.7,label='Training projection score / crop energy; not SNR')
    fig.savefig(a.output/'long-versus-short-acquisition.png',dpi=150);plt.close(fig)
    rows=[]
    for record in records:
        c=record['case'];loss=c['phase_only_five_band_loss_db'];minimum=min(c['phase_only_five_band_coherent_fraction_by_point'])
        rows.append(f'<tr><td>{record["label"]}</td><td>{loss:.6f}</td><td>{-10*np.log10(minimum):.6f}</td><td>{c["objective_gap_from_known_truth"]:.6g}</td></tr>')
    html='''<!doctype html><meta charset="utf-8"><title>Fading-ship acquisition validation</title><style>body{font:17px/1.5 system-ui;max-width:1250px;margin:40px auto}img{width:100%}td,th{padding:9px;border-bottom:1px solid #ccc}table{border-collapse:collapse}</style><h1>Acquisition versus coherent image loss</h1>
<p>This is a known-material-point control, not a successful full-recording ship reconstruction. Geometry, point support and attitude are given. Actual recorded waveform kernels, times, five VHF frequencies, conditional noise and fitted real brightness envelopes enter independently synthesized Cartesian returns. The estimator must recover one common physical path without knowing the fading envelopes.</p>
<p>The five-second joint acquisition reaches the same best likelihood as near-truth initialization. The direct optimizer can instead stop at a worse basin. A high fitted-mixture coherence score alone cannot detect this; the plots therefore use each known material point and the frozen fitted band calibration.</p>
<table><tr><th>Initializer</th><th>Mean phase-only loss (dB)</th><th>Worst material-point loss (dB)</th><th>Profile cost gap from truth</th></tr>'''+''.join(rows)+'''</table>
<p>Losses isolate propagation/calibration phase. They exclude range-envelope mismatch, other image-formation losses and errors in the given support/attitude. They are not an unconditional radar sensitivity bound. A slightly lower fitted training cost than the true parameters is expected when noise is fitted.</p>
<img src="acquisition-basin-and-coherent-power.png"><img src="long-versus-short-acquisition.png">
<p>Unknown constant reflector phases are profiled. Acquisition amplitudes/phases are nuisance parameters only; final refinement evaluates the full physical range, RF phase, Doppler and original-bin attenuation with fixed reflector phases. No independent time-varying band phase correction enters the final model.</p>'''
    (a.output/'index.html').write_text(html)
    (a.output/'analysis.json').write_text(json.dumps(dict(complete=True,scope=__doc__,records=records,real_recording_focus_achieved=False),indent=2)+'\n')


if __name__=='__main__':main()
