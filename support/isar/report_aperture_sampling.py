"""Show the temporal ambiguity introduced by legacy validation masks."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from native_folds import training_mask
from recorded_injection_fields import load_fields
from recovery_experiment import NAMES


def response(time,weight,frequency):
    """Unit-peak power response of the declared weighted temporal aperture."""
    return abs(np.exp(2j*np.pi*np.outer(frequency,time-time.mean()))@weight/weight.sum())**2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    saved=json.loads((args.fit/'analysis.json').read_text());first,last=saved['interval_s']
    observed=saved['observed_range_samples']
    fields,*_=load_fields(Path(saved['profiles']),offsets=np.arange(observed[0],observed[1]+1),
        fixed_epoch_s=saved['midpoint_s'],interval_s=(first,last))
    frequency=np.linspace(-3.,3.,3001);rows=[]
    fig,axes=plt.subplots(5,1,figsize=(12,12),sharex=True,layout='constrained')
    for c,f in enumerate(fields):
        subset=training_mask(f.time,first);weight=f.root_information**2
        record=dict(channel=NAMES[c],samples=len(f.time),subsets=[])
        for mask,label,colour in [(np.ones(len(f.time),bool),'Complete aperture','black'),
                (subset,'Legacy fitting subset','tab:blue'),(~subset,'Legacy unused subset','tab:orange')]:
            power=response(f.time[mask],weight[mask],frequency)
            axes[c].plot(frequency,10*np.log10(np.maximum(power,1e-12)),label=label,color=colour,lw=1.)
            record['subsets'].append(dict(label=label,samples=int(mask.sum()),
                information_fraction=float(weight[mask].sum()/weight.sum()),
                one_hz_replica_power_db=float(10*np.log10(response(f.time[mask],weight[mask],[1.])[0]))))
        axes[c].set(ylabel=NAMES[c]+' (dB)',ylim=(-70,1),xlim=(-3,3))
        axes[c].axvline(1.,color='#aaa',lw=.6);axes[c].axvline(-1.,color='#aaa',lw=.6)
        rows.append(record)
    axes[0].legend(ncol=3);axes[-1].set_xlabel('Residual Doppler offset (Hz)')
    fig.suptitle('Temporal sampling ambiguity from the actual native observation epochs\nUnit peak; known information weights; no target-dependent brightness or fitted phase')
    fig.savefig(out/'aperture-window.png',dpi=160);plt.close(fig)
    result=dict(scope=__doc__,source_fit=str(args.fit),interval_s=[first,last],channels=rows,
        formula='|sum w(t) exp(i 2 pi delta_f t)|^2 / (sum w(t))^2',
        interpretation='Sampling-window diagnostic, not the full bistatic point-spread function or a target-SNR bound.')
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    table=''.join(f'<tr><td>{r["channel"]}</td>'+''.join(f'<td>{g["one_hz_replica_power_db"]:.2f}</td>' for g in r['subsets'])+'</tr>' for r in rows)
    html=f'''<!doctype html><html><meta charset="utf-8"><title>Continuous aperture and validation masks</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;line-height:1.5}}img{{width:100%}}td,th{{padding:8px;text-align:right}}</style>
<h1>Continuous aperture and validation masks</h1>
<p>Motion estimation and image formation are deterministic statistical estimation, not machine-learning training. Withheld samples can diagnose model mismatch, but a periodic mask changes the measurement aperture and its ambiguity function. It cannot establish the full-aperture resolution or SNR.</p>
<p>The previous diagnostic retained one quarter-second each second as unused observations. For dense uniform sampling, that unused aperture has a 1 Hz replica at 20 log<sub>10</sub>|sinc(0.25)| = −0.912 dB. The other three quarters have a replica at −10.455 dB. The measured sampling responses below use the actual epochs and known information weights from {first:g}–{last:g} s.</p>
<table><tr><th>Channel</th><th>Complete aperture, +1 Hz (dB)</th><th>Legacy fitting subset</th><th>Legacy unused subset</th></tr>{table}</table><img src="aperture-window.png">
<p>Both motion fitting and final imaging now default to every available sample. Legacy masks remain explicit diagnostic options so old likelihoods can still be reproduced. Final fit residuals are labelled as in-sample diagnostics, never independent recovery evidence.</p>
<p>Primary validation uses independent physical forward models, known injections in recorded data, native-symbol/GPU agreement, parameter recovery, and the continuous-aperture point response. The continuous point response still has physical waveform/bandwidth sidelobes; removing the validation mask does not remove those or certify a resolved ship image.</p>
<p><a href="../five-point-continuous-nuisance-image/point-image.png">Continuous-aperture point control</a> · <a href="../five-point-continuous-nuisance-image/observed-image.png">Continuous-aperture recorded scene before the all-sample refit</a> · <a href="analysis.json">Sampling measurements</a></p></html>'''
    (out/'index.html').write_text(html)
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
