"""Show the data-fit/smoothness trade-off of the explicit physical prior."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from continuous_ship_motion import ContinuousShipMotion
from continuous_ship_fit import MotionCoordinates
from kinematic_prior import derivative_quadrature


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--fits',type=Path,nargs='+',required=True);p.add_argument('--labels',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if len(a.fits)!=len(a.labels):raise ValueError('One label per fit')
    a.output.mkdir(parents=True,exist_ok=True);base=ContinuousShipMotion(a.motion);model=MotionCoordinates(base);t=np.linspace(*base.interval,6001)
    fig,ax=plt.subplots(2,2,figsize=(14,9),layout='constrained');records=[]
    for directory,label in zip(a.fits,a.labels):
        final=directory/'analysis.json';path=final if final.exists() else directory/'best-progress.json';d=json.loads(path.read_text());finished=path==final
        latest=d if finished else d['history'][-1];coefficient=np.asarray(d['parameters']);pc,ac=model.coefficients(coefficient[:model.n]);curve=BSpline(model.pk,pc,3)
        v=curve(t,1);acc=curve(t,2);jerk=curve(t,3)
        _,w,design=derivative_quadrature(model.pk,3,3);integral=float(np.sum(w[:,None]*(design@pc)**2));rms=np.sqrt(integral/np.ptp(t))
        rss=latest.get('radar_rss_fraction',latest['objective']);iterations=latest.get('iteration',latest.get('iterations'))
        record=dict(label=label,fit=str(directory),fit_finished=finished,iteration=iterations,radar_rss_fraction=rss,explained_data_energy_fraction=1-rss,
            integrated_squared_jerk_m2_s5=integral,rms_jerk_mps3=float(rms),maximum_horizontal_acceleration_mps2=float(np.linalg.norm(acc[:,:2],axis=1).max()))
        records.append(record);suffix='' if finished else f' (iteration {iterations})'
        ax[0,0].plot(t,v[:,1],label=label+suffix);ax[0,1].plot(t,np.linalg.norm(acc[:,:2],axis=1),label=label)
        ax[1,0].plot(t,np.linalg.norm(jerk,axis=1),label=label);ax[1,1].scatter(1-rss,rms,s=80,label=label)
    ax[0,0].set(ylabel='North velocity (m/s)',xlabel='Capture time (s)');ax[0,1].set(ylabel='Horizontal acceleration (m/s²)',xlabel='Capture time (s)')
    ax[1,0].set(ylabel='Jerk magnitude (m/s³)',xlabel='Capture time (s)');ax[1,1].set(xlabel='Explained data energy / total data energy',ylabel='RMS jerk (m/s³)',yscale='log')
    for x in ax.ravel():x.legend()
    fig.suptitle('Physical smoothness versus the unchanged radar likelihood');fig.savefig(a.output/'jerk-prior-comparison.png',dpi=150);plt.close(fig)
    report=dict(scope='In-sample diagnostic; explained data energy is not target SNR. A motion prior does not establish geographic identifiability.',fits=records)
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Ship jerk prior</title><style>body{font:17px system-ui;max-width:1400px;margin:2em auto}img{width:100%}pre{white-space:pre-wrap}</style><h1>Explicit kinematic regularisation</h1><p>The previous fit had hard acceleration and jerk limits but no smoothness preference inside those limits. The added prior penalises the time integral of squared translational jerk and squared angular acceleration. It is evaluated on the shared physical trajectory, with analytic gradients and exact polynomial-span integrals. No phase correction or normalisation is applied to the observations or images.</p><p>The Cartesian prior is a finite-basis version of the <a href="https://arxiv.org/abs/1809.06518">white-noise-on-jerk motion prior</a>. Its strength is specified by the standard deviation of acceleration/rate increments over 30 seconds. These scales are provisional; the trade-off must be inspected alongside waveform focus and raw radar residuals.</p><img src="jerk-prior-comparison.png"><p>Curves marked with an iteration are intermediate results. Explained data energy is an in-sample model-fit statistic, not a measurement of target SNR or independent validation.</p><pre>'+json.dumps(report,indent=2)+'</pre>')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
