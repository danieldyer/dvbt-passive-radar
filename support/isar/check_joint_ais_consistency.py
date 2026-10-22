"""Check frozen radar fits against actual AIS positions with rigid lever/delay nuisance.

No radar parameter, image or antenna coordinate is changed. AIS is evaluated
only at its original report positions; no interpolated position is a datum.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from motion import ecef
from continuous_ship_motion import ContinuousShipMotion
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fit_lever_delay(state,time,observed,local,limit_s=30.,grid_step_s=.25):
    """Profile a constant two-axis rigid lever analytically at each delay.

    Positive delay means radar_time = AIS_report_time - delay. Lever axes
    are the fitted object's fixed reference axes, requiring no AIS heading.
    No lever bound is imposed: its fitted magnitude is reported explicitly.
    """
    def at(delay):
        s=state(np.asarray(time)-delay)
        design=np.einsum('tij,ik->tkj',s['body_to_ecef'][:,:,:2],local[:,:2]).reshape(-1,2)
        rhs=((observed-s['position_ecef_m'])@local[:,:2]).ravel()
        lever=np.linalg.lstsq(design,rhs,rcond=None)[0];residual=(design@lever-rhs).reshape(-1,2)
        return float(np.sum(residual**2)),lever,residual
    grid=np.linspace(-limit_s,limit_s,max(3,int(np.ceil(2*limit_s/grid_step_s))+1))
    scores=np.array([at(delay)[0] for delay in grid]);candidates=[(-limit_s,scores[0]),(limit_s,scores[-1])]
    for i in range(1,len(grid)-1):
        if scores[i]<=scores[i-1] and scores[i]<=scores[i+1]:
            result=minimize_scalar(lambda delay:at(delay)[0],bounds=grid[i-1:i+2:2],method='bounded',options={'xatol':1e-8})
            candidates.append((float(result.x),float(result.fun)))
    delay,_=min(candidates,key=lambda x:x[1]);cost,lever,residual=at(delay)
    return dict(delay_s=delay,lever_reference_axes_m=lever.tolist(),lever_length_m=float(np.linalg.norm(lever)),
        rms_horizontal_error_m=float(np.sqrt(cost/len(time))),maximum_horizontal_error_m=float(np.linalg.norm(residual,axis=1).max()),
        residual_east_north_m=residual.tolist(),delay_at_search_boundary=bool(abs(delay)>=limit_s-1e-5),
        profile_delay_s=grid.tolist(),profile_rms_horizontal_error_m=np.sqrt(scores/len(time)).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',nargs=2,action='append',required=True,metavar=('LABEL','MOTION_JSON'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--delay-limit-s',type=float,default=30.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    reference=ContinuousShipMotion(a.case[0][1]);track=reference.track;local=reference.rotation.T
    time=track.seconds-reference.epoch
    keep=(time>=reference.interval[0]+a.delay_limit_s)&(time<=reference.interval[1]-a.delay_limit_s)
    time=time[keep];observed=ecef(track.latitude[keep],track.longitude[keep],track.reference_height_ellipsoid_m)
    if len(time)<3:raise ValueError('Need at least three original AIS reports with complete delay-search support')
    fig,ax=plt.subplots(2,2,figsize=(14,10),layout='constrained');records=[]
    for label,path in a.case:
        model=ContinuousShipMotion(path)
        if time[0]-a.delay_limit_s<model.interval[0] or time[-1]+a.delay_limit_s>model.interval[1]:raise ValueError('All cases need identical original AIS reports')
        result=fit_lever_delay(model.state,time,observed,local,a.delay_limit_s)
        result.update(label=label,motion=str(path));records.append(result)
        residual=np.asarray(result['residual_east_north_m'])
        ax[0,0].plot(result['profile_delay_s'],result['profile_rms_horizontal_error_m'],label=label)
        for c in range(2):ax[1,c].plot(time,residual[:,c],'o-',label=label)
        ax[0,1].scatter(*result['lever_reference_axes_m'],label=label)
    ax[0,0].set(xlabel='Delay: radar time = AIS report time minus delay (s)',ylabel='Profiled horizontal position RMS error (m)')
    ax[0,1].set(xlabel='Fitted lever along reference axis 1 (m)',ylabel='Fitted lever along reference axis 2 (m)')
    for c,name in enumerate(['East','North']):ax[1,c].set(xlabel='Original AIS report time within capture (s)',ylabel=name+' position residual (m)');ax[1,c].axhline(0,color='grey',lw=.7)
    fig.legend(*ax[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=8,frameon=False)
    fig.suptitle('Frozen radar trajectories versus original AIS reports; rigid lever and constant delay profiled')
    fig.savefig(a.output/'ais-consistency.png',dpi=150);plt.close(fig)
    report=dict(complete=True,original_report_times_s=time.tolist(),original_reports=len(time),delay_limit_s=a.delay_limit_s,
        scope=__doc__,cases=records,
        interpretation='Geographic consistency diagnostic, not an AIS likelihood in the radar fits. No GPS error variance, heading measurement, timing prior or lever bound is assumed. Delay/lever can be weakly identifiable; a low residual with a very large lever or boundary delay is not confirmation. Reports cover only the stated times, not the remaining AIS gap.')
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    rows=''.join('<tr>'+''.join('<td>'+str(value)+'</td>' for value in [r['label'],f"{r['rms_horizontal_error_m']:.2f}",f"{r['maximum_horizontal_error_m']:.2f}",f"{r['lever_length_m']:.1f}",f"{r['delay_s']:.3f}",r['delay_at_search_boundary']])+'</tr>' for r in records)
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Joint fit AIS consistency</title><style>body{font:16px system-ui;max-width:1500px;margin:2em auto}img{width:100%}td,th{padding:.5em;border:1px solid #aaa}pre{white-space:pre-wrap}</style><h1>Original AIS report consistency</h1><p>'+report['interpretation']+'</p><table><tr><th>Frozen fit</th><th>RMS horizontal error (m)</th><th>Maximum error (m)</th><th>Lever length (m)</th><th>Delay (s)</th><th>Delay on boundary</th></tr>'+rows+'</table><img src="ais-consistency.png"><pre>'+json.dumps(report,indent=2)+'</pre>')
    print(json.dumps(dict(output=str(a.output),original_reports=len(time),cases=[{k:r[k] for k in ['label','rms_horizontal_error_m','lever_length_m','delay_s']} for r in records]),indent=2))


if __name__=='__main__':main()
