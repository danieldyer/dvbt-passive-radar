"""Review a continuous physical trajectory without fitting image pixels."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from continuous_ship_motion import ContinuousShipMotion
from continuous_ship_fit import MotionCoordinates
from motion import excess_path,C
from render_injected_point import fitted_state


def snapshot(initial,checkpoint,out):
    m=ContinuousShipMotion(initial);model=MotionCoordinates(m);fit=json.loads(Path(checkpoint).read_text());q=np.asarray(fit['parameters'])[:model.n]
    p,a=model.coefficients(q);d=dict(m.result);d.update(position_coefficients_enu_m=p.tolist(),angle_coefficients_rad=a.tolist(),
        complete=False,stage='Saved full-capture radar iteration; numerical optimisation continues',checkpoint=str(checkpoint),radar_fit=fit['history'][-1])
    Path(out).write_text(json.dumps(d,indent=2)+'\n')


def review(path,out):
    m=ContinuousShipMotion(path);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    t=np.linspace(*m.interval,6001);s=m.state(t);ais=m.track.sample(m.epoch+t,max_gap_s=400.)
    position=(s['position_ecef_m']-m.rx)@m.rotation.T;prior=(ais['position_ecef_m']-m.rx)@m.rotation.T;velocity=s['velocity_ecef_mps']@m.rotation.T
    angle=np.degrees(m.angle_curve(t));acceleration=m.position_curve(t,2)
    fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    ax[0,0].plot(prior[:,0],prior[:,1],color='#00a7ae',label='AIS antenna prior');ax[0,0].plot(position[:,0],position[:,1],label='Radar reference scatterer')
    ax[0,0].set(xlabel='East from receiver (m)',ylabel='North (m)',aspect='equal');ax[0,0].legend()
    for i,label in enumerate(['East','North']):
        ax[0,1].plot(t,position[:,i]-prior[:,i],label=label);ax[0,2].plot(t,velocity[:,i],label=label);ax[1,0].plot(t,acceleration[:,i],label=label)
    ax[0,1].set(ylabel='Radar reference minus AIS antenna (m)');ax[0,1].legend();ax[0,2].set(ylabel='Ground velocity (m/s)');ax[0,2].legend()
    ax[1,0].set(ylabel='Ground acceleration (m/s²)');ax[1,0].legend();ax[1,1].plot(t,angle);ax[1,1].set(ylabel='Physical rotation from reference N/E (degrees)')
    for c,tx in enumerate(m.tx):
        _,g=excess_path(s['position_ecef_m'],tx,m.rx);rate=(g*s['velocity_ecef_mps']).sum(axis=1)
        ax[1,2].plot(t,-m.rf[c]/C*rate,label=['Seven','SBS','Nine','Ten','ABC'][c])
    ax[1,2].set(ylabel='Reference-scatterer geometric Doppler (Hz)');ax[1,2].legend(ncol=2)
    for a in ax.ravel()[1:]:a.set(xlabel='Capture time (s)');a.axvspan(142,172,color='gray',alpha=.12)
    fig.suptitle('One physical trajectory for every band; fixed inter-band calibration');fig.savefig(out/'continuous-motion.png',dpi=150);plt.close(fig)
    d=json.loads((Path(m.result['reference_fit'])/'render-fit.json').read_text());rt=np.linspace(*d['interval_s'],2049)
    ref=fitted_state(rt,np.asarray(d['coarse_state_enu']),d['midpoint_s'],m.rotation,m.rx,d);current=m.state(rt);loss=[]
    for q in m.result['reference_points_east_north_up_m']:
        old=ref['position_ecef_m']+ref['body_to_ecef']@np.asarray(q);new=current['position_ecef_m']+current['body_to_ecef']@np.asarray(q)
        row=[]
        for c,tx in enumerate(m.tx):
            delta=excess_path(new,tx,m.rx)[0]-excess_path(old,tx,m.rx)[0]
            row.append(float(-20*np.log10(abs(np.exp(2j*np.pi*m.rf[c]/C*delta).mean()))))
        loss.append(row)
    record=dict(motion=str(path),stage=m.result.get('stage'),reference_motion_only_mismatch_db=loss,
        reference_metric_scope='Constant-amplitude template correlation against the reviewed local fit; not a measured target SNR or a proof of absolute geographic accuracy',radar_fit=m.result.get('radar_fit'))
    (out/'analysis.json').write_text(json.dumps(record,indent=2)+'\n')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Continuous ship motion</title><style>body{font:17px system-ui;max-width:1400px;margin:2em auto}img{width:100%}pre{white-space:pre-wrap}</style><h1>Continuous ship motion</h1><p>One C2 physical position and orientation curve is shared by all five channels. The shaded interval is the reviewed reference aperture. AIS is a gross motion initializer; its course is not a measured hull heading. Relative band phases remain fixed.</p><img src="continuous-motion.png"><p>The radar reference and the AIS antenna need not coincide. This analysis is conditional on the existing transmitter/receiver calibration and reference-scatterer support. It does not establish a unique absolute position.</p><pre>'+json.dumps(record,indent=2)+'</pre>')
    print(json.dumps(record,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    path=a.motion
    if a.checkpoint:path=a.output/'checkpoint-motion.json';snapshot(a.motion,a.checkpoint,path)
    review(path,a.output)

if __name__=='__main__':main()
