"""Do common, rigidly translating unresolved points mimic inconsistent band motion?"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from field import Field
from flash_geometry import observables
from recovery_experiment import Experiment,SCALE,NAMES,load_fields
from report_measured_scatterer_phases import serial
from run_quadratic_motion_audit import execute,slim
from quadratic_motion_kernel import QuadraticMotionKernel
from waveform_kernels import KernelField


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    fields,rot,rx,tx,state,mid=load_fields(149.,179.,8)
    truth=np.array([25.,-40.,.07,-.06,.002,-.001])/SCALE
    points=np.array([[0.,0.,0.],[-55.,110.,0.],[55.,-110.,0.]])
    beta=np.array([1.,.65*np.exp(.7j),.5*np.exp(-.9j)])
    scene=Experiment(fields,rot,rx,tx,state,mid,offsets=points)
    primary=Experiment(fields,rot,rx,tx,state,mid);synth=[];histories=[]
    for c,f in enumerate(fields):
        h=primary.templates(truth,c)[:,0];gain=np.sqrt(10*len(f.time)/np.vdot(h,h).real)
        y=scene.independent_signal(truth,c,gain*beta)
        raw=np.linalg.solve(f.whiten,(y.reshape(f.y.shape)/f.root_information[:,None]).T).T
        data={**f.data,'value':raw};base=Field(data,f.pulse.samples)
        synth.append(KernelField(base,f.pulse.kernels,f.pulse.row_blocks))
        hh=h.reshape(f.y.shape);z=np.sum(hh.conj()*y.reshape(f.y.shape),axis=1)/np.sum(abs(hh)**2,axis=1)
        histories.append(dict(name=NAMES[c],time_s=f.time,phase_relative_to_true_point_deg=np.degrees(np.angle(z)),magnitude=abs(z)/gain))
    pve=[s.copy() for s in state]
    for i in range(3):pve[i][:2]+=truth[2*i:2*i+2]*SCALE[2*i:2*i+2]
    local_tx=(tx-rx)@rot.T
    expected=[observables(pve[0]+point,pve[1],pve[2],local_tx,np.zeros(3)) for point in points]
    result=dict(scope=__doc__,complete=False,noise_added=False,band_dependent_material_reflectivity=False,
        point_positions_enu_m=points,truth_parameters=truth,true_point_observables=expected,histories=histories,windows=[])
    def save():(out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    save()
    for duration in (2.,4.,8.,16.,30.):
        window=[f.windowed(mid-duration/2,mid+duration/2) for f in synth]
        ex=Experiment(window,rot,rx,tx,state,mid)
        for c,f in enumerate(window):
            k=QuadraticMotionKernel(f,ex.training[c],ex.designs[c]);half=.5 if duration<=4 else .15 if duration<=8 else .08
            fit=execute(k,f.y.ravel(),acceleration_halfwidth=half)
            result['windows'].append(dict(name=NAMES[c],channel=c,duration_s=duration,**slim(fit)))
            print(NAMES[c],duration,fit['parameters'],flush=True);save()
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for c in range(5):
        entries=[e for e in result['windows'] if e['channel']==c]
        time=[e['duration_s'] for e in entries]
        axes[0,0].plot(time,[e['parameters'][1] for e in entries],'o-',label=NAMES[c])
        axes[0,1].plot(time,[e['parameters'][2] for e in entries],'o-')
        axes[0,2].plot(histories[c]['time_s'],histories[c]['magnitude'],label=NAMES[c])
        axes[1,c//2 if c<4 else 2].plot(histories[c]['time_s'],histories[c]['phase_relative_to_true_point_deg'],'.',ms=2,label=NAMES[c])
    axes[0,0].set(xlabel='Aperture (s)',ylabel='Apparent one-point path rate (m/s)');axes[0,0].legend(fontsize=8)
    axes[0,1].set(xlabel='Aperture (s)',ylabel='Apparent path acceleration (m/s²)')
    axes[0,2].set(xlabel='Time (s)',ylabel='Matched magnitude relative to isolated primary')
    for ax in axes[1]:ax.set(xlabel='Time (s)',ylabel='Phase relative to true primary (°)');ax.legend(fontsize=8)
    fig.suptitle('Noiseless unresolved scene: same three physical points and material gains in every band\nExactly shared planar translation, no roll/pitch/yaw, no multipath, no receiver impairment. One-point fitting is deliberately wrong.')
    fig.savefig(out/'unresolved-controls.png',dpi=150);plt.close(fig)
    result['complete']=True;save()
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Unresolved-scene control</title><style>body{font:16px/1.55 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}img{width:100%}</style>
<h1>Unresolved common-scatterer scene control</h1><p>All five bands see the same three points, all points share one known translation, and their material reflectivities are identical across bands. Exact carrier phases, waveform pulses and original sampling differ physically between bands. No noise, variable brightness, yaw, multipath or receiver errors are introduced.</p><img src="unresolved-controls.png"><p>The one-point kernel is deliberately applied to this extended scene. Any aperture-dependent or interband apparent motion here is model mismatch, not different material targets or a failed numerical forward operator. This is a constructive test of one possible failure mechanism; it is not a fit to the actual ship or proof that this particular three-point layout explains the recording. The saved true path/rate/acceleration for every point permits direct comparison.</p><p><a href="analysis.json">Truth and measured arrays</a> · <a href="../quadratic-motion-kernel-recursive/index.html">Real-data kernel analysis</a></p>''')


if __name__=='__main__':main()
