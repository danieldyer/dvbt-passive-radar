"""Resolve band-phase consistency in time without changing fitted trajectories."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from coherent_fit import OTHER
from complex_likelihood import projected_profile
from road_fit import load_trial


def block_field(field, start, end):
    return field.windowed(start,end)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fit','histories','patches','templates','routes','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=3.)
    args=parser.parse_args()
    if args.seconds<=0:parser.error('Positive block duration required')
    saved=json.loads(args.fit.read_text())
    joint=load_trial(saved,args.routes,args.histories,args.patches,args.templates,allow_time_dependent=True)
    state=np.array(saved['state']);beta=np.zeros(5)
    beta[OTHER]=state[joint.beta_offset:joint.magnitude_offset]
    biases=np.zeros(5)
    if joint.range_bias:biases[OTHER]=state[joint.bias_offset:]
    result={'source_fit':str(args.fit),'seconds':args.seconds,'targets':[],
        'scope':'Non-overlapping short-window phase contrasts with fitted geometry, calibration and paths fixed. Every band is measured independently. No phase correction or calibration update. Conditional energy uses the existing empirical covariance; it is not an absolute SNR measurement or a held-out validation.'}
    colours=['#2373bb','#da8a15','#228d59','#9a58b5','#cf4b48']
    names=['Seven','SBS','Nine','Ten','ABC']
    fig,axes=plt.subplots(joint.m,2,figsize=(12,2.25*joint.m),layout='constrained')
    axes=np.asarray(axes).reshape(joint.m,2)
    for index,problem in enumerate(joint.problems):
        local=np.r_[state[9+index*joint.n:9+(index+1)*joint.n],state[:9]]
        start=problem.model.seed['epoch']-problem.model.half
        end=start+2*problem.model.half
        boundaries=np.arange(start,end+.001,args.seconds)
        if end-boundaries[-1]>=args.seconds/2:boundaries=np.r_[boundaries,end]
        paths=[problem.model.paths(local,field.time,np.full(len(field.time),ch),problem.sites,True)[0]
               for ch,field in enumerate(problem.fields)]
        phases=[];energy=[]
        for first,last in zip(boundaries[:-1],boundaries[1:]):
            amplitudes=[];norms=[]
            for ch,field in enumerate(problem.fields):
                sliced=block_field(field,first,last)
                keep=(field.time>=first)&(field.time<last)
                u,_=sliced.template(paths[ch][keep],range_bias=biases[ch])
                _,a=projected_profile(sliced.y,u)
                amplitudes.append(a);norms.append(np.vdot(u,u).real)
            a=np.asarray(amplitudes)
            phases.append(np.degrees(np.angle(a*np.conj(a[1])*np.exp(-1j*beta))))
            energy.append(abs(a)**2*np.array(norms))
        times=(boundaries[:-1]+boundaries[1:])/2
        phases=np.array(phases);energy=np.array(energy)
        for ch in range(5):
            if ch!=1:
                axes[index,0].scatter(times,phases[:,ch],c=colours[ch],s=18,label=names[ch])
            axes[index,1].plot(times,10*np.log10(np.maximum(energy[:,ch],1e-30)),
                               '.-',color=colours[ch],ms=4,lw=.8,label=names[ch])
        for ax in axes[index]:ax.set_xlabel('Capture time (s)')
        axes[index,0].axhline(0,color='#b8bdc1',lw=.7)
        axes[index,0].set(ylim=(-185,185),yticks=[-180,-90,0,90,180],
                         ylabel=saved['labels'][index]+' phase (degrees)')
        axes[index,1].set(ylabel='Focused energy (dB)')
        result['targets'].append({'label':saved['labels'][index],'times_s':times.tolist(),
            'phase_contrast_degrees':phases.tolist(),'independent_focused_energy':energy.tolist()})
    axes[0,0].legend(ncol=4,fontsize=8)
    axes[0,1].legend(ncol=5,fontsize=8)
    axes[0,0].set_title('Phase relative to SBS after the fitted calibration')
    axes[0,1].set_title('Independent-band signal energy in each short block')
    fig.suptitle(f'{args.seconds:g} s blocks · same fitted path and calibration throughout\nConditional model diagnostic; these curves are not used to retime or rephase the recording')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'),dpi=150);plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
