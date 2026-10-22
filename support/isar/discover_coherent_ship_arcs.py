"""Full-capture acquisition of ship arcs visible coherently in all five bands.

This is candidate discovery, not a geographic calibration likelihood. A
three-second constant residual-rate template profiles a stationary tone with
the archived AIS deramp. The 16-symbol midpoint approximation is used only
here; selected candidates must be checked by the exact original-row model.
Training samples alone supply the scores. No one-second flash is privileged.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular

from compare_ais_phase import NAMES
from motion import C


def scores(product,weight,elapsed,static_phase,k,rates,spatial_norm):
    """Weighted tone likelihood after one complex stationary nuisance."""
    weight=np.asarray(weight);total=weight.sum()
    if total<=0:raise ValueError('Acquisition needs nonzero sample information')
    static=np.exp(1j*static_phase)
    stationary_coefficient=np.sum(weight*static.conj()*product)/total
    clean=product-static*stationary_coefficient
    tone=np.exp(-1j*k*elapsed[:,None]*rates)
    overlap=(weight*static.conj())@tone
    retained=np.maximum(0.,1-abs(overlap)**2/total**2)
    cross=(weight*clean)@tone.conj()
    energy=total*spatial_norm*retained
    value=np.divide(abs(cross)**2,energy,out=np.zeros_like(energy),where=retained>1e-10)
    return value,retained


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fields',type=Path,default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'discovery.json').exists():raise ValueError('Use a fresh output')
    origins=np.arange(-6,7);rates=np.linspace(-3.,3.,97);seconds=np.arange(4.,598.)
    per_band=np.zeros((5,len(origins),len(seconds),len(rates)),np.float32)
    fraction=np.zeros((5,len(seconds),len(rates)),np.float32)
    sources=[Path(__file__)]
    for ch in range(5):
        path=args.fields/f'channel-{ch}.npz';sources.append(path)
        with np.load(path) as data:
            time,info,value=data['time'],data['information'],data['value']
            pulse,offsets=data['pulse_samples'],data['offset_samples']
            k=2*np.pi*json.loads(str(data['metadata_json']))['rf_hz']/C
            path_at_bin=np.interp(time,data['row_time'],data['centre_path_m'])
            for j,origin in enumerate(origins):
                use=np.flatnonzero(abs(offsets-origin)<=2);r=offsets[use]
                covariance=pulse[r[:,None]-r[None,:]+64]
                whiten=solve_triangular(np.linalg.cholesky((covariance+covariance.conj().T)/2),np.eye(len(r)),lower=True)
                h=whiten@pulse[r-origin+64]
                product=(value[:,use]@whiten.T)@h.conj();norm=float(np.vdot(h,h).real)
                for n,first in enumerate(seconds):
                    selected=(time>=first)&(time<first+3)&((np.floor(time/.125).astype(int)%2)==0)&(info>0)
                    if np.count_nonzero(selected)<30:continue
                    elapsed=time[selected]-(first+1.5);weight=8192*info[selected]
                    phase=k*(path_at_bin[selected]-np.interp(first+1.5,time,path_at_bin))
                    per_band[ch,j,n],fraction[ch,n]=scores(product[selected],weight,elapsed,phase,k,rates,norm)
        print('full-capture acquisition',NAMES[ch],flush=True)
    useful=np.min(fraction,axis=0)>=.2
    # A minimum-band score selects candidates visible in every band. It is
    # an association screen, not the joint GLRT or a calibrated detection p-value.
    minimum=np.min(per_band,axis=0)
    objective=np.where(useful[None,:,:],minimum,-np.inf)
    index=objective.argmax(axis=2);maximum=np.take_along_axis(objective,index[:,:,None],axis=2)[:,:,0]
    available=np.ones(maximum.shape,bool);candidates=[]
    for number in range(12):
        j,n=np.unravel_index(np.argmax(np.where(available,maximum,-np.inf)),maximum.shape)
        if not np.isfinite(maximum[j,n]):break
        r=index[j,n]
        candidates.append({'label':f'F{number+1:02d}','first_s':float(seconds[n]),'last_s':float(seconds[n]+3),
            'centre_sample':int(origins[j]),'seed_residual_path_rate_mps':float(rates[r]),
            'minimum_band_training_score':float(maximum[j,n]),'band_training_scores':per_band[:,j,n,r].tolist(),
            'band_template_energy_retained':fraction[:,n,r].tolist()})
        available&=~((abs(origins[:,None]-origins[j])<3)&(abs(seconds[None,:]-seconds[n])<8))
    np.savez_compressed(args.output/'scores.npz',per_band=per_band,stationary_retained_fraction=fraction,
        origins=origins,seconds=seconds,rates=rates)
    result={'scope':__doc__,'complete':True,'candidates':candidates,'band_order':NAMES,
        'minimum_template_energy_retained':.2,'source_sha256':{str(q):hashlib.file_digest(q.open('rb'),'sha256').hexdigest() for q in sources}}
    (args.output/'discovery.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    fig,ax=plt.subplots(figsize=(15,5),layout='constrained')
    values=10*np.log10(np.maximum(maximum,1e-12))
    image=ax.imshow(values,origin='lower',aspect='auto',extent=[seconds[0],seconds[-1]+3,-6.5*C/8e6,6.5*C/8e6],cmap='magma')
    for c in candidates:
        t=(c['first_s']+c['last_s'])/2;y=c['centre_sample']*C/8e6
        ax.plot(t,y,'s',mfc='none',mec='#00ddff',ms=10)
        ax.annotate(c['label'],(t,y),xytext=(4,6),textcoords='offset points',color='#00ddff',fontsize=8)
    ax.set(xlabel='Capture time (s)',ylabel='Excess-path offset from AIS (m)',title='Three-second training-only discovery across the full capture\nMinimum-band matched score; static nuisance profiled; at least 20% template energy retained in every band')
    fig.colorbar(image,ax=ax,label='Weakest-band conditional acquisition score (dB)')
    fig.savefig(args.output/'candidates.png',dpi=160);plt.close(fig)
    print(json.dumps(candidates,indent=2),flush=True)


if __name__=='__main__':main()
