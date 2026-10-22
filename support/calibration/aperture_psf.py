"""Ideal point response of the actual RF/site geometry, without calibration error."""
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from geometry import C,RF,unit
from site_hypotheses import licensed_geometry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['terrain','output','work','executable']:
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();args.work.mkdir(parents=True,exist_ok=True)
    target=np.array([-5500.,-5000.,0.])
    base,offsets,_=licensed_geometry(args.output,args.terrain,[0,0,0,0,2])
    gradient=unit(target)+unit(target-base[0]);along=unit(np.r_[gradient[:2],0.])
    across=np.array([-along[1],along[0],0.])
    cross=np.arange(-1500.,1500.);radial=np.arange(-160.,160.)
    xx,yy=np.meshgrid(cross,radial)
    point=target+xx[:,:,None]*across+yy[:,:,None]*along
    lag=np.arange(1024);lag[lag>=960]-=1024
    fig,axes=plt.subplots(2,1,figsize=(14,6.8),layout='constrained')
    metrics=[]
    for ax,(name,mapping) in zip(axes,[('Previous three-mast assumption',[0,2,1,2,0]),('Licensed main-site configuration',[0,0,0,0,2])]):
        sites,offsets,_=licensed_geometry(args.output,args.terrain,mapping)
        transmitters=sites[mapping]+offsets
        geometry=np.ones((xx.size,5,4),np.float32);profiles=np.empty((5,1024),np.complex64)
        for ch,tx in enumerate(transmitters):
            length=np.linalg.norm(point-tx,axis=-1)+np.linalg.norm(point,axis=-1)-np.linalg.norm(tx)
            origin=np.linalg.norm(target-tx)+np.linalg.norm(target)-np.linalg.norm(tx)
            phase=np.remainder(2*np.pi*RF[ch]/C*length,2*np.pi)
            geometry[:,ch,0]=(length*8e6/C).ravel();geometry[:,ch,1]=np.cos(phase).ravel();geometry[:,ch,2]=np.sin(phase).ravel()
            profiles[ch]=np.sinc((6817/8192)*(lag-origin*8e6/C))*np.exp(-2j*np.pi*RF[ch]/C*origin)
        path=args.work/('psf-'+str(len(metrics))+'.bin');geometry.tofile(path)
        weights=np.tile([1,1,0,1],(5,1)).astype(np.float32)
        run=subprocess.run([str(args.executable),str(path),'5'],input=profiles.tobytes()+weights.tobytes(),stdout=subprocess.PIPE,check=True)
        result=np.frombuffer(run.stdout,np.float32).reshape(len(radial),len(cross),4)
        power=(result[:,:,0]**2+result[:,:,1]**2)/25
        reference=float(power[np.flatnonzero(radial==0)[0],np.flatnonzero(cross==0)[0]])
        db=10*np.log10(np.maximum(power/reference,1e-8))
        image=ax.imshow(db,origin='lower',extent=[cross[0],cross[-1],radial[0],radial[-1]],cmap='inferno',vmin=-40,vmax=0,aspect='auto',interpolation='nearest')
        ax.set(title=name,ylabel='Along-range displacement (m)')
        distant=abs(xx)>300
        metrics.append({'configuration':name,'normalized_peak_power':reference,
                        'maximum_response_beyond_300m_cross_range_db':float(db[distant].max())})
    axes[-1].set_xlabel('Cross-range displacement (m)')
    fig.colorbar(image,ax=axes,label='Power relative to the true target (dB)',shrink=.85)
    fig.suptitle('Ideal coherent point response — perfect calibration, equal amplitudes, flat local surface\n'
                 'Five occupied DVB-T bands; no noise, terrain error or target motion')
    fig.savefig(args.output/'ideal-aperture-psf.png',dpi=150);plt.close(fig)
    summary={'target_enu_m':target.tolist(),'range_sample_spacing_m':1.,'cross_range_sample_spacing_m':1.,
             'assumptions':'Perfect calibration; nondispersive point scatterer; equal channel amplitudes/noise; rectangular occupied band; flat local surface',
             'metrics':metrics}
    (args.output/'ideal-aperture-psf.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
