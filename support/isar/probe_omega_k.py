"""Independent two-leg point scene versus numerical-reference and Stolt images."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from omega_k import ParallelGeometry,BistaticOmegaK,sinc_sample
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    u=np.arange(-128,129)*.5;k=np.linspace(4.-.075,4.+.075,64)
    geo=ParallelGeometry(np.array([-35000.,9000.,700.]),np.array([-8000.,-2500.,30.]))
    model=BistaticOmegaK(geo,u,k,maximum_along_offset_m=160.)
    x=np.arange(-100,101,5.);points=np.array([[0.,0.],[35.,-40.],[-60.,75.]])
    gains=np.array([1.,.7*np.exp(.3j),.5*np.exp(-.9j)])
    # Independent explicit Cartesian ranges, no reference.path calls here.
    data=np.zeros((len(u),len(k)),complex)
    for (xx,yy),gain in zip(points,gains):
        pos=np.c_[np.full(len(u),xx),u+yy,np.zeros(len(u))]
        length=sum(np.sqrt(np.sum((pos-ant)**2,axis=1)) for ant in (geo.transmitter,geo.receiver))
        length-=np.linalg.norm(geo.transmitter-geo.receiver)
        data+=gain*np.exp(-1j*(length-model.reference)[:,None]*k)
    start=time.monotonic();reference,y=model.reference_image(data,x);print('reference',time.monotonic()-start,flush=True)
    start=time.monotonic();stolt,ys,meta=model.stolt_image(data,x,weight_points=points.tolist());weights=meta.pop('operator_weights')
    print('stolt',time.monotonic()-start,meta,flush=True)
    # One deterministic reference-point gain for the numerical quadrature.
    unit=np.exp(-1j*(geo.path(0.,u)-model.reference)[:,None]*k)
    zero=np.argmin(abs(y));scale=np.sum(abs(unit)**2)/np.sum(weights['0,0']*unit);stolt*=scale
    keep=abs(y)<=140
    errors=[]
    for point in points:
        col=np.argmin(abs(x-point[0]));row=np.argmin(abs(y-point[1]));direct=model.direct(data,[point])[0]
        errors.append(dict(point=point,direct_real_imag=[direct.real,direct.imag],
            exact_complex_relative_error=abs(reference[row,col]-direct)/max(abs(direct),1e-100),
            stolt_amplitude_error_db=20*np.log10(abs(stolt[row,col])/abs(direct)),
            stolt_phase_error_deg=np.degrees(np.angle(stolt[row,col]/direct))))
        w=weights[f'{point[0]:g},{point[1]:g}'];h=np.exp(-1j*(geo.path(point[0],u+point[1])-model.reference)[:,None]*k)
        efficiency=abs(np.sum(w*h))**2/np.sum(abs(w)**2)/np.sum(abs(h)**2)
        errors[-1]['stolt_true_matched_snr_loss_db']=-10*np.log10(efficiency)
        errors[-1]['operator_weight_replay_error']=abs(np.sum(w*data)*scale-stolt[row,col])/max(abs(stolt[row,col]),1e-100)
    result=dict(points=points,gains_real_imag=np.c_[gains.real,gains.imag],errors=errors,metadata=meta,
        stolt_gain_real_imag=[scale.real,scale.imag],relative_image_error=np.linalg.norm(stolt[keep]-reference[keep])/np.linalg.norm(reference[keep]))
    print(json.dumps(serial(result),indent=2),flush=True)
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'analysis.json').write_text(json.dumps(serial(result),indent=2))
    np.savez(a.output/'images.npz',reference=reference[keep],stolt=stolt[keep],x=x,y=y[keep])


if __name__=='__main__':main()
