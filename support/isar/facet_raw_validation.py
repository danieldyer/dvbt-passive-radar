"""Test a finite-side summary fit on unchanged withheld complex flash samples."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv
from cluster_phase import crop
from compare_ais_phase import NAMES
from facet_geometry import FacetFlash
from facet_raw import template
from flash_geometry import Flash,assignments
from flash_siting_report import EPOCH
from flash_siting_validation import phase_origin
from imaging import sites
from ship_site_signature import licensed_options


def check(field,h,training):
    test=~training
    baseline=float(np.sum(abs(field.y[test])**2))
    gain=np.vdot(h[training],field.y[training])/np.vdot(h[training],h[training]).real
    prediction=gain*h
    q,information=phase_origin(h,field.y,training)
    observed=np.sum(h.conj()*field.y,axis=1)/np.maximum(information,1e-300)
    rotation=np.sum(observed[training]*q[training].conj());rotation/=abs(rotation)
    strengths=np.interp(field.time,field.time[training],np.maximum(q[training].real,0.))
    amplitude_prediction=strengths[:,None]*h*rotation
    phase=np.angle(q)
    weight=abs(q[test])**2*information[test]
    return {'constant_gain_prediction_fraction':float(1-np.sum(abs(field.y[test]-prediction[test])**2)/baseline),
        'training_amplitude_prediction_fraction':float(1-np.sum(abs(field.y[test]-amplitude_prediction[test])**2)/baseline),
        'test_phase_rms_deg':float(np.degrees(np.sqrt(np.average(phase[test]**2,weights=weight)))),
        'training_constant_gain_phase_deg':float(np.degrees(np.angle(gain))),
        'time_s':field.time.tolist(),'training':training.tolist(),'phase_residual_deg':np.degrees(phase).tolist(),
        'observed_matched_amplitude':abs(observed).tolist(),
        'constant_gain_template_norm':(abs(gain)*np.sqrt(information)).tolist(),
        'observed_projection_norm':(abs(observed)*np.sqrt(information)).tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--fields',type=Path,default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--geometry',type=Path,default=Path('docs/data/calibration_20261016_sites'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    record=json.loads(args.fit.read_text())
    options=assignments(licensed_options(args.geometry))
    _,rx,_=sites(args.geometry)
    rotation=np.array(json.loads((args.geometry/'geometry-prior.json').read_text())['ecef_to_enu'])
    track=read_csv('docs/data/isar_20261013/rio-delta-ais.csv',reference_height_ellipsoid_m=5.)
    base=Flash(record['measurements'],record['covariance'],track,EPOCH+164.5,rotation,rx,options[record['index']],
        receiver_sigma=(0.,0.,7.5),receiver_up_bounds_m=(-10.,10.))
    model=FacetFlash(base,np.array([177.5,184.5,191.625,219.5,226.5])*1e6,
        length_m=record['dimensions_m'][0],height_m=record['dimensions_m'][1])
    parameters=np.array(record['fit']['parameters']);state=base.state(parameters[:base.count])
    q=parameters[base.nantenna:base.count]*base.nuisance_sigma
    transmitters=rx+state[3]@rotation;receiver=rx+state[4]@rotation
    up=(model.frame@np.array([0.,0.,1.]))@rotation
    results=[]
    fig,axes=plt.subplots(5,2,figsize=(14,14),layout='constrained')
    for ch in range(5):
        with np.load(args.fields/f'channel-{ch}.npz') as data:
            field=crop(data,164.,165.,np.arange(-2,3))
        t=field.data['row_time']-164.5
        position=np.polynomial.polynomial.polyval(t+q[-1],base.ais_coefficients).T+q[:3]+t[:,None]*q[3:6]+.5*t[:,None]**2*q[6:9]
        velocity=np.polynomial.polynomial.polyval(t+q[-1],np.polynomial.polynomial.polyder(base.ais_coefficients)).T+q[3:6]+t[:,None]*q[6:9]
        position=rx+position@rotation;velocity=velocity@rotation
        turn=parameters[-1]*model.scale[1]
        heading=model.heading+parameters[-2]*model.scale[0]+turn*t
        along=np.column_stack([np.sin(heading),np.cos(heading),np.zeros(len(t))])@model.frame.T@rotation
        along_rate=turn*np.column_stack([np.cos(heading),-np.sin(heading),np.zeros(len(t))])@model.frame.T@rotation
        h=template(field,position,velocity,along,along_rate,up,transmitters[ch],receiver,model.length,model.height)
        training=((field.time-164.)//.125).astype(int)%2==0
        fitted=check(field,h,training)
        point=check(field,field.trajectory(position,velocity,transmitter=transmitters[ch],receiver=receiver),training)
        row={'channel':NAMES[ch],'facet':fitted,'physical_centre_point_control':point}
        results.append(row)
        (args.output/'channels.json').write_text(json.dumps(results,indent=2,allow_nan=False))
        test=~training
        for series,label in [(point,'Centre point'),(fitted,'Finite side')]:
            axes[ch,0].plot(field.time[test],np.array(series['phase_residual_deg'])[test],'.-',label=label)
        axes[ch,1].plot(field.time,np.array(fitted['constant_gain_template_norm']),label='Constant-gain physical prediction')
        axes[ch,1].plot(field.time[test],np.array(fitted['observed_projection_norm'])[test],'.',label='Observed withheld projection')
        axes[ch,0].set_ylabel(NAMES[ch]+' phase error (degrees)');axes[ch,1].set_ylabel('Matched amplitude (whitened units)')
        print(NAMES[ch],'facet phase',round(fitted['test_phase_rms_deg'],2),'constant prediction',round(fitted['constant_gain_prediction_fraction'],3),
            'training-amplitude prediction',round(fitted['training_amplitude_prediction_fraction'],3),flush=True)
    for ax in axes[-1]:ax.set_xlabel('Recording time (s)')
    for ax in axes[0]:ax.legend(fontsize=8)
    fig.suptitle('Physical finite-side prediction on untouched complex data\nNo fitted per-band delay or rate; phase origin and strengths learned on training bins only')
    fig.savefig(args.output/'prediction.png',dpi=170);plt.close(fig)
    result={'fit':str(args.fit),'fields':str(args.fields),'channels':results,'accepted_calibration':False,
        'scope':__doc__,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [args.fit,Path(__file__),Path(__file__).with_name('facet_raw.py')]}}
    (args.output/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
