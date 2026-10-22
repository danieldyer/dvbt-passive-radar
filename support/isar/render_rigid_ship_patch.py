"""Exact original-symbol north-up ship BP with the shared rigid-target model."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv,utc_seconds
from combine_images import combine
from extract_full_aperture import load
from imaging import geometry_from_state,motion_state,read_image,write_input,sites
from rigid_patch import RigidPatch
from ship_site_signature import assigned_rays
from spline_motion import SplineMotion


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--geometry-directory',type=Path)
    parser.add_argument('--spacing-m',type=float,default=1.);parser.add_argument('--unrefined',action='store_true')
    parser.add_argument('--training-only',action='store_true')
    parser.add_argument('--assignment',default='00002')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    fitted=json.loads(args.fit.read_text());model=fitted['reference_motion'];epoch=utc_seconds(model['nominal_start_utc']);mid=fitted['epoch_s']
    curve=SplineMotion(model);track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    reference_tx,reference_rx,_=sites(Path(model['geometry_directory']))
    tx=np.asarray(fitted['transmitter_ecef_m']);rx=np.asarray(fitted['receiver_ecef_m'])
    if args.geometry_directory:tx,rx=assigned_rays(args.geometry_directory,args.assignment)
    local=np.asarray(fitted['local_to_ecef']);reference=fitted['reference']
    phase_scene=None
    if 'image_autofocus_fit' in fitted:
        record=fitted['image_autofocus_fit'];parameters=np.asarray(record['motion_parameters'])
        gains=np.asarray([complex(*z) for z in record['gains_real_imag']])
    elif 'phase_coherent_fit' in fitted:
        record=fitted['phase_coherent_fit'];parameters=np.asarray(record['motion_parameters'])
        phase_scene=np.asarray(record['coefficient_real_imag']);phase_scene=phase_scene[...,0]+1j*phase_scene[...,1]
        gains=np.asarray([complex(*z) for z in record['gain_phase_real_imag']])
    elif 'coherent_fit' in fitted:
        parameters=np.asarray(fitted['coherent_fit']['motion_parameters']);gains=np.asarray([complex(*z) for z in fitted['coherent_fit']['gains_real_imag']])
    else:
        best=min(fitted['trials'],key=lambda r:r['objective']);parameters=np.asarray(best['parameters']);gains=np.ones(5,complex)
    if args.unrefined:parameters=np.zeros(5)
    anchor=curve.apply(np.array([mid]),motion_state(track,epoch+mid,model),reference_tx[3],reference_rx)
    gauge=anchor['body_to_ecef'][0].T@local
    east=np.arange(-250.,250.01,args.spacing_m);north=np.arange(-350.,350.01,args.spacing_m)
    xx,yy=np.meshgrid(east,north);pixels=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,20.),np.zeros(xx.size)]
    result=dict(interval_s=fitted['interval_s'],fit_source=str(args.fit),reference=reference,geometry_directory=str(args.geometry_directory),
        parameters=parameters.tolist(),gains_real_imag=np.c_[gains.real,gains.imag].tolist(),
        transmitter_ecef_m=tx.tolist(),receiver_ecef_m=rx.tolist(),coordinate_frame='North-up east/north relative to coarse ship reference at aperture midpoint',
        operator='Exact original native C0/C1 bistatic BP, shared rigid motion, one complex gain per band, complex sum before power',
        phase_centres_calibrated=False,channels=[])
    image=[];information=[]
    for c in range(5):
        tic=time.monotonic();a=load(Path(f'build/isar-20261015/full-geographic-aperture/channel-{c}'));meta=json.loads(a['metadata_json'])
        t=a['time_s'];keep=(t>=fitted['interval_s'][0])&(t<fitted['interval_s'][1]);t=t[keep]
        base=track.sample(epoch+t,max_gap_s=400.);options={}
        if reference!='ais':
            base=curve.apply(t,motion_state(track,epoch+t,model),reference_tx[3],reference_rx)
            options=dict(reference_frame=base['body_to_ecef']@gauge,reference_frame_rate=base['frame_velocity']@gauge)
        residual_tx=np.asarray(fitted.get('translation_reference_transmitter_ecef_m',fitted['transmitter_ecef_m'][3]))
        residual_rx=np.asarray(fitted.get('translation_reference_receiver_ecef_m',fitted['receiver_ecef_m']))
        kin=RigidPatch(t,mid,base['position_ecef_m'],base['velocity_ecef_mps'],local,residual_tx,residual_rx,**options)
        centre,velocity=kin.point_state(parameters,[0.,0.,0.]);frame,fd=kin.orientation(parameters)
        state=dict(position_ecef_m=centre,velocity_ecef_mps=velocity,body_to_ecef=frame,frame_velocity=fd)
        weight=a['information'][keep]*a['valid'][keep]
        if args.training_only:
            with np.load(f'build/isar-20261015/full-fields/channel-{c}.npz') as field:
                times=field['row_time'];bins=field['time'];block=json.loads(str(field['metadata_json']))['block_rows']
            index=np.minimum(np.searchsorted(times,t),len(times)-1);before=np.maximum(index-1,0)
            index=np.where(abs(times[before]-t)<abs(times[index]-t),before,index)
            if np.max(abs(times[index]-t))>2e-8:raise ValueError('Training rows do not match archived native epochs')
            bt=bins[index//block];train=(np.floor((bt-142.)/.5).astype(int)%2==0)&(bt>=142.02)&(bt<171.98)
            weight=weight*train
        rf=meta['channel_configuration']['rf_hz'];g,_=geometry_from_state(state,tx[c],rx,rf,meta['first_range_sample'],weight)
        source=out/f'channel-{c}.bin';target=out/f'channel-{c}-image.bin';write_input(source,g,a['moments'][:,keep],pixels,rf)
        subprocess.run(['build/release/support/isar/isar_backprojection',str(source),str(target)],check=True)
        v=read_image(target,len(pixels));z=(v[:,0]+1j*v[:,1])/v[:,2]
        image.append(z.reshape(xx.shape));information.append(v[:,2].reshape(xx.shape))
        np.savez(out/f'channel-{c}.npz',east_m=east,north_m=north,complex_image=image[-1],information=information[-1])
        source.unlink()
        record=dict(channel=c,rf_hz=rf,seconds=time.monotonic()-tic,maximum_matched_power_db=float(10*np.log10(np.max(8192*v[:,2]*abs(z)**2))))
        result['channels'].append(record);print(record,flush=True)
    image=np.asarray(image);information=np.asarray(information)
    if phase_scene is not None:
        # Rayleigh-Ritz: maximize summed unit-noise coherent power for the
        # training scatterer response vectors, with one global band weight.
        # The amplitudes are nonnegative, so the leading mode has one sign.
        raw_information=8192*information[:,0,0]
        response=np.sqrt(raw_information[:,None])*abs(phase_scene)
        _,vectors=np.linalg.eigh(response@response.T);weight=vectors[:,-1]
        if weight.sum()<0:weight=-weight
        amplitude=weight/np.sqrt(raw_information);amplitude/=amplitude.max()
        gains=gains*amplitude
        result['combiner']='Constant band phases from all training scatterers; global amplitudes maximize summed training-point coherent power at unit noise (Rayleigh-Ritz); frozen for the entire image'
        result['gains_real_imag']=np.c_[gains.real,gains.imag].tolist()
    combined,total,efficiency=combine(image,information,gains)
    power=np.r_[8192*information*abs(image)**2,(8192*total*abs(combined)**2)[None]]
    np.savez(out/'complex-images.npz',east_m=east,north_m=north,individual=image,information=information,
        combined=combined,combined_information=total,coherent_to_independent_energy=efficiency)
    limits=[5.,50.];names=['Seven','SBS','Nine','Ten','ABC','Coherent complex sum']
    fig,axes=plt.subplots(2,3,figsize=(14,13),layout='constrained',sharex=True,sharey=True)
    for c,ax in enumerate(axes.ravel()):
        im=ax.imshow(10*np.log10(np.maximum(power[c],1e-30)),extent=[east[0],east[-1],north[0],north[-1]],origin='lower',aspect='equal',vmin=limits[0],vmax=limits[1],cmap='magma',interpolation='nearest')
        ax.plot(0,0,'+',color='cyan',ms=8);ax.set(title=names[c],xlabel='East (m)',ylabel='North (m)')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Matched power / original conditional thermal variance (dB)',shrink=.8)
    fig.suptitle('Restored rigid-target imaging · actual 142–172 s recording · north up\nOriginal symbols and full range/phase backprojection; no Doppler notch or local normalization')
    fig.savefig(out/'images.png',dpi=160);plt.close(fig)
    result['colour_scale_db']=limits;result['complete']=True
    result['training_only']=args.training_only;result['assignment']=args.assignment
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
