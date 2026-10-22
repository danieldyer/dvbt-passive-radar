"""Form bistatic omega-k ship patches from all retained recording observations.

The field noise metric and its native-row averaging adjoint are retained.
Known waveform spectra are applied before azimuth FFT. Every input row enters
the spectral integrand; small along-track quadrature blocks are sums, not
sample selection. Their complete operator can be pulled back to the original
whitened data to measure true sensitivity loss against exact backprojection.
"""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft,fftfreq

from ais import read_csv
from cluster_phase import crop
from compare_ais_phase import path_geometry
from extract_full_aperture import load as load_aperture
from field import FS,average
from fixed_range_field import fixed_reference
from flash_siting_report import EPOCH
from motion import C
from omega_k import ParallelGeometry,BistaticOmegaK,sinc_sample
from piecewise_ais import LocalAisReference
from report_measured_scatterer_phases import load_fields,serial
from ship_site_signature import licensed_options
from symbol_moments import moments
from waveform_kernels import read_kernels,bind_subset


class FieldSpectralAdapter:
    def __init__(self,field,geometry,basis,centre,receiver,transmitter,velocity,*,path_parameters,
            path_epoch,path_reference,phase_polynomial=None,midpoint=164.,native_group=128,period=256):
        self.field=field;self.geometry=geometry;self.basis=basis;self.centre=centre
        self.receiver,self.transmitter,self.velocity=receiver,transmitter,velocity
        self.midpoint=midpoint;self.parameters=np.asarray(path_parameters);self.epoch=path_epoch;self.path_reference=path_reference
        self.phase_polynomial=phase_polynomial;self.group=native_group;self.period=period
        self.speed=np.linalg.norm(velocity);self.time=field.data['row_time'];self.u=self.speed*(self.time-midpoint)
        n=len(self.time);block=field.metadata['block_rows'];nb=len(field.time)
        if n!=block*nb or native_group%block:raise ValueError('Complete retained/native grouping required')
        self.bin_group=np.arange(nb)//(native_group//block);ng=self.bin_group[-1]+1
        dt=np.median(np.diff(self.time))
        self.group_time=self.time[0]+(np.arange(ng)*native_group+(native_group-1)/2)*dt
        self.group_u=self.speed*(self.group_time-midpoint)
        mode=(fftfreq(period)*period).astype(int);order=np.argsort(mode);mode=mode[order]
        self.radial=2*np.pi*mode/(period*C/FS);self.k=field.k+self.radial
        self.fourier=np.exp(-2j*np.pi*field.offsets[:,None]*mode[None,:]/period)
        samples=field.pulse.kernels.samples
        padded=np.zeros((len(samples),period),complex);padded[:,np.arange(-64,65)%period]=samples
        pcoef=fft(padded,axis=1)[:,order]/period
        row_blocks=field.pulse.row_blocks.reshape(nb,block)
        if np.any(row_blocks!=row_blocks[:,0,None]):raise ValueError('Known waveform must retain original block identity')
        self.pulse_coeff=pcoef[row_blocks[:,0]]
        length0=geometry.path(0.,self.u);dt=self.time-path_epoch
        fitted=path_reference+self.parameters[0]+self.parameters[1]*dt+.5*self.parameters[2]*dt**2
        midpoint_path=path_reference+self.parameters[0]+self.parameters[1]*(midpoint-path_epoch)+.5*self.parameters[2]*(midpoint-path_epoch)**2
        # Preserve the measured constant delay and carrier phase. Only dynamic
        # bulk motion is mapped to the straight reference trajectory.
        self.correction=fitted-length0-(midpoint_path-geometry.path(0.,0.))
        rate0=geometry.path(0.,self.u,True)[2]*self.speed
        self.correction_rate=self.parameters[1]+self.parameters[2]*dt-rate0
        if phase_polynomial is None:self.phase=np.zeros(n);phase_rate=np.zeros(n)
        else:
            self.phase=np.polynomial.polynomial.polyval((self.time-midpoint)/15,phase_polynomial)
            phase_rate=np.polynomial.polynomial.polyval((self.time-midpoint)/15,np.polynomial.polynomial.polyder(phase_polynomial))/15
        omega=-field.k*.001024*(self.parameters[1]+self.parameters[2]*dt)+.001024*phase_rate
        m0,m1,_,_=moments(omega);factor=m0-1j*field.data['centre_omega']*m1
        self.carrier=np.array([field.k*geometry.path(0.,0.,True)[1],field.k*geometry.path(0.,0.,True)[2]])
        native_bin=np.arange(n)//block;native_group=np.arange(n)//self.group
        carrier_reference=field.data.get('carrier_reference_path_m',field.data['centre_path_m'])
        shift=self.correction+geometry.path(0.,0.)-field.data['centre_path_m']
        known_phase=field.k*(field.data['centre_path_m']-carrier_reference)-self.phase
        # Store one accumulated native quadrature coefficient per retained bin
        # and radial Fourier mode. No target-dependent data normalisation.
        self.coefficient=np.empty((nb,len(self.k)),complex)
        for first in range(0,nb,128):
            last=min(first+128,nb);section=slice(first*block,last*block)
            angle=shift[section,None]*self.k[None,:]+known_phase[section,None]
            angle+=self.carrier[1]*(self.u[section,None]-self.group_u[native_group[section],None])
            w=np.exp(1j*angle)*factor[section,None].conj()*field.data['row_information'][section,None]
            self.coefficient[first:last]=w.reshape(last-first,block,-1).sum(axis=1)*self.pulse_coeff[first:last].conj()
        self.scale=np.divide(field.root_information,field.data['information'],out=np.zeros(nb),where=field.data['information']>0)

    def apply(self,whitened_data):
        f=self.field;per_bin=(np.asarray(whitened_data)@f.whiten.conj())*self.scale[:,None]
        value=(per_bin@self.fourier)*self.coefficient
        out=np.zeros((len(self.group_u),len(self.k)),complex)
        np.add.at(out,self.bin_group,value)
        return out

    def pullback(self,spectral_weights):
        value=np.asarray(spectral_weights)[self.bin_group]*self.coefficient
        return ((value@self.fourier.T)@self.field.whiten.conj().T)*self.scale[:,None]

    def exact_template(self,x,y):
        f=self.field;length=self.geometry.path(x,self.u+y)
        rate=self.geometry.path(x,self.u+y,True)[2]*self.speed+self.correction_rate
        path=length+self.correction
        delta=path-f.data['centre_path_m']
        phase_rate=np.zeros(len(path)) if self.phase_polynomial is None else np.polynomial.polynomial.polyval(
            (self.time-self.midpoint)/15,np.polynomial.polynomial.polyder(self.phase_polynomial))/15
        omega=-f.k*.001024*rate+.001024*phase_rate
        raw=f.path_response(delta,omega)*np.exp(1j*self.phase[:,None])
        return f.apply(average(raw,f.data['row_information'],f.metadata['block_rows'])[0])


def project_stationary(field,values,selected=None):
    """Profile one constant complex range profile in fixed coordinates.

    This is an orthogonal nuisance projection in the declared whitened noise
    metric. Its basis is fixed by the observation operator, not the image.
    Separate epoch sets have separate projections, so validation cannot borrow
    a fitted static coefficient from an unused epoch.
    """
    use=np.ones(len(field.time),bool) if selected is None else np.asarray(selected,bool)
    q=field.root_information*use
    q=q/np.linalg.norm(q)
    value=np.asarray(values)*use[:,None]
    return value-q[:,None]*(q.conj()@value)[None,:]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--channels',type=int,nargs='+',default=[0,1,2,3,4]);parser.add_argument('--spacing',type=float,default=2.)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    geometry_dir=Path('docs/data/calibration_20261016_sites');prior=json.loads((geometry_dir/'geometry-prior.json').read_text())
    rx=np.asarray(prior['receiver_ecef_m']);rotation=np.asarray(prior['ecef_to_enu']);options=licensed_options(geometry_dir)
    track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    reference=LocalAisReference(track,EPOCH+164.,rotation,rx);p=reference(0.);v=reference(0.,1)
    ey=v.copy();ey[2]=0;ey/=np.linalg.norm(ey);ex=np.array([ey[1],-ey[0],0.]);up=np.array([0.,0.,1.])
    basis=rotation.T@np.column_stack([ex,ey,up]);centre=rx+(p+20*up)@rotation
    velocity=np.linalg.norm(v[:2])*basis[:,1]
    source_root=Path('docs/data/isar_20261017/quadratic-motion-kernel-recursive')
    source=json.loads((source_root/'analysis.json').read_text());autofocus=json.loads((source_root/'continued-autofocus.json').read_text())
    records=load_fields(Path('build/isar-20261015/full-fields'),geometry_dir)
    kernel=Path('build/isar-20261017/waveform-kernels-full');_,krecords,ksamples=read_kernels(kernel/'response.kernels');grid=np.load(kernel/'rows.npz')
    x=np.arange(-220.,220.01,args.spacing);result=dict(complete=False,operator='Bistatic numerical-reference omega-k with Stolt mapping and differential group delay',
        interval_s=[149.,179.],phase_centres_calibrated=False,heading_fitted=False,centre_ecef_m=centre,basis_x_along_up_ecef=basis,
        velocity_ecef_mps=velocity,reference_height_above_AIS_m=20.,channels=[])
    def save():(out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    save()
    for c in args.channels:
        tic=time.monotonic();old=crop(records[c]['data'],149.,179.,np.arange(-8,9))
        tx=np.asarray(next(o for o in options if o['channel']==c and o['site_id']==('12014' if c==4 else '12013'))['transmitter_ecef_m'])
        _,rate=path_geometry(old.data['centre_ecef_m'],old.data['centre_velocity_ecef_mps'],tx,rx)
        omega=-old.k*.001024*np.interp(164.,old.data['row_time'],rate)
        raw_aperture=load_aperture(Path(f'build/isar-20261015/full-geographic-aperture/channel-{c}'))
        fixed,_=fixed_reference(old,raw_aperture,164.,np.arange(-16,17),omega_reference=omega)
        f=bind_subset(fixed,grid[f'channel_{c}']['time'],grid[f'channel_{c}']['information'],ksamples[c],np.array([r['valid'] for r in krecords[c]]))
        selected=next(e for e in source['recorded'] if e['channel']==c and e['duration_s']==30.);fit=selected['continuation'] or selected
        phase=autofocus['channels'][c];ts=np.asarray(phase['time_s']);epoch=(ts[0]+ts[-1])/2
        path_reference=np.interp(epoch,old.data['row_time'],old.data['centre_path_m'])
        polynomial=np.polynomial.polynomial.polyfit((ts-164.)/15,np.asarray(phase['phase_rad']),8)
        geo=ParallelGeometry((tx-centre)@basis,(rx-centre)@basis)
        adapter=FieldSpectralAdapter(f,geo,basis,centre,rx,tx,velocity,path_parameters=fit['parameters'],
            path_epoch=epoch,path_reference=path_reference,phase_polynomial=polynomial)
        # Keep the exact epochs used by the earlier motion/phase estimator out
        # of the image-validation data. The other seven of eight retained
        # epochs were never used by that fit either.
        training=np.zeros(len(f.time),bool)
        for t in ts[np.asarray(phase['training'],bool)]:
            j=int(np.argmin(abs(f.time-t)))
            if abs(f.time[j]-t)>2e-7:raise ValueError('Autofocus epoch does not match original retained data')
            training[j]=True
        integrand=adapter.apply(f.y)
        training_integrand=adapter.apply(project_stationary(f,f.y,training))
        unused_integrand=adapter.apply(project_stationary(f,f.y,~training))
        projected_integrand=adapter.apply(project_stationary(f,f.y))
        model=BistaticOmegaK(geo,adapter.group_u,adapter.k,maximum_along_offset_m=250.)
        points=[(0.,0.),(-120.,60.),(120.,-60.)]
        image,y,meta=model.stolt_image(integrand,x,range_period_m=1600.,weight_points=points)
        weights=meta.pop('operator_weights');checks=[]
        for px,py in points:
            key=f'{px:g},{py:g}';h=adapter.exact_template(px,py)
            w=adapter.pullback(weights[key]);efficiency=abs(np.sum(w*h))**2/np.sum(abs(w)**2)/np.sum(abs(h)**2)
            reference_weights=np.exp(1j*(geo.path(px,adapter.group_u+py)-model.reference)[:,None]*adapter.k)
            reference_weights*=np.exp(-1j*(model.carrier@[px,py]))
            wr=adapter.pullback(reference_weights)
            reference_efficiency=abs(np.sum(wr*h))**2/np.sum(abs(wr)**2)/np.sum(abs(h)**2)
            exact=np.vdot(h,f.y)*np.exp(-1j*(model.carrier@[px,py]));actual=np.sum(w*f.y)
            checks.append(dict(point=[px,py],matched_snr_loss_db=-10*np.log10(efficiency),
                spectral_reference_loss_db=-10*np.log10(reference_efficiency),
                true_complex_gain_real_imag=[(np.sum(w*h)/np.sum(abs(h)**2)).real,(np.sum(w*h)/np.sum(abs(h)**2)).imag],
                real_exact_value_real_imag=[exact.real,exact.imag],real_stolt_value_real_imag=[actual.real,actual.imag]))
        reference_image,yr=model.reference_image(integrand,x)
        training_image,_=model.reference_image(training_integrand,x)
        unused_image,_=model.reference_image(unused_integrand,x)
        projected_image,_=model.reference_image(projected_integrand,x)
        no_phase_adapter=FieldSpectralAdapter(f,geo,basis,centre,rx,tx,velocity,path_parameters=fit['parameters'],
            path_epoch=epoch,path_reference=path_reference,phase_polynomial=None)
        quadratic_image,_=model.reference_image(no_phase_adapter.apply(project_stationary(f,f.y)),x)
        h0=adapter.exact_template(0.,0.);information=np.sum(abs(h0)**2,axis=1)
        psf_image,_=model.reference_image(adapter.apply(project_stationary(f,h0)),x)
        projected_information=[np.vdot(h0*mask[:,None],project_stationary(f,h0,mask)).real
            for mask in (np.ones(len(f.time),bool),training,~training)]
        keep=abs(y)<=250.
        np.savez(out/f'channel-{c}.npz',stolt=image[keep],reference=reference_image[keep],x_m=x,y_m=y[keep],carrier_rad_per_m=model.carrier,
            training=training_image[keep],unused=unused_image[keep],stationary_profiled=projected_image[keep],
            quadratic_only=quadratic_image[keep],unit_point_response=psf_image[keep],
            spectral_integrand=integrand,u_m=adapter.group_u,radial_wavenumber_rad_per_m=adapter.k,
            origin_information=projected_information[0],training_information=projected_information[1],unused_information=projected_information[2])
        record=dict(channel=c,name=selected['name'],rf_hz=f.metadata['rf_hz'],metadata=meta,checks=checks,
            exact_native_rows=len(f.data['row_time']),retained_bins=len(f.time),summed_native_rows_per_spectral_node=adapter.group,
            spectral_nodes=len(adapter.group_u),seconds=time.monotonic()-tic,source_path_parameters=fit['parameters'],autofocus_phase_polynomial=polynomial)
        record.update(training_epochs=int(training.sum()),unused_epochs=int((~training).sum()),
            split='Exact motion/autofocus fitting epochs are training; all other original retained epochs are unused by that fit',
            unprojected_origin_information=information.sum(),
            origin_information=projected_information[0],training_information=projected_information[1],unused_information=projected_information[2],
            stationary_nuisance='One constant complex profile in fixed range coordinates; orthogonal projection in the conditional whitened metric, separately for each epoch set; no Doppler interval is masked')
        result['channels'].append(record);save();print(record['name'],record['seconds'],'checks',checks,flush=True)
    result['complete']=True;save()


if __name__=='__main__':main()
