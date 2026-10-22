"""Dense known-waveform matched images and exact pose derivatives on Vulkan.

The fixed data-side polynomial is the adjoint range whitening followed by
the original known pulse interpolation. No FFT or single-point phase screen
is involved. All retained epochs contribute, including their own waveform,
information and carrier reference. Final original-symbol BP stays separate.
"""
import json
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
from scipy.optimize import minimize,LinearConstraint

from imaging import geometry_from_state
from motion import C
from rigid_patch import profile_static


def matched_polynomials(field,selected,first_cell=-15,cells=31,groups=None):
    if field.metadata['block_rows']!=1:raise ValueError('Use a declared representative-epoch model, validated against native symbols')
    use=np.asarray(selected,bool);y=profile_static(field.y*use[:,None],field.root_information,use,groups)
    adjoint=(y@field.whiten.conj())*field.root_information[:,None]
    coefficients=field.pulse.kernels.coefficients;blocks=field.pulse.row_blocks
    degree=len(coefficients)-1;out=np.empty((len(field.time),cells,degree+1),complex)
    for cell in range(first_cell,first_cell+cells):
        at=field.offsets-cell-1+field.pulse.kernels.radius
        if at.min()<0 or at.max()>=2*field.pulse.kernels.radius:raise ValueError('Range polynomial requests unretained pulse support')
        for k in range(degree+1):
            pulse=coefficients[k,blocks[:,None],at[None,:]]
            out[:,cell-first_cell,k]=(-1)**k*np.sum(pulse.conj()*adjoint,axis=1)
    return out


def trial_geometry(band,parameters,selected,stride):
    f=band.field;kin=band.kinematics
    centre,velocity,dp,dv=kin.point_state(parameters,[0.,0.,0.],derivatives=True)
    frame,fd=kin.orientation(parameters)
    state=dict(position_ecef_m=centre,velocity_ecef_mps=velocity,body_to_ecef=frame,frame_velocity=fd)
    base,details=geometry_from_state(state,band.tx,band.rx,f.metadata['rf_hz'],0.,np.asarray(selected,float))
    observer=f.data['centre_path_m'];carrier=f.data.get('carrier_reference_path_m',observer)
    base[:,9,0]=(details['path_m']-observer+band.origin)*8e6/C
    phase=f.k*(details['path_m']-carrier);base[:,9,1]=np.cos(phase);base[:,9,2]=np.sin(phase)
    base[:,5,3]=f.k
    result=np.zeros((stride,50,4),dtype='<f4');result[:len(centre),:10]=base
    for j in range(5):
        result[:len(centre),10+8*j,:3]=dp[:,:,j];result[:len(centre),11+8*j,:3]=dv[:,:,j]
    for axis in range(3):
        point=np.eye(3)[axis];p=kin.point_state(parameters,point,derivatives=True)
        for j in range(5):
            result[:len(centre),12+8*j+axis,:3]=p[2][:,:,j]-dp[:,:,j]
            result[:len(centre),15+8*j+axis,:3]=p[3][:,:,j]-dv[:,:,j]
    return result


def read_exact(stream,size):
    value=bytearray(size);view=memoryview(value)
    while view:
        count=stream.readinto(view)
        if not count:raise RuntimeError('Dense focus worker closed its output')
        view=view[count:]
    return value


class DenseImages:
    def __init__(self,bands,pixels,directory,*,training=True,full_pulse_support=False):
        self.bands=bands;self.pixels=np.asarray(pixels);self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.selected=[(b.training if training else ~b.training)&(b.field.data['information']>0) for b in bands]
        self.stride=((max(len(b.field.time) for b in bands)+127)//128)*128
        self.first_cell=-15;self.cells=31;self.degree=bands[0].field.pulse.kernels.degree
        if full_pulse_support:
            self.first_cell=int(max(b.field.offsets.max()-b.field.pulse.kernels.radius for b in bands))
            last=int(min(b.field.offsets.min()+b.field.pulse.kernels.radius for b in bands))
            self.cells=last-self.first_cell
            if self.cells<=0:raise ValueError('No common retained pulse-interpolation support')
        self.coefficients=np.zeros((len(bands),self.stride,self.cells,self.degree+1),dtype='<c8')
        for c,b in enumerate(bands):self.coefficients[c,:len(b.field.time)]=matched_polynomials(b.field,self.selected[c],self.first_cell,self.cells,getattr(b,'groups',None))
        path=self.directory/'model.bin'
        with path.open('xb') as output:
            output.write(struct.pack('<8sIIIIIi',b'SHIPAF2\0',self.stride,len(bands),self.cells,self.degree,len(pixels),self.first_cell))
            self.coefficients.tofile(output)
            np.c_[pixels,np.zeros(len(pixels))].astype('<f4').tofile(output)
        self.log=(self.directory/'gpu.log').open('w')
        self.worker=subprocess.Popen(['build/release/support/isar/isar_focus_patch',str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log)
        self.calls=0;self.seconds=0.;self.derivatives=5;self.archive_pixels=len(pixels)
        self.archive_pixel_coordinates=np.c_[pixels,np.zeros(len(pixels))].astype('<f4')

    def replace_coefficients(self,coefficients):
        """Replace linear observation data without recompiling the operator.

        Geometry must be supplied again, as with replace_archive. Original
        archive dimensions and the full proposal pixel grid are retained.
        """
        values=np.asarray(coefficients,dtype='<c8')
        expected=(len(self.bands),self.stride,self.cells,self.degree+1)
        if values.shape!=expected or not np.isfinite(values).all():raise ValueError('Inconsistent observation-polynomial replacement')
        self.worker.stdin.write(struct.pack('<I',2));self.worker.stdin.write(values.tobytes())
        self.worker.stdin.write(self.archive_pixel_coordinates.tobytes());self.worker.stdin.flush()
        status,=struct.unpack('<I',read_exact(self.worker.stdout,4))
        if status:raise RuntimeError('Observation-polynomial replacement failed')
        self.pixels=self.archive_pixel_coordinates[:,:3].copy()

    @classmethod
    def reopen(cls,path,*,log_path=None,derivatives=5):
        """Reopen an immutable polynomial image operator with bounded GPU RAM."""
        self=cls.__new__(cls);path=Path(path)
        with path.open('rb') as source:
            magic,stride,bands,cells,degree,pixels,first_cell=struct.unpack('<8sIIIIIi',source.read(32))
        if magic!=b'SHIPAF2\0':raise ValueError('Unexpected cached focus operator')
        self.stride,self.cells,self.degree,self.first_cell=stride,cells,degree,first_cell
        self.bands=[None]*bands;self.pixels=np.empty((pixels,0));self.calls=0;self.seconds=0.;self.derivatives=derivatives;self.archive_pixels=pixels
        self.log=Path(log_path or path.parent/'gpu-reopen.log').open('a')
        self.worker=subprocess.Popen(['build/release/support/isar/isar_focus_patch',str(path),str(derivatives)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log)
        return self

    def replace_archive(self,path):
        """Upload another fixed window while retaining GPU pipelines/buffers."""
        with Path(path).open('rb') as source:
            header=struct.unpack('<8sIIIIIi',source.read(32))
            expected=(b'SHIPAF2\0',self.stride,len(self.bands),self.cells,self.degree,self.archive_pixels,self.first_cell)
            if header!=expected:raise ValueError('Replacement operator has different dimensions')
            payload=source.read()
        self.worker.stdin.write(struct.pack('<I',2));self.worker.stdin.write(payload);self.worker.stdin.flush()
        status,=struct.unpack('<I',read_exact(self.worker.stdout,4))
        if status:raise RuntimeError('Replacement operator upload failed')
        self.pixels=np.empty((self.archive_pixels,0))

    def set_pixels(self,pixels):
        points=np.asarray(pixels)
        if points.ndim!=2 or points.shape[1] not in [3,4] or not 0<len(points)<=self.archive_pixels:raise ValueError('Invalid focus pixel coordinates')
        if points.shape[1]==3:points=np.c_[points,np.zeros(len(points))]
        self.worker.stdin.write(struct.pack('<II',5,len(points)));self.worker.stdin.write(np.asarray(points,dtype='<f4').tobytes());self.worker.stdin.flush()
        status,=struct.unpack('<I',read_exact(self.worker.stdout,4))
        if status:raise RuntimeError('Focus pixel update failed')
        self.pixels=points[:,:3].copy()

    def evaluate_cached(self,*,derivatives=True):
        started=time.monotonic()
        self.worker.stdin.write(struct.pack('<I',6 if derivatives else 7));self.worker.stdin.flush()
        status,=struct.unpack('<I',read_exact(self.worker.stdout,4));width=self.derivatives+1 if derivatives else 1
        raw=read_exact(self.worker.stdout,len(self.bands)*width*len(self.pixels)*8)
        if status:raise ValueError('Updated focus pixels exceed fixed waveform support')
        self.calls+=1;self.seconds+=time.monotonic()-started
        return np.frombuffer(raw,'<c8').reshape(len(self.bands),width,len(self.pixels)).astype(complex)

    def histories(self):
        """Unreduced complex matched contributions at every retained epoch.

        Uses exactly the last physical geometry and current pixels; summing
        rows reproduces the image. No phase smoothing or rephasing is applied.
        """
        self.worker.stdin.write(struct.pack('<I',8));self.worker.stdin.flush()
        status,=struct.unpack('<I',read_exact(self.worker.stdout,4))
        raw=read_exact(self.worker.stdout,len(self.bands)*self.stride*len(self.pixels)*8)
        if status:raise ValueError('History points exceed fixed waveform support')
        return np.frombuffer(raw,'<c8').reshape(len(self.bands),self.stride,len(self.pixels)).astype(complex)

    def evaluate(self,parameters):
        geometry=np.stack([trial_geometry(b,parameters,s,self.stride) for b,s in zip(self.bands,self.selected)])
        return self.evaluate_geometry(geometry)

    def evaluate_geometry(self,geometry,*,derivatives=True):
        started=time.monotonic()
        if geometry.shape!=(len(self.bands),self.stride,10+8*self.derivatives,4):raise ValueError('Incorrect focus geometry shape')
        self.worker.stdin.write(struct.pack('<I',0 if derivatives else 3));self.worker.stdin.write(np.asarray(geometry,dtype='<f4').tobytes());self.worker.stdin.flush()
        status,=struct.unpack('<I',read_exact(self.worker.stdout,4))
        width=self.derivatives+1 if derivatives else 1
        raw=read_exact(self.worker.stdout,len(self.bands)*width*len(self.pixels)*8)
        if status:raise ValueError(f'{status} trial row/pixel pairs exceed the fixed known-waveform support')
        self.calls+=1;self.seconds+=time.monotonic()-started
        return np.frombuffer(raw,'<c8').reshape(len(self.bands),width,len(self.pixels)).astype(complex)

    def adjoint(self,cotangent,*,active_derivatives=None):
        """2 Re(sum(conj(cotangent) dI)) for each row's five pose variables.

        This differentiates the last geometry packet. It permits a continuous
        motion spline with many coefficients without a per-coefficient image.
        """
        weight=np.asarray(cotangent,dtype='<c8')
        if weight.shape!=(len(self.bands),len(self.pixels)):raise ValueError('Incorrect image cotangent shape')
        self.worker.stdin.write(struct.pack('<I',1) if active_derivatives is None else struct.pack('<II',4,active_derivatives));self.worker.stdin.write(weight.tobytes());self.worker.stdin.flush()
        status,=struct.unpack('<I',read_exact(self.worker.stdout,4))
        raw=read_exact(self.worker.stdout,len(self.bands)*self.stride*self.derivatives*4)
        if status:raise ValueError('Adjoint exceeds fixed waveform support')
        return np.frombuffer(raw,'<f4').reshape(len(self.bands),self.stride,self.derivatives).astype(float)

    def close(self):
        self.worker.stdin.close();code=self.worker.wait(timeout=60);self.worker.stdout.close();self.log.close()
        if code:raise RuntimeError(f'Dense focus worker failed with {code}')


def entropy(image,derivative=None):
    power=abs(image)**2;total=power.sum();squared=np.sum(power**2)
    value=2*np.log(total)-np.log(squared)
    if derivative is None:return float(value)
    dp=2*np.real(image.conj()[:,None]*derivative)
    gradient=2*dp.sum(axis=0)/total-2*np.sum(power[:,None]*dp,axis=0)/squared
    return float(value),gradient


def focus(model,initial,gains,bounds,*,iterations=100,fit_phases=False,checkpoint=None):
    """Minimize order-two image entropy; validate on different epochs later.

    This is a focusing objective, not a CRLB claim. It changes one common
    physical pose and optionally constant interband phases, never image rows,
    pixel phases, noise spectra or independent band motion histories.
    """
    gains=np.asarray(gains);free=[c for c in range(len(gains)) if c!=3]
    initial=np.r_[initial,np.zeros(len(free)) if fit_phases else []]
    scale=np.r_[.05,.005,.0005,.05,.005,np.ones(len(free)) if fit_phases else []]
    bounds=(np.r_[bounds[0],np.full(len(free),-np.pi) if fit_phases else []],
        np.r_[bounds[1],np.full(len(free),np.pi) if fit_phases else []])
    cache=[None,None];trace=[];normal=np.sqrt(sum(abs(g)**2*np.sum(b.field.root_information**2) for g,b in zip(gains,model.bands)))
    def evaluate(p):
        if cache[0] is None or not np.array_equal(p,cache[0]):
            image=model.evaluate(p[:5]);gain=gains.copy()
            if fit_phases:gain[free]*=np.exp(1j*p[5:])
            z=np.einsum('c,cp->p',gain.conj(),image[:,0])/normal
            derivative=np.einsum('c,cjp->pj',gain.conj(),image[:,1:])/normal
            if fit_phases:
                phase=np.stack([-1j*gain[c].conjugate()*image[c,0]/normal for c in free],axis=1)
                derivative=np.c_[derivative,phase]
            value,gradient=entropy(z,derivative)
            cache[:]=[p.copy(),dict(value=value,gradient=gradient,image=image,gains=gain,combined=z)]
        return cache[1]
    def objective(q):
        r=evaluate(q*scale);return r['value'],r['gradient']*scale
    def callback(q):
        r=evaluate(q*scale);record=dict(objective=r['value'],parameters=(q*scale).tolist(),maximum_gradient=float(np.max(abs(r['gradient']*scale))),evaluations=model.calls)
        trace.append(record)
        if checkpoint:checkpoint(record)
    before=evaluate(initial)
    fitted=minimize(objective,initial/scale,jac=True,method='L-BFGS-B',bounds=list(zip(np.asarray(bounds[0])/scale,np.asarray(bounds[1])/scale)),
        callback=callback,options={'maxiter':iterations,'maxcor':20,'ftol':1e-12,'gtol':1e-6,'maxls':20})
    p=fitted.x*scale;after=evaluate(p)
    gradient=fitted.jac.copy();low=np.asarray(bounds[0]);high=np.asarray(bounds[1])
    gradient=np.where(p<low+1e-8,np.minimum(gradient,0),np.where(p>high-1e-8,np.maximum(gradient,0),gradient))
    gradient=np.where(low==high,0.,gradient)
    return p,dict(parameters=p.tolist(),motion_parameters=p[:5].tolist(),gains_real_imag=np.c_[after['gains'].real,after['gains'].imag].tolist(),
        initial_entropy=before['value'],final_entropy=after['value'],maximum_scaled_gradient=float(np.max(abs(fitted.jac))),
        maximum_projected_gradient=float(np.max(abs(gradient))),
        active_physical_bounds=np.flatnonzero(((p<low+1e-8)|(p>high-1e-8))&(low<high)).tolist(),
        frozen_parameters=np.flatnonzero(low==high).tolist(),
        converged=bool(fitted.success and np.max(abs(gradient))<1e-5),message=fitted.message,trace=trace,
        individual_entropy_before=[entropy(a) for a in before['image'][:,0]],individual_entropy_after=[entropy(a) for a in after['image'][:,0]],
        calls=model.calls,seconds=model.seconds),after
