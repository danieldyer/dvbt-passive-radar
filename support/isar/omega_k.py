"""Bistatic range-migration imaging with a numerical point-reference spectrum.

Coordinates are cross-track x and along-track u in metres. Both stationary
antennas move along the same straight track in the target frame. The exact
two-leg reference is used, never a monostatic equivalent range. A numerical
spectral correlator is retained as the reference; the fast processor performs
Stolt resampling, differential range mapping and phase/amplitude correction.

Input is the range-frequency matched-data integrand, not a magnitude image.
Output is a complex image with the declared spatial carrier removed.
"""
from dataclasses import dataclass

import numpy as np
from scipy.fft import fft,ifft,fftshift,ifftshift,fftfreq,next_fast_len


@dataclass
class ParallelGeometry:
    """Antenna coordinates relative to the target reference, axes x/u/up."""
    transmitter: np.ndarray
    receiver: np.ndarray

    def path(self,x,u,derivatives=False):
        x,u=np.broadcast_arrays(np.asarray(x),np.asarray(u));length=np.zeros(x.shape)
        gx=np.zeros_like(length);gu=np.zeros_like(length);huu=np.zeros_like(length);hxu=np.zeros_like(length)
        for antenna in (self.transmitter,self.receiver):
            dx=x-antenna[0];dy=u-antenna[1];dz=-antenna[2]
            distance=np.sqrt(dx*dx+dy*dy+dz*dz);length+=distance
            gx+=dx/distance;gu+=dy/distance
            huu+=(dx*dx+dz*dz)/distance**3;hxu-=dx*dy/distance**3
        length-=np.linalg.norm(self.transmitter-self.receiver)
        return (length,gx,gu,huu,hxu) if derivatives else length

    def stationary(self,ky,k,*,x=0.):
        """Solve k*dL/du=ky on the exact double hyperbola."""
        ky,k=np.broadcast_arrays(np.asarray(ky),np.asarray(k));u=np.zeros_like(ky)
        for _ in range(18):
            _,_,gradient,curvature,_=self.path(x,u,True)
            step=(gradient-ky/k)/curvature
            u-=np.clip(step,-5000,5000)
        _,_,gradient,_,_=self.path(x,u,True)
        valid=(abs(gradient-ky/k)<1e-9)&np.isfinite(u)
        return u,valid


def sinc_stencil(coordinate,size,*,radius=8):
    q=np.asarray(coordinate);base=np.floor(q).astype(int);tap=np.arange(-radius+1,radius+1)
    at=base[...,None]+tap;d=q[...,None]-at
    weight=np.sinc(d)*np.i0(8*np.sqrt(np.maximum(0,1-(d/radius)**2)))/np.i0(8)
    valid=(at>=0)&(at<size);at=np.clip(at,0,size-1)
    return at,weight*valid


def sinc_sample(values,coordinate,*,radius=8):
    """Complex bandlimited interpolation on a regular grid, zero outside it."""
    at,weight=sinc_stencil(coordinate,len(values),radius=radius)
    # No local renormalisation at missing support. These are zero-extended data.
    return np.sum(values[at]*weight,axis=-1)


class BistaticOmegaK:
    def __init__(self,geometry,u,k,*,maximum_along_offset_m=250.,padding=2):
        self.geometry=geometry;self.u=np.asarray(u,float);self.k=np.asarray(k,float)
        if len(u)<8 or len(k)<4 or np.any(np.diff(k)<=0) or np.any(np.diff(u)<=0):
            raise ValueError('Increasing, nonempty physical sample coordinates required')
        self.du=float(np.diff(u).mean());self.dk=float(np.diff(k).mean())
        if np.max(abs(np.diff(u)-self.du))>1e-8 or np.max(abs(np.diff(k)-self.dk))>1e-10:
            raise ValueError('Regular along-track and radial-wavenumber grids required')
        self.u_offset=self.u[len(u)//2]
        self.reference_inner=np.max(abs(u))+maximum_along_offset_m+2*self.du
        needed=2*self.reference_inner/self.du+128
        self.n=next_fast_len(int(np.ceil(max(needed,padding*len(u)))))
        self.s=(np.arange(self.n)-self.n//2)*self.du
        outer=max(abs(self.s))-self.du
        fraction=np.clip((abs(self.s)-self.reference_inner)/(outer-self.reference_inner),0,1)
        self.extension=np.cos(np.pi*fraction/2)**2
        self.kappa=2*np.pi*fftfreq(self.n,d=self.du)
        self.kc=float((k[0]+k[-1])/2)
        self.reference,self.gx,self.gu,_,_=geometry.path(0.,0.,True)
        self.carrier=np.array([self.kc*self.gx,self.kc*self.gu])
        self.h0,self.dh0=self.reference_spectrum(0.,self.k)

    def reference_spectrum(self,x,k,*,derivative=False):
        """Exact numerical transfer, including finite-support padding.

        An extended reference covers every requested image shift. No observed
        data are extrapolated; input observations remain zero outside aperture.
        """
        k=np.atleast_1d(k);path=self.geometry.path(x,self.s)-self.reference
        raw=self.extension[:,None]*np.exp(-1j*path[:,None]*k[None,:]+1j*self.carrier[1]*self.s[:,None])
        h=fft(ifftshift(raw,axes=0),axis=0)
        dh=fft(ifftshift(-1j*path[:,None]*raw,axes=0),axis=0) if derivative else None
        return h,dh

    def data_spectrum(self,data):
        if np.shape(data)!=(len(self.u),len(self.k)):raise ValueError('Time/radial-frequency data shape mismatch')
        padded=np.zeros((self.n,len(self.k)),complex)
        start=self.n//2-len(self.u)//2
        padded[start:start+len(self.u)]=data*np.exp(1j*self.carrier[1]*self.u[:,None])
        return fft(ifftshift(padded,axes=0),axis=0)*np.exp(-1j*self.kappa[:,None]*self.u_offset)

    def reference_image(self,data,x):
        """Exact spectral correlation; independent of Stolt/stationary-phase mapping."""
        d=self.data_spectrum(data);output=[]
        for xx in x:
            h,_=self.reference_spectrum(xx,self.k)
            image=fftshift(fft(np.sum(d*h.conj(),axis=1))/self.n)
            output.append(image*np.exp(-1j*self.carrier[0]*xx))
        return np.asarray(output).T,self.s

    def direct(self,data,points):
        """Independent exact time/range-frequency matched filter at selected pixels."""
        out=[]
        for x,y in points:
            path=self.geometry.path(x,self.u+y)-self.reference
            h=np.exp(-1j*path[:,None]*self.k)
            out.append(np.sum(data*h.conj())*np.exp(-1j*(self.carrier@[x,y])))
        return np.asarray(out)

    def stolt_image(self,data,x,*,range_padding=4,interpolation_radius=8,weight_points=(),range_period_m=None):
        """Stolt mapping followed by differential range/phase correction.

        The phase ratio H_x/H_0 is expanded about carrier frequency, not about
        pixel position. Numerical group delay corrects residual range curvature;
        the full ratio phase and amplitude at carrier are retained. The neglected
        range-band curvature is quantified by the validation driver.
        """
        d=self.data_spectrum(data);filtered=d*self.h0.conj()
        ky=self.carrier[1]-self.kappa
        # dKx/dk and Kx are evaluated at the exact bistatic stationary point.
        s,valid=self.geometry.stationary(ky[:,None],self.k[None,:])
        valid&=(abs(s)<=max(abs(self.s)))
        _,gx,gu,huu,hxu=self.geometry.path(0.,s,True)
        mapped=self.k[None,:]*gx;jac=gx-hxu*gu/huu
        differences=np.diff(mapped,axis=1);pairs=valid[:,1:]&valid[:,:-1]
        if np.any(abs(differences[pairs])<1e-14) or (np.any(differences[pairs]>0) and np.any(differences[pairs]<0)):
            raise ValueError('Folded Stolt mapping: this geometry requires another patch/chart')
        source_dkx=np.median(abs(differences[pairs]))
        dkx=source_dkx/range_padding if range_period_m is None else 2*np.pi/range_period_m
        dx=max(np.diff(x).min(),.1) if len(x)>1 else 1.
        nx=next_fast_len(max(256,int(np.ceil(2*np.pi/(dkx*dx)))))
        # Make the physical range FOV cover the requested interval, while
        # preserving oversampled spectral interpolation rather than enlarging it.
        nx=next_fast_len(max(nx,int(np.ceil((mapped[valid].max()-mapped[valid].min())/dkx))+8))
        qx=self.carrier[0]+2*np.pi*fftfreq(nx,d=2*np.pi/(nx*dkx))
        spectral=np.zeros((self.n,nx),complex);mapped_rows=0;plans=[None]*self.n
        for row in range(self.n):
            ok=valid[row]&np.isfinite(jac[row])&(abs(jac[row])>1e-12)
            if ok.sum()<4:continue
            indices=np.flatnonzero(ok);coordinate=(mapped[row,indices]-self.carrier[0])/dkx
            base=np.floor(coordinate).astype(int);tap=np.arange(-interpolation_radius+1,interpolation_radius+1)
            at=base[:,None]+tap;delta=coordinate[:,None]-at
            weight=np.sinc(delta)*np.i0(14*np.sqrt(np.maximum(0,1-(delta/interpolation_radius)**2)))/np.i0(14)
            # Partition of unity is an interpolation identity, fixed entirely
            # by sample coordinates. It preserves the original discrete
            # quadrature; no density/noise weighting is fitted to the data.
            weight/=weight.sum(axis=1)[:,None];at%=nx
            for t in range(at.shape[1]):np.add.at(spectral[row],at[:,t],filtered[row,indices]*weight[:,t])
            plans[row]=(indices,at,weight)
            mapped_rows+=1
        range_axis=fftshift(2*np.pi*fftfreq(nx,d=dkx))
        range_image=fftshift(ifft(spectral,axis=1)*nx,axes=1)
        range_moment=fftshift(ifft(spectral*(qx-self.carrier[0])[None,:],axis=1)*nx,axes=1)
        h0,dh0=self.reference_spectrum(0.,[self.kc],derivative=True)
        ss,ok=self.geometry.stationary(ky,np.full(self.n,self.kc))
        _,ggx,ggu,hhuu,hhxu=self.geometry.path(0.,ss,True)
        j0=ggx-hhxu*ggu/hhuu;kxc=self.kc*ggx
        output=[];weights={}
        for xx in x:
            hx,dhx=self.reference_spectrum(xx,[self.kc],derivative=True)
            usable=ok&(abs(h0[:,0])>abs(h0[:,0]).max()*1e-8)&(abs(hx[:,0])>abs(hx[:,0]).max()*1e-8)&(abs(j0)>1e-12)
            ratio=np.divide(hx[:,0],h0[:,0],out=np.zeros(self.n,complex),where=usable)
            slope=np.zeros(self.n)
            log_derivative=np.zeros(self.n,complex)
            log_derivative[usable]=dhx[usable,0]/hx[usable,0]-dh0[usable,0]/h0[usable,0]
            slope[usable]=np.imag(log_derivative[usable])
            coordinate=np.divide(-slope,j0,out=np.zeros_like(slope),where=usable)
            selected=np.flatnonzero(usable);samples=np.zeros(self.n,complex);moment=np.zeros_like(samples)
            at,w=sinc_stencil((coordinate[selected]-range_axis[0])/(range_axis[1]-range_axis[0]),nx,radius=interpolation_radius)
            samples[selected]=np.sum(range_image[selected[:,None],at]*w,axis=1)
            moment[selected]=np.sum(range_moment[selected[:,None],at]*w,axis=1)
            amplitude_slope=np.divide(log_derivative.real,j0,out=np.zeros(self.n),where=usable)
            samples+=amplitude_slope*(moment-(kxc-self.carrier[0])*samples)
            correction=ratio.conj()*np.exp(-1j*(kxc-self.carrier[0])*coordinate-1j*self.carrier[0]*xx)
            sample=samples*correction;sample[~usable]=0
            output.append(fftshift(fft(sample)/self.n))
            for px,py in weight_points:
                if abs(px-xx)>1e-9:continue
                fourier_weights=np.zeros_like(d)
                for row,plan in enumerate(plans):
                    if plan is None or not usable[row]:continue
                    sources,at,weight=plan
                    r_at,r_weight=sinc_stencil((coordinate[row]-range_axis[0])/(range_axis[1]-range_axis[0]),nx,radius=interpolation_radius)
                    phase=np.exp(1j*(qx[at]-self.carrier[0])[:,:,None]*range_axis[r_at][None,None,:])@r_weight
                    phase*=correction[row]*np.exp(-1j*self.kappa[row]*py)/self.n
                    phase*=1+amplitude_slope[row]*(qx[at]-kxc[row])
                    fourier_weights[row,sources]=np.sum(phase*weight,axis=1)
                    fourier_weights[row]*=self.h0[row].conj()
                index=(np.arange(len(self.u))-len(self.u)//2)%self.n
                fourier_weights*=np.exp(-1j*self.kappa[:,None]*self.u_offset)
                original_weights=fft(fourier_weights,axis=0)[index]*np.exp(1j*self.carrier[1]*self.u[:,None])
                weights[f'{px:g},{py:g}']=original_weights
        return np.asarray(output).T,self.s,dict(radial_fft_size=nx,azimuth_fft_size=self.n,
            mapped_azimuth_rows=mapped_rows,range_interpolation_radius=interpolation_radius,
            operator_weights=weights,
            method='Exact numerical bistatic reference; adjoint Stolt k->Kx preserving discrete quadrature; differential group-delay and complex-ratio correction')
