"""Noise-metric Radon/chirp matched filter and exact quadratic-path refinement.

The chirp sum is the nonuniform fractional-Fourier kernel, parameterized by
physical excess-path rate and acceleration. Coarse range migration and fine
carrier phase use separate grids. Original observations are never resampled
for acquisition. The exact C0/C1, known-pulse, original-symbol operator scores
and refines candidates; a coarse chirp approximation is never a final result.

All ranges are TOTAL bistatic excess path; there is no monostatic factor two.
References: Li et al., arXiv:2403.06788; Tian et al., arXiv:1511.03122;
Li et al., arXiv:1904.00169 (variable coherent apertures).
"""
import numpy as np
from scipy.fft import fft, ifft, fftfreq, next_fast_len
from scipy.optimize import least_squares

from field import FS, average
from motion import C


def fractional_angle(acceleration_mps2, rf_hz, time_scale_s):
    """FRFT convention exp(+i*pi*u²*cot(angle)), t=time_scale*u."""
    return np.arctan2(1., rf_hz/C*acceleration_mps2*time_scale_s**2)


def wavenumber_focus(profiles, spacing_m, path_change_m, rf_hz, *, pad_factor=4):
    """Range-frequency migration correction plus RF dechirp, at supplied epochs.

    x(r-L(t))*exp(-i*k*L(t)) -> x(r-L0)*exp(-i*k*L0).
    No Stolt mapping or fictitious monostatic geometry is introduced. With no
    padding this is unitary on the periodic discrete model. Padding supplies
    guard space for a finite crop; it cannot invent unrecorded range support.
    """
    values=np.asarray(profiles,complex);delta=np.asarray(path_change_m,float)
    if values.ndim!=2 or delta.shape!=(len(values),) or spacing_m<=0 or pad_factor<1:
        raise ValueError('Need complex time/range profiles, one path increment per epoch, positive spacing and padding')
    n=values.shape[1];size=n if pad_factor==1 else next_fast_len(pad_factor*n);left=(size-n)//2
    padded=np.pad(values,((0,0),(left,size-n-left)))
    frequency=fftfreq(size,d=spacing_m)
    correction=np.exp(2j*np.pi*frequency[None,:]*delta[:,None])
    shifted=ifft(fft(padded,axis=1)*correction,axis=1)[:,left:left+n]
    return shifted*np.exp(2j*np.pi*rf_hz/C*delta[:,None])


class QuadraticMotionKernel:
    def __init__(self,field,training,stationary_design):
        self.field=field;self.training=np.asarray(training,bool)
        self.selected=np.repeat(self.training,len(field.offsets));self.design=np.asarray(stationary_design)
        u,s,_=np.linalg.svd(self.design[self.selected],full_matrices=False)
        self.q=u[:,s>s[0]*1e-10] if len(s) else u
        self.epoch=(field.time[0]+field.time[-1])/2
        self.duration=field.time[-1]-field.time[0]
        self.reference=float(np.interp(self.epoch,field.data['row_time'],field.data['centre_path_m']))
        self.dt=field.data['row_time']-self.epoch
        self.scale=np.array([C/FS,2*np.pi/field.k/self.duration,2*np.pi/field.k/self.duration**2])
        # A quadratic reference supplies search centres only; all three path
        # coefficients are fitted freely, independently in each band.
        self.nominal=np.polynomial.polynomial.polyfit(self.dt,field.data['centre_path_m']-self.reference,2)*[1,1,2]

    def response(self,p,derivatives=False):
        f=self.field;dt=self.dt;r,v,a=np.asarray(p)
        delta=self.reference+r+v*dt+.5*a*dt**2-f.data['centre_path_m']
        omega=-f.k*.001024*(v+a*dt)
        value=f.path_response(delta,omega,derivatives=derivatives)
        if derivatives:
            raw,path,doppler,_=value
            derivative=np.stack([path,path*dt[:,None]-f.k*.001024*doppler,
                .5*path*dt[:,None]**2-f.k*.001024*dt[:,None]*doppler],axis=-1)
        else:raw=value
        h=f.apply(average(raw,f.data['row_information'],f.metadata['block_rows'])[0]).ravel()
        if not derivatives:return h
        dj=average(derivative,f.data['row_information'],f.metadata['block_rows'])[0]
        return h,np.column_stack([f.apply(dj[:,:,i]).ravel() for i in range(3)])

    def projected(self,value):return value-self.q@(self.q.conj().T@value)

    def evaluate(self,p,data,derivatives=True):
        value=self.response(p,derivatives=derivatives);h,dh=value if derivatives else (value,None)
        a=self.projected(h[self.selected]);y=self.projected(np.asarray(data)[self.selected])
        energy=np.vdot(a,a).real
        if energy<=0:raise ValueError('No retained information for this hypothesis')
        alpha=np.vdot(a,y)/energy;r=y-a*alpha
        if not derivatives:return r,None,alpha,h
        da=self.projected(dh[self.selected]);v=da*alpha
        dc=(-a.conj()@v+da.conj().T@r)/energy
        dr=-v-a[:,None]*dc
        return r,dr,alpha,h

    def search(self,data,range_grid_m,*,rate_halfwidth=.6,acceleration_halfwidth=.05,keep=5):
        """Dual-scale Radon-FRFT search with the original projected noise metric."""
        f=self.field;t=f.time-self.epoch;train=self.training;nr=len(f.offsets)
        y=self.projected(np.asarray(data)[self.selected]).reshape(-1,nr)
        q=self.q.reshape(train.sum(),nr,-1)
        # Coarse envelope displacement < quarter of a range sample at an edge.
        vstep=C/FS/(2*self.duration);astep=2*C/FS/self.duration**2
        coarse_v=np.arange(-rate_halfwidth,rate_halfwidth+vstep,vstep)+self.nominal[1]
        coarse_a=np.arange(-acceleration_halfwidth,acceleration_halfwidth+astep,astep)+self.nominal[2]
        # Fine phase grids; exact continuous refinement removes grid loss.
        dv=max(2*np.pi/f.k/(4*self.duration),1e-6)
        da=max(2*np.pi/f.k/(self.duration**2),1e-8)
        proposals=[]
        for v in coarse_v:
            for a in coarse_a:
                fine_v=np.arange(-vstep/2,vstep/2+dv,dv)
                fine_a=np.arange(-astep/2,astep/2+da,da)
                vv,aa=np.meshgrid(fine_v,fine_a,indexing='ij');vv,aa=vv.ravel(),aa.ravel()
                valid=(abs(v+vv-self.nominal[1])<=rate_halfwidth+dv)&(abs(a+aa-self.nominal[2])<=acceleration_halfwidth+da)
                vv,aa=vv[valid],aa[valid]
                if not len(vv):continue
                # Direct nonuniform FRFT: no uniform-epoch interpolation.
                tone=np.exp(-1j*f.k*(t[train,None]*vv+.5*t[train,None]**2*aa))
                for origin in range_grid_m:
                    h=self.response([origin,v,a])[self.selected].reshape(-1,nr)
                    cross=np.sum(h.conj()*y,axis=1)
                    nuisance=np.einsum('nrk,nr->kn',q.conj(),h)@tone
                    norm=np.sum(abs(h)**2)-np.sum(abs(nuisance)**2,axis=0)
                    product=tone.conj().T@cross
                    score=np.divide(abs(product)**2,norm,out=np.zeros_like(norm),where=norm>np.sum(abs(h)**2)*1e-12)
                    for i in np.argsort(-score)[:keep]:
                        proposals.append(dict(parameters=np.array([origin,v+vv[i],a+aa[i]]),
                            score=score[i],retained_fraction=norm[i]/max(np.sum(abs(h)**2),1e-100)))
        selected=[]
        for item in sorted(proposals,key=lambda p:-p['score']):
            if any(np.linalg.norm((item['parameters']-s['parameters'])/self.scale)<1. for s in selected):continue
            selected.append(item)
            if len(selected)==keep:break
        return selected

    def refine(self,data,seed,*,bounds,max_evaluations=100):
        cache=[None,None]
        def evaluate(q):
            if cache[0] is None or not np.array_equal(cache[0],q):cache[:]=[q.copy(),self.evaluate(q*self.scale,data)]
            return cache[1]
        fit=least_squares(lambda q:np.r_[evaluate(q)[0].real,evaluate(q)[0].imag],np.asarray(seed)/self.scale,
            jac=lambda q:np.r_[evaluate(q)[1].real,evaluate(q)[1].imag]*self.scale,
            bounds=(np.asarray(bounds[0])/self.scale,np.asarray(bounds[1])/self.scale),
            max_nfev=max_evaluations,ftol=1e-11,xtol=1e-11,gtol=1e-8,x_scale='jac')
        p=fit.x*self.scale;r,j,alpha,h=self.evaluate(p,data)
        beta=np.linalg.lstsq(self.design[self.selected],np.asarray(data)[self.selected]-h[self.selected]*alpha,rcond=1e-10)[0]
        pred=h*alpha+self.design@beta;test=~self.selected
        data_residual=np.asarray(data)-pred
        return dict(parameters=p,training_rss=np.vdot(r,r).real,test_rss=np.sum(abs(data_residual[test])**2),
            score=abs(alpha)**2*np.vdot(self.projected(h[self.selected]),self.projected(h[self.selected])).real,
            alpha_real_imag=[alpha.real,alpha.imag],gradient=fit.optimality,nfev=fit.nfev,success=fit.success,
            termination=fit.message,frft_angle_degrees=np.degrees(fractional_angle(p[2],self.field.metadata['rf_hz'],self.duration)),
            prediction=pred,residual=data_residual,unit_template=h)
