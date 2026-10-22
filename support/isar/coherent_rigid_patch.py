"""Shared complex ship scene, interband gains and continuous point positions.

Unlike a registration at one pixel, one band gain must explain every selected
point and all unused epochs. All reflectors share the same rigid motion.
Reflector amplitudes are profiled jointly; the nonlinear optimizer sees exact
complex variable-projection derivatives, including coefficient refitting.
"""
import numpy as np
from scipy.optimize import least_squares


class CoherentPatch:
    def __init__(self,bands,initial_motion,*,reference_band=3):
        self.bands=bands;self.points=bands[0].points.copy();self.count=len(self.points)
        if any(not np.array_equal(b.points,self.points) for b in bands):raise ValueError('One physical point set is required')
        self.reference_band=reference_band;self.gain_bands=[c for c in range(len(bands)) if c!=reference_band]
        self.ng=2*len(self.gain_bands);self.nposition=2*(self.count-1);self.size=5+self.nposition+self.ng
        self.energy=sum(b.energy for b in bands)
        coefficients=np.asarray([b.evaluate(initial_motion,derivatives=False)['coefficient'] for b in bands])
        u,s,vh=np.linalg.svd(coefficients,full_matrices=False);gain=u[:,0]
        if abs(gain[reference_band])<1e-8:raise ValueError('The reference band has no supported scene response')
        gain=gain/gain[reference_band]
        self.initial=np.r_[initial_motion,np.zeros(self.nposition),np.c_[gain[self.gain_bands].real,gain[self.gain_bands].imag].ravel()]
        self.last=None

    def unpack(self,p):
        points=self.points.copy();points[1:,:2]+=np.asarray(p)[5:5+self.nposition].reshape(-1,2)
        gain=np.ones(len(self.bands),complex);v=np.asarray(p)[-self.ng:].reshape(-1,2)
        gain[self.gain_bands]=v[:,0]+1j*v[:,1]
        return np.asarray(p)[:5],points,gain

    def evaluate(self,p,*,derivatives=True):
        motion,points,gain=self.unpack(p);local=[];design=[];data=[]
        for c,b in enumerate(self.bands):
            h,dh=b.design(motion,derivatives=derivatives,points=points,point_derivatives=True)
            a=b.project(h).reshape(-1,self.count)[b.selected]
            da=b.project(dh).reshape(-1,self.count,7)[b.selected] if derivatives else None
            local.append((a,da));design.append(gain[c]*a);data.append(b.data)
        a=np.concatenate(design);y=np.concatenate(data)
        u,s,vh=np.linalg.svd(a,full_matrices=False)
        if s[-1]<s[0]*1e-10:raise ValueError('Coherent scene has unresolved columns')
        alpha=vh.conj().T@((u.conj().T@y)/s);residual=y-a@alpha
        result=dict(residual=residual,coefficient=alpha,gain=gain,points=points,condition=s[0]/s[-1],value=np.vdot(residual,residual).real)
        if not derivatives:return result
        fixed=[];cross=np.zeros((self.count,self.size),complex);first=0
        for c,((ac,da),b) in enumerate(zip(local,self.bands)):
            r=residual[first:first+len(ac)];first+=len(ac)
            v=np.zeros((len(ac),self.size),complex)
            v[:,:5]=gain[c]*np.einsum('ncp,c->np',da[:,:,:5],alpha)
            cross[:,:5]+=gain[c].conjugate()*np.einsum('ncp,n->cp',da[:,:,:5].conj(),r)
            for point in range(1,self.count):
                where=slice(5+2*(point-1),5+2*point)
                v[:,where]=gain[c]*alpha[point]*da[:,point,5:]
                cross[point,where]+=gain[c].conjugate()*(da[:,point,5:].conj().T@r)
            if c!=self.reference_band:
                j=5+self.nposition+2*self.gain_bands.index(c);pred=ac@alpha
                v[:,j]=pred;v[:,j+1]=1j*pred
                cross[:,j]=ac.conj().T@r;cross[:,j+1]=-1j*(ac.conj().T@r)
            fixed.append(v)
        v=np.concatenate(fixed)
        result['jacobian']=-v+u@(u.conj().T@v)-u@((vh@cross)/s[:,None])
        return result

    def validation(self,p):
        fitted=self.evaluate(p,derivatives=False);motion,points,gains=self.unpack(p);records=[]
        for c,b in enumerate(self.bands):
            h,_=b.design(motion,points=points);test=~b.training;selected=np.repeat(test,len(b.field.offsets))
            a=b.project(h,test).reshape(-1,self.count)[selected];y=b.project(b.field.y,test).ravel()[selected]
            predicted=gains[c]*(a@fitted['coefficient']);free=np.linalg.lstsq(a,y,rcond=1e-10)[0];best=a@free
            rss=np.vdot(y-predicted,y-predicted).real
            records.append(dict(unused_rss=rss,unused_data_energy=np.vdot(y,y).real,
                unused_prediction_energy=np.vdot(predicted,predicted).real,unused_independent_scene_energy=np.vdot(best,best).real,
                unused_complex_prediction_overlap=abs(np.vdot(predicted,best))**2/(np.vdot(predicted,predicted).real*np.vdot(best,best).real),
                free_coefficient_real_imag=np.c_[free.real,free.imag].tolist()))
        return records

    def fit(self,*,bounds,max_evaluations=150):
        scale=np.r_[.05,.005,.0005,.05,.005,np.full(self.nposition,.5),np.full(self.ng,.1)]
        cache=[None,None]
        def evaluate(q):
            if cache[0] is None or not np.array_equal(q,cache[0]):cache[:]=[q.copy(),self.evaluate(q*scale)]
            return cache[1]
        root=np.sqrt(self.energy)
        r=least_squares(lambda q:np.r_[evaluate(q)['residual'].real,evaluate(q)['residual'].imag]/root,
            self.initial/scale,jac=lambda q:np.r_[evaluate(q)['jacobian'].real,evaluate(q)['jacobian'].imag]*scale/root,
            bounds=(np.asarray(bounds[0])/scale,np.asarray(bounds[1])/scale),x_scale='jac',
            max_nfev=max_evaluations,ftol=1e-11,xtol=1e-11,gtol=1e-9)
        p=r.x*scale;motion,points,gains=self.unpack(p);local=self.evaluate(p,derivatives=False)
        return p,dict(parameters=p.tolist(),motion_parameters=motion.tolist(),points_east_north_up_m=points.tolist(),
            gains_real_imag=np.c_[gains.real,gains.imag].tolist(),scene_coefficient_real_imag=np.c_[local['coefficient'].real,local['coefficient'].imag].tolist(),
            objective=float(2*r.cost),maximum_scaled_gradient=float(r.optimality),success=bool(r.success and r.optimality<1e-7),
            evaluations=r.nfev,condition=float(local['condition']),validation=self.validation(p))
