"""Joint range/phase fits of several selected echoes across received bands.

Every echo has a shared residual quadratic path and an independent constant
phase and envelope origin in each band. The latter are measurements for later
geometric calibration, not fitted antenna coordinates. Original OFDM moments,
range covariance and row averaging are supplied by the existing Field operator.
No new stationary cancellation or sea/multipath model is introduced.
"""
import numpy as np
from scipy.optimize import least_squares

from amplitude_arc import AmplitudeArc
from amplitude_pair import AmplitudeCluster
from cluster_phase import PolynomialPath
from field import FS
from motion import C


class ReferencedAmplitudeArc(AmplitudeArc):
    def __init__(self,field,training,knots,first,last,known_path,known_rate,*,stationary_design=None,curve=None):
        super().__init__(field,training,knots,stationary_paths=[] if stationary_design is None else None,
            stationary_design=stationary_design)
        curve=PolynomialPath(first,last,2) if curve is None else curve
        self.epoch=self.base.epoch=curve.epoch;self.scale=self.base.scale=curve.scale
        self.base.curve=curve
        self.base.basis=curve(field.data['row_time']);self.base.rate_basis=curve(field.data['row_time'],1)
        self.known_path=np.asarray(known_path);self.known_rate=np.asarray(known_rate)
        if self.known_path.shape!=field.data['row_time'].shape or self.known_rate.shape!=self.known_path.shape:
            raise ValueError('Known path and rate require the exact original-row support')

    def response(self,parameters):
        f,b=self.field,self.base
        delta=self.known_path+b.basis@parameters[2:]
        omega=f.data['centre_omega']-f.k*.001024*(self.known_rate+b.rate_basis@parameters[2:])
        raw,path,rate,envelope=f.path_response(delta,omega,range_bias_m=parameters[0]*C/FS,derivatives=True)
        rotation=np.exp(1j*parameters[1])
        derivative=np.concatenate([(envelope*C/FS)[:,:,None],(1j*raw)[:,:,None],
            path[:,:,None]*b.basis[:,None,:]-rate[:,:,None]*(f.k*.001024*b.rate_basis[:,None,:])],axis=2)
        return raw*rotation,derivative*rotation


class SharedEchoCluster:
    def __init__(self,bands,origins,*,origin_radius=1.,shared_components=None):
        if not bands or not len(origins) or origin_radius<=0:raise ValueError('Bands, selected origins and positive radius required')
        self.bands=bands;self.origins=np.asarray(origins,float);self.components=len(origins)
        self.clusters=[AmplitudeCluster(b,self.components) for b in bands]
        n=self.components;shared=2*n*len(bands)
        selected=np.arange(n) if shared_components is None else np.asarray(shared_components,int)
        if selected.ndim!=1 or len(np.unique(selected))!=len(selected) or np.any((selected<0)|(selected>=n)):
            raise ValueError('Shared components must be distinct valid indices')
        self.shared_components=tuple(sorted(selected.tolist()))
        # Preserve the original all-shared layout. Other modes give each
        # unshared return its own band trajectory, then append shared paths.
        groups=[(c,s) for c in range(len(bands)) for s in range(n) if s not in self.shared_components]
        groups += [(None,s) for s in self.shared_components]
        lookup={key:shared+2*j for j,key in enumerate(groups)}
        self.phase_indices=np.array([[np.arange(lookup[(None if s in self.shared_components else c,s)],
            lookup[(None if s in self.shared_components else c,s)]+2) for s in range(n)] for c in range(len(bands))])
        self.count=shared+2*len(groups);self.energy=sum(b.energy for b in bands);self.last=None
        self.indices=[np.array([j for s in range(n) for j in (2*n*c+2*s,2*n*c+2*s+1,*self.phase_indices[c,s])]) for c in range(len(bands))]
        self.lower=np.full(self.count,-np.inf);self.upper=-self.lower
        for c in range(len(bands)):
            self.lower[2*n*c+np.arange(n)*2]=self.origins-origin_radius
            self.upper[2*n*c+np.arange(n)*2]=self.origins+origin_radius
        bound=np.tile([3*bands[0].scale,bands[0].scale**2],len(groups))
        self.lower[shared:]=-bound;self.upper[shared:]=bound

    def evaluate(self,parameters):
        if self.last is not None and np.array_equal(parameters,self.last[0]):return self.last[1:]
        residuals=[];jacobians=[]
        for cluster,index in zip(self.clusters,self.indices):
            value=cluster.evaluate(parameters[index]);scale=np.sqrt(cluster.energy/self.energy)
            residuals.append(value['residual']*scale)
            jac=np.zeros((len(value['residual']),self.count));jac[:,index]=value['jacobian']*scale;jacobians.append(jac)
        residual=np.concatenate(residuals);jacobian=np.concatenate(jacobians)
        self.last=(np.asarray(parameters).copy(),residual,jacobian)
        return residual,jacobian

    def fit(self,seed,max_evaluations=180,*,free_indices=None):
        initial=np.clip(seed,self.lower,self.upper)
        free=np.arange(self.count) if free_indices is None else np.asarray(free_indices,int)
        if free.ndim!=1 or not len(free) or len(np.unique(free))!=len(free) or np.any((free<0)|(free>=self.count)):
            raise ValueError('Distinct valid free coordinates required')
        def unpack(x):
            p=initial.copy();p[free]=x;return p
        fitted=least_squares(lambda x:self.evaluate(unpack(x))[0],initial[free],jac=lambda x:self.evaluate(unpack(x))[1][:,free],
            bounds=(self.lower[free],self.upper[free]),x_scale='jac',ftol=2e-12,xtol=2e-12,gtol=2e-10,max_nfev=max_evaluations)
        p=unpack(fitted.x);r,j=self.evaluate(p);gradient=2*j.T@r
        active=((p<self.lower+1e-5)&(gradient>0))|((p>self.upper-1e-5)&(gradient<0));gradient[active]=0
        condition=[];ratios=[]
        for cluster,index in zip(self.clusters,self.indices):
            value=cluster.evaluate(p[index]);matrix=cluster.base.nuisance.remove(value['design'][cluster.base.selected])[:,value['amplitude']>0]
            matrix=np.r_[matrix.real,matrix.imag];s=np.linalg.svd(matrix,compute_uv=False)
            condition.append(float(s[0]/s[-1]) if len(s) and s[-1]>0 else None)
            k=len(cluster.base.knots);parts=np.stack([value['design'][:,i*k:(i+1)*k]@value['amplitude'][i*k:(i+1)*k] for i in range(self.components)])
            parts=parts[:,cluster.base.selected];ratios.append(float(np.sum(abs(parts)**2)/max(np.sum(abs(parts.sum(axis=0))**2),1e-300)))
        return dict(parameters=p.tolist(),cost=float(r@r*self.energy),data_energy=self.energy,evaluations=fitted.nfev,
            termination=str(fitted.message),maximum_projected_gradient=float(abs(gradient[free]).max()),free_indices=free.tolist(),
            bound_indices=free[(p[free]<self.lower[free]+1e-5)|(p[free]>self.upper[free]-1e-5)].tolist(),
            active_brightness_condition=condition,component_to_sum_energy_ratio=ratios)

    def prediction(self,parameters,testing):
        return [c.prediction(parameters[i],t) for c,i,t in zip(self.clusters,self.indices,testing)]
