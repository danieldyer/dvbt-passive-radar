"""A physical primary point with separately moving nuisance returns.

All components share the original observation operator and positive brightness
profiling. Only the selected primary is coupled through physical propagation;
the nuisance returns are not assumed to be points on the same object.
"""
import numpy as np

from amplitude_pair import AmplitudeCluster
from amplitude_arc import AmplitudeArc
from physical_pair import fit_parameters
from motion import C


class EnvelopeOffsetArc(AmplitudeArc):
    """Sensitivity control: fixed band envelope offsets, with carrier unchanged.

    This tests the point-envelope assumption. It does not identify an extra
    propagation delay or provide a correction to the recorded observations.
    """
    def __init__(self,physical):
        super().__init__(physical.field,physical.training,physical.knots,stationary_paths=[])
        self.physical=physical;self.geometry=physical.geometry
        self.channel,self.midpoint=physical.channel,physical.midpoint
        self.primary_count=self.geometry.count+10
        self.envelope_indices=np.arange(self.geometry.count+5,self.primary_count)

    def response(self,parameters):
        core=self.geometry.count+5
        raw,change=self.physical.response(parameters[:core],
            range_bias_m=parameters[core+self.channel]*C/8e6,envelope_derivative=True)
        extra=np.zeros(raw.shape+(5,),complex)
        extra[:,:,self.channel]=change[:,:,-1]*C/8e6
        return raw,np.concatenate([change[:,:,:core],extra],axis=2)


class PhysicalNuisanceCluster(AmplitudeCluster):
    def __init__(self,physical,empirical,nuisance_components,*,nuisance_arcs=None):
        arcs=(empirical,)*nuisance_components if nuisance_arcs is None else tuple(nuisance_arcs)
        if len(arcs)!=nuisance_components:raise ValueError('One arc per nuisance component required')
        for arc in (empirical,*arcs):
            if (physical.field is not arc.field or
                    not np.array_equal(physical.training,arc.training) or
                    not np.array_equal(physical.knots,arc.knots) or
                    not np.array_equal(physical.design,arc.design) or
                    type(physical.nuisance) is not type(arc.nuisance)):
                raise ValueError('Require identical observations, splits and declared background operator')
        super().__init__(physical,1+nuisance_components)
        self.empirical=empirical
        self.primary_count=getattr(physical,'primary_count',physical.geometry.count+5)
        self.nuisance_arcs=arcs;self.nuisance_slices=[];next_index=self.primary_count
        for arc in arcs:
            count=2+arc.base.curve.degree
            self.nuisance_slices.append(slice(next_index,next_index+count));next_index+=count
        self.count=next_index

    def response_blocks(self,parameters):
        n=self.primary_count
        return [self.base.response(parameters[:n]),
            *[arc.response(np.asarray(parameters)[block]) for arc,block in zip(self.nuisance_arcs,self.nuisance_slices)]]


class PhysicalEchoCluster:
    def __init__(self,bands,empirical,nuisance_origins,*,origin_radius=2.,nuisance_arcs=None,nuisance_jerk_std=None):
        if (not bands or len(bands)!=len(empirical) or origin_radius<=0 or
                len({b.channel for b in bands})!=len(bands)):
            raise ValueError('Require distinct corresponding bands and a positive range radius')
        self.bands=bands;self.geometry=bands[0].geometry
        if any(b.geometry is not self.geometry for b in bands):raise ValueError('One geometry is required')
        self.origins=np.asarray(nuisance_origins,float);n=len(self.origins)
        self.primary_count=getattr(bands[0],'primary_count',self.geometry.count+5)
        if nuisance_arcs is not None and len(nuisance_arcs)!=len(bands):raise ValueError('Corresponding nuisance arcs required')
        self.clusters=[PhysicalNuisanceCluster(b,e,n,nuisance_arcs=None if nuisance_arcs is None else nuisance_arcs[c])
            for c,(b,e) in enumerate(zip(bands,empirical))]
        self.indices=[];self.count=self.primary_count
        for cluster in self.clusters:
            count=cluster.count-self.primary_count
            self.indices.append(np.r_[np.arange(self.primary_count),self.count+np.arange(count)]);self.count+=count
        self.energy=sum(b.energy for b in bands);self.last=None
        lo,hi=self.geometry.bounds()
        extra=self.primary_count-self.geometry.count
        self.lower=np.r_[lo,np.full(extra,-np.inf),np.empty(self.count-self.primary_count)]
        self.upper=np.r_[hi,np.full(extra,np.inf),np.empty(self.count-self.primary_count)]
        if hasattr(bands[0],'envelope_indices'):
            self.lower[bands[0].envelope_indices]=-origin_radius
            self.upper[bands[0].envelope_indices]=origin_radius
        factors=[]
        if nuisance_jerk_std is not None:
            if not np.isfinite(nuisance_jerk_std) or nuisance_jerk_std<=0:raise ValueError('Positive nuisance jerk scale required')
            from smooth_echo_cluster import jerk_factor
        for c,(cluster,index) in enumerate(zip(self.clusters,self.indices)):
            for origin,arc,block in zip(self.origins,cluster.nuisance_arcs,cluster.nuisance_slices):
                indices=index[block];curve=arc.base.curve
                self.lower[indices[:2]]=[origin-origin_radius,-np.inf]
                self.upper[indices[:2]]=[origin+origin_radius,np.inf]
                limit=(np.array([3*arc.scale,arc.scale**2]) if nuisance_arcs is None else
                    np.array([6*arc.scale,3*arc.scale**2]) if curve.degree==2 else np.full(curve.degree,6*curve.scale))
                self.lower[indices[2:]],self.upper[indices[2:]]=-limit,limit
                if nuisance_jerk_std is not None and curve.degree>2:
                    factor=jerk_factor(curve)/nuisance_jerk_std
                    prior=np.zeros((len(factor),self.count));prior[:,indices[2:]]=factor;factors.append(prior)
        self.nuisance_prior=np.concatenate(factors) if factors else np.zeros((0,self.count))

    def evaluate(self,parameters):
        if self.last is not None and np.array_equal(parameters,self.last[0]):return self.last[1:]
        residual=[];jacobian=[]
        for cluster,index in zip(self.clusters,self.indices):
            value=cluster.evaluate(parameters[index]);scale=np.sqrt(cluster.energy/self.energy)
            residual.append(value['residual']*scale)
            j=np.zeros((len(value['residual']),self.count));j[:,index]=value['jacobian']*scale
            jacobian.append(j)
        n=self.geometry.count
        # Proper-complex unit noise has NLL ||y-mu||². A standard real
        # Gaussian prior contributes ||q||²/2, exactly once across bands.
        residual.append(parameters[:n]/np.sqrt(2*self.energy))
        jacobian.append(np.eye(n,self.count)/np.sqrt(2*self.energy))
        if len(self.nuisance_prior):
            residual.append(self.nuisance_prior@parameters/np.sqrt(2*self.energy))
            jacobian.append(self.nuisance_prior/np.sqrt(2*self.energy))
        result=np.concatenate(residual),np.concatenate(jacobian)
        self.last=(np.asarray(parameters).copy(),*result)
        return result

    def objective(self,parameters):
        r,j=self.evaluate(parameters)
        return float(r@r),2*j.T@r

    def fit(self,seed,*,free_antennas=False,max_evaluations=240):
        first=0 if free_antennas else self.geometry.nantenna
        free=np.arange(first,self.count)
        present={b.channel for b in self.bands}
        free=np.array([i for i in free if not (self.geometry.count<=i<self.primary_count and
            (i-self.geometry.count)%5 not in present)])
        return fit_parameters(self,seed,free,max_evaluations=max_evaluations)

    def prediction(self,parameters,testing):
        return [cluster.prediction(parameters[index],test)
            for cluster,index,test in zip(self.clusters,self.indices,testing)]

    def prior_cost(self,parameters):
        p=np.asarray(parameters);jerk=self.nuisance_prior@p
        return float((p[:self.geometry.count]@p[:self.geometry.count]+jerk@jerk)/2)

    def mean_derivative(self,parameters):
        rows=[]
        for cluster,index in zip(self.clusters,self.indices):
            local=cluster.evaluate(parameters[index])['mean_derivative']*np.sqrt(2.)
            j=np.zeros((len(local),self.count));j[:,index]=local;rows.append(j)
        return np.concatenate(rows)

    def information(self,parameters):
        j=self.mean_derivative(parameters)
        n=self.geometry.count
        nuisance=j[:,n:];norm=np.linalg.norm(nuisance,axis=0);nonzero=norm>0
        u,s,_=np.linalg.svd(nuisance[:,nonzero]/norm[nonzero],full_matrices=False)
        keep=s>np.finfo(float).eps*max(nuisance.shape)*s[0] if len(s) else np.zeros(0,bool)
        q=u[:,keep];primary=j[:,:n]-q@(q.T@j[:,:n])
        return dict(matrix=(primary.T@primary).tolist(),nuisance_rank=int(sum(keep)),
            scope='Conditional proper-complex mean information after real brightness, constant band phases and all empirical nuisance envelope/phase/trajectory parameters. No priors, active-bound information or model-discrepancy calibration.')
