"""Physical scatterers with common NEU velocity/acceleration over a short arc.

Point positions, per-band constant phases and positive brightness are separate.
No yaw, moving phase screen or independent range origin is fitted. Every point
uses exact two-leg geometry for delay, phase and symbol Doppler. A linear map
ties the original disjoint response parameters before profiling brightness.
"""
from types import SimpleNamespace

import numpy as np

from amplitude_pair import AmplitudeCluster
from component_phase import conditional_components,restore_physical_phase
from flash_geometry import observables
from joint_path_information import expected_jacobian
from physical_amplitude_arc import PhysicalAmplitudeArc
from physical_pair import fit_parameters
from smooth_echo_cluster import jerk_factor


class OffsetPointGeometry:
    """A static relative horizontal position and separate absolute height.

    Relative horizontal coordinates are radius / scale and angle from east.
    This is a point-coordinate parameterization, not a fitted vessel heading.
    Radius bounds give explicit compact spatial support without a penalty wall.
    """
    def __init__(self,base,relative_scale_m):
        self.base=base;self.count=base.count+3;self.relative_scale_m=float(relative_scale_m)

    def __getattr__(self,name):return getattr(self.base,name)

    def state(self,parameters):
        p,v,a,tx,rx=self.base.state(np.asarray(parameters)[:self.base.count])
        radius,angle,height=np.asarray(parameters)[self.base.count:]
        up=self.base.nuisance_sigma[2]*(height-parameters[20])
        offset=np.array([self.relative_scale_m*radius*np.cos(angle),
            self.relative_scale_m*radius*np.sin(angle),up])
        return p+offset,v,a,tx,rx

    def prediction(self,parameters):return observables(*self.state(parameters))


class DisjointArcCluster(AmplitudeCluster):
    def __init__(self,arcs):
        if not arcs:raise ValueError('At least one response required')
        base=arcs[0]
        for arc in arcs:
            if (arc.field is not base.field or not np.array_equal(arc.training,base.training) or
                    not np.array_equal(arc.knots,base.knots) or not np.array_equal(arc.design,base.design)):
                raise ValueError('Responses must share original observations, masks, brightness knots and background')
        super().__init__(base,len(arcs));self.arcs=tuple(arcs);self.slices=[];self.count=0
        for arc in arcs:
            count=arc.geometry.count+5 if isinstance(arc,PhysicalAmplitudeArc) else 2+arc.base.curve.degree
            self.slices.append(slice(self.count,self.count+count));self.count+=count

    def response_blocks(self,parameters):
        return [a.response(np.asarray(parameters)[s]) for a,s in zip(self.arcs,self.slices)]


class SharedTranslationScene:
    def __init__(self,bands,nuisance_arcs,nuisance_origins,*,points=2,origin_radius=2.5,
            jerk_std=None,relative_scale_m=150.,relative_radius_m=300.):
        if (not bands or not isinstance(points,int) or points<1 or origin_radius<=0 or
                relative_scale_m<=0 or relative_radius_m<=0 or len(nuisance_arcs)!=len(bands)):
            raise ValueError('Corresponding bands/neighbours, positive supports and point count required')
        self.bands=tuple(bands);self.geometry=bands[0].geometry;g=self.geometry
        if g.count!=28 or g.nantenna!=18 or any(b.geometry is not g for b in bands):
            raise ValueError('One 18-antenna/10-state geometry required')
        if len({b.channel for b in bands})!=len(bands):raise ValueError('Distinct channels required')
        if jerk_std is not None and (not np.isfinite(jerk_std) or jerk_std<=0):raise ValueError('Positive jerk scale required')
        self.points=points;self.origins=np.asarray(nuisance_origins,float)
        if any(len(a)!=len(self.origins) for a in nuisance_arcs):raise ValueError('One origin per neighbour required')
        self.offset_indices=[np.empty(0,int),*[28+3*s+np.arange(3) for s in range(points-1)]]
        self.phase_start=28+3*(points-1);self.phase_indices=self.phase_start+np.arange(5*points).reshape(points,5)
        self.primary_count=self.phase_start+5*points;self.count=self.primary_count
        self.point_geometries=[g,*[OffsetPointGeometry(g,relative_scale_m) for _ in range(points-1)]]
        self.point_arcs=[];self.clusters=[];self.maps=[];self.nuisance_indices=[]
        local_indices=[];expanded_count=0
        for b,neighbours in zip(bands,nuisance_arcs):
            physical=[b,*[PhysicalAmplitudeArc(b.field,b.training,b.knots,pg,b.channel,b.midpoint,
                stationary_design=b.design) for pg in self.point_geometries[1:]]]
            self.point_arcs.append(physical);cluster=DisjointArcCluster([*physical,*neighbours]);self.clusters.append(cluster)
            indices=[]
            for s in range(points):indices.extend([*np.arange(28),*self.offset_indices[s],*self.phase_indices[s]])
            private=[]
            for arc in neighbours:
                n=2+arc.base.curve.degree;index=self.count+np.arange(n);self.count+=n;private.append(index);indices.extend(index)
            local_indices.append(np.asarray(indices));self.nuisance_indices.append(private)
            self.maps.append(None);expanded_count+=cluster.count
        self.expanded_indices=[];start=0
        for c,(cluster,index) in enumerate(zip(self.clusters,local_indices)):
            self.maps[c]=np.eye(self.count)[index]
            self.expanded_indices.append(start+np.arange(cluster.count));start+=cluster.count
        self.expanded=SimpleNamespace(count=expanded_count,clusters=self.clusters,indices=self.expanded_indices)
        self.mapping_matrix=np.concatenate(self.maps)
        self.energy=sum(b.energy for b in bands);self.last=None
        self.lower=np.full(self.count,-np.inf);self.upper=-self.lower
        self.lower[:28],self.upper[:28]=g.bounds()
        prior=[np.eye(28,self.count)]
        for index in self.offset_indices[1:]:
            self.lower[index]=[0.,-np.inf,g.lower[20]]
            self.upper[index]=[relative_radius_m/relative_scale_m,np.inf,g.upper[20]]
            rows=np.zeros((2,self.count));rows[0,index[0]]=1.;rows[1,index[2]]=1.;rows[1,20]=-1.;prior.append(rows)
        for arcs,private in zip(nuisance_arcs,self.nuisance_indices):
            for arc,index,origin in zip(arcs,private,self.origins):
                self.lower[index[:2]]=[origin-origin_radius,-np.inf];self.upper[index[:2]]=[origin+origin_radius,np.inf]
                curve=arc.base.curve
                limit=np.array([6*curve.scale,3*curve.scale**2]) if curve.degree==2 else np.full(curve.degree,6*curve.scale)
                self.lower[index[2:]],self.upper[index[2:]]=-limit,limit
                if jerk_std is not None and curve.degree>2:
                    factor=jerk_factor(curve)/jerk_std;rows=np.zeros((len(factor),self.count));rows[:,index[2:]]=factor;prior.append(rows)
        self.prior_factor=np.concatenate(prior)
        self.relative_radius_m=relative_radius_m;self.relative_scale_m=relative_scale_m

    def evaluate(self,parameters):
        if self.last is not None and np.array_equal(parameters,self.last[0]):return self.last[1:]
        residual=[];jacobian=[]
        for cluster,transform in zip(self.clusters,self.maps):
            value=cluster.evaluate(transform@parameters);scale=np.sqrt(cluster.energy/self.energy)
            residual.append(value['residual']*scale);jacobian.append((value['jacobian']@transform)*scale)
        scale=np.sqrt(2*self.energy)
        residual.append(self.prior_factor@parameters/scale);jacobian.append(self.prior_factor/scale)
        result=np.concatenate(residual),np.concatenate(jacobian)
        self.last=(np.asarray(parameters).copy(),*result)
        return result

    def objective(self,parameters):
        r,j=self.evaluate(parameters)
        return float(r@r),2*j.T@r

    def prior_cost(self,parameters):
        value=self.prior_factor@parameters
        return float(value@value/2)

    def fit(self,seed,free,*,max_evaluations=180):
        return fit_parameters(self,seed,free,max_evaluations=max_evaluations)

    def prediction(self,parameters,testing):
        return [c.prediction(t@parameters,test) for c,t,test in zip(self.clusters,self.maps,testing)]

    def expected_jacobian(self,parameters):
        return expected_jacobian(self.expanded,self.mapping_matrix@parameters,self.mapping_matrix)

    def measurements(self,parameters):
        result=[]
        for arcs,cluster,transform in zip(self.point_arcs,self.clusters,self.maps):
            local=transform@parameters;conditional=conditional_components(cluster,local)
            result.append([restore_physical_phase(conditional[s],arc,local[cluster.slices[s]]) for s,arc in enumerate(arcs)])
        return result

    def point_summary(self,parameters):
        output=[]
        for s,g in enumerate(self.point_geometries):
            core=np.r_[np.asarray(parameters)[:28],np.asarray(parameters)[self.offset_indices[s]]]
            output.append(dict(position_enu_m=g.state(core)[0],observables=g.prediction(core)))
        return output
