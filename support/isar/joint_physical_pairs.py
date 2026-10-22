"""One antenna solution shared by several explicit two-echo intervals.

Each interval retains its own physical primary state and scattering phase(s),
and secondary-echo nuisance. Shared antenna priors occur once. All operations
reuse the exact original-row local likelihood and its derivative.
"""
import itertools

import numpy as np
from scipy.optimize import least_squares

from physical_pair import fit_parameters


class JointPhysicalPairs:
    def __init__(self,arcs):
        if not arcs or any(a.geometry.count!=28 or a.geometry.nantenna!=18 for a in arcs):
            raise ValueError('Each interval must have 18 shared antenna and 10 local state coordinates')
        reference=arcs[0].geometry
        for arc in arcs[1:]:
            for name in ('receiver','rotation','local_transmitters'):
                if not np.array_equal(getattr(arc.geometry,name),getattr(reference,name)):
                    raise ValueError('Shared antennas require identical coordinate frames and site assignments')
            if not np.array_equal(arc.geometry.antennas.factor,reference.antennas.factor):
                raise ValueError('Antenna priors differ between intervals')
        self.arcs=arcs;self.count=18;self.indices=[];self.last=None
        for arc in arcs:
            self.indices.append(np.r_[np.arange(18),self.count+np.arange(arc.count-18)])
            self.count+=arc.count-18
        self.energy=sum(a.energy for a in arcs)
        self.prior=np.zeros(self.count,bool);self.prior[:18]=True
        self.lower=np.full(self.count,-np.inf);self.upper=-self.lower
        for arc,index in zip(arcs,self.indices):
            self.prior[index[18:28]]=True
            self.lower[index]=np.maximum(self.lower[index],arc.lower)
            self.upper[index]=np.minimum(self.upper[index],arc.upper)

    def evaluate(self,parameters):
        if self.last is not None and np.array_equal(parameters,self.last[0]):return self.last[1:]
        residuals=[];jacobians=[]
        for arc,index in zip(self.arcs,self.indices):
            residual,jacobian=arc.evaluate(parameters[index]);scale=np.sqrt(arc.energy/self.energy)
            residuals.append(residual[:-28]*scale)
            derivative=np.zeros((len(residual)-28,self.count));derivative[:,index]=jacobian[:-28]*scale
            jacobians.append(derivative)
        residuals.append(parameters[self.prior]/np.sqrt(self.energy))
        jacobians.append(np.eye(self.count)[self.prior]/np.sqrt(self.energy))
        residual=np.concatenate(residuals);jacobian=np.concatenate(jacobians)
        self.last=(parameters.copy(),residual,jacobian)
        return residual,jacobian

    def objective(self,parameters):
        residual,jacobian=self.evaluate(parameters)
        return float(residual@residual),2*jacobian.T@residual

    def fit(self,seed,*,free_antennas=True,max_evaluations=300):
        return fit_parameters(self,seed,np.arange(0 if free_antennas else 18,self.count),max_evaluations=max_evaluations)

    def prediction(self,parameters,testing):
        return [arc.prediction(parameters[index],test) for arc,index,test in zip(self.arcs,self.indices,testing)]

    def mean_derivative(self,parameters):
        rows=[]
        for arc,index in zip(self.arcs,self.indices):
            local=arc.mean_derivative(parameters[index]);j=np.zeros((len(local),self.count));j[:,index]=local;rows.append(j)
        return np.concatenate(rows)

    def information(self,parameters):
        j=self.mean_derivative(parameters);ranks=[]
        phases=np.concatenate([index[28:arc.primary_count] for arc,index in zip(self.arcs,self.indices)])
        secondary=np.concatenate([index[arc.primary_count:] for arc,index in zip(self.arcs,self.indices)])
        motion=np.concatenate([index[18:27] for index in self.indices])
        for columns in (phases,secondary,motion):
            a=j[:,columns];norms=np.linalg.norm(a,axis=0);good=norms>0
            if not np.any(good):ranks.append(0);continue
            a=a[:,good]/norms[good];u,s,_=np.linalg.svd(a,full_matrices=False)
            keep=s>np.finfo(float).eps*max(a.shape)*s[0];q=u[:,keep]
            j-=q@(q.T@j);ranks.append(int(sum(keep)))
        antenna=j[:,np.r_[np.arange(15),17]];singular=np.linalg.svd(antenna,compute_uv=False)
        return dict(matrix=(antenna.T@antenna).tolist(),eigenvalues=(singular**2).tolist(),nuisance_ranks=ranks,
            scope='Conditional data-only antenna information after all primary phases, secondaries and nine Cartesian state coordinates per interval. No geographical priors or active-bound information.')

    def phase_initializers(self,independent_parameters):
        """Joint cycle lifting with the same shared antenna coordinates.

        These are initializers only. Their 0.1 rad scaling does not weight the
        measured-data fit. Position and common physical phase may adjust in
        every interval, while velocity/acceleration and secondaries stay fixed.
        """
        if any(a.primary_count!=29 for a in self.arcs):raise ValueError('Cycle lifting needs common-phase arcs')
        shared=np.mean([p[:18] for p in independent_parameters],axis=0)
        base=np.zeros(self.count);base[:18]=np.clip(shared,self.lower[:18],self.upper[:18])
        references=[];pattern_sets=[]
        for arc,index,independent in zip(self.arcs,self.indices,independent_parameters):
            local=np.r_[base[:18],independent[18:28],0.,independent[33:]]
            local=np.clip(local,arc.lower,arc.upper);base[index]=local
            length=arc.geometry.prediction(local[:28])[:,0]
            phase=np.angle(np.exp(1j*(independent[28:33]+arc.phase_map.k*(length-arc.phase_map.reference))))
            edges=np.unique(np.r_[-np.pi,np.sort(np.angle(np.exp(1j*(phase+np.pi)))),np.pi])
            origins=(edges[1:]+edges[:-1])/2
            pattern_sets.append([(origin,np.rint((origin-phase)/(2*np.pi)).astype(int)) for origin in origins])
            references.append((length,phase))
        coordinates=np.r_[np.arange(15),17,np.concatenate([index[18:21] for index in self.indices]),
            np.array([index[28] for index in self.indices])]
        prior_coordinates=coordinates[:-len(self.arcs)]
        records=[]
        for choices in itertools.product(*pattern_sets):
            seed=base.copy()
            for arc,index,(origin,cycles) in zip(self.arcs,self.indices,choices):
                physical=np.r_[seed[index][:28],origin]
                seed[index[28]]=arc.phase_map.from_physical(physical)[-1]
            def expand(x):
                p=seed.astype(np.result_type(seed,x));p[coordinates]=x;return p
            def residual(x):
                p=expand(x);terms=[]
                for arc,index,(initial,phase),(_,cycles) in zip(self.arcs,self.indices,references,choices):
                    local=p[index];length=arc.geometry.prediction(local[:28])[:,0]
                    phi=arc.phase_map.physical_parameters(local[:29])[-1]
                    terms.append((phi-arc.phase_map.k*(length-initial)-phase-2*np.pi*cycles)/.1)
                return np.r_[np.concatenate(terms),p[prior_coordinates]-base[prior_coordinates]]
            fitted=least_squares(residual,seed[coordinates],jac='cs',bounds=(self.lower[coordinates],self.upper[coordinates]),
                x_scale='jac',max_nfev=150,ftol=1e-12,xtol=1e-12,gtol=1e-10)
            records.append(dict(parameters=expand(fitted.x),carrier_cycles=[c.tolist() for _,c in choices],
                phase_lift_cost=float(fitted.fun@fitted.fun),phase_lift_optimality=float(fitted.optimality)))
        return records
