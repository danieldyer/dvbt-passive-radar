"""Shared antenna calibration across disjoint, independently moving scenes.

Original-data rows occur once. Antenna priors occur once; each scene retains
its motion, scattering phase(s), independent neighbours and white-jerk prior.
Prior ownership is explicit: no assumed number of residual-tail rows is cut.
"""
import numpy as np

from joint_path_information import expected_jacobian
from physical_pair import fit_parameters


def independent_arc(arc):
    return getattr(arc,'independent',arc)


class JointPhysicalEchoes:
    def __init__(self,arcs):
        if not arcs:raise ValueError('At least one scene required')
        self.arcs=tuple(arcs);self.count=18;self.indices=[];self.data_rows=[];self.last=None
        reference=arcs[0].geometry;intervals={}
        for arc in arcs:
            base=independent_arc(arc);g=arc.geometry
            if g.count!=28 or g.nantenna!=18:raise ValueError('Require 18 shared antenna and 10 local motion coordinates')
            for name in ('receiver','rotation','local_transmitters'):
                if not np.array_equal(getattr(g,name),getattr(reference,name)):
                    raise ValueError('All scenes must use one coordinate frame and site assignment')
            if not np.array_equal(g.antennas.factor,reference.antennas.factor):raise ValueError('Antenna priors differ')
            for b in arc.bands:
                first,last=b.field.data['row_time'][[0,-1]]
                if any(first<=end+2e-8 and last>=start-2e-8 for start,end in intervals.get(b.channel,())):
                    raise ValueError('Shared original observations cannot be counted as independent scenes')
                intervals.setdefault(b.channel,[]).append((first,last))
            self.indices.append(np.r_[np.arange(18),self.count+np.arange(arc.count-18)])
            self.count+=arc.count-18
            self.data_rows.append(sum(2*len(c.base.clean) for c in base.clusters))
        self.energy=sum(a.energy for a in arcs)
        self.lower=np.full(self.count,-np.inf);self.upper=-self.lower
        rows=[np.eye(18,self.count)]
        for arc,index in zip(arcs,self.indices):
            self.lower[index]=np.maximum(self.lower[index],arc.lower)
            self.upper[index]=np.minimum(self.upper[index],arc.upper)
            state=np.zeros((10,self.count));state[:,index[18:28]]=np.eye(10);rows.append(state)
            base=independent_arc(arc)
            if np.any(base.nuisance_prior[:,:base.primary_count]):
                raise ValueError('Neighbour prior must not contain physical motion or phase columns')
            local=np.zeros((len(base.nuisance_prior),arc.count))
            local[:,arc.primary_count:]=base.nuisance_prior[:,base.primary_count:]
            jerk=np.zeros((len(local),self.count));jerk[:,index]=local;rows.append(jerk)
        self.prior_factor=np.concatenate(rows)

    def evaluate(self,parameters):
        if self.last is not None and np.array_equal(parameters,self.last[0]):return self.last[1:]
        residuals=[];jacobians=[]
        for arc,index,rows in zip(self.arcs,self.indices,self.data_rows):
            r,j=arc.evaluate(np.asarray(parameters)[index]);scale=np.sqrt(arc.energy/self.energy)
            residuals.append(r[:rows]*scale)
            derivative=np.zeros((rows,self.count));derivative[:,index]=j[:rows]*scale;jacobians.append(derivative)
        scale=np.sqrt(2*self.energy)
        residuals.append(self.prior_factor@parameters/scale);jacobians.append(self.prior_factor/scale)
        result=np.concatenate(residuals),np.concatenate(jacobians)
        self.last=(np.asarray(parameters).copy(),*result)
        return result

    def objective(self,parameters):
        r,j=self.evaluate(parameters)
        return float(r@r),2*j.T@r

    def prior_cost(self,parameters):
        p=self.prior_factor@parameters
        return float(p@p/2)

    def fit(self,seed,free,*,max_evaluations=180):
        return fit_parameters(self,seed,free,max_evaluations=max_evaluations)

    def prediction(self,parameters,testing):
        return [arc.prediction(np.asarray(parameters)[index],test)
            for arc,index,test in zip(self.arcs,self.indices,testing)]

    def expected_jacobian(self,parameters):
        """Data-only means and original norms, with no prior/bound information."""
        rows=[];norm2=np.zeros(self.count)
        for arc,index in zip(self.arcs,self.indices):
            local=np.asarray(parameters)[index];base=independent_arc(arc)
            if arc is base:mapped,transform=local,None
            else:mapped,transform=arc.mapping(local)
            mean,norm=expected_jacobian(base,mapped,transform)
            j=np.zeros((len(mean),self.count));j[:,index]=mean;rows.append(j);norm2[index]+=norm**2
        return np.concatenate(rows),np.sqrt(norm2)
