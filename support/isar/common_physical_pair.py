"""Common physical primary phase, with explicit secondary echoes retained.

This is a testable restriction of SharedPhysicalPair, not a phase alignment
applied to measured data. The existing referenced carrier-phase coordinate
map supplies all geometric chain-rule terms. Independent secondary phases
and their original-row likelihood are unchanged.
"""
import numpy as np

from physical_amplitude_arc import PhysicalAmplitudeFit
from physical_pair import fit_parameters
from referenced_common_arc import ReferencedCommonArc


class CommonPhysicalPair:
    def __init__(self,independent,reference_band=3):
        if [b.channel for b in independent.bands]!=list(range(5)):
            raise ValueError('The common-phase control requires all five ordered bands')
        self.independent=independent;self.geometry=independent.geometry
        self.bands=independent.bands;self.energy=independent.energy
        self.phase_map=ReferencedCommonArc(PhysicalAmplitudeFit(self.bands),reference_band)
        self.primary_count=self.geometry.count+1
        self.count=independent.count-4
        n=self.geometry.count;m=independent.primary_count
        self.lower=np.r_[independent.lower[:n],-np.inf,independent.lower[m:]]
        self.upper=np.r_[independent.upper[:n],np.inf,independent.upper[m:]]
        self.last=None

    def mapping(self,parameters):
        n=self.primary_count;m=self.independent.primary_count
        primary,change=self.phase_map.mapping(parameters[:n])
        mapped=np.r_[primary,parameters[n:]]
        transform=np.zeros((self.independent.count,self.count))
        transform[:m,:n]=change;transform[m:,n:]=np.eye(self.count-n)
        return mapped,transform

    def evaluate(self,parameters):
        if self.last is not None and np.array_equal(parameters,self.last[0]):return self.last[1:]
        mapped,transform=self.mapping(parameters)
        residual,jacobian=self.independent.evaluate(mapped)
        jacobian=jacobian@transform
        self.last=(parameters.copy(),residual,jacobian)
        return residual,jacobian

    def objective(self,parameters):
        residual,jacobian=self.evaluate(parameters)
        return float(residual@residual),2*jacobian.T@residual

    def prediction(self,parameters,testing):
        return self.independent.prediction(self.mapping(parameters)[0],testing)

    def mean_derivative(self,parameters):
        mapped,transform=self.mapping(parameters)
        return self.independent.mean_derivative(mapped)@transform

    def phase_branches(self,independent_parameters,*,adjacent=True,free_coordinates=None):
        n=self.independent.primary_count
        records=self.phase_map.phase_branches(independent_parameters[:n],adjacent=adjacent,free_coordinates=free_coordinates)
        for record in records:record['parameters']=np.r_[record['parameters'],independent_parameters[n:]]
        return records

    def fit(self,seed,*,free_antennas=True,max_evaluations=300):
        first=0 if free_antennas else self.geometry.nantenna
        return fit_parameters(self,seed,np.arange(first,self.count),max_evaluations=max_evaluations)


def antenna_information(model,parameters,phase_columns,secondary_columns):
    """Data-only antenna information after every scene nuisance coordinate."""
    j=model.mean_derivative(parameters);ranks=[]
    for columns in (phase_columns,secondary_columns,np.arange(18,27)):
        a=j[:,columns];norms=np.linalg.norm(a,axis=0);keep=norms>0
        if not np.any(keep):ranks.append(0);continue
        a=a[:,keep]/norms[keep];u,s,_=np.linalg.svd(a,full_matrices=False)
        supported=s>np.finfo(float).eps*max(a.shape)*s[0];q=u[:,supported]
        j-=q@(q.T@j);ranks.append(int(sum(supported)))
    antenna=j[:,np.r_[np.arange(15),17]];singular=np.linalg.svd(antenna,compute_uv=False)
    return dict(matrix=(antenna.T@antenna).tolist(),eigenvalues=(singular**2).tolist(),nuisance_ranks=ranks,
        scope='Conditional thermal information after stationary/active brightness, specified primary phases, all secondary coordinates and nine Cartesian target coordinates. No priors or active-bound information; AIS time is redundant with free p/v/a.')
