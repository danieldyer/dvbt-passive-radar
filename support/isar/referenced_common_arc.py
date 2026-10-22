"""Condition a common physical scattering phase at one band's midpoint.

This is an exact coordinate change, not a new per-band phase parameter.
Subtracting one band's geometric phase from the common scalar avoids forcing
the optimizer to follow its large curved position/phase correlation.
"""
import numpy as np

from common_physical_arc import CommonPhysicalFit
from physical_amplitude_arc import state_derivatives


class ReferencedCommonArc(CommonPhysicalFit):
    def __init__(self,independent,reference_band=3):
        super().__init__(independent)
        if not 0<=reference_band<5:raise ValueError('Choose one of the five bands')
        self.reference_band=reference_band

    def physical_parameters(self,parameters):
        result=np.asarray(parameters).copy();c=self.reference_band
        length=self.geometry.prediction(result[:-1])[c,0]
        result[-1]+=self.k[c]*(length-self.reference[c])
        return result

    def from_physical(self,parameters):
        result=np.asarray(parameters).copy();c=self.reference_band
        length=self.geometry.prediction(result[:-1])[c,0]
        result[-1]-=self.k[c]*(length-self.reference[c])
        return result

    def mapping(self,parameters):
        n=self.geometry.count;c=self.reference_band
        _,_,length,derivative=state_derivatives(self.geometry,parameters[:n])
        phases=parameters[-1]-self.k*(length-self.reference)+self.k[c]*(length[c]-self.reference[c])
        transform=np.zeros((n+5,n+1));transform[:n,:n]=np.eye(n)
        transform[n:,:n]=-self.k[:,None]*derivative+self.k[c]*derivative[c]
        transform[n:,-1]=1.
        return np.r_[parameters[:n],phases],transform

    def phase_branches(self,independent_parameters,*,adjacent=True,free_coordinates=None):
        result=super().phase_branches(independent_parameters,adjacent=adjacent,free_coordinates=free_coordinates)
        for record in result:record['parameters']=self.from_physical(record['parameters'])
        return result
