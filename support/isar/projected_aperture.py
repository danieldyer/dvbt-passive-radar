"""Reversible whole-record stationary nuisance projection for native images.

The original archive remains untouched. The weighted mean is estimated in
the original carrier reference and transformed back into each saved bin.
This is one constant range vector over the recording, not a Doppler notch
per image. A projected matched-filter noise norm requires its Schur term;
these comparison images retain the original norm and report its conditional
upper bound from the aperture/recording information fraction.
"""
import numpy as np
from motion import C


class Moments:
    def __init__(self,source,reference_path,information,k,*,chunk=2048):
        self.source=source;self.reference=np.asarray(reference_path);self.k=k;self.shape=source.shape
        if source.shape[:2]!=(2,len(self.reference)) or len(information)!=len(self.reference):raise ValueError('Projection requires all original bin identities')
        information=np.asarray(information);total=information.sum()
        if total<=0:raise ValueError('No supported stationary nuisance observations')
        mean=np.zeros((2,source.shape[2]),complex)
        for first in range(0,len(information),chunk):
            section=slice(first,first+chunk);rotation=np.exp(-1j*k*self.reference[section])
            mean+=np.einsum('mtr,t->mr',np.asarray(source[:,section],complex),information[section]*rotation)
        self.mean=mean/total;self.total_information=float(total)

    def __getitem__(self,item):
        if not isinstance(item,tuple) or len(item)!=2 or item[0]!=slice(None):raise ValueError('Select both moments and explicit native rows')
        value=np.asarray(self.source[item],complex);phase=np.exp(1j*self.k*self.reference[item[1]])
        return value-self.mean[:,None,:]*phase[None,:,None]


def apply(apertures,rf):
    output=[]
    for c,a in enumerate(apertures):
        record=dict(a);record['moments']=Moments(a['moments'],a['carrier_reference_path_m'],a['information'],2*np.pi*rf[c]/C)
        record['stationary_projection']=dict(scope=__doc__,total_information=record['moments'].total_information)
        output.append(record)
    return output
