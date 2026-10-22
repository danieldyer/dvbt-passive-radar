"""One flexible measured trajectory, with independent neighbouring returns.

This changes the temporal path basis, not the observation operator. Range
migration, carrier phase and C0/C1 Doppler all use that one path and derivative.
Other returns retain their independently measured quadratic paths.
"""
import numpy as np

from amplitude_pair import AmplitudeCluster
from shared_echo_cluster import SharedEchoCluster,ReferencedAmplitudeArc


class MixedPathCluster(AmplitudeCluster):
    def __init__(self,base,selected,selected_arc,components):
        super().__init__(base,components)
        selected=(selected,) if np.isscalar(selected) else tuple(selected)
        self.arcs=[selected_arc if s in selected else base for s in range(components)]
        self.slices=[];start=0
        for arc in self.arcs:
            count=2+arc.base.curve.degree
            self.slices.append(slice(start,start+count));start+=count

    def response_blocks(self,parameters):
        return [arc.response(parameters[block]) for arc,block in zip(self.arcs,self.slices)]


class FlexibleEchoCluster(SharedEchoCluster):
    def __init__(self,bands,origins,selected,curve,*,shared=True,origin_radius=2.,shared_components=None):
        chosen=(int(selected),) if np.isscalar(selected) else tuple(selected)
        if (not bands or not chosen or len(set(chosen))!=len(chosen) or
                any(s not in range(len(origins)) for s in chosen)):
            raise ValueError('Bands and distinct valid selected components required')
        common=chosen if shared and shared_components is None else (() if shared_components is None else tuple(shared_components))
        if len(set(common))!=len(common) or not set(common)<=set(chosen):
            raise ValueError('Shared flexible paths must be selected components')
        self.bands=bands;self.origins=np.asarray(origins,float);self.components=len(origins)
        self.selected=selected;self.selected_components=chosen;self.shared_selected_components=common
        self.curve=curve;self.degree=curve.degree
        self.clusters=[];self.indices=[];self.local_path_indices=[]
        n=len(origins);next_index=2*n*len(bands)
        shared_paths={}
        for s in common:
            shared_paths[s]=np.arange(next_index,next_index+self.degree);next_index+=self.degree
        for c,b in enumerate(bands):
            selected_arc=ReferencedAmplitudeArc(b.field,b.training,b.knots,curve.epoch-curve.scale,curve.epoch+curve.scale,
                b.known_path,b.known_rate,stationary_design=b.design,curve=curve)
            self.clusters.append(MixedPathCluster(b,selected,selected_arc,n))
            local=[];paths=[]
            for s in range(n):
                count=self.degree if s in chosen else 2
                if s in common:path=shared_paths[s]
                else:path=np.arange(next_index,next_index+count);next_index+=count
                paths.append(path);local.extend([2*n*c+2*s,2*n*c+2*s+1,*path])
            self.indices.append(np.asarray(local));self.local_path_indices.append(paths)
        self.count=next_index;self.energy=sum(b.energy for b in bands);self.last=None
        self.lower=np.full(self.count,-np.inf);self.upper=-self.lower
        for c,b in enumerate(bands):
            for s,origin in enumerate(origins):
                self.lower[2*n*c+2*s]=origin-origin_radius;self.upper[2*n*c+2*s]=origin+origin_radius
                indices=self.local_path_indices[c][s]
                limit=np.full(len(indices),6*curve.scale) if s in chosen else np.array([3*b.scale,b.scale**2])
                self.lower[indices]=-limit;self.upper[indices]=limit
