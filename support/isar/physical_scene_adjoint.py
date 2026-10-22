"""Vulkan residual-image proposals for a saved physical point-path model.

This is an acquisition operator at representative epochs. Final support
scoring uses PointBand's original-bin carrier attenuation and exact complex
likelihood. Candidate generation never rephases a band independently.
"""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from bistatic_path_lift import lift
from dense_patch_image import DenseImages,matched_polynomials
from imaging import geometry_from_state
from motion import C


def geometry(bands,stride):
    result=np.zeros((len(bands),stride,50,4),dtype='<f4')
    for c,b in enumerate(bands):
        s=lift(b.anchor,b.velocity,b.direction,b.direction_rate,b.base_path,b.base_rate,b.motion.tx[3],b.motion.rx)
        state=dict(position_ecef_m=s['position_ecef_m']-b.frame@b.reference_point,
            velocity_ecef_mps=s['velocity_ecef_mps']-b.frame_rate@b.reference_point,
            body_to_ecef=b.frame,frame_velocity=b.frame_rate)
        f=b.field;g,d=geometry_from_state(state,b.motion.tx[c],b.motion.rx,f.metadata['rf_hz'],0.,np.ones(len(b.t)))
        g[:,9,0]=(d['path_m']-f.data['centre_path_m'])*8e6/C
        phase=f.k*(d['path_m']-f.data.get('carrier_reference_path_m',f.data['centre_path_m']))
        g[:,9,1]=np.cos(phase);g[:,9,2]=np.sin(phase);g[:,5,3]=f.k
        result[c,:len(b.t),:10]=g
    return result


class PhysicalSceneAdjoint:
    def __init__(self,bands,pixels,directory):
        self.bands=bands
        views=[SimpleNamespace(field=b.field,training=b.training,groups=b.groups) for b in bands]
        self.worker=DenseImages(views,pixels,Path(directory),full_pulse_support=True)
        self.geometry=geometry(bands,self.worker.stride)
        self.last_values=None

    def evaluate(self,values):
        if len(values)!=len(self.bands):raise ValueError('Every band needs its original-coordinate residual')
        coefficients=np.zeros_like(self.worker.coefficients)
        for c,(b,y) in enumerate(zip(self.bands,values)):
            if np.shape(y)!=b.field.y.shape:raise ValueError('Residual changed the original field shape')
            f=copy.copy(b.field);f.y=np.asarray(y)
            coefficients[c,:len(b.t)]=matched_polynomials(f,b.training,self.worker.first_cell,self.worker.cells,b.groups)
        self.worker.replace_coefficients(coefficients)
        return self.worker.evaluate_geometry(self.geometry,derivatives=False)[:,0]

    def close(self):self.worker.close()
