"""Physical ship hypotheses with explicitly observed stationary-range support.

Use all seventeen recorded range cells while retaining the original, narrower
delay grid for nuisance clutter. The observation operator and user geometry
priors are unchanged. The wider data must constrain every nuisance column.
"""
from pathlib import Path

import numpy as np

from cluster_phase import crop
from physical_amplitude_arc import PhysicalAmplitudeArc
from physical_pair import SharedPhysicalPair
from propagated_pair import primary_propagation
from stationary import make_design


def model_for_interval(case,index,*,family='direct'):
    description=case['descriptions'][index]
    midpoint=(description['first_s']+description['last_s'])/2
    geometry=case['geometries'][index];bands=[]
    for ch,(narrow,training,knots) in enumerate(zip(case['fields'][index],case['training'][index],case['knots'][index])):
        paths=make_design(narrow)[1]
        with np.load(Path(f'build/isar-20261015/full-fields/channel-{ch}.npz')) as data:
            field=crop(data,description['first_s'],description['last_s'],np.arange(-8,9))
        band=PhysicalAmplitudeArc(field,training,knots,geometry,ch,midpoint,stationary_paths=paths)
        if band.nuisance.rank!=len(paths):raise ValueError('Surrounding ranges do not resolve the declared nuisance columns')
        bands.append(band)
    model=SharedPhysicalPair(bands,origin_bounds=(-2.8,2.8))
    return model if family=='direct' else primary_propagation(model,case['water'],family)


def background_status(model):
    return [dict(channel=b.channel,range_offsets=b.field.offsets.tolist(),delay_paths_m=b.delays.tolist(),
        rank=b.nuisance.rank,columns=len(b.delays),
        condition=float(b.nuisance.singular_values[0]/b.nuisance.singular_values[-1])) for b in model.bands]
