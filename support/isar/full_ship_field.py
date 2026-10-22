"""Memory-mapped known-waveform fields for one continuous capture reference."""
import json
from pathlib import Path
import numpy as np
from block_pulse import BlockPulse
from field import Field
from waveform_kernels import KernelField


def select_range(field,first,last):
    """Fixed linear range selection with its exact conditional covariance.

    Carrier reference, original epochs, known kernels and complex raw values
    are unchanged. This is for reversible local acquisition; native image
    formation and the final full-field likelihood remain separate.
    """
    keep=(field.offsets>=first)&(field.offsets<=last)
    if np.count_nonzero(keep)<2:raise ValueError('A range acquisition gate needs at least two retained cells')
    data=dict(field.data);data['value']=data['value'][:,keep];data['offset_samples']=field.offsets[keep]
    base=Field(data,field.pulse.samples)
    return KernelField(base,field.pulse.kernels,field.pulse.row_blocks) if isinstance(field,KernelField) else base


def load(directory):
    directory=Path(directory);description=json.loads((directory/'analysis.json').read_text())
    if not description['complete']:raise ValueError('Full field export is incomplete')
    if 'composite_range_reference' in description:
        from paired_range_reference import combine
        selection=description['composite_range_reference'];outer,coordinate,_=load(selection['outer']);middle,other,_=load(selection['middle'])
        result=[];epochs=[]
        for a,b,x,y in zip(outer,middle,coordinate,other):
            for key in ['native_time_s','native_information','row_blocks','training']:np.testing.assert_array_equal(x[key],y[key])
            field,epoch=combine(a,b,x,selection['middle_interval_s']);result.append(field);epochs.append(epoch)
        return result,epochs,description
    fields=[];epochs=[]
    for record in description['channels']:
        root=directory/f'channel-{record["channel"]}'
        with np.load(root/'field.npz') as source:data=dict(source)
        with np.load(root/'epochs.npz') as source:coordinate=dict(source)
        kernel=BlockPulse.__new__(BlockPulse);kernel.degree=record['kernel_degree'];kernel.radius=record['kernel_radius']
        kernel.samples=np.load(root/'kernel-samples.npy',mmap_mode='r')
        kernel.coefficients=np.load(root/'kernel-coefficients.npy',mmap_mode='r')
        kernel.derivative_coefficients=np.load(root/'kernel-derivatives.npy',mmap_mode='r')
        base=Field(data,np.load(root/'mean-pulse.npy'));field=KernelField(base,kernel,coordinate['row_blocks'])
        if kernel.coefficients.shape!=(kernel.degree+1,len(kernel.samples),2*kernel.radius):raise ValueError('Known-waveform polynomial layout changed')
        if len(coordinate['native_time_s'])!=16*len(field.time):raise ValueError('Native bin identities are incomplete')
        np.testing.assert_allclose(coordinate['native_information'].reshape(-1,16).sum(axis=1),field.data['information'],rtol=1e-14,atol=1e-30)
        fields.append(field);epochs.append(coordinate)
    return fields,epochs,description
