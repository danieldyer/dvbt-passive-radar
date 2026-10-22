"""Reader for the support-only static profile/PSF replay format."""
import json
import struct

import numpy as np


def read_profiles(path, allow_in_progress=False):
    with path.open('rb') as f:
        if f.read(8)!=b'GEOProf1':
            raise ValueError('Not a geographic profile export')
        size=struct.unpack('<I',f.read(4))[0]
        if not 0<size<1024**2:
            raise ValueError('Invalid geographic header length')
        header=json.loads(f.read(size));start=f.tell()
    fields=[('channel','<u4'),('reserved','<u4'),('bin','<u8'),('segment','<u8'),
            ('first','<f8'),('last','<f8'),('information','<f8'),('time','<f8'),('values','<c8',1024)]
    if header.get('psf_samples',0):
        samples=header['psf_samples']
        if not isinstance(samples,int) or samples<129 or samples>8191 or samples%2!=1:
            raise ValueError('Unsupported autocorrelation support')
        if header.get('psf_lag_radius',(samples-1)//2)!=(samples-1)//2:
            raise ValueError('Autocorrelation count and lag radius disagree')
        fields.append(('psf','<c8',samples))
    dtype=np.dtype(fields);count,tail=divmod(path.stat().st_size-start,dtype.itemsize)
    if tail and not allow_in_progress:
        raise ValueError('Incomplete geographic export record')
    return header,np.memmap(path,mode='r',offset=start,dtype=dtype,shape=count)
