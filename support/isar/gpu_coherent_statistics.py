"""Vulkan implementation of the independently tested positive-amplitude GLRT."""
from pathlib import Path
import struct,subprocess
import numpy as np
from dense_patch_image import read_exact


class Statistics:
    def __init__(self,directory,max_batch_pixels=4_000_000):
        if not 0<max_batch_pixels<=4_000_000:raise ValueError('Invalid bounded statistic batch')
        self.max_batch_pixels=max_batch_pixels
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);self.log=(directory/'statistics-gpu.log').open('w')
        self.worker=subprocess.Popen(['build/release/support/isar/isar_coherent_statistics'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log)

    def evaluate(self,values):
        z=np.asarray(values,dtype='<c8',order='C')
        if z.ndim<2 or z.shape[0]!=5 or not np.isfinite(z).all():raise ValueError('Expected five finite whitened, phase-calibrated fields')
        shape=z.shape[1:];count=int(np.prod(shape))
        if not count:raise ValueError('Expected nonempty coherent fields')
        flat=z.reshape(5,count);result=np.empty((count,2),dtype='<f4')
        for first in range(0,count,self.max_batch_pixels):
            last=min(first+self.max_batch_pixels,count)
            result[first:last]=self._batch(flat[:,first:last])
        result=result.reshape(*shape,2)
        return result[...,0],result[...,1]

    def _batch(self,z):
        count=z.shape[1]
        self.worker.stdin.write(struct.pack('<I',count));self.worker.stdin.write(z.tobytes());self.worker.stdin.flush()
        actual,=struct.unpack('<I',read_exact(self.worker.stdout,4))
        if actual!=count:raise ValueError('Coherent-statistic response has a different shape')
        return np.frombuffer(read_exact(self.worker.stdout,count*8),'<f4').reshape(count,2)

    def close(self):
        try:self.worker.stdin.close()
        except BrokenPipeError:pass
        code=self.worker.wait(timeout=60);self.worker.stdout.close();self.log.close()
        if code:raise RuntimeError(f'Coherent-statistic worker failed with {code}')
