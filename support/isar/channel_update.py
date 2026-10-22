"""Bounded-memory five-channel complex sliding-window BP updates."""
from pathlib import Path
import struct,subprocess
import numpy as np
from dense_patch_image import read_exact


class ChannelUpdates:
    def __init__(self,pixels,directory,*,active_channels=True,executable='build/release/support/isar/isar_backprojection'):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.pixels=len(pixels)
        path=self.directory/'pixels.bin'
        with path.open('xb') as output:output.write(struct.pack('<I',len(pixels)));np.asarray(pixels,dtype='<f4').tofile(output)
        self.log=(self.directory/'gpu.log').open('w')
        self.active_channels=active_channels;self.transferred_bytes=0
        mode='--update-active-channels' if active_channels else '--update-channels'
        self.worker=subprocess.Popen([str(executable),mode,str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log)

    def contributions(self,geometry,moments):
        geometry=np.asarray(geometry,dtype='<f4');moments=np.asarray(moments,dtype='<c8');rows=len(geometry)
        if not 0<rows<=4096 or geometry.shape!=(rows,10,4) or moments.shape[:2]!=(2,rows):raise ValueError('Inconsistent update packet')
        self.worker.stdin.write(struct.pack('<II',rows,moments.shape[2]));self.worker.stdin.write(geometry.tobytes());self.worker.stdin.write(moments.tobytes());self.worker.stdin.flush()
        count,=struct.unpack('<I',read_exact(self.worker.stdout,4))
        expected=set(np.unique(geometry[:,6,3]).astype(int)) if self.active_channels else set(range(5))
        if count!=len(expected):raise ValueError('Active channel count differs from the scheduled observations')
        seen=set()
        for index in range(count):
            c=struct.unpack('<I',read_exact(self.worker.stdout,4))[0] if self.active_channels else index
            if c not in expected or c in seen:raise ValueError('Unexpected or duplicate channel image')
            seen.add(c);information=struct.unpack('<d',read_exact(self.worker.stdout,8))[0]
            image=np.frombuffer(read_exact(self.worker.stdout,self.pixels*8),'<c8');self.transferred_bytes+=image.nbytes
            yield c,image,information

    def update(self,geometry,moments):
        images=np.zeros((5,self.pixels),complex);information=np.zeros(5)
        for c,value,w in self.contributions(geometry,moments):images[c]=value;information[c]=w
        return images,information

    def accumulate(self,geometry,moments,images,information):
        for c,value,w in self.contributions(geometry,moments):images[c]+=value;information[c]+=w

    def close(self):
        try:self.worker.stdin.close()
        except BrokenPipeError:pass
        code=self.worker.wait(timeout=60);self.worker.stdout.close();self.log.close()
        if code:
            detail=(self.directory/'gpu.log').read_text()[-2000:]
            raise RuntimeError(f'Channel update worker failed with {code}: {detail}')
