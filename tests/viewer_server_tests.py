"""Native complex maps and preview crops must agree in power, axes and pooling."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('serve',Path(__file__).resolve().parents[1]/'web/serve.py')
serve=importlib.util.module_from_spec(spec);spec.loader.exec_module(serve)

class PreviewTest(unittest.TestCase):
    def test_dc_projection_matches_time_domain_and_crops(self):
        samples,size,dt=1013,1024,.004352
        t=np.arange(samples)
        window=(.5-.5*np.cos(2*np.pi*t/samples)).astype(np.float32)
        signal=np.array([8-4j+.03*np.exp(1j*(.37+2*np.pi*23*t/size)),
                         2+3j+.02*np.exp(1j*(-.19-2*np.pi*1.7*t/samples))])
        weights=window.sum(dtype=np.float64)
        z=np.fft.fftshift(np.fft.fft(signal*window,n=size,axis=1),axes=1)/weights
        mean=(signal*window).sum(axis=1)/weights
        expected=np.fft.fftshift(np.fft.fft((signal-mean[:,None])*window,n=size,axis=1),axes=1)/weights
        meta=dict(ranges=2,frequencies=size,range_step_m=37.5,frequency_step_hz=1/(size*dt),frequency_start_hz=-.5/dt,
                  samples=samples,fft_size=size,integration_s=samples*dt,window='hann',format='cf32le',dc_removed=False)
        # Test both current and legacy map metadata, and a crop excluding DC.
        for legacy in (False,True):
            m=dict(meta)
            if legacy:del m['fft_size']
            for first,last in ((0,size),(550,800)):
                zero,kernel=serve.dc_projection(m,first,last)
                actual=z[:,first:last]-z[:,zero,None]*kernel[None,:]
                np.testing.assert_allclose(actual,expected[:,first:last],atol=3e-15)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'latest.rdm'
            for native_dc in (False,True):
                m=dict(meta,dc_removed=native_dc)
                source=(expected if native_dc else z).astype('<c8')
                header=json.dumps(m).encode()
                path.write_bytes(struct.pack('<4sIQ',b'RDM1',len(header),source.nbytes)+header+source.tobytes())
                packet=serve.preview(path,{'remove_dc':['1'],'width':['16'],'height':['2048']})
                _,length,_=struct.unpack('<4sIQ',packet[:16]);outmeta=json.loads(packet[16:16+length])
                out=np.frombuffer(packet,dtype='<f4',offset=16+length).reshape(size,2)
                self.assertTrue(outmeta['dc_removed'])
                self.assertEqual(outmeta['native_dc_removed'],native_dc)
                # Compare linear amplitudes: dB error near an exact null is
                # unbounded, while native cf32 quantization is small and known.
                np.testing.assert_allclose(10**(out.T/20),np.abs(expected),atol=4e-7)
                self.assertLess(out[size//2].max(),-140)

    def test_native_crop_and_peak_pool(self):
        z=(np.arange(35*101).reshape(35,101)+1j*2).astype('<c8')
        z[17,42]=10000+300j
        meta=dict(ranges=35,frequencies=101,range_step_m=37.5,frequency_step_hz=.1,frequency_start_hz=-5,format='cf32le')
        header=json.dumps(meta).encode()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'latest.rdm'
            path.write_bytes(struct.pack('<4sIQ',b'RDM1',len(header),z.nbytes)+header+z.tobytes())
            for q in ({},{'width':['16'],'height':['16']},{'rmin':['375'],'rmax':['750'],'fmin':['-1'],'fmax':['1']}):
                packet=serve.preview(path,q)
                magic,length,size=struct.unpack('<4sIQ',packet[:16])
                m=json.loads(packet[16:16+length]);out=np.frombuffer(packet,dtype='<f4',offset=16+length).reshape(m['frequencies'],m['ranges'])
                self.assertEqual(size,out.nbytes);self.assertEqual(magic,b'RDM1')
                r0=round(m['range_start_m']/37.5);f0=round((m['frequency_start_hz']+5)/.1)
                rd=round(m['range_step_m']/37.5);fd=round(m['frequency_step_hz']/.1)
                for f in range(out.shape[0]):
                    for r in range(out.shape[1]):
                        rs=slice(r0+r*rd,min(35,r0+(r+1)*rd));fs=slice(f0+f*fd,min(101,f0+(f+1)*fd))
                        expected=10*np.log10(np.max(np.abs(z[rs,fs])**2))
                        self.assertAlmostEqual(float(out[f,r]),float(expected),places=4)
                self.assertAlmostEqual(float(out.max()),float(10*np.log10(abs(z[17,42])**2)),places=4)

if __name__=='__main__':unittest.main()
