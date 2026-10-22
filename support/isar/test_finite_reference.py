import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from finite_reference import (BINS, coefficient_planes, covariance, echo, probes,
                              range_weights, references, segment)


class FiniteReferenceTests(unittest.TestCase):
    def test_reference_against_scalar_fourier_sum_and_direct_interpolation(self):
        n = 128
        rng = np.random.default_rng(173)
        s, p = rng.normal(size=(2, 65))+1j*rng.normal(size=(2, 65))
        origin, location = 4.23, 43.37
        offsets = np.arange(-2, 3)
        actual = references(s, p, origin, location, offsets, nfft=n)
        centre, taps, weights = range_weights(location)
        k = np.arange(65)-32
        expected = np.zeros_like(actual)
        for r, offset in enumerate(offsets):
            for j in range(n):
                for tap, w in zip(taps, weights):
                    t = j-centre-offset-tap-int(np.floor(origin))
                    previous = t < -n//16
                    t = t+(n+n//16 if previous else 0)-(origin-np.floor(origin))
                    spectrum = p if previous else s
                    expected[r, j] += w*np.sum(spectrum*np.exp(2j*np.pi*k*t/n))/n
        np.testing.assert_allclose(actual, expected, atol=3e-15)

    def test_coefficients_against_direct_time_correlation(self):
        n = 256
        rng = np.random.default_rng(891)
        s, p, y = rng.normal(size=(3, 129))+1j*rng.normal(size=(3, 129))
        h = np.exp(.3*rng.normal(size=129)+.3j*rng.normal(size=129))
        v = np.exp(rng.normal(size=129));origin=3.27;location=43.18
        offsets=np.arange(-3, 4);k=np.arange(129)-64
        q = coefficient_planes(s,p,h,v,origin,location,offsets,nfft=n)
        packed=np.zeros(n,complex);packed[k % n]=y*h.conj()/v
        work=np.fft.ifft(packed);gain=np.sum(abs(h)**2/v)/n
        ref=references(s,p,origin,location,offsets,nfft=n)
        times=(np.arange(n)-(n-1)/2)/n
        expected=np.stack((ref.conj()@work,ref.conj()@(work*times)))/gain
        np.testing.assert_allclose(q@y,expected,atol=3e-15)
        c=q[0]-.4j*q[1]
        noise=(rng.normal(size=(129,6000))+1j*rng.normal(size=(129,6000)))*np.sqrt(v[:,None]/2)
        observed=c@noise
        predicted=covariance(c,v)
        self.assertLess(np.linalg.norm(observed@observed.conj().T/6000-predicted)/np.linalg.norm(predicted),.05)

    def test_echo_and_clock_fraction_wrapping(self):
        n=128;k=7;s=np.zeros(65,complex);s[k+32]=1
        # The single-tone previous symbol has the phase needed for continuity.
        p=s*np.exp(-2j*np.pi*k*(n+n//16)/n)
        for origin in (4.999,5.001):
            for delay in (0,20.3):
                a=echo(s,p,np.ones(65),origin,delay,0,nfft=n)
                expected=s*np.exp(-2j*np.pi*k*(origin+delay)/n)
                np.testing.assert_allclose(a,expected,atol=2e-14)
        with self.assertRaises(ValueError):segment(s,p,0,-200,128,nfft=n)

    def test_archive_completion_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'probe.fref'
            header=b'{}';meta=json.dumps({'information':1}).encode()
            payload=b'\0'*(24*BINS)+np.ones(BINS,dtype='<f4').tobytes()+b'\0'*(8*BINS+16*1024)
            done=b'{"rows":1}'
            value=(b'FREF0001'+struct.pack('<I',len(header))+header+
                   struct.pack('<4sIQ',b'ROW0',len(meta),len(payload))+meta+payload+
                   struct.pack('<4sIQ',b'DONE',len(done),0)+done)
            p.write_bytes(value);self.assertEqual(len(list(probes(p))),1)
            p.write_bytes(value[:-1])
            with self.assertRaises(ValueError):list(probes(p))


if __name__=='__main__':unittest.main()
