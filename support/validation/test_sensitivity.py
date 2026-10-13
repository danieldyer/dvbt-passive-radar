"""Independent checks of the offline audit's phase and noise conventions."""
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from sensitivity import BINS, DT, FS, K, NFFT, ROW, ROW2, blocks, map_at, summarize, window
from matched_filter_probe import efficiency
from covariance_bound import bound
from doppler_bound import white_local_bound, white_local_model


class SensitivityTests(unittest.TestCase):
    def test_source_clock_archive_coordinates(self):
        rows, origin = 16, 128.375
        coordinates = np.zeros(rows, dtype=ROW2)
        coordinates['sample'] = np.arange(rows)*8704
        coordinates['source_offset'] = np.remainder(.37+.02*np.arange(rows)+.5, 1)-.5
        coordinates['source_rate'] = .04e-6
        s = np.ones((rows, BINS), dtype='<c8')
        x = s*np.exp(-2j*np.pi*K*(origin-coordinates['source_offset'][:, None])/NFFT)
        h = np.full(BINS, .7+.4j, dtype='<c8')
        q = .01*np.exp(.73j-2j*np.pi*K*37.25/NFFT)
        y = (x*(h+q)).astype('<c8')
        ratio = float(np.sum(abs(y-x*h)**2)/np.sum(abs(y)**2))
        body = struct.pack('<6Idd', 0, 0, rows, 0, rows, 0, origin, ratio)
        body += coordinates.tobytes()+h.tobytes()+y.tobytes()+s.tobytes()
        body += np.ones((rows, BINS), dtype='<u4').tobytes()
        archive = b'DVBTSAR2'+struct.pack('<4IQdd', 56000000, 7, 1, 0, 56000000, 553e6, 3.)
        archive += struct.pack('<ddII', 557625000., 4625000., 16, 0)
        archive += struct.pack('<4sQ', b'SPEC', len(body))+body+struct.pack('<4sQQ', b'DONE', 8, 1)
        with tempfile.TemporaryDirectory() as directory:
            path, cache = Path(directory)/'clock.sar', Path(directory)/'clock.npz'
            path.write_bytes(archive)
            decoded = list(blocks(path))[0]
            np.testing.assert_array_equal(decoded['source_offsets'], coordinates['source_offset'])
            summarize(path, cache)
            result = np.load(cache)['after']
            expected = np.array([q @ np.exp(2j*np.pi*K*r/NFFT)/NFFT for r in range(801)])
            np.testing.assert_allclose(result, np.broadcast_to(expected, result.shape), atol=5e-8)

    def test_hann_local_doppler_information_bound(self):
        # Analytic tridiagonal signal and pentadiagonal noise covariance.
        for radius in (0,1,2,4,8,16):
            result=white_local_bound('hann',radius)
            cells=2*radius+1
            self.assertAlmostEqual(result['information_vs_untapered'],
                                   (cells+1)/(cells+2),places=11)
            q,c=white_local_model('hann',radius)
            distance=radius-np.abs(np.arange(-radius,radius+1))
            weights=(distance+1)*(distance+2)/(2*radius+3)
            np.testing.assert_allclose(c@weights,q,atol=1e-13)
        q,c=white_local_model('hann',4)
        np.testing.assert_allclose(q[3:6],[-.5,1,-.5],atol=1e-15)
        np.testing.assert_allclose(c[4,4:7],[1.5,-1,.25],atol=1e-15)

    def test_doppler_covariance_complex_gain_and_noise(self):
        rng=np.random.default_rng(531);n=512;radius=4
        q,c=white_local_model('hann',radius)
        taper=window('hann',n);offsets=np.arange(-radius,radius+1)
        noise=(rng.normal(size=(4000,n))+1j*rng.normal(size=(4000,n)))/np.sqrt(2)
        bins=np.fft.fft(noise*taper,axis=1)[:,offsets%n]/taper.sum()
        actual=bins.T@bins.conj()/len(bins)
        self.assertLess(np.linalg.norm(actual-c/n)/np.linalg.norm(c/n),.06)
        weights=np.linalg.solve(c,q);weights/=np.vdot(weights,q).conjugate()
        phase=.04*np.exp(1.234j)
        injected=np.fft.fft(np.full(n,phase)*taper)[offsets%n]/taper.sum()
        self.assertAlmostEqual(abs(np.vdot(weights,injected)-phase),0,places=14)
        predicted=np.vdot(weights,c@weights).real/n
        self.assertAlmostEqual(np.var(bins@weights.conj())/predicted,1,delta=.04)

    def test_independent_noise_floor_is_an_information_upper_bound(self):
        rng=np.random.default_rng(1341)
        d=np.diag(rng.uniform(.1,2,24))
        u=rng.normal(size=(24,4))+1j*rng.normal(size=(24,4))
        q=rng.normal(size=24)+1j*rng.normal(size=24)
        upper=np.vdot(q,np.linalg.solve(d,q)).real
        actual=np.vdot(q,np.linalg.solve(d+u@u.conj().T,q)).real
        self.assertLessEqual(actual,upper)

    def test_low_rank_complex_and_real_information_agree(self):
        # An orthonormal, real mode basis has a closed-form eigendecomposition.
        # Compare that result to the Woodbury and augmented-real implementations.
        modes=np.column_stack((np.ones(BINS)/np.sqrt(BINS),K/np.linalg.norm(K)))
        eigenvalues=np.array([30.,2.])
        model=dict(d=np.ones(BINS),h=np.ones(BINS),p=np.ones(BINS),modes=modes,
                   modes_cov=np.diag(eigenvalues),real_cov=np.diag(np.tile(eigenvalues/2,2)))
        delay=1
        q=np.exp(-2j*np.pi*K*delay/NFFT)
        projections=abs(modes.T@q)**2
        information=BINS-np.sum(eigenvalues/(1+eigenvalues)*projections)
        variance=BINS+np.sum(eigenvalues*projections)
        expected=10*np.log10(information*variance/BINS**2)
        actual=bound(model,delay)
        self.assertAlmostEqual(actual['two_mode_conditional_gain_db'],expected,places=11)
        np.testing.assert_allclose(actual['improper_noise_gain_db_phase_min_max'],expected,atol=1e-11)

    def test_covariance_matched_filter_bound(self):
        rng=np.random.default_rng(9021)
        a=rng.normal(size=(20,20))+1j*rng.normal(size=(20,20))
        covariance=a@a.conj().T+np.eye(20)
        q=rng.normal(size=20)+1j*rng.normal(size=20)
        optimum=np.linalg.solve(covariance,q)
        bound=np.vdot(q,optimum).real
        def snr(w):return abs(np.vdot(w,q))**2/np.vdot(w,covariance@w).real
        self.assertAlmostEqual(snr(optimum)/bound,1,places=13)
        self.assertLess(snr(q)/bound,.9)
        for _ in range(30):
            w=rng.normal(size=20)+1j*rng.normal(size=20)
            self.assertLessEqual(snr(w),bound*(1+1e-13))
        self.assertAlmostEqual(efficiency(q,q),1,places=13)

    def test_hann_noise_bandwidth(self):
        w = window('hann', 4096)
        self.assertAlmostEqual(len(w) * np.dot(w, w) / w.sum()**2, 1.5, places=13)

    def test_temporal_information_weighting_attains_known_noise_bound(self):
        # Unit-amplitude estimates with three different noise levels. Their
        # information weights must survive per-batch amplitude normalization.
        variance=np.repeat([1.,10.,.1],256)
        weights=1/variance
        information=weights.sum()
        snr=lambda w: w.sum()**2/np.dot(w*w,variance)
        self.assertAlmostEqual(snr(weights)/information,1,places=14)
        self.assertLess(snr(np.ones(len(variance)))/information,.1)
        rng=np.random.default_rng(932)
        noise=(rng.normal(size=(3000,len(variance)))+1j*rng.normal(size=(3000,len(variance))))*np.sqrt(variance/2)
        estimate=noise@weights/weights.sum()
        self.assertAlmostEqual(np.var(estimate)*information,1,delta=.06)

    def test_dc_projection_phase_and_noise(self):
        n = 512
        t = np.arange(n)
        s = np.exp(2j * np.pi * 3 * t / n)
        w = window('hann', n)
        c = np.sum(w * s.conj()) / w.sum()
        g = w * (s.conj() - c) / w.sum()
        rng = np.random.default_rng(6402)
        noise = (rng.normal(size=(3000, n)) + 1j*rng.normal(size=(3000, n))) / np.sqrt(2)
        amplitude = .02 * np.exp(.73j)
        data = noise + amplitude*s + (100+70j)
        # This implementation matches the production weighted-mean operator.
        demeaned = data - (data @ w / w.sum())[:, None]
        actual = demeaned @ (w*s.conj()) / w.sum()
        expected = data @ g
        np.testing.assert_allclose(actual, expected, atol=1e-13)
        predicted_noise = np.sum(abs(g)**2)
        self.assertAlmostEqual(np.var(actual) / predicted_noise, 1, delta=.06)
        self.assertAlmostEqual(abs(np.sum(g*s)), 1, places=13)
        self.assertLess(abs(np.mean(actual)-amplitude), 3*np.sqrt(predicted_noise/len(noise)))

    def test_archive_replay_delay_and_complex_phase(self):
        rows = 16
        origin = 128.375
        s = np.ones((rows, BINS), dtype=np.complex64)
        x = s*np.exp(-2j*np.pi*K*origin/NFFT)
        h = np.full(BINS, .7+.4j, dtype=np.complex64)
        a = .01*np.exp(.73j)
        q = a*np.exp(-2j*np.pi*K*37.25/NFFT)
        y = (x*(h+q)).astype(np.complex64)
        ratio = float(np.sum(abs(y-x*h)**2)/np.sum(abs(y)**2))
        coordinates = np.zeros(rows, dtype=ROW)
        coordinates['sample'] = np.arange(rows)*8704
        body = struct.pack('<6Idd', 0, 0, rows, 0, rows, 0, origin, ratio)
        body += coordinates.tobytes() + h.tobytes() + y.tobytes() + s.tobytes()
        body += np.ones((rows, BINS), dtype='<u4').tobytes()
        archive = b'DVBTSAR1' + struct.pack('<4IQdd', 56000000, 7, 1, 0, 56000000, 553e6, 3.)
        archive += struct.pack('<ddII', 557625000., 4625000., 16, 0)
        archive += struct.pack('<4sQ', b'SPEC', len(body)) + body
        archive += struct.pack('<4sQQ', b'DONE', 8, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.sar'
            cache = Path(directory)/'test.npz'
            path.write_bytes(archive)
            summary = summarize(path, cache)
            result = np.load(cache)
            np.testing.assert_allclose(np.diff(result['times']), DT, atol=2e-16)
            expected = np.array([q @ np.exp(2j*np.pi*K*r/NFFT)/NFFT for r in range(801)])
            np.testing.assert_allclose(result['after'][0], expected, atol=5e-8)
            self.assertEqual(np.argmax(abs(result['after'][0])), 37)
            self.assertLess(summary['max_gpu_cpu_energy_fraction_error'], 1e-10)
            path.write_bytes(archive[:-1])
            with self.assertRaises(ValueError):
                list(blocks(path))

    def test_fractional_doppler_and_dc_are_complex_dfts(self):
        n = 73
        seconds = (n+.1)*DT
        t = np.arange(n)
        rows = (3+5j + .04*np.exp(1.17j+2j*np.pi*2.3*t/n))[:, None]
        actual, frequencies = map_at(rows, seconds, remove_dc=True)
        w = window('hann', n)
        e = rows[:, 0] - w@rows[:, 0]/w.sum()
        expected = np.array([np.sum(w*e*np.exp(-2j*np.pi*f*t*DT))/w.sum() for f in frequencies])
        np.testing.assert_allclose(actual[:, 0], expected, atol=3e-15)


if __name__ == '__main__':
    unittest.main()
