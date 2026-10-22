import unittest

import numpy as np
from scipy.signal.windows import dpss

from secondary_spectrum import tile_statistics


class SecondarySpectrumTests(unittest.TestCase):
    def test_stationary_projection_noise_propagates_to_fourier_covariance(self):
        rng = np.random.default_rng(917)
        n, r, k = 40, 3, 5
        q = np.linalg.qr(rng.normal(size=(n*r, k))+1j*rng.normal(size=(n*r, k)))[0]
        tapers = dpss(n, 2., Kmax=3)
        bands = [np.arange(n)%3 == 0]
        y = (rng.normal(size=(n, r))+1j*rng.normal(size=(n, r)))/np.sqrt(2)
        clean = (y.ravel()-q@(q.conj().T@y.ravel())).reshape(n, r)
        result = tile_statistics(y, clean, q.reshape(n, r, k), tapers, bands)
        expected = np.zeros((r, r), complex)
        projector = np.eye(n*r)-q@q.conj().T
        for taper in tapers:
            for frequency in np.flatnonzero(bands[0]):
                row = taper*np.exp(-2j*np.pi*frequency*np.arange(n)/n)
                transform = np.kron(row[None], np.eye(r))
                expected += transform@projector@transform.conj().T/(len(tapers)*bands[0].sum())
        np.testing.assert_allclose(result['nominal_covariance'][0], expected, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(result['covariance'][0]).min(), 0.)


if __name__ == '__main__':
    unittest.main()
