import unittest
import numpy as np
from scatterer_chirp_diagnostic import acquire, chirp_score


class TestChirp(unittest.TestCase):
    def test_original_epoch_gradient_and_recovery(self):
        rng = np.random.default_rng(19)
        t = np.sort(rng.uniform(-15, 15, 1100))
        w = rng.uniform(.2, 3, len(t)); k = 4.1
        truth = np.array([.314159, -.02718])
        y = (.3+.7j)*np.exp(-1j*k*(truth[0]*t+.5*truth[1]*t*t))
        p = truth + [.01, .002]
        value, gradient, _ = chirp_score(t, y, w, k, p)
        for j in range(2):
            step = np.eye(2)[j]*1e-7
            numeric = (chirp_score(t,y,w,k,p+step)[0]-chirp_score(t,y,w,k,p-step)[0])/2e-7
            self.assertLess(abs(numeric-gradient[j])/max(1,abs(numeric)), 1e-7)
        result = acquire(t,y,w,k, maximum_rate=1., maximum_acceleration=.08)
        np.testing.assert_allclose(result['parameters'], truth, atol=1e-8)
        self.assertAlmostEqual(result['coherence'], 1., places=12)


if __name__ == '__main__': unittest.main()
