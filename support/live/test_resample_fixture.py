import unittest

try:
    import numpy as np
    from scipy.signal import resample_poly
    from resample_fixture import blocks
except ImportError:
    np = None


@unittest.skipIf(np is None, 'SciPy is only required for fixture resampling')
class FixtureTests(unittest.TestCase):
    def test_every_block_preserves_global_rational_phase(self):
        rng = np.random.default_rng(19283)
        x = rng.normal(size=(5003, 2)).astype(np.float32)
        expected = resample_poly(x, 24, 25, axis=0, window=('kaiser', 8.6))
        for size in [25, 100, 500, 5000]:
            actual = np.concatenate(list(blocks(x, size)))
            np.testing.assert_array_equal(actual, expected)
        cyclic = x[:5000]
        expected = resample_poly(cyclic, 24, 25, axis=0, window=('kaiser', 8.6), padtype='wrap')
        explicit = np.concatenate(list(blocks(cyclic, 5000, cyclic=True)))
        # SciPy's wrap kernel and constant-context kernel accumulate float32
        # products differently. Their difference is bounded by float rounding;
        # changing our block boundaries must remain bit-exact.
        np.testing.assert_allclose(explicit, expected, rtol=0, atol=1e-6)
        for size in [25, 100, 500, 5000]:
            np.testing.assert_array_equal(np.concatenate(list(blocks(cyclic, size, cyclic=True))), explicit)


if __name__ == '__main__':
    unittest.main()
