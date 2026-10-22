import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from aperture import extract, FS
from phase_history import index_archive


class ApertureTests(unittest.TestCase):
    def make_archive(self, path, *, segments=(0, 0)):
        expected = []
        with path.open('wb') as output:
            header = json.dumps({'corrected_rate_hz': 64e6, 'decimation': 8,
                                 'applied_channels': [{'rf_hz': 177.5e6}]}).encode()
            output.write(b'DVBTLIV1'+struct.pack('<I', len(header))+header)
            for record, segment in enumerate(segments):
                t = 10+(record*4+np.arange(4))*.001088
                # Large, varying window_phase must not be applied a second time.
                rows = [[i+record*4, float(v-4095.5/FS), 0, 0, 2.1+i, int(i != 1)] for i, v in enumerate(t)]
                profile = (np.arange(2*4*1024).reshape(2, 4, 1024)+1j*(record+3)).astype('<c8')
                meta = {'channel': 0, 'count': 4, 'moments': 2, 'size': 1024, 'segment': segment,
                        'information': 2.5+record, 'rows': rows}
                raw = json.dumps(meta).encode()
                output.write(struct.pack('<4sIQ', b'PROF', len(raw), profile.nbytes)+raw+profile.tobytes())
                expected.append(profile)
            output.write(struct.pack('<4sIQ', b'DONE', 2, 0)+b'{}')
        return np.concatenate(expected, axis=1)

    def test_crop_preserves_complex_phase_both_moments_erasures_and_epochs(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'test.dvblive'
            expected = self.make_archive(path)
            archive = index_archive(path, Path(directory)/'index.json')
            data = extract(path, archive, 0, 10, 10.009, 400, 450)
            np.testing.assert_array_equal(data['moments'], expected[:, :, 400:451])
            np.testing.assert_allclose(data['time_s'], 10+np.arange(8)*.001088, atol=2e-15, rtol=0)
            np.testing.assert_array_equal(data['valid'], [1, 0, 1, 1, 1, 0, 1, 1])
            np.testing.assert_array_equal(data['information'], [2.5]*4+[3.5]*4)
            np.testing.assert_array_equal(data['record_index'], [0]*4+[1]*4)

    def test_acquisition_reset_requires_explicit_aperture_split(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'test.dvblive'
            self.make_archive(path, segments=(0, 1))
            archive = index_archive(path, Path(directory)/'index.json')
            with self.assertRaisesRegex(ValueError, 'segment'):
                extract(path, archive, 0, 10, 10.009, 400, 450)


if __name__ == '__main__':
    unittest.main()
