import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from captured_response import captured_rows, observations


class CapturedResponseTests(unittest.TestCase):
    def make_capture(self, directory):
        path=Path(directory)/'capture.dvblive'
        dt=.001088
        rows=[[r*8704,r*dt,0,0,0,True,r*.01,0] for r in range(4)]
        meta={'channel':0,'segment':3,'count':4,'moments':2,'size':1024,
              'stationary_carriers':6817,'response_carriers':6817,'noise_carriers':6817,
              'cancellation_variance_carriers':6817,'information':123.,'reference_origin':127.5,'rows':rows}
        profile=np.empty((2,4,1024),'<c8')
        for r in range(4):profile[:,r]=np.array([r+2j,2*r-3j])[:,None]
        payload=(profile.tobytes()+np.zeros(6817,'<c8').tobytes()+
                 np.full(6817,2+3j,'<c8').tobytes()+np.full(6817,1.25,'<f4').tobytes()+
                 np.full(6817,.002,'<f8').tobytes())
        encoded=json.dumps(meta).encode()
        path.write_bytes(struct.pack('<4sIQ',b'PROF',len(encoded),len(payload))+encoded+payload)
        stat=path.stat();index=Path(directory)/'index.json'
        index.write_text(json.dumps({'identity':[str(path),stat.st_size,stat.st_mtime_ns],
                                     'records':[[0,0,0,3*dt]]}))
        return path,index,dt

    def test_original_clock_context_and_weights(self):
        with tempfile.TemporaryDirectory() as d:
            _,index,dt=self.make_capture(d)
            rows=captured_rows(index,0,np.array([dt,3*dt])+4095.5/8e6,dt)
            self.assertEqual([r['source_offset'] for r in rows],[.01,.03])
            self.assertEqual([r['previous_source_offset'] for r in rows],[0,.02])
            np.testing.assert_array_equal(rows[0]['response'],np.full(6817,2+3j))
            np.testing.assert_array_equal(rows[1]['cancellation_variance'],np.full(6817,.002))
            self.assertEqual(rows[0]['segment'],3)

    def test_saved_moment_order_and_complex_doppler_combination(self):
        with tempfile.TemporaryDirectory() as d:
            _,index,dt=self.make_capture(d)
            rows=captured_rows(index,0,np.array([dt,3*dt])+4095.5/8e6,dt)
            rate=np.array([.07,-.13]);offsets=np.arange(-8,9)
            actual=observations(rows,[300.31,400.73],rate,offsets)
            r=np.array([1,3]);expected=r+2j-1j*rate*(2*r-3j)
            np.testing.assert_allclose(actual,np.broadcast_to(expected[:,None],actual.shape),rtol=0,atol=3e-15)

    def test_changed_capture_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path,index,dt=self.make_capture(d)
            with path.open('ab') as f:f.write(b'x')
            with self.assertRaisesRegex(ValueError,'identity changed'):
                captured_rows(index,0,np.array([dt])+4095.5/8e6,dt)

    def test_truncated_coefficient_variance_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path,index,dt=self.make_capture(d)
            path.write_bytes(path.read_bytes()[:-8]);j=json.loads(index.read_text());s=path.stat()
            j['identity'][1:]=[s.st_size,s.st_mtime_ns];index.write_text(json.dumps(j))
            with self.assertRaisesRegex(ValueError,'Truncated'):
                captured_rows(index,0,np.array([dt])+4095.5/8e6,dt)


if __name__=='__main__':
    unittest.main()
