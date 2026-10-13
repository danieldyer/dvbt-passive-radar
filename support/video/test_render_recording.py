import io
import json
import struct
import subprocess
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import render_recording as recording
import render_video as video


class RecordingVideoTest(unittest.TestCase):
    def test_native_pool_matches_scientific_peak_pool(self):
        exe=video.ROOT/'build/release/rdm_preview'
        if not exe.is_file():self.skipTest('Optional native display reducer not built')
        rng=np.random.default_rng(728)
        for shape in ((103,281),(3247,5021),(3203,2509)):
            values=(rng.normal(size=shape)+1j*rng.normal(size=shape)).astype('<c8')
            meta=dict(channel=0,sequence=1,time_end_s=1,ranges=shape[0],frequencies=shape[1],format='cf32le',order='range,doppler')
            def frame():
                header=json.dumps(meta).encode()
                return struct.pack('<4sIQ',b'RDM1',len(header),values.nbytes)+header+values.tobytes()
            result=subprocess.run([str(exe)],input=frame(),capture_output=True,check=True)
            m,actual=next(recording.read_pooled(io.BytesIO(result.stdout)))
            expected=video.power_preview(values,1624,740)
            np.testing.assert_allclose(actual,expected,rtol=0,atol=4e-6)
            self.assertEqual(m['channel'],0)
            values[0,0]=complex(float('nan'),0)
            invalid=subprocess.run([str(exe)],input=frame(),capture_output=True)
            self.assertNotEqual(invalid.returncode,0)
            self.assertIn(b'Nonfinite',invalid.stderr)

    def test_display_cache_keeps_channel_phase_metadata(self):
        meta=dict(channel=3,sequence=17,time_end_s=29.7,phase_epoch_s=26.7,physical_rf_hz=219500000,
                  noise_range_factor=[1,2],noise_doppler_power=[3,4],map_peak_db=41.37)
        db=np.array([[-450,1.723,41.37],[-24.5,12.57,2.888]],dtype='<f4')
        stream=io.BytesIO();recording.write_preview(stream,meta,db);stream.seek(0)
        records=list(recording.read_previews(stream));self.assertEqual(len(records),1)
        m,actual=records[0]
        for key in ('channel','sequence','time_end_s','phase_epoch_s','physical_rf_hz','map_peak_db'):
            self.assertEqual(m[key],meta[key])
        self.assertNotIn('noise_range_factor',m)
        np.testing.assert_allclose(actual,db,rtol=0,atol=.016)
        truncated=io.BytesIO(stream.getvalue()[:-1])
        with self.assertRaisesRegex(ValueError,'Truncated'):list(recording.read_previews(truncated))


if __name__=='__main__':unittest.main()
