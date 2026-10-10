import importlib.util
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('video',Path(__file__).resolve().with_name('render_video.py'))
video=importlib.util.module_from_spec(spec);spec.loader.exec_module(video)


def packet(sequence=1,end=1):
    values=np.array([[1+2j,3+4j],[5+6j,7+8j]],dtype='<c8')
    meta=dict(sequence=sequence,time_end_s=end,ranges=2,frequencies=2,format='cf32le',order='range,doppler')
    header=json.dumps(meta).encode()
    return struct.pack('<4sIQ',b'RDM1',len(header),values.nbytes)+header+values.tobytes()


class Chunked(io.BytesIO):
    def read(self,size=-1):return super().read(min(size,7))
    def readinto(self,buffer):return super().readinto(buffer[:7])


class VideoTest(unittest.TestCase):
    def test_stream_short_reads_and_sequence(self):
        frames=list(video.read_frames(Chunked(packet()+packet(2,1.2))))
        self.assertEqual(len(frames),2)
        self.assertEqual(frames[1][1][1,0],5+6j)
        with self.assertRaisesRegex(ValueError,'Missing'):
            list(video.read_frames(io.BytesIO(packet()+packet(3,1.2))))
        with self.assertRaisesRegex(ValueError,'Truncated'):
            list(video.read_frames(io.BytesIO(packet()[:-1])))

    def test_doppler_orientation_and_peak_pool(self):
        values=np.full((4,8),.001+0j,dtype=np.complex64)
        values[2,7]=2  # positive Doppler must be at the top, range to the right
        db=video.power_preview(values,2,4)
        self.assertEqual(np.unravel_index(np.argmax(db),db.shape),(0,1))
        self.assertAlmostEqual(float(db[0,1]),20*np.log10(2),places=5)

    def test_pool_partial_edges(self):
        rng=np.random.default_rng(507)
        values=(rng.normal(size=(103,281))+1j*rng.normal(size=(103,281))).astype('<c8')
        actual=video.power_preview(values,17,39)
        expected=np.array([[10*np.log10(np.max(abs(values[r:min(r+7,103),f:min(f+8,281)])**2))
                            for f in range(0,281,8)] for r in range(0,103,7)])
        np.testing.assert_allclose(actual,expected.T[::-1],rtol=1e-6,atol=2e-6)

    def test_duration_matches_integer_samples_and_wrapped_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'iq.wav'
            for samples in (16000,2**29+16):
                length=samples*8
                with p.open('wb') as f:
                    f.write(struct.pack('<4sI4s4sIHHIIHH4sI',b'RIFF',(length+36)&0xffffffff,b'WAVE',b'fmt ',16,3,2,8000000,64000000,8,32,b'data',length&0xffffffff))
                    f.truncate(44+length)  # sparse file: no gigabytes of sample allocation
                self.assertAlmostEqual(video.recording_duration(p),samples/8000000)

    def test_clock_and_cli_limits(self):
        self.assertEqual(video.clock_label(301.84),'05:01.840')
        a=video.arguments(['input.wav','--output','movie.mp4'])
        self.assertTrue(a.remove_dc);self.assertFalse(a.auto_scale)
        enabled=video.arguments(['input.wav','--output','movie.mp4','--remove-dc','--auto-scale'])
        self.assertTrue(enabled.remove_dc);self.assertTrue(enabled.auto_scale)
        self.assertEqual(a.integration,3)
        self.assertEqual(a.range_km,30)
        self.assertEqual(a.doppler_hz,200)
        self.assertGreaterEqual(a.fps,30)
        self.assertAlmostEqual(a.fps,3125/102)

    def test_production_and_control_commands(self):
        a=video.arguments(['input.wav','--output','movie.mp4','--no-remove-dc'])
        command=video.processor_command(a,Path('work'))
        self.assertEqual(Path(command[0]).name,'dvbt_detect')
        self.assertIn('--keep-dc',command)
        self.assertEqual(command[command.index('--channel')+1],'557625000,4625000,16')
        self.assertNotIn('--slow-decimation',command)
        legacy=video.arguments(['input.wav','--output','movie.mp4','--backend','legacy'])
        command=video.processor_command(legacy,Path('work'))
        self.assertEqual(Path(command[0]).name,'dvbt_reconstruct')
        self.assertIn('--remove-dc',command)
        self.assertIn('--slow-decimation',command)

    def test_quicktime_output_selection(self):
        args=video.arguments(['input.wav','--output','movie.mov','--encoding','prores','--integration','10'])
        self.assertEqual(args.encoding,'prores')
        self.assertEqual(args.integration,10)
        for options in (['--output','movie.mp4','--encoding','prores'],['--output','movie.mov']):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                video.arguments(['input.wav',*options])


if __name__=='__main__':unittest.main()
