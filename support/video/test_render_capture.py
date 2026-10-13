import json
from pathlib import Path
import struct
import tempfile
import unittest

import render_capture as capture


class CaptureVideoTest(unittest.TestCase):
    @staticmethod
    def fixture(path, mode='profiles', done=True):
        header=json.dumps(dict(archive_mode=mode,applied_channels=[dict(guard=16)])).encode()
        meta=json.dumps(dict(channel=0,count=2,size=1024,moments=2,rows=[[123,4.0,123,0,0,True],[124,4.001088,124,0,0,False]])).encode()
        payload=bytes(4*1024*8)
        body=b'DVBTLIV1'+struct.pack('<I',len(header))+header
        channel=json.dumps(dict(first=0,count=3,channels=1,order='channel,sample')).encode()
        iq=struct.pack('<4sIQ',b'CHIQ',len(channel),24)+channel+bytes(24)
        if mode=='both':body+=iq
        body+=struct.pack('<4sIQ',b'PROF',len(meta),len(payload))+meta+payload
        if mode=='both':body+=iq
        if done:body+=struct.pack('<4sIQ',b'DONE',2,0)+b'{}'
        path.write_bytes(body)

    def test_duration_keeps_capture_epoch_and_invalid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'capture.dvblive'
            for mode in ('profiles','both'):
                with self.subTest(mode=mode):
                    self.fixture(path,mode=mode)
                    header,duration=capture.archive_layout(path)
                    self.assertEqual(header['archive_mode'],mode)
                    self.assertAlmostEqual(duration,4.002176,places=12)

    def test_rejects_wrong_archive_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'capture.dvblive';self.fixture(path,mode='channels')
            with self.assertRaisesRegex(ValueError,'profiles archive'):capture.archive_layout(path)

    def test_rejects_incomplete_or_trailing_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'capture.dvblive';self.fixture(path,done=False)
            with self.assertRaisesRegex(ValueError,'Truncated'):capture.archive_layout(path)
            self.fixture(path);path.write_bytes(path.read_bytes()+b'junk')
            with self.assertRaisesRegex(ValueError,'after DONE'):capture.archive_layout(path)

    def test_rejects_truncated_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'capture.dvblive';self.fixture(path)
            path.write_bytes(path.read_bytes()[:-20000])
            with self.assertRaisesRegex(ValueError,'Truncated'):capture.archive_layout(path)


if __name__=='__main__':unittest.main()
