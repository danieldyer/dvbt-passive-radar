import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from extract_full_aperture import window
from field import interpolate


class ProfileWindowTests(unittest.TestCase):
    def fixture(self,directory):
        path=Path(directory)/'capture.dvblive';time=np.arange(4)*.001088
        rows=[[i*8704,t,0,0,0,i!=2,0,0] for i,t in enumerate(time)]
        logical=np.r_[np.arange(929),np.zeros(31),np.arange(-64,0)]
        profile=np.empty((2,4,1024),'<c8')
        for m in range(2):
            for r in range(4):profile[m,r]=np.exp(2j*np.pi*.13*logical)*(1+m+.1j*r)
        profile[:,:,929:960]=np.nan
        meta=dict(channel=0,segment=1,count=4,moments=2,size=1024,information=12.,rows=rows)
        encoded=json.dumps(meta).encode();payload=profile.tobytes()
        path.write_bytes(struct.pack('<4sIQ',b'PROF',len(encoded),len(payload))+encoded+payload)
        stat=path.stat();index=dict(identity=[str(path),stat.st_size,stat.st_mtime_ns],
            records=[[0,0,time[0],time[-1]]])
        return path,index,time+4095.5/8e6

    def test_negative_profile_slots_and_original_erasures_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path,index,time=self.fixture(directory)
            data=window(path,index,0,time[1],time[-1])
            np.testing.assert_array_equal(data['time_s'],time[1:])
            np.testing.assert_array_equal(data['valid'],[True,False,True])
            self.assertEqual(data['moments'].shape,(2,3,993))
            self.assertTrue(np.isfinite(data['moments']).all())
            logical=np.arange(-64,929)
            np.testing.assert_allclose(data['moments'][1,0],np.exp(2j*np.pi*.13*logical)*(2+.1j),rtol=8e-8,atol=8e-8)
            coordinate=np.full(3,64+.37)
            value=interpolate(data['moments'],coordinate,np.array([-1,0,1]))
            expected=np.exp(2j*np.pi*.13*(np.array([-1,0,1])+.37))
            for m in range(2):
                for r in range(3):
                    np.testing.assert_allclose(value[m,r],expected*(1+m+.1j*(r+1)),rtol=2e-5,atol=2e-5)

    def test_capture_identity_and_supported_range_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path,index,time=self.fixture(directory)
            with self.assertRaises(ValueError):window(path,index,0,*time[[0,-1]],first_range_sample=-65)
            with path.open('ab') as output:output.write(b'x')
            with self.assertRaisesRegex(ValueError,'identity changed'):window(path,index,0,*time[[0,-1]])


if __name__=='__main__':unittest.main()
