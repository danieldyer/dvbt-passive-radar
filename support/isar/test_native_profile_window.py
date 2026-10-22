import json,struct,tempfile,unittest
from pathlib import Path
import numpy as np
from recorded_injection_fields import read_profiles,write_profiles
from native_folds import native_indices


def record(output,tag,meta,payload=b''):
    text=json.dumps(meta).encode();output.write(struct.pack('<4sIQ',tag,len(text),len(payload)))
    output.write(text);output.write(payload)


class NativeProfileWindowTests(unittest.TestCase):
    def test_bounded_reader_preserves_complex_payload_and_global_fold_identities(self):
        rng=np.random.default_rng(821)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'profiles.bin';header=json.dumps(dict(range_count=9)).encode()
            with path.open('wb') as output:
                output.write(b'INJPROF1'+struct.pack('<I',len(header))+header)
                for block in range(20):
                    for c in range(2):
                        index=np.arange(block*17,(block+1)*17);rows=np.zeros((17,6))
                        rows[:,0]=index*8192;rows[:,1]=index*.01+c*.0001;rows[:,4]=1
                        rows[(index>=128)&(index<144),4]=0
                        values=(rng.normal(size=(17,2,9))+1j*rng.normal(size=(17,2,9))).astype('<c8')
                        pulse=(rng.normal(size=(17,129))+1j*rng.normal(size=(17,129))).astype('<c8')
                        record(output,b'PROF',dict(channel=c,rows=rows.tolist(),information=1+block*.01),values.tobytes()+pulse.tobytes())
                record(output,b'DONE',{})
            _,full=read_profiles(path);_,part=read_profiles(path,interval_s=(1.,1.7))
            copy=Path(directory)/'roundtrip.bin';write_profiles(copy,dict(range_count=9),full)
            _,copied=read_profiles(copy)
            for c in full:
                for key in ('values','pulse','rows','information'):np.testing.assert_array_equal(copied[c][key],full[c][key])
            for c in range(2):
                a,b=full[c],part[c];start=b['first_row_index'];stop=start+len(b['rows'])
                self.assertEqual(start%16,0);self.assertGreater(start,0)
                np.testing.assert_array_equal(b['values'],a['values'][:,start:stop])
                for key in ('pulse','rows','information'):np.testing.assert_array_equal(b[key],a[key][start:stop])
                for fold in ('all','training','testing'):
                    wanted=native_indices(a['rows'],a['information'],1.,1.7,fold)
                    actual=native_indices(b['rows'],b['information'],1.,1.7,fold)+start
                    np.testing.assert_array_equal(actual,wanted)


if __name__=='__main__':unittest.main()
