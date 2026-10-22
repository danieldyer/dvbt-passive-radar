import unittest

import numpy as np

from phase_window import rebase_parameters,window_band
from test_component_phase import fixture
from shared_echo_cluster import SharedEchoCluster


class PhaseWindowTests(unittest.TestCase):
    def test_restriction_preserves_original_rows_and_each_complex_response(self):
        model,p=fixture();old=model.bands[0]
        value=np.divide(old.field.y,old.field.root_information[:,None],
            out=np.zeros_like(old.field.y),where=old.field.root_information[:,None]>0)
        old.field.data['value']=np.linalg.solve(old.field.whiten,value.T).T
        first,last=old.epoch-.65*old.scale,old.epoch+.25*old.scale
        new,part,rows=window_band(old,first,last)
        transformed=rebase_parameters(p,3,old,new)
        sub=SharedEchoCluster([new],model.origins)
        a=model.clusters[0].response_blocks(p[model.indices[0]])
        b=sub.clusters[0].response_blocks(transformed[sub.indices[0]])
        for (expected,_),(actual,_) in zip(a,b):
            np.testing.assert_allclose(actual,expected[rows],atol=5e-15,rtol=1e-12)
        np.testing.assert_array_equal(new.field.data['row_information'],old.field.data['row_information'][rows])
        np.testing.assert_array_equal(new.known_path,old.known_path[rows])
        self.assertFalse(np.any(new.training&(part==3)))


if __name__=='__main__':unittest.main()
