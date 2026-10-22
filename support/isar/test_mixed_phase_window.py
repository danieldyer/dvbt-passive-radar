import unittest

import numpy as np

from cluster_phase import SplinePath,PolynomialPath
from flexible_echo_cluster import FlexibleEchoCluster
from phase_window import rebase_cluster,window_band
from test_component_phase import fixture as fixed_fixture


class MixedPhaseWindowTests(unittest.TestCase):
    def fixture(self):
        original,previous=fixed_fixture();old=original.bands[0]
        source=FlexibleEchoCluster([old],original.origins,1,
            PolynomialPath(old.epoch-old.scale,old.epoch+old.scale,3),shared=False)
        p=np.zeros(source.count);p[:6]=previous[:6]
        for s,indices in enumerate(source.local_path_indices[0]):
            value=previous[original.phase_indices[0,s]]
            p[indices]=np.r_[value,.12] if s==1 else value
        field=old.field
        # The synthetic fixture writes whitened observations directly. Restore
        # their data representation before exercising the real Field crop.
        raw=np.divide(field.y,field.root_information[:,None],out=np.zeros_like(field.y),where=field.root_information[:,None]>0)
        field.data['value']=np.linalg.solve(field.whiten,raw.T).T
        first,last=old.epoch-.65*old.scale,old.epoch+.25*old.scale
        new,part,rows=window_band(old,first,last)
        curve=SplinePath(first,last,(last-first)/2)
        target=FlexibleEchoCluster([new],source.origins,1,curve,shared=False)
        return source,p,target,rows

    def test_all_components_preserve_envelope_carrier_and_symbol_response(self):
        source,p,target,rows=self.fixture();cluster=source.clusters[0]
        q,details=rebase_cluster(cluster,p[source.indices[0]],target.clusters[0])
        self.assertGreater(max(abs(d['carrier_intercept_shift_rad']) for d in details),.1)
        for (expected,_),(actual,_) in zip(cluster.response_blocks(p[source.indices[0]]),target.clusters[0].response_blocks(q)):
            np.testing.assert_allclose(actual,expected[rows],rtol=2e-12,atol=2e-14)
        np.testing.assert_allclose(target.bands[0].field.y,source.bands[0].field.y[
            (source.bands[0].field.time>=target.curve.epoch-target.curve.scale)&
            (source.bands[0].field.time<target.curve.epoch+target.curve.scale)],atol=2e-13)

    def test_unrepresentable_path_is_rejected_instead_of_silently_smoothed(self):
        source,p,target,rows=self.fixture()
        # A truly cubic path cannot be represented by a quadratic replacement.
        from shared_echo_cluster import SharedEchoCluster
        target=SharedEchoCluster(target.bands,source.origins)
        # Expose its explicit quadratic local layout to this mixed-path helper.
        for cluster in target.clusters:
            cluster.arcs=[cluster.base]*cluster.components
            cluster.slices=[slice(4*s,4*s+4) for s in range(cluster.components)]
        with self.assertRaisesRegex(ValueError,'temporal basis'):
            rebase_cluster(source.clusters[0],p[source.indices[0]],target.clusters[0])

    def test_complete_window_keeps_whole_original_bins_inside_the_basis_support(self):
        source,p,_,_=self.fixture();old=source.bands[0]
        first,last=old.epoch-.64*old.scale,old.epoch+.26*old.scale
        new,part,rows=window_band(old,first,last,complete_rows=True)
        self.assertGreaterEqual(new.field.data['row_time'][0],first)
        self.assertLess(new.field.data['row_time'][-1],last)
        np.testing.assert_array_equal(new.field.data['row_information'],old.field.data['row_information'][rows])
        keep=np.isin(old.field.time,new.field.time)
        np.testing.assert_array_equal(new.field.data['value'],old.field.data['value'][keep])
        from amplitude_arc import linear_basis
        amplitude=.7+.2*np.sin(np.arange(len(old.knots)))
        restricted=linear_basis(new.knots,old.knots)@amplitude
        np.testing.assert_allclose(new.amplitude_basis@restricted,old.amplitude_basis[rows]@amplitude,atol=3e-16)


if __name__=='__main__':unittest.main()
