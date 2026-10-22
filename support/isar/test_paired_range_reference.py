import json,unittest
import numpy as np
from field import Field
from full_path_point import PointBand
from paired_range_reference import combine
from rigid_patch import profile_static
from sparse_positive_envelope import Normal
from scipy.sparse import csr_matrix
from audit_ship_scattering_envelopes import linear_basis
from test_full_path_point import problem
from motion import excess_path


class PairedRangeReferenceTests(unittest.TestCase):
    def test_two_range_grids_preserve_physical_carrier_across_all_bands(self):
        base=problem();old=base.field;t=old.time;centre=base.anchor;velocity=base.velocity
        for channel,rf in enumerate([177.5e6,184.5e6,191.625e6,219.5e6,226.5e6]):
            sources=[]
            tx=base.motion.tx[channel]
            for shift in [-40.,90.]:
                data=dict(old.data);observer=centre[0]+base.direction[0]*shift
                length=excess_path(observer[None],tx,base.motion.rx)[0][0]
                data.update(centre_ecef_m=np.broadcast_to(observer,centre.shape).copy(),centre_path_m=np.full(len(t),length),centre_omega=np.zeros(len(t)),centre_velocity_ecef_mps=np.zeros_like(centre),transmitter_ecef_m=tx)
                meta=json.loads(data['metadata_json']);meta['rf_hz']=rf;data['metadata_json']=json.dumps(meta)
                f=Field(data,old.pulse.samples);raw=f.raw_trajectory(centre,velocity)
                f.data['value']=raw;sources.append(f)
            before=[s.data['value'].copy() for s in sources]
            merged,coordinate=combine(*sources,base.coordinate,[4.,12.])
            expected=merged.raw_trajectory(centre,velocity)
            np.testing.assert_allclose(merged.data['value'],expected,rtol=1e-10,atol=2e-12)
            for source,value in zip(sources,before):np.testing.assert_array_equal(source.data['value'],value)
            selected=(t>=4)&(t<12);np.testing.assert_array_equal(coordinate['stationary_groups'],selected.astype(int))

    def test_grouped_normal_matrix_matches_explicit_projected_dictionary(self):
        rng=np.random.default_rng(600);t=np.linspace(0,20,501);basis,_=linear_basis(t,[0,20],2.);b=csr_matrix(basis)
        h=rng.normal(size=(len(t),4,2))+1j*rng.normal(size=(len(t),4,2));y=rng.normal(size=(len(t),4))+1j*rng.normal(size=(len(t),4))
        root=1+rng.uniform(size=len(t));mask=np.arange(len(t))%4!=3;group=((t>=4)&(t<15)).astype(int)
        normal=Normal(h,y,root,mask,b,group)
        dictionary=(h[:,:,None,:]*basis[:,None,:,None]).reshape(len(t),4,-1)
        dictionary=profile_static(dictionary,root,mask,group)[mask].reshape(-1,dictionary.shape[-1]);data=profile_static(y,root,mask,group)[mask].ravel()
        value=rng.normal(size=dictionary.shape[-1])+1j*rng.normal(size=dictionary.shape[-1])
        np.testing.assert_allclose(normal.multiply(value),dictionary.conj().T@(dictionary@value),rtol=1e-12,atol=1e-10)
        np.testing.assert_allclose(normal.rhs,dictionary.conj().T@data,rtol=1e-12,atol=1e-10)

if __name__=='__main__':unittest.main()
