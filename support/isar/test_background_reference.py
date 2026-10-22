import unittest

import numpy as np

from background_reference import profile_means,combine_means,field_reference
from field import average,interpolate,FS
from motion import C
from test_physical_pair import fixture


class BackgroundReferenceTests(unittest.TestCase):
    def test_weighted_mean_ignores_erasures_and_outside_intervals(self):
        rng=np.random.default_rng(701);values=rng.normal(size=(2,30,100))+1j*rng.normal(size=(2,30,100))
        time=np.arange(30)/10;information=1+time;valid=np.ones(30,bool);valid[6:8]=False
        aperture=dict(time_s=time,information=information,valid=valid,moments=values)
        rows=profile_means(aperture,[(.2,1.2),(1.8,2.5)],chunk_rows=3)
        select=((time>=.2)&(time<1.2))|((time>=1.8)&(time<2.5));w=information*valid*select
        expected=np.einsum('n,mnr->mr',w,values)/w.sum()
        np.testing.assert_allclose(combine_means(rows),expected,atol=1e-15)
        altered=values.copy();altered[:,~(select&valid),:]=1e10
        changed=profile_means({**aperture,'moments':altered},[(.2,1.2),(1.8,2.5)],chunk_rows=7)
        np.testing.assert_allclose(combine_means(changed),expected,atol=1e-15)

    def test_reference_preserves_phase_c1_and_original_linear_operator(self):
        model,_=fixture();f=model.bands[0].field;d=f.data
        coordinate=d['centre_path_m']*FS/C
        first=int(np.floor(coordinate.min()))-50;size=int(np.ceil(coordinate.max()))-first+51
        rng=np.random.default_rng(88);mean=rng.normal(size=(2,size))+1j*rng.normal(size=(2,size))
        time=np.linspace(0,2,20);scale=1+time
        moments=np.stack([mean*(1+.2*np.sin(t)) for t in time],axis=1)
        reference=profile_means(dict(time_s=time,information=scale,valid=np.ones(20,bool),moments=moments),[(0,3)])[0]['mean']
        expected=np.zeros_like(f.y)
        # Transform individual reference realizations before averaging, using
        # the exact field's moment/phase operator. It must commute with this
        # independent fixed-profile mean, even for nonzero C1.
        for i in range(len(time)):
            constant=np.broadcast_to(moments[:,i,None,:],(2,len(coordinate),size))
            zero,one=interpolate(constant,coordinate-first,f.offsets)
            raw=(zero-1j*d['centre_omega'][:,None]*one)*np.exp(1j*f.k*d['centre_path_m'][:,None])
            expected+=scale[i]*f.apply(average(raw,d['row_information'],f.metadata['block_rows'])[0])
        expected/=scale.sum()
        np.testing.assert_allclose(field_reference(f,reference,first),expected,atol=2e-12,rtol=1e-12)


if __name__=='__main__':unittest.main()
