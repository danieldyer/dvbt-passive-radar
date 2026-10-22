import unittest

import numpy as np

from field import Field,average
import test_waveform_kernels as kernel_fixture
from waveform_kernels import bind_subset


class KernelSubsetTests(unittest.TestCase):
    def test_original_kernel_identity_survives_different_averaging_boundaries(self):
        original=kernel_fixture.KernelFieldTests().field();d=original.data;rows=slice(8,56)
        selected=dict(d)
        for key in ('row_time','row_information','centre_ecef_m','centre_path_m',
                'centre_velocity_ecef_mps','centre_omega','body_to_ecef'):
            selected[key]=d[key][rows]
        selected['time'],selected['information']=average(selected['row_time'],selected['row_information'],16)
        selected['value']=np.zeros((3,17),complex)
        base=Field(selected,original.pulse.samples)
        mapped=bind_subset(base,d['row_time'],d['row_information'],original.pulse.kernels.samples,
            np.ones(4,bool))
        np.testing.assert_array_equal(mapped.pulse.row_blocks,np.arange(8,56)//16)
        position=d['centre_ecef_m']+np.array([31.,-14.,8.]);velocity=d['centre_velocity_ecef_mps']
        raw=original.raw_trajectory(position,velocity)
        expected=base.apply(average(raw[rows],selected['row_information'],16)[0])
        np.testing.assert_allclose(mapped.trajectory(position[rows],velocity[rows]),expected,atol=2e-12,rtol=2e-13)
        np.testing.assert_array_equal(mapped.y,base.y)
        np.testing.assert_array_equal(mapped.whiten,base.whiten)

    def test_missing_waveforms_or_changed_row_information_are_rejected(self):
        field=kernel_fixture.KernelFieldTests().field();d=field.data
        for valid,information in (([True,False,True,True],d['row_information']),
                (np.ones(4,bool),d['row_information']*1.01)):
            with self.assertRaises(ValueError):
                bind_subset(field,d['row_time'],information,field.pulse.kernels.samples,valid)


if __name__=='__main__':unittest.main()
