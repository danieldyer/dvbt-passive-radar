import json
import unittest
import numpy as np
from ship_arc_coherence import archived_observables
from motion import C


class ArchivedReferenceTests(unittest.TestCase):
    def test_reference_rate_curvature_and_wavelength_come_from_saved_rows(self):
        time=400.+np.arange(2001)*.001
        coefficients=np.array([14500.,-.7,.03,-.001])
        u=time-401.
        path=np.polynomial.polynomial.polyval(u,coefficients)
        rate=np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyder(coefficients))
        for rf in (177.5e6,226.5e6):
            data={'row_time':time,'centre_path_m':path,'centre_omega':-2*np.pi*rf/C*.001024*rate,
                  'metadata_json':json.dumps({'rf_hz':rf})}
            np.testing.assert_allclose(archived_observables(data,401.),[14500.,-.7,.06],atol=1e-11)
            with self.assertRaises(ValueError):archived_observables(data,399.)


if __name__=='__main__':unittest.main()
