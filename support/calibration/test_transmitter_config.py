"""Site-policy and physical-height regressions shared by road and ship fitting."""
import copy
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]/'isar'))
from geometry import ecef, enu_rotation
from ship_site_signature import licensed_options
from site_hypotheses import licensed_geometry
from transmitter_config import candidate_indices, configuration, nominal_indices


class TransmitterConfigurationTests(unittest.TestCase):
    def fixture(self, directory):
        config = copy.deepcopy(configuration())
        receiver = ecef(-37.81040454194318, 144.98744506389454, 41.438749)
        rotation = enu_rotation(-37.81040454194318, 144.98744506389454)
        absolute = np.array([ecef(site['location']['latitude_deg'], site['location']['longitude_deg'],
                                  605.+site['antenna_agl_m']) for site in config['transmitters']])
        positions = (absolute-receiver)@rotation.T
        descriptions = [{'id':s['id'], 'acma_site_id':s['acma_site_id'],
                         'antenna_agl_m':s['antenna_agl_m']} for s in config['transmitters']]
        geometry = {'receiver_ecef_m':receiver.tolist(), 'ecef_to_enu':rotation.tolist(),
                    'enu_transmitters_m':positions.tolist(), 'transmitters':descriptions,
                    'site_configuration':config}
        source = {'sites':{}, 'devices':[]}
        for site in config['transmitters']:
            source['sites'][site['acma_site_id']] = {'LATITUDE':site['location']['latitude_deg'],
                                                   'LONGITUDE':site['location']['longitude_deg']}
        for signal in config['signals']:
            for site in config['transmitters']:
                # Deliberately conflicting licensed heights and an extra site:
                # neither may override the supplied physical priors/policy.
                source['devices'].append({'CALL_SIGN':signal['callsign'], 'SITE_ID':site['acma_site_id'],
                    'HEIGHT':'42', 'STATION_TYPE':'Main', 'SDD_ID':'synthetic'})
        main = {'site_positions_enu_m':positions.tolist(), 'channel_offsets_enu_m':np.zeros((5,3)).tolist(),
                'channel_sites':nominal_indices(geometry)}
        for name, data in (('geometry-prior',geometry),('sites-main',main),('acma-transmitters',source)):
            (directory/(name+'.json')).write_text(json.dumps(data))
        return geometry, absolute

    def test_all_32_assignments_use_exact_same_physical_site_centres(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            geometry, absolute=self.fixture(directory)
            choices=list(itertools.product(*candidate_indices(geometry)))
            self.assertEqual(len(choices),32)
            options=licensed_options(directory)
            self.assertEqual(len(options),10)
            for assignment in choices:
                centres, offsets, _=licensed_geometry(directory, directory, assignment)
                np.testing.assert_array_equal(offsets,0.)
                resolved=np.array(geometry['receiver_ecef_m'])+(centres[list(assignment)]+offsets)@geometry['ecef_to_enu']
                np.testing.assert_allclose(resolved,absolute[list(assignment)],rtol=0,atol=2e-9)
                ship=np.array([next(o['transmitter_ecef_m'] for o in options if o['channel']==channel and
                    o['site_id']==geometry['transmitters'][site]['acma_site_id']) for channel,site in enumerate(assignment)])
                np.testing.assert_allclose(ship,resolved,rtol=0,atol=2e-9)
            with self.assertRaisesRegex(ValueError,'candidate set'):
                licensed_geometry(directory,directory,[0,2,0,2,2])

    def test_missing_licensed_candidate_fails_instead_of_silently_shrinking_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            self.fixture(directory)
            path=directory/'acma-transmitters.json'
            source=json.loads(path.read_text())
            source['devices']=[d for d in source['devices'] if (d['CALL_SIGN'],d['SITE_ID'])!=('HSV6','12015')]
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError,'exactly one licensed device'):
                licensed_options(directory)


if __name__=='__main__':
    unittest.main()
