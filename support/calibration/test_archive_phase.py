"""Independent point-echo checks across the archived-profile analysis boundary.

These start after production range compression. They complement, and do not
replace, the GPU matched-filter and retiming tests against time-domain oracles.
"""
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from phase_history import C, FS, extract, index_archive
from range_likelihood import extract as extract_patch


class ArchivePhaseTests(unittest.TestCase):
    def test_five_band_physical_echo_keeps_phase_through_fractional_extraction(self):
        rf = np.array([177.5, 184.5, 191.625, 219.5, 226.5])*1e6
        transmitters = np.array([[31600., -3030., 640.], [31700., -2815., 625.],
                                 [31800., -2750., 610.]])
        assignment = [0, 2, 1, 2, 0]
        epoch, start, end = 60., 59.9, 60.1
        frequency = np.fft.fftfreq(1024)
        spectrum = (abs(frequency) < .37)*(1+.55*np.sin(9*frequency))
        # Asymmetric positive spectral power gives a complex range PSF; a
        # real sinc alone would miss some conjugation and phase-origin errors.
        inner_time = (np.arange(64)+.5)/64-.5
        records, seeds, expected, expected_patches = [], [], [], []
        offsets = np.arange(-8, 9)
        for channel, carrier in enumerate(rf):
            time = epoch+np.linspace(-.0895, .0895, 180)+channel*.000023
            dt = time-epoch
            position = np.array([-650., 120., 5.])[None, :]+dt[:, None]*[9., -2., .1]
            position += .5*dt[:, None]**2*[.7, .2, 0.]
            tx = transmitters[assignment[channel]]
            length = np.linalg.norm(position-tx, axis=1)+np.linalg.norm(position, axis=1)-np.linalg.norm(tx)
            offset = (channel-2)*3.7
            coefficients = np.polynomial.polynomial.polyfit(dt/15, length+offset, 4)
            seed = {'epoch': epoch, 'scale': 15., 'duration_s': end-start,
                    'path': coefficients.tolist(), 'rf_hz': carrier}
            speed = np.polynomial.polynomial.polyval(dt/15, np.polynomial.polynomial.polyder(coefficients))/15
            doppler = -carrier*speed/C
            omega = 2*np.pi*doppler*.001024
            tone = np.exp(1j*omega[:, None]*inner_time)
            moment0, moment1 = tone.mean(axis=1), (tone*inner_time).mean(axis=1)
            amplitude = (1+.2*channel)*np.exp(.73j)
            pulse = np.fft.ifft(spectrum[None, :]*np.exp(-2j*np.pi*(length*FS/C)[:, None]*frequency), axis=1)*1024/spectrum.sum()
            echo = amplitude*np.exp(-2j*np.pi*carrier/C*length)[:, None]*pulse
            profile = np.array([moment0[:, None]*echo, moment1[:, None]*echo], dtype='<c8')
            valid = np.arange(len(time)) % 11 != 3
            rows = [[int(i), float(t-4095.5/FS), 0, 0, float(1.9+.03*i), int(v)]
                    for i, (t, v) in enumerate(zip(time, valid))]
            metadata = {'channel': channel, 'moments': 2, 'size': 1024, 'count': len(time),
                        'information': 1.5, 'rows': rows}
            records.append((metadata, profile.tobytes()))
            seeds.append(seed)
            fractional_response = np.sum(spectrum*np.exp(2j*np.pi*frequency*offset*FS/C))/spectrum.sum()
            phase_origin = np.exp(-2j*np.pi*carrier/C*(coefficients[0]-offset))
            expected.append(amplitude*phase_origin*fractional_response*(moment0-1j*omega*moment1)[valid])
            patch_response = np.sum(spectrum[None, :]*np.exp(2j*np.pi*frequency[None, :]
                *(offset*FS/C+offsets[:, None])),axis=1)/spectrum.sum()
            expected_patches.append(amplitude*phase_origin*(moment0-1j*omega*moment1)[valid,None]
                                    *patch_response[None,:])
        with TemporaryDirectory() as directory:
            path = Path(directory)/'point.dvblive'
            with path.open('wb') as output:
                header = json.dumps({'applied_channels': [{'rf_hz': float(f)} for f in rf]}).encode()
                output.write(b'DVBTLIV1'+struct.pack('<I', len(header))+header)
                for metadata, payload in records:
                    raw = json.dumps(metadata).encode()
                    output.write(struct.pack('<4sIQ', b'PROF', len(raw), len(payload))+raw+payload)
                output.write(struct.pack('<4sIQ', b'DONE', 2, 0)+b'{}')
            index = index_archive(path, Path(directory)/'index.json')
            phase_errors = []
            for channel in range(5):
                result = extract(path, index, channel, seeds[channel], start, end, .01)
                ratio = result['row_value']/expected[channel]
                phase_errors.extend(np.angle(ratio))
                self.assertLess(np.max(abs(abs(ratio)-1)), 2e-4)
                self.assertEqual(len(ratio), 163)
                patch = extract_patch(path,index,{'start':start,'end':end,'channels':seeds},channel,offsets)
                np.testing.assert_allclose(patch['row_value'],expected_patches[channel],rtol=2e-4,atol=4e-5)
            self.assertLess(np.max(abs(np.degrees(phase_errors))), .01)


if __name__ == '__main__':
    unittest.main()
