"""Measure the production clock controller's deterministic sinusoidal response.

This is a noiseless clock-following test, not an oscillator noise calibration or
an optimum bandwidth claim. Original observation epochs are passed to the same
controller replay executable used for the recording audit.
"""
import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    a = p.parse_args()
    a.scratch.mkdir(parents=True, exist_ok=True)
    cases = []
    fs, channels, dt = 64e6, 5, .001
    record = np.dtype([('time', '<f8'), ('value', '<f8'), ('variance', '<f8'), ('channel', '<u8')])
    with tempfile.TemporaryDirectory(prefix='clock-bandwidth-', dir=a.scratch) as temporary:
        root = Path(temporary)
        for amplitude_ns in [1., 10., 100.]:
            for frequency in [.03, .1, .3, .5, 1., 1.5, 2., 3., 5., 10.]:
                duration = max(20., 12/frequency)
                time = np.arange(1, int(duration/dt)+1)*dt
                rows = np.empty(len(time)*channels, dtype=record)
                rows['time'] = np.repeat(time, channels)
                rows['value'] = np.repeat(fs*amplitude_ns*1e-9*np.sin(2*np.pi*frequency*time), channels)
                rows['variance'] = 1.
                rows['channel'] = np.tile(np.arange(channels), len(time))
                with (root/'observations.bin').open('wb') as f:
                    f.write(b'CLKOBS01'+struct.pack('<IdQ', channels, fs, len(rows)))
                    rows.tofile(f)
                subprocess.run([str(a.probe.resolve()), str(root/'observations.bin'), str(root/'response.csv')],
                               check=True, stdout=subprocess.DEVNULL)
                result = np.genfromtxt(root/'response.csv', delimiter=',', names=True)
                result = result[result['time_s'] >= max(5., duration/2)]
                t = result['time_s']
                design = np.column_stack([np.sin(2*np.pi*frequency*t), np.cos(2*np.pi*frequency*t),
                                          np.ones(len(t)), t-t.mean()])
                fit = np.linalg.lstsq(design, result['offset_samples'], rcond=None)[0]
                gain = np.hypot(fit[0], fit[1])/(fs*amplitude_ns*1e-9)
                cases.append(dict(amplitude_ns=amplitude_ns, frequency_hz=frequency, gain=float(gain),
                                  phase_degrees=float(np.degrees(np.arctan2(fit[1], fit[0]))),
                                  residual_rms_ns=float(np.std(result['offset_samples']-design@fit)/fs*1e9)))
    a.output.write_text(json.dumps(dict(scope=__doc__,observation_interval_s=dt,update_interval_s=.032,
        command_lead_s=.002,channels=channels,noise_injected=False,cases=cases),indent=2)+'\n')
    print(json.dumps(cases),flush=True)


if __name__ == '__main__':
    main()
