"""Plot conditional point responses and measure connected half-power support."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC', 'Coherent five-channel point response']
    files = [args.directory/f'channel-{i}-image.npz' for i in range(5)]+[args.directory/'coherent-point.npz']
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True, sharey=True, layout='constrained')
    result = []
    for axis, name, path in zip(axes.flat, names, files):
        with np.load(path) as data:
            z, x, y = data['complex_image'], data['along_m'], data['port_m']
        power = abs(z)**2
        peak = np.unravel_index(np.argmax(power), power.shape)
        components, _ = label(power >= power[peak]/2, np.ones((3, 3)))
        main = components == components[peak]
        xx, yy = np.meshgrid(x, y)
        result.append({'name': name, 'half_power_axis_aligned_extent_m':
                       [float(np.ptp(xx[main])+x[1]-x[0]), float(np.ptp(yy[main])+y[1]-y[0])],
                       'half_power_area_m2': float(main.sum()*(x[1]-x[0])*(y[1]-y[0]))})
        handle = axis.imshow(10*np.log10(np.maximum(power, 1e-10)), origin='lower',
                             extent=(x[0], x[-1], y[0], y[-1]), vmin=-45, vmax=0,
                             cmap='magma', aspect='equal', interpolation='nearest')
        axis.set(title=name, xlim=(-150, 150), ylim=(-60, 60))
        axis.set_ylabel('Port offset (m)')
    for axis in axes[-1]:
        axis.set_xlabel('Forward offset (m)')
    fig.colorbar(handle, ax=axes.ravel().tolist(), label='Point response power (dB relative to unit echo)')
    fig.suptitle('Known point through the actual 90–210 s aperture and archived spectral weights\n'
                 'Conditional on exact motion, attitude, phase centres and a common five-band scatterer response.', fontsize=12)
    fig.savefig(args.output, dpi=150)
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
