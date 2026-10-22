"""Display independent complex ISAR images on one fixed, labelled power scale."""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--title', required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    channels, images = [], []
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    for channel in range(5):
        path = args.directory/f'channel-{channel}-image.npz'
        if not path.exists():
            continue
        with np.load(path) as data:
            images.append(8192*abs(data['complex_image'])**2*data['information'])
            along, port = data['along_m'], data['port_m']
        channels.append(channel)
    db = [10*np.log10(np.maximum(a, 1e-30)) for a in images]
    upper = max(a.max() for a in db)
    fig, axes = plt.subplots(len(db), 1, figsize=(13, 3.7*len(db)), sharex=True, sharey=True, layout='constrained')
    for axis, channel, image in zip(np.atleast_1d(axes), channels, db):
        handle = axis.imshow(image, origin='lower', extent=(along[0], along[-1], port[0], port[-1]),
                             vmin=upper-35, vmax=upper, cmap='magma', aspect='equal', interpolation='nearest')
        axis.set_title(names[channel], fontsize=11)
        axis.set_ylabel('Port offset (m)')
    np.atleast_1d(axes)[-1].set_xlabel('Forward offset from AIS antenna prior (m)')
    fig.colorbar(handle, ax=np.atleast_1d(axes).tolist(), label='Matched power / conditional thermal variance (dB)')
    fig.suptitle(args.title+'\nCourse-based attitude hypothesis; antenna calibration and motion refinement pending.\n'
                 'Fixed colour scale across the displayed channels; no DC notch.', fontsize=12)
    fig.savefig(args.output, dpi=140)


if __name__ == '__main__':
    main()
