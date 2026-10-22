"""Coherent addition of complex BP images on one physical body-coordinate grid.

Linearity makes this equal to one backprojection over the concatenated channels.
Default gains are one: a declared nominal calibration, not a phase fit to peaks.
No magnitudes, colour scales, or local image normalization enter the sum.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def combine(amplitudes, information, gains=None):
    amplitudes, information = np.asarray(amplitudes), np.asarray(information)
    if (amplitudes.shape != information.shape or amplitudes.ndim < 2 or
            not np.isfinite(amplitudes).all() or not np.isfinite(information).all() or
            np.any(information <= 0)):
        raise ValueError('Complex images need matching finite positive information')
    if gains is None:
        gains = np.ones(amplitudes.shape[0], complex)
    gains = np.asarray(gains)
    if gains.shape != (amplitudes.shape[0],) or not np.isfinite(gains).all() or not np.any(abs(gains)):
        raise ValueError('Need one finite response coefficient per channel')
    gains = gains.reshape((-1,)+(1,)*(amplitudes.ndim-1))
    total = np.sum(information*abs(gains)**2, axis=0)
    image = np.sum(gains.conj()*information*amplitudes, axis=0)/total
    independent_energy = np.sum(information*abs(amplitudes)**2, axis=0)
    coherent_energy = total*abs(image)**2
    efficiency = np.divide(coherent_energy, independent_energy, out=np.zeros_like(total),
                           where=independent_energy > 0)
    return image, total, efficiency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    amplitudes, weights = [], []
    along = port = None
    for channel in range(5):
        with np.load(args.images/f'channel-{channel}-image.npz') as source:
            if along is not None and (not np.array_equal(along, source['along_m']) or
                                      not np.array_equal(port, source['port_m'])):
                raise ValueError('Coherent images must use exactly the same physical grid')
            along, port = source['along_m'], source['port_m']
            amplitudes.append(source['complex_image'])
            weights.append(source['information'])
    amplitudes, weights = np.array(amplitudes), np.array(weights)
    image, information, efficiency = combine(amplitudes, weights)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output/'combined-image.npz', along_m=along, port_m=port,
             complex_image=image, information=information, coherent_to_independent_energy=efficiency)
    power = np.concatenate([8192*weights*abs(amplitudes)**2,
                            (8192*information*abs(image)**2)[None]])
    db = 10*np.log10(np.maximum(power, 1e-30))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey=True, layout='constrained')
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC', 'Coherent sum · nominal calibration']
    for axis, name, values in zip(axes.ravel(), names, db):
        handle = axis.imshow(values, origin='lower', extent=[along[0], along[-1], port[0], port[-1]],
                             vmin=db.max()-35, vmax=db.max(), cmap='magma', interpolation='nearest', aspect='equal')
        axis.set(title=name, xlabel='Forward from AIS antenna prior (m)', ylabel='Port (m)')
    figure.colorbar(handle, ax=axes.ravel().tolist(), label='Matched power / conditional thermal variance (dB)')
    figure.suptitle('Complex five-channel backprojection baseline\nOriginal carrier phases; licensed antenna coordinates; no fitted channel gains or autofocus')
    figure.savefig(args.output/'combined-images.png', dpi=150)
    plt.close(figure)
    result = {'source_images': str(args.images), 'gains_real_imag': [[1., 0.]]*5,
              'operator': 'Sum of information-weighted complex BP images, divided by total information. Equivalent to the concatenated-sample matched filter for unit channel responses.',
              'maximum_matched_power_db': dict(zip(names, db.max(axis=(1, 2)).tolist())),
              'calibration_validated': False,
              'scope': 'Nominal coherent baseline. Preserves relative phases and physical coordinates; this does not establish a shared scattering response or accepted phase centres. No post-hoc phase alignment to the brightest pixel.'}
    (args.output/'combination.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
