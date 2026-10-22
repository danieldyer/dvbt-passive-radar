"""Independent observation-space reference for the saved finite C0/C1 operator.

Noise is proper-complex and diagonal in the measured occupied FFT carriers.
This explicitly retains the realised symbols, previous-symbol context, fitted
response and fixed range/Doppler combination. It does not estimate a channel,
motion, clutter covariance or phase correction from the tested observation.
"""
import json
import struct

import numpy as np

NFFT, BINS, FS = 8192, 6817, 8e6


def _read(stream, size):
    value = stream.read(size)
    if len(value) != size:
        raise ValueError('Truncated finite-reference archive')
    return value


def probes(path):
    """Read every time-selected probe, checking dimensions and completion."""
    with path.open('rb') as stream:
        if _read(stream, 8) != b'FREF0001':
            raise ValueError('Not a finite-reference archive')
        size, = struct.unpack('<I', _read(stream, 4))
        if not 0 < size <= 1024*1024:
            raise ValueError('Invalid archive header length')
        json.loads(_read(stream, size))
        count = 0
        while True:
            tag, length, size = struct.unpack('<4sIQ', _read(stream, 16))
            if not 0 < length <= 1024*1024:
                raise ValueError('Invalid record header length')
            meta = json.loads(_read(stream, length))
            if tag == b'DONE':
                if size or meta['rows'] != count or stream.read(1):
                    raise ValueError('Invalid completion record')
                return
            if tag != b'ROW0' or size != 36*BINS+16*1024:
                raise ValueError('Invalid finite-reference record dimensions')
            data, offset = _read(stream, size), 0
            for name, dtype, n in [('previous', '<c8', BINS), ('current', '<c8', BINS),
                    ('response', '<c8', BINS), ('variance', '<f4', BINS),
                    ('measured', '<c8', BINS), ('zero', '<c8', 1024), ('one', '<c8', 1024)]:
                meta[name] = np.frombuffer(data, dtype, n, offset)
                offset += n*np.dtype(dtype).itemsize
            if not np.isfinite(meta['information']) or meta['information'] <= 0 or np.any(meta['variance'] <= 0):
                raise ValueError('Invalid fixed likelihood weights')
            count += 1
            yield meta


def useful(spectrum, fraction, nfft=NFFT):
    k = np.arange(len(spectrum))-(len(spectrum)-1)//2
    packed = np.zeros(nfft, complex)
    packed[k % nfft] = spectrum*np.exp(-2j*np.pi*k*fraction/nfft)
    return np.fft.ifft(packed)


def segment(current, previous, origin, start, stop, *, guard=16, nfft=NFFT,
            previous_fraction=None):
    """Reference at integer coordinates, using production fractional-origin convention.

    An optional previous_fraction reproduces the GPU's linear correction of
    the adjacent row's fractional clock. Otherwise the previous symbol is
    evaluated with an exact Fourier phase shift at the current row's clock.
    """
    integer = int(np.floor(origin))
    fraction = origin-integer
    current_time = useful(current, fraction, nfft)
    previous_time = useful(previous, fraction if previous_fraction is None else previous_fraction, nfft)
    coordinate = np.arange(start, stop)-integer
    if coordinate.min() < -nfft-nfft//guard or coordinate.max() >= nfft:
        raise ValueError('Reference needs more than previous/current symbol context')
    old = coordinate < -nfft//guard
    index = (coordinate+nfft//guard) % nfft
    if previous_fraction is None:
        old_values = previous_time[index]
    else:
        delta = previous_fraction-fraction
        shift = int(np.floor(delta))
        mix = delta-shift
        old_values = (1-mix)*previous_time[(index+shift) % nfft]+mix*previous_time[(index+shift+1) % nfft]
    return np.where(old, old_values, current_time[coordinate % nfft])


def range_weights(coordinate):
    integer = int(np.floor(coordinate))
    tap = np.arange(-31, 33)
    delta = coordinate-integer-tap
    weight = np.sinc(delta)*np.i0(8*np.sqrt(np.maximum(0, 1-(delta/32)**2)))/np.i0(8)
    return integer, tap, weight/weight.sum()


def references(current, previous, origin, coordinate, offsets, *, guard=16,
               nfft=NFFT, previous_fraction=None):
    """The exact 64-tap linear combination of integer-lag finite references."""
    offsets = np.asarray(offsets)
    if offsets.ndim != 1 or not len(offsets) or not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError('Integer range offsets required')
    centre, tap, weight = range_weights(coordinate)
    first, last = centre+offsets.min()+tap[0], centre+offsets.max()+tap[-1]
    values = segment(current, previous, origin, -last, nfft-first, guard=guard,
                     nfft=nfft, previous_fraction=previous_fraction)
    filtered = np.convolve(values, weight, mode='valid')
    return filtered[np.arange(nfft)[None, :]+offsets.max()-offsets[:, None]]


def coefficient_planes(current, previous, response, variance, origin, coordinate,
                       offsets, *, guard=16, nfft=NFFT, previous_fraction=None):
    reference = references(current, previous, origin, coordinate, offsets, guard=guard,
                           nfft=nfft, previous_fraction=previous_fraction)
    k = np.arange(len(current))-(len(current)-1)//2
    gain = np.sum(abs(response)**2/variance)/nfft
    time = (np.arange(nfft)-(nfft-1)/2)/nfft
    spectra = np.fft.fft(np.stack((reference, reference*time)), axis=-1)[..., k % nfft]
    return spectra.conj()*response.conj()/variance/(nfft*gain)


def echo(current, previous, response, origin, delay, doppler, *, guard=16, nfft=NFFT, rate=FS):
    samples = segment(current, previous, origin+delay, 0, nfft, guard=guard, nfft=nfft)
    time = (np.arange(nfft)-(nfft-1)/2)/rate
    k = np.arange(len(current))-(len(current)-1)//2
    return np.fft.fft(samples*np.exp(2j*np.pi*doppler*time))[k % nfft]*response


def covariance(coefficients, variance):
    return (coefficients*variance)@coefficients.conj().T


def circular_profile(spectral_information, delay, coordinate, offsets, *, nfft=NFFT):
    """Interpolated circular mean; input information already divided by gain."""
    k = np.arange(len(spectral_information))-(len(spectral_information)-1)//2
    packed = np.zeros(nfft, complex)
    packed[k % nfft] = spectral_information*np.exp(-2j*np.pi*k*delay/nfft)
    pulse = np.fft.ifft(packed)
    centre, tap, weight = range_weights(coordinate)
    return pulse[(centre+np.asarray(offsets)[:, None]+tap) % nfft]@weight
