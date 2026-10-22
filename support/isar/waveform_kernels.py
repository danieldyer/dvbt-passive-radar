"""Read immutable known-response kernels and bind them to existing fields."""
import json
from pathlib import Path
import struct

import numpy as np

from block_pulse import BlockPulse, RowPulse
from field import Field
from scene_dictionary import fingerprint


def read_exact(stream, size):
    value = stream.read(size)
    if len(value) != size:
        raise ValueError('Truncated waveform kernel archive')
    return value


def read_kernels(path):
    with Path(path).open('rb') as stream:
        if read_exact(stream, 8) != b'WFKERN01':
            raise ValueError('Not a waveform kernel archive')
        length, = struct.unpack('<I', read_exact(stream, 4))
        if not 0 < length <= 1024*1024:
            raise ValueError('Invalid kernel header length')
        header = json.loads(read_exact(stream, length))
        rows = [[None]*c['blocks'] for c in header['channels']]
        pulses = [np.zeros((c['blocks'], 129), complex) for c in header['channels']]
        count = 0
        while True:
            tag, length, size = struct.unpack('<4sIQ', read_exact(stream, 16))
            if not 0 < length <= 1024*1024:
                raise ValueError('Invalid kernel record metadata')
            meta = json.loads(read_exact(stream, length))
            if tag == b'DONE':
                if (size or meta['blocks'] != count or count != sum(map(len, rows)) or stream.read(1)):
                    raise ValueError('Incomplete or inconsistent kernel export')
                return header, rows, pulses
            if tag != b'KERN' or size != 129*8:
                raise ValueError('Unexpected kernel record')
            channel, block = meta['channel'], meta['block']
            if channel not in range(len(rows)) or block not in range(len(rows[channel])) or rows[channel][block] is not None:
                raise ValueError('Invalid or duplicate block identity')
            pulse = np.frombuffer(read_exact(stream, size), '<c8')
            if not np.isfinite(pulse).all() or meta['information'] < 0:
                raise ValueError('Invalid kernel value/weight')
            rows[channel][block] = meta; pulses[channel][block] = pulse; count += 1


class KernelField(Field):
    """Change only the known forward response; retain recorded data and metric."""
    def __init__(self, base, kernels, row_blocks):
        self.__dict__.update(base.__dict__)
        row_blocks = np.asarray(row_blocks)
        if row_blocks.shape != base.data['row_time'].shape:
            raise ValueError('Kernel identity must accompany every original symbol row')
        self.pulse = RowPulse(kernels, row_blocks, base.pulse.samples)

    def windowed(self, first_s, last_s):
        base = super().windowed(first_s, last_s)
        index = np.searchsorted(self.data['row_time'], base.data['row_time'])
        np.testing.assert_array_equal(self.data['row_time'][index], base.data['row_time'])
        return KernelField(base, self.pulse.kernels, self.pulse.row_blocks[index])


def bind_subset(base, times, information, samples, valid, *, block_rows=16):
    """Bind original known kernels to a field with possibly different bin edges.

    Kernel identities follow original symbol epochs, not the field's retained
    bin number. Missing waveforms are an error here; neither data nor rows are
    silently replaced or dropped for a template comparison.
    """
    times,information,samples,valid=map(np.asarray,(times,information,samples,valid))
    if (times.ndim!=1 or information.shape!=times.shape or np.any(np.diff(times)<=0) or
            len(times)!=block_rows*len(samples) or valid.shape!=(len(samples),)):
        raise ValueError('Kernel blocks and original row identities disagree')
    rows=base.data['row_time'];index=np.minimum(np.searchsorted(times,rows),len(times)-1)
    previous=np.maximum(index-1,0)
    index=np.where(abs(times[previous]-rows)<abs(times[index]-rows),previous,index)
    if np.any(np.diff(index)<=0) or np.max(abs(times[index]-rows))>2e-8:
        raise ValueError('Requested original rows are outside the known kernel archive')
    if not np.array_equal(information[index],base.data['row_information']):
        raise ValueError('Known kernel and observation information disagree')
    blocks=index//block_rows;used,inverse=np.unique(blocks,return_inverse=True)
    if not np.all(valid[used]):raise ValueError('A requested reconstructed waveform is unavailable')
    return KernelField(base,BlockPulse(samples[used]),inverse)


def paired_fields(base, request, header, records, samples, channel):
    """Use identical observations under both kernel hypotheses, with explicit erasures."""
    description = header['channels'][channel]
    if fingerprint(Path(description['field'])) != description['field_sha256']:
        raise ValueError('Original field changed since waveform request')
    with np.load(Path(request)/'rows.npz') as grids:
        grid = grids[f'channel_{channel}']
    np.testing.assert_array_equal(grid['time'], base.data['row_time'])
    np.testing.assert_array_equal(grid['information'], base.data['row_information'])
    block = base.metadata['block_rows']
    if block != header['block_rows'] or len(records) != len(base.time):
        raise ValueError('Coherent averaging identities changed')
    valid = np.array([r['valid'] for r in records])
    expected = grid['information'].reshape(-1, block).sum(axis=1)
    np.testing.assert_allclose([r['information'] for r in records], expected, rtol=2e-15)
    if not valid.all():
        # A missing known waveform excludes the same observed bins in both
        # models. It cannot become a target-dependent normalization or fill.
        data = dict(base.data)
        data['row_information'] = data['row_information']*np.repeat(valid, block)
        data['information'] = data['information']*valid
        base = Field(data, base.pulse.samples)
    kernels = BlockPulse(samples)
    corrected = KernelField(base, kernels, np.arange(len(grid))//block)
    np.testing.assert_array_equal(base.y, corrected.y)
    np.testing.assert_array_equal(base.whiten, corrected.whiten)
    return base, corrected, valid
