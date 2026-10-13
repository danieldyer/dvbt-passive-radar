#!/usr/bin/env python3
"""Streaming reader for scientific live archives; no video/display dependency."""
import argparse
import json
import pathlib
import struct


def exact(source, count):
    value = source.read(count)
    if len(value) != count:
        raise ValueError('Truncated live archive')
    return value


def read_header(source):
    if exact(source, 8) != b'DVBTLIV1':
        raise ValueError('Invalid live archive magic')
    size, = struct.unpack('<I', exact(source, 4))
    if size > 1024 * 1024:
        raise ValueError('Oversized live header')
    return json.loads(exact(source, size))


def records(source):
    """Yield (tag, metadata, payload) with owned little-endian payload bytes.

    PROF: complex64[2, count, 1024], then stationary complex64[6817]
          when stationary_carriers is nonzero, response complex64[6817], and
          physical carrier-noise float32[6817]. Row validity stays explicit.
    CHIQ: complex64[channels, count]. CLCK gives immutable [start, offset, rate].
    """
    while True:
        tag = source.read(4)
        if not tag:
            raise ValueError('Missing DONE record: capture was interrupted')
        if len(tag) != 4:
            raise ValueError('Truncated record tag')
        size, count = struct.unpack('<IQ', exact(source, 12))
        if size > 1024 * 1024 or count > 64 * 1024 * 1024:
            raise ValueError('Oversized live record')
        metadata = json.loads(exact(source, size))
        payload = exact(source, count)
        if tag == b'PROF':
            expected = (metadata['moments'] * metadata['count'] * metadata['size'] + metadata['stationary_carriers'] + metadata.get('response_carriers', 0) + metadata.get('cancellation_variance_carriers', 0)) * 8 + metadata.get('noise_carriers', 0) * 4
            if count != expected or len(metadata['rows']) != metadata['count']:
                raise ValueError('Invalid profile record dimensions')
        elif tag == b'CHIQ' and count != metadata['channels'] * metadata['count'] * 8:
            raise ValueError('Invalid channel record dimensions')
        yield tag.decode('ascii'), metadata, payload
        if tag == b'DONE':
            if source.read(1):
                raise ValueError('Unexpected bytes after DONE')
            break


def summarize(path):
    with open(path, 'rb') as source:
        header = read_header(source)
        summary = {'mode': header['archive_mode'], 'records': {}, 'profiles': {}, 'gaps': [], 'clock_segments': 0}
        for tag, metadata, payload in records(source):
            summary['records'][tag] = summary['records'].get(tag, 0) + 1
            if tag == 'GAPS':
                summary['gaps'].append(metadata)
            elif tag == 'CLCK':
                summary['clock_segments'] += len(metadata['segments'])
            elif tag == 'PROF':
                channel = summary['profiles'].setdefault(str(metadata['channel']), {'rows': 0, 'valid': 0, 'segments': {}})
                channel['rows'] += metadata['count']
                channel['valid'] += sum(row[5] for row in metadata['rows'])
                segment = channel['segments'].setdefault(str(metadata['segment']), [float('inf'), -float('inf')])
                for row in metadata['rows']:
                    segment[0] = min(segment[0], row[1])
                    segment[1] = max(segment[1], row[1])
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=pathlib.Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.archive), indent=2))
