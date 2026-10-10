#!/usr/bin/env python3
"""Local live/replay server. DSP output is replaced atomically; clients never read torn maps."""
import argparse
import functools
import http.server
import json
import math
import pathlib
import struct
import tempfile
import urllib.parse
import numpy as np


def read_map(path):
    # Keep this descriptor open: an atomic producer rename may replace the pathname.
    source = open(path, 'rb')
    magic, length, size = struct.unpack('<4sIQ', source.read(16))
    if magic != b'RDM1' or length > 1024*1024:
        source.close()
        raise ValueError('Invalid RDM header')
    meta = json.loads(source.read(length))
    if size != meta['ranges'] * meta['frequencies'] * 8:
        source.close()
        raise ValueError('Invalid RDM dimensions')
    data = np.memmap(source, dtype='<c8', mode='r', offset=16+length,
                     shape=(meta['ranges'], meta['frequencies']))
    return source, meta, data


def dc_projection(meta, first, last):
    """Spectrum of a unit constant through the native, possibly padded Hann FFT.

    Z(f) - Z(0) W(f)/W(0) is exactly the transform of Hann-weighted mean
    subtraction in slow time. Use the original zero bin even when zoomed away
    from DC, and subtract complex amplitudes before pooling or taking power.
    """
    if meta.get('window') != 'hann':
        raise ValueError('DC removal requires a Hann-window complex map')
    samples = meta['samples']
    size = meta.get('fft_size', round(samples/(meta['integration_s']*meta['frequency_step_hz'])))
    zero = round(-meta['frequency_start_hz']/meta['frequency_step_hz'])
    if not 8 <= samples <= size or not 0 <= zero < meta['frequencies']:
        raise ValueError('Invalid DC projection dimensions')
    # Match the stored float window coefficients, including their normalization.
    window = (.5-.5*np.cos(2*np.pi*np.arange(samples)/samples)).astype(np.float32)
    spectrum = np.fft.fft(window.astype(np.float64), n=size)/window.sum(dtype=np.float64)
    kernel = spectrum[(np.arange(first,last)-zero)%size]
    return zero, kernel


def preview(path, query):
    source, meta, data = read_map(path)
    try:
        def number(key, default):
            v = float(query.get(key, [default])[0])
            if not math.isfinite(v):
                raise ValueError('Non-finite limit')
            return v
        nr, nf = data.shape
        rs, fs = meta['range_step_m'], meta['frequency_step_hz']
        f0 = meta['frequency_start_hz']
        r0 = max(0, min(nr-1, math.floor(number('rmin', 0)/rs)))
        r1 = max(r0+1, min(nr, math.ceil(number('rmax', nr*rs)/rs)+1))
        d0 = max(0, min(nf-1, math.floor((number('fmin', f0)-f0)/fs)))
        d1 = max(d0+1, min(nf, math.ceil((number('fmax', f0+nf*fs)-f0)/fs)+1))
        width = max(16, min(2048, int(number('width', 1024))))
        height = max(16, min(2048, int(number('height', 768))))
        remove_dc = number('remove_dc', 0)
        if remove_dc not in (0, 1):
            raise ValueError('remove_dc must be 0 or 1')
        native_dc = meta.get('dc_removed', False)
        kernel = None
        if remove_dc and not native_dc:
            zero, kernel = dc_projection(meta, d0, d1)
        # Maximum power pooling preserves weak narrow peaks when the display has
        # fewer pixels than the scientific map. Zoom queries use the native bins.
        rd, fd = max(1, math.ceil((r1-r0)/width)), max(1, math.ceil((d1-d0)/height))
        # Bound working memory even for several-minute native maps. Pool a
        # range tile at a time instead of materializing the entire power matrix.
        pooled = np.empty((math.ceil((r1-r0)/rd), math.ceil((d1-d0)/fd)), dtype=np.float32)
        for begin in range(0, pooled.shape[0], 32):
            end = min(begin+32, pooled.shape[0])
            z = data[r0+begin*rd:min(r1,r0+end*rd), d0:d1]
            if kernel is not None:
                dc = data[r0+begin*rd:min(r1,r0+end*rd), zero]
                z = z-dc[:,None]*kernel[None,:]
            power = z.real*z.real + z.imag*z.imag
            if fd > 1:
                power = np.maximum.reduceat(power, np.arange(0, power.shape[1], fd), axis=1)
            if rd > 1:
                power = np.maximum.reduceat(power, np.arange(0, power.shape[0], rd), axis=0)
            pooled[begin:end] = power
        db = (10*np.log10(np.maximum(pooled, 1e-30))).T.astype('<f4')
        meta = dict(meta, native_range_step_m=rs, native_frequency_step_hz=fs,
                    range_start_m=r0*rs, range_step_m=rs*rd, frequency_start_hz=f0+d0*fs,
                    frequency_step_hz=fs*fd, ranges=db.shape[1], frequencies=db.shape[0],
                    format='db32le', order='doppler,range', peak_db=float(db.max()),
                    full_range_max_m=(nr-1)*rs, full_doppler_max_hz=f0+(nf-1)*fs,
                    native_dc_removed=native_dc, dc_removed=bool(remove_dc or native_dc),
                    dc_removal=meta.get('dc_removal','hann_weighted_mean') if native_dc else ('hann_weighted_mean' if remove_dc else 'none'),
                    display_pool='maximum', finished=(path.parent/'metadata.json').exists())
        header = json.dumps(meta, separators=(',', ':')).encode()
        return struct.pack('<4sIQ', b'RDM1', len(header), db.nbytes) + header + db.tobytes()
    finally:
        del data
        source.close()


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if args and str(args[1]) not in ('200', '204', '304'):
            super().log_message(fmt, *args)

    def send(self, code, body=b'', content='application/octet-stream'):
        self.send_response(code)
        self.send_header('Content-Type', content)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        if url.path == '/favicon.ico':
            self.send(204); return
        if url.path == '/api/frame':
            try:
                self.send(200, preview(self.server.maps/'latest.rdm', urllib.parse.parse_qs(url.query)))
            except FileNotFoundError:
                self.send(204)
            except (ValueError, KeyError, struct.error) as error:
                self.send(400, str(error).encode(), 'text/plain')
            return
        if url.path == '/api/native':
            try:
                with open(self.server.maps/'latest.rdm', 'rb') as src:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/octet-stream')
                    self.send_header('Content-Disposition', 'attachment; filename="range-doppler.rdm"')
                    self.send_header('Content-Length', str(src.seek(0, 2)))
                    self.end_headers(); src.seek(0)
                    import shutil
                    shutil.copyfileobj(src, self.wfile)
            except FileNotFoundError:
                self.send(404)
            return
        super().do_GET()

    def do_POST(self):
        # Local UI only; no cross-origin write requests.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            self.send(403); return
        if self.path != '/api/integration':
            self.send(404); return
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 256:
                raise ValueError('Invalid body length')
            seconds = float(json.loads(self.rfile.read(length))['seconds'])
            if not math.isfinite(seconds) or not 0 <= seconds <= 3600:
                raise ValueError('Integration must be 0 (auto) or positive seconds up to 3600')
            if (self.server.maps/'metadata.json').exists():
                self.send(409, b'Recording finished; integration controls apply while the processor is running', 'text/plain'); return
            try:
                source, meta, data = read_map(self.server.maps/'latest.rdm')
                del data; source.close()
                if seconds > meta.get('history_limit_s', 300):
                    raise ValueError('Integration exceeds the processor history limit')
            except FileNotFoundError:
                pass
            target = self.server.maps/'integration.txt'
            with tempfile.NamedTemporaryFile(mode='w', dir=target.parent, delete=False) as file:
                file.write(str(seconds)+'\n')
                temporary = pathlib.Path(file.name)
            temporary.replace(target)
            self.send(200, b'{"accepted":true}', 'application/json')
        except (ValueError, KeyError) as error:
            self.send(400, str(error).encode(), 'text/plain')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('maps', type=pathlib.Path)
    parser.add_argument('--web', type=pathlib.Path, default=pathlib.Path('build/web'))
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    args.maps.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(Handler, directory=str(args.web.resolve()))
    with http.server.ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        server.maps = args.maps.resolve()
        print(f'Viewer: http://127.0.0.1:{args.port}/  |  maps: {server.maps}', flush=True)
        server.serve_forever()

if __name__ == '__main__':
    main()
