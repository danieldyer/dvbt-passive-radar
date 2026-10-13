#!/usr/bin/env python3
"""Exercise the actual FTXUI terminal, continuous replay, native map key and drain."""
import argparse
import codecs
import fcntl
import json
import os
import pathlib
import pty
import select
import signal
import struct
import subprocess
import termios
import time
import pyte
from read_archive import summarize


def snapshot(screen, path):
    path.with_suffix('.txt').write_text('\n'.join(screen.display))
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 17)
    cw, ch = round(font.getlength('M')), 23
    image = Image.new('RGB', (screen.columns*cw, screen.lines*ch), '#0f1319')
    draw = ImageDraw.Draw(image)
    palette = {'black': '#000000', 'red': '#f28482', 'green': '#92ca99', 'brown': '#e6bc5e',
               'blue': '#5281dd', 'magenta': '#b28fd9', 'cyan': '#58c4d3', 'white': '#dce4ef',
               'brightblack': '#808c9b', 'brightwhite': '#ffffff'}
    def colour(value, default):
        if value == 'default':
            return default
        return palette.get(value, '#'+value if len(value) == 6 else default)
    for y in range(screen.lines):
        for x in range(screen.columns):
            cell = screen.buffer[y][x]
            fg, bg = colour(cell.fg, '#dce4ef'), colour(cell.bg, '#0f1319')
            if cell.reverse:
                fg, bg = bg, fg
            draw.rectangle((x*cw, y*ch, (x+1)*cw, (y+1)*ch), fill=bg)
            if len(cell.data) == 1 and 0x2800 <= ord(cell.data) <= 0x28ff:
                # Pillow has no automatic fallback for Menlo's missing Braille
                # glyphs. Draw their exact dot pattern instead of tofu boxes.
                mask = ord(cell.data)-0x2800
                for bit, (dx, dy) in enumerate([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (0, 3), (1, 3)]):
                    if mask & (1 << bit):
                        px, py = x*cw+2+dx*(cw-4), y*ch+3+dy*5
                        draw.ellipse((px, py, px+1, py+1), fill=fg)
            else:
                draw.text((x*cw, y*ch), cell.data, fill=fg, font=font)
    image.save(path.with_suffix('.png'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=pathlib.Path, default=pathlib.Path('build/release/dvbt_capture'))
    parser.add_argument('--metadata', type=pathlib.Path, default=pathlib.Path('recordings/live_mt_dandenong.json'))
    parser.add_argument('--fixture', type=pathlib.Path)
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--archive', choices=['profiles', 'channels', 'both', 'none'], default='none')
    parser.add_argument('--seconds', type=float, default=13)
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--ctrl-c', action='store_true')
    args = parser.parse_args()
    if bool(args.fixture) == args.hardware:
        parser.error('Choose exactly one of --fixture or --hardware')
    args.output.mkdir(parents=True, exist_ok=False)
    master, slave = pty.openpty()
    width, height = 120, 38
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', height, width, 0, 0))
    original = termios.tcgetattr(slave)
    environment = os.environ.copy(); environment['TERM'] = 'xterm-256color'
    environment['COLORTERM'] = 'truecolor'; environment.pop('NO_COLOR', None)
    sdk = pathlib.Path('build/sdk/macOS/share/vulkan/icd.d/MoltenVK_icd.json').resolve()
    if sdk.exists() and not environment.get('VK_ICD_FILENAMES'):
        environment['VK_ICD_FILENAMES'] = str(sdk)
    command = [str(args.binary), '--metadata', str(args.metadata),
               '--output', str(args.output/'capture'), '--archive', args.archive]
    if args.fixture:
        command.extend(['--simulate', str(args.fixture)])
    process = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave, env=environment)
    screen = pyte.Screen(width, height); stream = pyte.Stream(screen)
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    received = bytearray(); running = None; stage = 0; stage_at = time.monotonic()
    def send(keys):
        os.write(master, keys)
    def resize(w, h):
        screen.resize(h, w)
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', h, w, 0, 0))
        process.send_signal(signal.SIGWINCH)
    try:
        deadline = time.monotonic()+100
        while time.monotonic() < deadline:
            now = time.monotonic()
            if select.select([master], [], [], .05)[0]:
                block = os.read(master, 65536)
                if not block:
                    break
                received.extend(block); stream.feed(decoder.decode(block))
                if b'\x1b[6n' in block:
                    send(b'\x1b[1;1R')
            text = '\n'.join(screen.display)
            if stage == 0 and 'DVB-T PASSIVE RADAR' in text:
                send(b'?'); stage = 1; stage_at = now
            elif stage == 1 and 'Keyboard guide' in text:
                snapshot(screen, args.output/'help'); send(b'o'); stage = 2
            elif stage == 2 and 'RUNNING' in text:
                running = now; send(b'5t'); stage = 3; stage_at = now
            elif stage == 3 and 'Shared-clock observer' in text:
                snapshot(screen, args.output/'timing-start'); resize(80, 24); send(b'o'); stage = 4; stage_at = now
            elif stage == 4 and now-stage_at > .4 and 'ABC' in text and 'Quit' in text:
                snapshot(screen, args.output/'compact'); resize(width, height); send(b'i'); stage = 5; stage_at = now
            elif stage == 5 and 'Applied SDR settings' in text:
                snapshot(screen, args.output/'source'); send(b'1vo' if args.native else b'1o'); stage = 6; stage_at = now
            elif stage == 6 and running is not None and now-running > 8:
                assert 'TRACK' in text, 'Channels never reached pilot tracking'
                snapshot(screen, args.output/'overview'); send(b't'); stage = 7; stage_at = now
            elif stage == 7 and now-stage_at > .4:
                snapshot(screen, args.output/'timing'); send(b'2vo' if args.native else b'o'); stage = 8
            elif stage == 8 and now-running > args.seconds:
                if args.ctrl_c:
                    send(b'\x03'); stage = 11
                else:
                    send(b's'); stage = 9
            elif stage == 9 and 'STOPPED' in text:
                assert process.poll() is None, 'Stop should leave results open'
                snapshot(screen, args.output/'stopped'); send(b'e'); stage = 10; stage_at = now
            elif stage == 10 and now-stage_at > .4:
                snapshot(screen, args.output/'events'); send(b'q'); stage = 11
            if process.poll() is not None:
                break
        assert stage == 11, f'Keyboard workflow stalled at stage {stage}:\n{text}'
        assert process.wait(timeout=30) == 0, 'Capture failed'
        restored = termios.tcgetattr(slave)
        (args.output/'termios.json').write_text(json.dumps({'original': original, 'restored': restored}, default=lambda x: x.hex(), indent=2))
        # Darwin sets PENDIN after restoring canonical input, to reprocess any
        # buffered characters. It is a transient driver flag, not a raw-mode leak.
        restored[3] &= ~getattr(termios, 'PENDIN', 0)
        original[3] &= ~getattr(termios, 'PENDIN', 0)
        assert restored == original, 'Raw terminal settings not restored'
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        while select.select([master], [], [], 0)[0]:
            try:
                block = os.read(master, 65536)
            except OSError:
                break
            if not block:
                break
            received.extend(block)
        os.close(slave); os.close(master)
        (args.output/'terminal.ansi').write_bytes(received)
    assert b'\x1b[?1049h' in received and b'\x1b[?1049l' in received, 'Alternate screen not restored'
    summary = json.loads((args.output/'capture/summary.json').read_text())
    assert 'error' not in summary and summary['missing_samples'] == summary['host_discarded_samples'] == summary['unquantified_gaps'] == 0, summary
    # RUNNING precedes UHD stream startup; hardware may start a fraction of a
    # second later. This checks the requested wall-clock run, not ADC coercion.
    assert summary['received_samples']/summary['actual_rate_hz'] >= args.seconds-(1 if args.hardware else 0), summary
    if args.fixture:
        assert summary['simulation_loop'] and summary['replay_loops'] >= 1, 'Replay did not continue beyond the file'
    else:
        assert not summary['simulation'] and summary['replay_loops'] == 0, 'Hardware test used replay'
    assert summary['processing_speed'] > 1, 'Capture cannot sustain realtime'
    for channel in summary['channels']:
        assert channel['locked'] and channel['rows'] > 500, channel
        if args.fixture:
            assert channel['rs_bad'] == 0 and channel['resets'] == summary['replay_loops'], channel
    if args.native:
        assert summary['channels'][0]['maps'] > 0 and summary['channels'][1]['maps'] > 0, summary
    if args.archive != 'none':
        archive = summarize(args.output/'capture/capture.dvblive')
        assert archive['records']['DONE'] == 1 and len(archive['gaps']) == summary['segments'], archive
        (args.output/'archive.json').write_text(json.dumps(archive, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
