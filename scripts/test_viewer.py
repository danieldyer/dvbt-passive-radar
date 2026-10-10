#!/usr/bin/env python3
"""Exercise the compiled ImGui/Wasm viewer in Chrome, including native-resolution zoom."""
import argparse
import json
import atexit
import functools
import importlib.util
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from playwright.sync_api import sync_playwright

parser=argparse.ArgumentParser()
parser.add_argument('--url',default='http://127.0.0.1:8765/')
parser.add_argument('--chrome',default='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
parser.add_argument('--live',action='store_true')
parser.add_argument('--launch',type=Path,help='Start a bounded recording run and an isolated server after the browser is ready')
args=parser.parse_args()
processor=None
if args.launch:
    args.live=True
    directory=Path(tempfile.mkdtemp(prefix='gui-live-',dir='build')).resolve()
    spec=importlib.util.spec_from_file_location('serve',Path('web/serve.py'))
    serve=importlib.util.module_from_spec(spec);spec.loader.exec_module(serve)
    handler=functools.partial(serve.Handler,directory=str(Path('build/web').resolve()))
    server=serve.http.server.ThreadingHTTPServer(('127.0.0.1',0),handler)
    server.maps=directory
    threading.Thread(target=server.serve_forever,daemon=True).start()
    args.url=f'http://127.0.0.1:{server.server_port}/'
    def cleanup():
        if processor and processor.poll() is None:
            processor.terminate();processor.wait(timeout=10)
        server.shutdown();server.server_close()
    atexit.register(cleanup)
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=args.chrome,headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1000})
    errors=[]
    page.on('pageerror',lambda error: errors.append(str(error)))
    page.goto(args.url)
    if args.launch:
        page.wait_for_function("typeof Module._receive_frame==='function'",timeout=30000)
        log=open(directory.with_suffix('.log'),'w')
        environment=dict(os.environ,TMP='/private/tmp',XDG_CACHE_HOME=str(Path('build/cache').resolve()),XDG_CONFIG_HOME=str(Path('build/config').resolve()))
        processor=subprocess.Popen(['build/release/dvbt_reconstruct',str(args.launch),'--output',str(directory),'--seconds','60','--map-interval','1'],stdout=log,stderr=subprocess.STDOUT,env=environment)
    page.wait_for_function('window.radar.frames>=2',timeout=30000)
    before=page.evaluate('window.radar.rmax-window.radar.rmin')
    page.mouse.move(900,450)
    page.mouse.wheel(0,-450)
    page.wait_for_timeout(800)
    after=page.evaluate('window.radar.rmax-window.radar.rmin')
    assert after<before,(before,after)
    page.wait_for_function('window.radar.meta.full_range_max_m > window.radar.rmax',timeout=10000)
    meta=page.evaluate('window.radar.meta')
    assert meta['native_frequency_step_hz']<=meta['frequency_step_hz']
    # Pan the image, then reset it with a double click.
    initial=page.evaluate('window.radar.rmin')
    page.mouse.move(900,450);page.mouse.down();page.mouse.move(800,450,steps=8);page.mouse.up()
    page.wait_for_timeout(300)
    assert page.evaluate('window.radar.rmin') != initial
    page.mouse.dblclick(900,450)
    page.wait_for_timeout(300)
    assert page.evaluate('window.radar.rmin')==0
    # The ImGui button changes fetching, not the processor's coherent history.
    page.mouse.click(130,115)
    page.wait_for_timeout(700)
    assert page.evaluate('window.radar.playing') is False
    count=page.evaluate('window.radar.frames');page.wait_for_timeout(700)
    assert page.evaluate('window.radar.frames')==count
    page.mouse.click(130,115)
    page.wait_for_function(f'window.radar.frames>{count}')
    if args.live:
        sequence=page.evaluate('window.radar.meta.sequence')
        page.mouse.click(48,253)  # Switch the actual ImGui control to manual.
        page.wait_for_timeout(150)
        page.mouse.click(120,290)
        page.keyboard.press('Control+A');page.keyboard.type('2');page.keyboard.press('Enter')
        page.wait_for_function('window.radar.meta.requested_integration_s===2',timeout=30000)
        assert page.evaluate('window.radar.meta.integration_s')<=2
        assert page.evaluate('window.radar.meta.sequence')>sequence
        page.mouse.click(48,253)  # Restore automatic integration through the UI.
        page.wait_for_function('window.radar.meta.requested_integration_s===0',timeout=30000)
        assert page.evaluate('window.radar.meta.integration_s')>2
    # Toggle the actual ImGui DC control, including while paused and after EOF.
    raw_peak=page.evaluate('window.radar.meta.peak_db')
    page.mouse.click(48,633)
    page.wait_for_function('window.radar.meta.dc_removed===true',timeout=10000)
    filtered_peak=page.evaluate('window.radar.meta.peak_db')
    if not args.live:
        assert filtered_peak<raw_peak-10,(raw_peak,filtered_peak)
    page.mouse.move(200,980)
    page.screenshot(path='build/viewer-dc-removed.png')
    page.mouse.click(130,115)
    page.wait_for_timeout(700)
    page.mouse.click(48,633)
    page.wait_for_function('window.radar.meta.dc_removed===false',timeout=10000)
    assert page.evaluate('window.radar.playing') is False
    page.mouse.click(48,633)
    page.wait_for_function('window.radar.meta.dc_removed===true',timeout=10000)
    page.mouse.click(130,115)
    # A narrow frequency view must request full native Doppler bins from the server.
    for x,y,value in ((225,429,'5'),(90,495,'-1'),(225,495,'1')):
        page.mouse.click(x,y);page.keyboard.press('Control+A');page.keyboard.type(value);page.keyboard.press('Enter')
    page.wait_for_function('window.radar.meta.frequency_start_hz>=-1.2',timeout=10000)
    page.wait_for_timeout(500)
    zoom=page.evaluate('window.radar.meta')
    assert zoom['frequency_step_hz']==zoom['native_frequency_step_hz']
    page.screenshot(path='build/viewer-zoom.png')
    print(json.dumps({'errors':errors,'frames':page.evaluate('window.radar.frames'),
                      'raw_peak_db':raw_peak,'dc_removed_peak_db':filtered_peak,'zoom':zoom},indent=2))
    assert not errors,errors
    browser.close()
    if processor:
        processor.terminate();processor.wait(timeout=10)
        print('Live UI control test completed; stopped its bounded test recording.')
