#!/usr/bin/env python3
"""Validate known target locations in actual native ImPlot framebuffer exports."""
import json
import pathlib
import sys
import numpy as np
from PIL import Image

directory = pathlib.Path(sys.argv[1])
results = json.loads((directory/'renderer.json').read_text())
for result in results:
    image = np.asarray(Image.open(directory/(result['id']+'.png')).convert('RGB'))
    x, y, width, height = result['plot_pixels']
    x0, y0 = int(x)+2, int(y)+2
    plot = image[y0:int(y+height)-2, x0:int(x+width)-2].astype(float)
    # Viridis's yellow target maximum, without relying on axes text or OS
    # screenshot permissions. The exported rectangle is the actual ImPlot area.
    score = plot[:, :, 0]+plot[:, :, 1]-2*plot[:, :, 2]
    py, px = np.unravel_index(np.argmax(score), score.shape)
    observed = [(x0+px-x)/width, (y0+py-y)/height]
    expected = [result['expected_x'], result['expected_y']]
    error = float(np.max(np.abs(np.array(observed)-expected)))
    assert error < .01, f"{result['id']}: range or signed Doppler flipped: {observed} != {expected}"
    result['observed'] = observed; result['maximum_fractional_error'] = error
(directory/'image_checks.json').write_text(json.dumps(results, indent=2))
print('Native target coordinates and signed Doppler orientation: PASS')
