"""Known-template Gaussian information with a constant-profile nuisance.

For a fixed range covariance and original weights, the nuisance is one complex
constant per observed range coordinate. The field's existing whitening defines
the metric. Its orthogonal projection is applied to both signal and data.
GPU polynomials evaluate the exact same retained Chebyshev pulse model; they
do not estimate a local background, fit a noise floor or change target phases.
Representative-epoch accuracy must be checked against native rows separately.
"""
from pathlib import Path
import struct
import subprocess

import numpy as np

from dense_patch_image import trial_geometry,matched_polynomials
from field import average
from fixed_range_field import stationary_profile_design
from stationary import Stationary


def norm_polynomials(field, first_cell, cells):
    """Exact degree-2d polynomial of ||W pulse(r-x)||² on each delay cell."""
    kernels, identity = field.pulse.kernels, field.pulse.row_blocks
    degree, radius = kernels.degree, kernels.radius
    coefficients = np.moveaxis(kernels.coefficients[:, identity], 0, -1)
    result = np.zeros((len(identity), cells, 2*degree+1))
    for i, cell in enumerate(range(first_cell, first_cell+cells)):
        at = field.offsets-cell-1+radius
        if at.min() < 0 or at.max() >= 2*radius:
            raise ValueError('Information polynomial exceeds the retained waveform support')
        a = coefficients[:, at]*((-1.)**np.arange(degree+1))
        white = np.einsum('sr,nrk->nsk', field.whiten, a)
        gram = np.einsum('nrk,nrl->nkl', white.conj(), white).real
        for k in range(degree+1):
            for j in range(degree+1):
                result[:, i, k+j] += .5*gram[:, k, j]
                result[:, i, abs(k-j)] += .5*gram[:, k, j]
    return result


def reference(band, parameters, pixels, *, selected=None, amplitude=None,moments=None):
    """Simple double-precision forward/projection cross-check, without polynomials."""
    f = band.field
    keep = band.training if selected is None else np.asarray(selected, bool)
    a = np.ones(len(f.data['row_time'])) if amplitude is None else np.asarray(amplitude)
    if a.shape!=f.data['row_time'].shape:raise ValueError('Template brightness must accompany the original model epochs')
    use = np.repeat(keep, len(f.offsets))
    nuisance = Stationary(stationary_profile_design(f), use)
    raw, projected = [], []
    for point in np.asarray(pixels):
        offset, velocity = band.kinematics.point_state(parameters, point, origin=f.data['centre_ecef_m'])
        response=f.raw_trajectory(f.data['centre_ecef_m']+offset, velocity, position_offset=offset)*a[:,None]
        if moments is not None:
            if f.metadata['block_rows']!=1 or amplitude is not None:raise ValueError('Carrier moments already include the native brightness')
            _,omega,_=f.path_kinematics(f.data['centre_ecef_m']+offset,velocity,position_offset=offset)
            factor=np.polynomial.polynomial.polyval(1j*omega/.001024,np.asarray(moments).T,tensor=False)
            response*=factor[:,None]
        h=f.apply(average(response,f.data['row_information'],f.metadata['block_rows'])[0]).ravel()[use]
        residual = nuisance.remove(h)
        raw.append(np.vdot(h, h).real); projected.append(np.vdot(residual, residual).real)
    return np.c_[raw, projected]


def bin_moments(native,amplitude=None):
    """Original weighted carrier-bin moments, with brightness before averaging.

    Coefficients multiply (i times angular Doppler)^j. Sixth order leaves a
    negligible remainder over the checked ship aperture; native-row tests
    bound propagation curvature and envelope migration omitted by this factor.
    """
    import math
    f=native.field if hasattr(native,'field') else native
    block=f.metadata['block_rows'];rows=f.data['row_time'].reshape(-1,block)
    weight=f.data['row_information'].reshape(rows.shape)
    a=np.ones_like(rows) if amplitude is None else np.asarray(amplitude).reshape(rows.shape)
    dt=rows-f.time[:,None];total=weight.sum(axis=1)
    return np.column_stack([np.divide(np.sum(weight*a*dt**j,axis=1),total,
        out=np.zeros(len(total)),where=total>0)/math.factorial(j) for j in range(7)])


def gpu(bands, parameters, pixels, directory, *, selected=None, amplitude=None, moments=None,score_values=None,reuse=False):
    """Evaluate each physical pixel's raw and nuisance-projected information."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=reuse)
    pixels = np.asarray(pixels)
    if pixels.ndim != 2 or pixels.shape[1] != 3: raise ValueError('Physical ENU pixel coordinates required')
    first = max(int(b.field.offsets.max()-b.field.pulse.kernels.radius) for b in bands)
    last = min(int(b.field.offsets.min()+b.field.pulse.kernels.radius) for b in bands)
    cells = last-first
    degree = bands[0].field.pulse.kernels.degree
    stride = max(len(b.field.time) for b in bands)
    ranges = len(bands[0].field.offsets)
    geometry = np.zeros((len(bands), stride, 10, 4), '<f4')
    pulses = np.zeros((len(bands), stride, 64, degree+1), '<c8')
    norms = np.zeros((len(bands), stride, cells, 2*degree+1), '<f4')
    carrier_moments=np.zeros((len(bands),stride,7),'<f4')
    data_coefficients=np.zeros((2,len(bands),stride,cells,degree+1),'<c8')
    whiten = np.empty((len(bands), ranges, ranges), '<c8')
    scales = []
    selected = [b.training for b in bands] if selected is None else selected
    amplitude = [np.ones(len(b.field.time)) for b in bands] if amplitude is None else amplitude
    score_values=[[b.field.y for b in bands],[np.zeros_like(b.field.y) for b in bands]] if score_values is None else score_values
    for c, (b, keep, a) in enumerate(zip(bands, selected, amplitude)):
        f = b.field; n = len(f.time)
        np.testing.assert_array_equal(f.offsets, bands[0].field.offsets)
        if (f.metadata['block_rows'] != 1 or f.pulse.kernels.degree != degree or
                np.any(f.data['centre_omega'] != 0) or 'carrier_reference_path_m' in f.data):
            raise ValueError('Use fixed-reference representative epochs and the same pulse degree')
        if np.shape(a) != f.time.shape or not np.isfinite(a).all() or np.any(np.asarray(a) < 0):
            raise ValueError('A finite nonnegative template amplitude is required at every epoch')
        g = trial_geometry(b, parameters, keep, stride)[:, :10]
        scale = np.max(f.root_information**2); scales.append(scale)
        g[:n, 8, 3] = f.root_information**2/scale
        geometry[c] = g
        pulses[c, :n] = np.moveaxis(f.pulse.kernels.coefficients[:, f.pulse.row_blocks], 0, -1)
        norms[c, :n] = norm_polynomials(f, first, cells)
        whiten[c] = f.whiten
        if moments is None:carrier_moments[c,:n,0]=a
        else:carrier_moments[c,:n]=moments[c]
        import copy
        for case in range(2):
            field=copy.copy(f);field.y=np.asarray(score_values[case][c])
            data_coefficients[case,c,:n]=matched_polynomials(field,keep,first,cells)
    model = directory/'model.bin'; output = directory/'information.bin'
    header=b'FIELDIN2'+struct.pack('<7Iif', stride, len(bands), ranges, cells, degree, len(pixels), 32, first, .001024)
    arrays=[geometry, pulses, norms, whiten, np.c_[pixels, np.zeros(len(pixels))].astype('<f4'),np.asarray(bands[0].field.offsets, '<i4'),carrier_moments,data_coefficients[0],data_coefficients[1]]
    cached=reuse and model.exists()
    if cached:
        with model.open('rb') as source:
            if source.read(len(header))!=header:raise ValueError('Cached template-information dimensions changed')
            for value in arrays:
                if source.read(value.nbytes)!=value.tobytes():raise ValueError('Cached template-information model changed')
            if source.read(1):raise ValueError('Trailing cached information-model data')
    else:
        with model.open('xb') as target:
            target.write(header)
            for value in arrays:value.tofile(target)
    if not (cached and output.exists()):
        with (directory/'gpu.log').open('w') as log:
            subprocess.run(['build/release/support/isar/isar_template_information', str(model), str(output)], check=True, stderr=log)
    payload=np.fromfile(output,'<f4').reshape(2,len(bands),len(pixels),4).astype(float)
    value=payload[0]
    value *= np.asarray(scales)[:, None, None]
    # fourth column is the sum of original information, not a target statistic.
    score=payload[1]
    return value,score[:,:,:2]@np.array([1.,1j]),score[:,:,2:]@np.array([1.,1j])


def evaluate_cached(model,pixels,scales,directory):
    """Query an immutable known-waveform model on independent physical cuts."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    with Path(model).open('rb') as source:
        if source.read(8)!=b'FIELDIN2':raise ValueError('Expected the carrier-preserving matched-statistic model')
        _,bands,*_=struct.unpack('<7Iif',source.read(36))
    points=directory/'pixels.bin';output=directory/'information.bin'
    with points.open('xb') as target:
        target.write(struct.pack('<I',len(pixels)));np.c_[pixels,np.zeros(len(pixels))].astype('<f4').tofile(target)
    with (directory/'gpu.log').open('w') as log:
        subprocess.run(['build/release/support/isar/isar_template_information',str(model),str(output),str(points)],check=True,stderr=log)
    data=np.fromfile(output,'<f4').reshape(2,bands,len(pixels),4).astype(float)
    data[0]*=np.asarray(scales)[:,None,None]
    return data[0],data[1,:,:,:2]@np.array([1.,1j]),data[1,:,:,2:]@np.array([1.,1j])
