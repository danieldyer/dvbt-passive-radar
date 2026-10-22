"""Local Gaussian information after active real brightness coefficients.

This is a local, conditional diagnostic. It holds scatterer associations,
attitude, antenna geometry, and the NNLS active set fixed. Original waveform
rows and the exact stationary projection are retained. It is not evidence of
global identifiability, and active-set boundaries are not regular CRLB cases.
"""
import numpy as np
from full_path_scene import RefinedScenePath,SceneBand,ScenePath
from rigid_patch import profile_static


def information(likelihood, parameters):
    parameters = np.asarray(parameters)
    if len(parameters) > 128:
        raise ValueError('The dense information reference is limited to local scenes')
    refined = isinstance(likelihood, RefinedScenePath)
    if refined:
        likelihood.set_points(parameters)
    n, count = likelihood.n, likelihood.p
    end = n+5+count-1
    phi = np.r_[0., parameters[n+5:end]]
    fisher = np.zeros((len(parameters), len(parameters)))
    gradient = np.zeros_like(parameters)
    records = []
    for c, band in enumerate(likelihood.bands):
        b = band.point
        solution = band.evaluate(parameters[:n], phi+parameters[n+c], c)
        values = band.design(parameters[:n], c, True, refined)
        h, dr, dv = values[:3]
        phase = band.phase_at_time(phi+parameters[n+c])
        gain = (b.amplitude@solution['amplitude'])*np.exp(1j*phase)
        model = h*gain[:, None, :]
        jacobian = np.zeros((*h.shape[:2], len(parameters)), complex)
        jacobian[:, :, :n] = (np.sum(dr*gain[:, None, :], axis=2)[:, :, None]*b.b.toarray()[:, None, :]
            +np.sum(dv*gain[:, None, :], axis=2)[:, :, None]*b.bd.toarray()[:, None, :])
        jacobian[:, :, n+c] = 1j*model.sum(axis=2)
        for window in range(len(band.active_points)):
            rows = band.window_index == window
            for point in range(band.points_per_window):
                if not band.active_points[window,point]:continue
                identity = band.phase_ids[window,point]
                if identity:
                    jacobian[rows, :, n+5+identity-1] = 1j*model[rows, :, point]
                if refined:
                    for axis in range(2):
                        jacobian[rows, :, end+2*identity+axis] = values[3+axis][rows, :, point]*gain[rows, None, point]
        selected = b.training
        projected = lambda x: profile_static(x, b.field.root_information, selected, b.groups)[selected]
        residual = projected(model.sum(axis=2))-b.data[True][selected]
        j = projected(jacobian).reshape(-1, len(parameters))
        brightness = (h*np.exp(1j*phase)[:, None, :])[:, :, None, :]*b.amplitude.toarray()[:, None, :, None]
        active = solution['amplitude'].ravel() > 0
        a = projected(brightness).reshape(-1, active.size)[:, active]
        real_j = np.r_[j.real, j.imag]
        real_a = np.r_[a.real, a.imag]
        real_r = np.r_[residual.real.ravel(), residual.imag.ravel()]
        if len(active) and active.any():
            u, singular, _ = np.linalg.svd(real_a, full_matrices=False)
            tolerance = np.finfo(float).eps*max(real_a.shape)*singular[0]
            supported = singular > tolerance
            real_j -= u[:, supported]@(u[:, supported].T@real_j)
            rank = int(supported.sum())
        else:
            singular = np.empty(0)
            rank = 0
        fisher += 2*(real_j.T@real_j)
        gradient += 2*(real_j.T@real_r)
        records.append(dict(channel=c, active_brightness_coefficients=int(active.sum()),
            numerical_brightness_rank=rank, brightness_singular_values=singular.tolist(),
            amplitude_kkt=solution['maximum_amplitude_kkt']))
    return (fisher+fisher.T)/2, gradient/likelihood.energy, records


def coordinates(fisher, phase_pivot, noise_degrees, condition_limit=1e4):
    """Invertible frozen optimizer scaling; never a likelihood modification.

    The finite condition cap controls numerical step size only. Every mode,
    physical constraint, likelihood term and output parameter is preserved.
    """
    if noise_degrees <= 0 or condition_limit <= 1:
        raise ValueError('Need positive noise degrees and a finite condition cap')
    matrix = phase_pivot.T@fisher@phase_pivot/noise_degrees
    diagonal = np.diag(matrix)
    scale = 1/np.sqrt(np.maximum(diagonal, max(diagonal.max(), 1e-300)/condition_limit))
    normalized = matrix*scale[:, None]*scale[None, :]
    eigen, vector = np.linalg.eigh((normalized+normalized.T)/2)
    floor = max(eigen[-1], 1e-300)/condition_limit
    transform = phase_pivot@(scale[:, None]*vector/np.sqrt(np.maximum(eigen, floor)))
    return transform, dict(normalized_information_eigenvalues=eigen.tolist(),
        optimizer_condition_cap=condition_limit, numerical_information_floor=floor,
        scope='Invertible frozen Gaussian-information coordinate change only; no mode dropped and no statistical loading or regularisation')


def aspect_information(likelihood, parameters, progress=None):
    """Small aspect-block approximation for full-capture optimizer scaling.

    Brightness and stationary nuisances are profiled within each declared
    aspect support. This extra nuisance freedom makes a conservative local
    preconditioner, not the final likelihood or a full-record CRLB. Every
    original epoch belongs to exactly one aspect block. Shared physical-path
    and band-phase coordinates are mapped back into the original layout.
    """
    import copy
    if isinstance(likelihood, RefinedScenePath) or likelihood.bands[0].points.ndim != 3:
        raise ValueError('Aspect information needs fixed nonoverlapping supports')
    parameters = np.asarray(parameters)
    n, count = likelihood.n, likelihood.p
    windows, points, _ = likelihood.bands[0].points.shape
    total = np.zeros((len(parameters), len(parameters)))
    reports = []
    for window in range(windows):
        active=np.flatnonzero(likelihood.bands[0].active_points[window])
        identities=likelihood.bands[0].phase_ids[window,active]
        local_points=len(active)
        path_columns = np.zeros(n, bool)
        for band in likelihood.bands:
            rows = band.window_index == window
            path_columns |= np.asarray(band.point.b[rows].getnnz(axis=0)).ravel() > 0
            path_columns |= np.asarray(band.point.bd[rows].getnnz(axis=0)).ravel() > 0
        path = np.flatnonzero(path_columns)
        mapping = np.zeros((len(path)+5+local_points-1, len(parameters)))
        mapping[np.arange(len(path)), path] = 1
        for c in range(5):
            mapping[len(path)+c, n+c] = 1
            if identities[0]:
                mapping[len(path)+c, n+5+identities[0]-1] = 1
        for point in range(1, local_points):
            mapping[len(path)+5+point-1, n+5+identities[point]-1] = 1
            if identities[0]:
                mapping[len(path)+5+point-1, n+5+identities[0]-1] = -1
        local = []
        for band in likelihood.bands:
            original = band.point
            index = np.flatnonzero(band.window_index == window)
            b = copy.copy(original)
            b.field = original.field.windowed(original.t[index[0]], np.nextafter(original.t[index[-1]], np.inf))
            np.testing.assert_array_equal(b.field.time, original.t[index])
            # Synthetic controls may supply whitened observations directly;
            # windowing must retain those exact values as well as raw captures.
            np.testing.assert_array_equal(b.field.whiten, original.field.whiten)
            b.field.y = original.field.y[index]
            for name in ['t', 'base_path', 'base_rate', 'frame', 'frame_rate', 'anchor', 'velocity', 'direction', 'direction_rate', 'training', 'groups']:
                setattr(b, name, getattr(original, name)[index])
            b.b, b.bd = original.b[index][:, path], original.bd[index][:, path]
            basis = original.amplitude[index]
            columns = np.flatnonzero(np.asarray(basis.getnnz(axis=0)).ravel())
            b.amplitude = basis[:, columns]
            b.amplitude_nodes = original.amplitude_nodes[columns]
            b.coordinate = {key: original.coordinate[key].reshape(-1, 16)[index].ravel() for key in ['native_time_s', 'native_information']}
            b.data, b.energy = {}, {}
            for training in [True, False]:
                selected = b.training if training else ~b.training
                y = profile_static(b.field.y, b.field.root_information, selected, b.groups)*selected[:, None]
                b.data[training] = y
                b.energy[training] = float(np.vdot(y, y).real)
            local.append(SceneBand(b, band.points[window,active]))
        model = ScenePath(local)
        fisher, _, channels = information(model, mapping@parameters)
        total += mapping.T@fisher@mapping
        record = dict(window=window, retained_path_columns=path.tolist(), channels=channels,
            original_rows_by_band=[len(b.point.t) for b in local])
        reports.append(record)
        if progress:
            progress(record)
    return (total+total.T)/2, reports
