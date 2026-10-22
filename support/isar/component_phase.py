"""Conditional measured phases after frozen subtraction of other components.

This is a diagnostic, not an independent new observation or a phase correction.
Its noise variance conditions on the fitted neighbouring returns and stationary
profile. Raw measurements must accompany it when assessing association.
"""
import numpy as np

from field import average


def conditional_components(cluster, parameters):
    """Matched coefficients after training-frozen subtraction of other echoes.

    The returned coefficient still uses its template's phase gauge. Keeping
    this operation independent of an empirical or physical path representation
    gives both diagnostics the same data selection and subtraction operator.
    """
    b = cluster.base
    value = cluster.evaluate(np.asarray(parameters))
    k = len(b.knots)
    design, amplitude = value['design'], value['amplitude']
    parts = np.stack([design[:, s*k:(s+1)*k]@amplitude[s*k:(s+1)*k]
        for s in range(cluster.components)])
    prediction = parts.sum(axis=0)
    y = b.field.y.ravel()
    beta = b.nuisance.solve(y[b.selected]-prediction[b.selected])
    stationary = b.design@beta
    records = []
    for s, (raw, _) in enumerate(cluster.response_blocks(parameters)):
        unit = b.reduced(raw[:, :, None])[:, 0].reshape(b.field.y.shape)
        norm = np.sum(abs(unit)**2, axis=1)
        clean = (y-stationary-prediction+parts[s]).reshape(b.field.y.shape)
        z = np.divide(np.sum(unit.conj()*clean, axis=1), norm,
            out=np.zeros(len(norm), complex), where=norm>0)
        records.append(dict(time_s=b.field.time, matched_complex_coefficient=z,
            conditional_information=norm/8192,
            frozen_component_prediction=parts[s].reshape(b.field.y.shape)))
    return records


def component_measurements(cluster, parameters):
    """Restore empirical path phase, with no fitted rate removed from data."""
    b=cluster.base;records=conditional_components(cluster,parameters)
    arcs=getattr(cluster,'arcs',[b]*cluster.components)
    blocks=getattr(cluster,'slices',[slice(4*s,4*s+4) for s in range(cluster.components)])
    for record,arc,block in zip(records,arcs,blocks):
        row_path=arc.known_path+arc.base.basis@np.asarray(parameters)[block][2:]
        path=average(row_path,b.field.data['row_information'],b.field.metadata['block_rows'])[0]
        record.update(physical_path_change_m=path,
            complex_coefficient=record.pop('matched_complex_coefficient')*np.exp(-1j*b.field.k*path))
    return records


def restore_physical_phase(record,arc,parameters):
    """Restore one physical point's path convention in a measured coefficient."""
    arc=getattr(arc,'physical',arc);g=arc.geometry
    record=record.copy()
    p,v,a,tx,rx=g.state(np.asarray(parameters)[:g.count]);f=arc.field
    dt=f.data['row_time']-arc.midpoint
    local=p+dt[:,None]*v+.5*dt[:,None]**2*a
    offset=(local-arc.reference_local)@g.rotation
    position=f.data['centre_ecef_m']+offset;velocity=(v+dt[:,None]*a)@g.rotation
    delta,_,_=f.path_kinematics(position,velocity,transmitter=g.receiver+tx[arc.channel]@g.rotation,
        receiver=g.receiver+rx@g.rotation,position_offset=offset)
    row_path=delta+f.data['centre_path_m'];row_path-=np.interp(arc.midpoint,f.data['row_time'],row_path)
    path=average(row_path,f.data['row_information'],f.metadata['block_rows'])[0]
    record.update(physical_path_change_m=path,
        complex_coefficient=record.pop('matched_complex_coefficient')*np.exp(-1j*f.k*path))
    return record


def physical_primary_measurements(cluster,parameters):
    """Restore the physical primary's exact two-leg path-change convention."""
    return restore_physical_phase(conditional_components(cluster,parameters)[0],cluster.base,parameters)
