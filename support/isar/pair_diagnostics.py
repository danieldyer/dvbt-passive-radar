"""Report cancellation and conditioning without changing the echo likelihood."""
import numpy as np


def components(pair, parameters):
    value=pair.evaluate(parameters);k=len(pair.base.knots)
    return np.stack([value['design'][:,j*k:(j+1)*k]@value['amplitude'][j*k:(j+1)*k]
                     for j in range(2)])


def diagnose(model, parameters):
    """Accept either independently phased or common-primary paired models."""
    if hasattr(model,'independent'):
        parameters=model.mapping(parameters)[0];model=model.independent
    records=[]
    for pair,index in zip(model.pairs,model.indices):
        p=parameters[index];value=pair.evaluate(p);b=pair.base
        z=components(pair,p)[:,b.selected]
        energies=np.sum(abs(z)**2,axis=1);total=np.sum(abs(z.sum(axis=0))**2)
        correlation=np.vdot(z[0],z[1])/max(np.sqrt(np.prod(energies)),1e-300)
        projected=b.nuisance.remove(value['design'][b.selected])
        matrix=np.r_[projected.real,projected.imag][:,value['amplitude']>0]
        singular=np.linalg.svd(matrix,compute_uv=False)
        condition=float(singular[0]/singular[-1]) if len(singular) and singular[-1]>0 else None
        records.append(dict(component_energy=energies.tolist(),sum_energy=float(total),
            component_to_sum_energy_ratio=float(energies.sum()/max(total,1e-300)),
            component_correlation_magnitude=float(abs(correlation)),
            component_correlation_phase_deg=float(np.degrees(np.angle(correlation))),
            active_brightness_condition=condition,active_brightness_count=int(len(singular)),
            largest_brightness_coefficient=float(max(value['amplitude']))))
    return records
