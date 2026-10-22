"""Plot frozen-model, jointly deblended phase diagnostics, without correcting data."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from recovery_experiment import NAMES
from stationary import Stationary


def template_support(time,unit,predicted,stationary):
    """Conditional pure-template information, never an observed-SNR estimate."""
    unit=np.asarray(unit);predicted=np.asarray(predicted);energy=np.sum(abs(predicted)**2,axis=1)
    total=energy.sum();unit_energy=np.sum(abs(unit)**2)
    if total==0 or unit_energy==0:return dict(supported=False)
    mean=float(energy@time/total);variance=float(energy@(np.asarray(time)-mean)**2/total)
    a=unit.ravel();b=predicted.ravel();pa=stationary.remove(a);pb=stationary.remove(b)
    norm=np.vdot(pb,pb).real;constant_norm=np.vdot(pa,pa).real
    ratio=(float(abs(np.vdot(pa,pb))**2/(constant_norm*norm)) if constant_norm>0 and norm>0 else None)
    return dict(supported=True,expected_energy=float(total),energy_time_centroid_s=mean,
        rectangular_moment_equivalent_aperture_s=float(np.sqrt(12*variance)),
        constant_strength_template_efficiency_after_stationary=ratio,
        target_information_retention_after_stationary=float(norm/total))


def aperture_diagnostics(band,parameters):
    """Compare finite native templates with/without fitted brightness weights.

    The pure-template comparison uses the exact same observed coordinates,
    whitening and a full-aperture stationary Schur projection on both sides.
    It is conditional on the inferred motion/brightness, which can be wrong.
    """
    components,_=band.frozen_components(parameters);raw,_=band.responses(parameters,False)
    nuisance=Stationary(band.design,np.ones(band.field.y.size,bool));result=[]
    for j,h in enumerate(raw):
        unit=band.reduced(h[:,:,None]).reshape(band.field.y.shape)
        predicted=components[:,j].reshape(unit.shape)
        result.append(template_support(band.field.time,unit,predicted,nuisance))
    return result


def brightness_model(bands,parameters,*,physical_phases=None):
    """Export physical constant phases without guessing parameter-tail layouts."""
    points=len(bands[0].points);m=bands[0].motion_size
    if physical_phases is None:
        band_phase=np.asarray([b.band_phase for b in bands])
        point_phase=np.asarray(parameters)[m:m+points].copy()
        if bands[0].position_gauge is not None:
            offsets=np.asarray(parameters)[m+points:m+points+2*(points-1)].reshape(-1,2)
            point_phase[1:]+=np.sum(bands[0].position_gauge[1:]*offsets,axis=1)
    else:
        band_phase,point_phase=map(np.asarray,physical_phases)
        if band_phase.shape!=(len(bands),) or point_phase.shape!=(points,):
            raise ValueError('Physical constant phases must match the bands and source support')
    return dict(point_phase_rad=point_phase,bands=[dict(rf_hz=b.field.metadata['rf_hz'],knots_s=b.knots,
        coefficients=b.evaluate_scene(parameters,full=True)['amplitude'].reshape(points,len(b.knots)),
        band_phase_rad=phase) for b,phase in zip(bands,band_phase)])


def audit(bands,parameters,first_s,directory,*,physical_phases=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    records=[b.component_histories(parameters,first_s) for b in bands]
    support=[aperture_diagnostics(b,parameters) for b in bands]
    count=len(bands[0].points)
    fig,axes=plt.subplots(5,count,figsize=(5*count,12),sharex=True,sharey=True,squeeze=False,layout='constrained')
    for c,rows in enumerate(records):
        t=np.asarray([r['time_s'] for r in rows]);training=np.asarray([r['training'] for r in rows])
        phases=np.asarray([[np.nan if z is None else z for z in r['phase_error_deg']] for r in rows])
        energy=np.asarray([r['expected_component_energy'] for r in rows])
        for j in range(count):
            ax=axes[c,j]
            finite=np.isfinite(phases[:,j]);maximum=energy[:,j].max()
            size=8+32*energy[:,j]/max(maximum,np.finfo(float).tiny)
            for mask,label,colour in [(training,'Fitting samples','tab:blue'),(~training,'Unused samples','tab:orange')]:
                use=mask&finite
                if np.any(use):ax.scatter(t[use],phases[use,j],s=size[use],c=colour,label=label)
            ax.axhline(0,color='#999',lw=.6);ax.set(ylim=(-180,180),ylabel=NAMES[c]+' phase error (degrees)',xlabel='Capture time (s)')
            if c==0:ax.set_title(f'Candidate {j}')
    axes[0,0].legend(loc='upper right',fontsize=8)
    fig.suptitle('Jointly deblended quarter-second phase histories against frozen native predictions\nMarker area follows expected component energy; local complex multipliers are diagnostic only')
    fig.savefig(directory/'phase-histories.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(5,1,figsize=(12,11),sharex=True,layout='constrained')
    for c,rows in enumerate(records):
        t=np.asarray([r['time_s'] for r in rows]);energy=np.asarray([r['expected_component_energy'] for r in rows])
        for j in range(count):
            total=energy[:,j].sum()
            if total:
                duration=support[c][j]['rectangular_moment_equivalent_aperture_s']
                axes[c].plot(t,energy[:,j]/total,label=f'Candidate {j}: moment-equivalent aperture {duration:.1f} s')
        axes[c].set(ylabel=NAMES[c]+' energy fraction');axes[c].legend(fontsize=8)
    axes[-1].set_xlabel('Capture time (s)')
    fig.suptitle('When the fitted components actually carry conditional information\nFitted real brightness; moment-equivalent duration is not demonstrated coherent resolution')
    fig.savefig(directory/'aperture-support.png',dpi=160);plt.close(fig)
    brightness=brightness_model(bands,parameters,physical_phases=physical_phases)
    return dict(channels=records,block_s=.25,local_complex_multipliers_applied=False,conditional_aperture_support=support,
        brightness_model=brightness,
        scope='Fitted motion, band phases, real brightness and stationary coefficients frozen. Local multipliers diagnose phase consistency, not target SNR or independent physical scatterer identity.')
