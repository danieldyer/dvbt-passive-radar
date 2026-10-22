"""Reviewable diagnostics for unaccepted physical ship-motion fits."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(work, destination):
    paths = [work/'rigid-motion-142-172/channel-3-final.json',
             work/'rigid-motion-142-172/shared-linear-final.json']
    if not all(path.exists() for path in paths):
        return ''
    single, shared = [json.loads(path.read_text()) for path in paths]
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout='constrained')
    loss = np.array([100*(after['prediction_error']/before['prediction_error']-1)
                     for before, after in zip(shared['initial'], shared['fitted'])])
    axes[0].bar(names, loss, color='#2373bb')
    axes[0].axhline(0, color='#888888', linewidth=.7)
    axes[0].set(title='Shared linear physical correction', ylabel='Held-out prediction-error change (%)')
    time = np.linspace(*single['model']['interval_s'], 301)
    u = (time-time.mean())/(np.ptp(time)/2)
    basis = u[:, None]**np.arange(1, single['model']['degree']+1)
    correction = basis@np.array(single['physical_coefficients']).T
    for axis, labels, values in ((axes[1], ['Forward', 'Port', 'Up'], correction[:, :3]),
                                 (axes[2], ['Roll', 'Pitch', 'Yaw'], np.degrees(correction[:, 3:]))):
        for column, label in enumerate(labels):
            axis.plot(time, values[:, column], label=label)
        axis.set_xlabel('Capture time (s)')
        axis.legend(fontsize=8)
    axes[1].set(title='Ten-only cubic trial: translation', ylabel='Correction (m)')
    axes[2].set(title='Ten-only cubic trial: attitude', ylabel='Correction (degrees)')
    figure.suptitle('Physical ship-motion tests: trial corrections, not accepted motion estimates\n'
                   'Antennas fixed; original row geometry; alternating two-second training/test blocks')
    figure.savefig(destination/'rigid-motion-validation.png', dpi=150)
    plt.close(figure)
    rows = ''.join(f'<tr><td>{name}</td><td>{before["conditional_efficiency_loss_db"]:.2f}</td>'
                   f'<td>{after["conditional_efficiency_loss_db"]:.2f}</td><td>{delta:+.1f}%</td></tr>'
                   for name, before, after, delta in zip(names, shared['initial'], shared['fitted'], loss))
    single_change = 100*(single['fitted'][0]['prediction_error']/single['initial'][0]['prediction_error']-1)
    nuisance_loss = max(max(band['stationary_nuisance_point_information_loss_db'])
                        for result in (single, shared) for kind in ('initial', 'fitted')
                        for band in result[kind])
    summary = {'independent_ten': single, 'shared_five_channel': shared,
               'accepted': False,
               'interpretation': 'The tested physical corrections increase training fit but worsen withheld prediction. They do not establish refined motion, phase centres or a coherent five-channel scattering model. These finite local tests do not rule out richer physical motion/scattering models; constant pose, scattering-point selection, conditional covariance and two-second validation gaps remain limitations.'}
    (destination/'rigid-motion-validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    return f'''<h2 id="physical-motion">Physical translation and attitude fitting</h2>
<p>A new forward model changes the ship's three-dimensional translation, roll, pitch
and yaw before computing each point's exact bistatic delay, RF phase and symbol
moments at every original sample epoch. The original range crop and temporal averaging
remain fixed. Antenna coordinates stay at the licensed hypothesis. Constant pose at
the aperture midpoint fixes the coordinate gauge; the AIS/course prior is not promoted
to measured heading.</p>
<p>Complex amplitudes for four trial scattering points per band and a constant-delay
background are solved jointly using
<a href="https://www.cs.umd.edu/users/oleary/software/varpro.pdf">variable projection</a>.
The nonlinear motion uses declared Gaussian regularisation scales, rather than
claiming measured AIS uncertainty. Different bands retain independent complex point
responses. Thus this estimates a shared physical motion hypothesis without assuming
that inter-band scattering phases are already calibrated.</p>
<figure><a href="short-aperture-five-prior.png"><img src="short-aperture-five-prior.png"
alt="Five independent thirty-second AIS-prior ship images, without accepted motion correction"></a>
<figcaption class="caption">The five 30-second prior images used for point selection,
on one fixed power scale. These use the original unnotched profiles and exact GPU
backprojection. The point-response ambiguities discussed above remain; the bright
ridges are not established physical hull details.</figcaption></figure>
<img src="rigid-motion-validation.png" alt="Held-out errors and rejected trial translation and attitude corrections">
<p>On 142–172 s, the Ten-only cubic correction increases withheld prediction error by
{single_change:.1f}%. A simpler linear correction fitted jointly across all five channels
also worsens each band's prediction. Both solvers stop on their relative-objective
tolerance and reach parameter bounds; this is not a global optimum certificate.
The cubic trial's normalized projected gradient is still
{single['maximum_projected_gradient']:.2g}. Neither correction is adopted.</p>
<table><tr><th>Band</th><th>Initial conditional mismatch (dB)</th><th>After shared motion fit (dB)</th><th>Held-out error change</th></tr>{rows}</table>
<p>The dB columns compare prediction from training amplitudes with amplitudes refitted
on the withheld samples under the same trial model. They are conditional model-mismatch
diagnostics, not production SNR loss or a CRLB gap. Initial point selection used complete
prior images; the test applies to the subsequent fit. A finite set of local polynomial
fits cannot rule out faster motion, extended/aspect-dependent scattering or different
physical associations. Two-second gaps in the training blocks also limit how rapidly
varying motion can be inferred.</p>
<p>No Doppler bin is removed. The stationary background is an explicit nuisance in
the same likelihood; its maximum conditional single-point information loss among
these trial templates is {nuisance_loss:.2f} dB. That quantifies overlap with unknown
stationary returns, not deletion from the archive or measured production SNR loss.</p>
<p>The ISAR suite now passes 26 tests, including independent rotation evaluation,
original-row range/phase/moment derivatives, erasures, profiled-likelihood gradients
and a physical-motion injection recovered on withheld samples. Original capture data
and production processing are unchanged.
<a href="rigid-motion-validation.json">Complete parameters, priors and validation results</a>.</p>'''
