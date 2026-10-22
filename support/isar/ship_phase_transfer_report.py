"""Publish carrier-cycle, separate-interval and HH reflection diagnostics."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_ais_phase import NAMES


def error(record):
    return sum(v['test_error'] for v in record['prediction'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('docs/data/isar_20261017'))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/ship-phase-transfer-review'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    paths={name:args.root/folder/'fits.json' for name,folder in {
        'branches':'ship-common-phase-polished','exact':'ship-common-phase-exact-residual',
        'early':'ship-frozen-calibration-transfer','later':'ship-frozen-calibration-transfer-later',
        'common':'ship-common-phase-transfer','independent':'ship-physical-arc-refined'}.items()}
    inputs={name:json.loads(path.read_text()) for name,path in paths.items()}
    for name in ('branches','exact','later','common','independent'):
        if not inputs[name]['complete']:raise ValueError(f'{name} is incomplete')
    if not inputs['early'].get('stopped_reason'):raise ValueError('Expected explicit early-part termination provenance')
    transfer=inputs['early']['trials']+inputs['later']['trials']
    labels=list(dict.fromkeys(r['label'] for r in transfer));calibrations=inputs['early']['calibrations']
    if len(labels)!=11 or len(transfer)!=33:raise ValueError('All eleven separate intervals and three geometries are required')
    lookup={(r['label'],r['calibration']):r for r in transfer}
    if len(lookup)!=len(transfer):raise ValueError('Duplicate interval/calibration result')
    log=(args.output/'tests.log').read_text()
    tests=re.search(r'Ran (\d+) tests',log)
    if not tests or not log.rstrip().endswith('OK'):raise ValueError('Passing numerical test log required')
    summary=dict(scope=__doc__,accepted_calibration=False,new_focused_videos=False,tests_passed=int(tests[1]),
                 separate_intervals=len(labels),frozen_geometry_trials=len(transfer),transfer=[],common_phase=[])
    # All local minima and failed searches remain visible. Do not silently rank
    # a failed branch as an accepted physical site fit.
    local=inputs['branches']['trials']+inputs['exact']['trials']
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    descriptions=[]
    for index,row in enumerate(local):
        reference=next(r['cost'] for r in inputs['independent']['trials'] if r['assignment']==row['assignment'])
        label=f"{row['assignment']} / cycle {row['branch_index']}"+(' / exact residual' if index>=6 else '')
        descriptions.append(label)
        colour='#24699b' if row['converged'] else '#b45f28'
        axes[0].plot(row['cost']-reference,index,'o' if row['converged'] else 'x',color=colour)
        axes[1].semilogx(row['polishing']['final_kkt_residual'],index,'o' if row['converged'] else 'x',color=colour)
    for ax in axes:ax.set_yticks(range(len(local)),descriptions,fontsize=8);ax.invert_yaxis()
    axes[0].set(xlabel='Training objective penalty vs independent band phases',title='Same observations and positive brightness model')
    axes[1].axvline(1e-6,color='#888',linestyle='--');axes[1].set(xlabel='Scaled stationarity residual',title='Filled: converged local minimum; cross: incomplete')
    fig.suptitle('Physical carrier phase restored explicitly · 163–166 s\n00002 = Seven/SBS/Nine/Ten Ornata, ABC Ridge; 02002 moves SBS to Ridge',fontsize=12)
    fig.savefig(args.output/'01-carrier-branches.png',dpi=170);plt.close(fig)
    converged=[r for r in local if r['converged']]
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for row in converged:
        offsets=np.r_[row['antenna_fit']['transmitter_offsets_enu_m'],[row['antenna_fit']['receiver_offset_enu_m']]]
        for component,ax in enumerate(axes):
            ax.plot(range(6),offsets[:,component],'o-',label=f"{row['assignment']} cycle {row['branch_index']}")
    for ax,name in zip(axes,['East','North','Up']):
        ax.axhline(0,color='#aaa',lw=.7);ax.set_xticks(range(6),[*NAMES,'RX'],rotation=40)
        ax.set(ylabel=f'{name} displacement from user prior (m)',title=f'{name} component')
    axes[0].legend(fontsize=8)
    fig.suptitle('Alternative converged conditional fits are not surveyed phase centres\nReceiver horizontal position is fixed; all retained fits still contact parameter bounds',fontsize=12)
    fig.savefig(args.output/'02-location-alternatives.png',dpi=170);plt.close(fig)
    improvement=np.empty((len(labels),2));valid=np.empty_like(improvement,dtype=bool)
    for i,label in enumerate(labels):
        base=lookup[label,calibrations[0]['label']]['independent']
        for j,calibration in enumerate(calibrations[1:]):
            record=lookup[label,calibration['label']]['independent']
            value=100*(error(base)-error(record))/error(base)
            improvement[i,j]=value;valid[i,j]=record['converged'] and base['converged']
            summary['transfer'].append(dict(label=label,interval_s=lookup[label,calibration['label']]['interval_s'],
                calibration=calibration['label'],test_error_reduction_percent=value,
                nominal_test_error=error(base),fitted_test_error=error(record),both_converged=bool(valid[i,j])))
    limit=max(abs(improvement).max(),1.)
    fig,ax=plt.subplots(figsize=(9,8),layout='constrained')
    im=ax.imshow(improvement,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    for i in range(len(labels)):
        for j in range(2):ax.text(j,i,f'{improvement[i,j]:+.2f}%'+(' *' if not valid[i,j] else ''),ha='center',va='center',
            color='white' if abs(improvement[i,j])>limit*.6 else 'black')
    ax.set_xticks(range(2),[c['label'] for c in calibrations[1:]])
    ax.set_yticks(range(len(labels)),[f"{label} · {lookup[label,calibrations[0]['label']]['interval_s'][0]:g} s" for label in labels])
    ax.set_title('Unused-block error reduction after freezing the flash-derived antennas\nPositive = better than nominal 00002; * marks an incomplete local optimization',fontsize=11)
    fig.colorbar(im,ax=ax,label='Reduction in squared prediction error (%)')
    fig.savefig(args.output/'03-interval-transfer.png',dpi=170);plt.close(fig)
    common_labels=list(dict.fromkeys(r['label'] for r in inputs['common']['trials']))
    fig,axes=plt.subplots(1,len(common_labels),figsize=(14,5),layout='constrained')
    for ax,label in zip(axes,common_labels):
        independent_values=[];common_values=[]
        for calibration in calibrations:
            row=next(r for r in inputs['common']['trials'] if r['label']==label and r['calibration']==calibration['label'])
            best=min(row['common_trials'],key=lambda t:t['cost'])
            independent_values.append(error(row['independent']));common_values.append(error(best))
            summary['common_phase'].append(dict(label=label,calibration=calibration['label'],
                independent_test_error=error(row['independent']),common_test_error=error(best),
                common_penalty_percent=100*(error(best)/error(row['independent'])-1),
                converged=best['converged'],branch_index=best['branch_index']))
        normal=independent_values[0]
        x=np.arange(3);ax.bar(x-.18,np.array(independent_values)/normal,.36,label='Independent constant band phase')
        ax.bar(x+.18,np.array(common_values)/normal,.36,label='Common physical scattering phase')
        ax.set_xticks(x,['Nominal','Cycle 1 cal.','Cycle 8 cal.'],rotation=25)
        ax.set(title=label,ylabel='Unused-block error / nominal independently phased error')
    axes[0].legend(fontsize=8)
    fig.suptitle('Same five channels, frozen antennas, locally refitted physical point\nThree local carrier starts per case; selected by training likelihood only',fontsize=12)
    fig.savefig(args.output/'04-common-phase-transfer.png',dpi=170);plt.close(fig)
    sea_path=Path('docs/data/isar_20261016/flash-reflection/analysis.json')
    sea=json.loads(sea_path.read_text())
    example=next(r for r in sea['ship_sea_paths'] if r['site']=='Ornata' and r['scatterer_height_above_water_m']==20.)
    summary['hh_example']=example
    summary['converged_carrier_fits']=[dict(assignment=r['assignment'],cycle=r['branch_index'],
        training_cost=r['cost'],bounds=r['bound_indices'],kkt=r['polishing']['final_kkt_residual']) for r in converged]
    summary['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
                              for p in [*paths.values(),sea_path,args.output/'tests.log',Path(__file__)]}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    rows=''.join(f"<tr><td>{r['label']}</td><td>{r['calibration']}</td><td>{r['test_error_reduction_percent']:+.3f}%</td><td>{r['both_converged']}</td></tr>" for r in summary['transfer'])
    minimum_rows=''.join(f"<tr><td>{r['assignment']}</td><td>{r['branch_index']}</td><td>{r['cost']:.3f}</td><td>{r['polishing']['final_kkt_residual']:.3g}</td><td>{html.escape(str(r['bound_indices']))}</td></tr>" for r in converged)
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ship phase and calibration transfer</title>
<style>body{{font:16px/1.55 system-ui;max-width:1200px;margin:32px auto;padding:0 20px}}img{{width:100%}}.note{{padding:16px;background:#fff1d3}}table{{border-collapse:collapse}}td,th{{padding:7px;border:1px solid #ccc}}</style>
<h1>Ship phase: carrier ambiguity, separate intervals and sea reflection</h1>
<p class="note">The flash-derived geometry helps some neighbouring intervals but does not improve prediction consistently across the capture. No phase-centre calibration is accepted. Converged local solutions, incomplete searches and the additional common-scattering-phase assumption are shown separately. This does not prove that the physical calibration is impossible.</p>
<h2>Horizontal polarisation: a strong sea path is plausible</h2>
<p>A smooth sea-reflected leg can be almost as strong as the corresponding direct leg. For the declared 20 m reflector height and Ornata geometry at 200 MHz, the illustrative HH Fresnel coefficients have losses of {example['legs'][0]['fresnel_power_db']:.3f} dB on the transmit leg and {example['legs'][1]['fresnel_power_db']:.3f} dB on the receive leg, with phase close to 180°. Convex-Earth spreading, roughness, shoreline, blockage, antenna patterns and directional ship scattering also matter. The permittivity and conductivity are illustrative values, not a measurement of the bay.</p>
<p>The four possibilities are Tx→ship→Rx, Tx→sea→ship→Rx, Tx→ship→sea→Rx, and a sea reflection on both legs. Their electric fields add coherently. A passive sea interface does not supply gain, but the sum can be stronger than the direct ship echo and the sea-assisted ray can illuminate a different strong ship facet. It can also cancel the direct echo. The resulting phase is frequency dependent even with fixed electronics and one physical ship structure. This is a reason to test a common cross-band scattering phase, not assume it from acquisition coherence.</p>
<p>The extra geometric paths in the 20 m example are {example['extra_paths_m'][1]:.3f}, {example['extra_paths_m'][2]:.3f} and {example['extra_paths_m'][3]:.3f} m. They are unresolved within the 37.47 m excess-path sample spacing. Across the tested 5–60 m heights, the largest two-sea-reflection extra path is 2.732 m. This mechanism does not by itself explain a separately resolved companion hundreds of metres behind the ship. The smooth-sea-only Tx→sea→Rx specular centre is inland in the current geometry.</p>
<img src="../../isar_20261016/flash-reflection/02-sea-path-scales.png" alt="Sea-path delay and conditional HH interference versus reflector height">
<p><a href="../../isar_20261016/flash-reflection/index.html">Reflection points, terrain and finite-side calculations</a>; <a href="../facet-sea/index.html">finite-side four-path phase dynamics</a>. The latter is a declared 20 m-height sensitivity control and does not exhaust all heights or unknown ship structures. The ideal infinite-water curve is conditional; a specular footprint can cross the shoreline.</p>
<p>Prior art: <a href="https://www.mdpi.com/2072-4292/11/7/748">Ship Profile Imaging Using Multipath Backscattering (2019)</a> develops the ship/sea four-path construction, and <a href="https://researchspace.csir.co.za/bitstream/handle/10204/1071/lord_2006.pdf">Lord, Nel and Abdul Gaffar (2006)</a> explicitly study sea multipath in ship ISAR. Their microwave numerical results are not VHF measurements of this capture.</p>
<h2>Restore the carrier before fitting shared phase</h2>
<p>The independently phased fit conditions each band at the interval midpoint. Before enforcing a common physical scattering phase, the fit restores −k[L(midpoint)−L(saved reference)] for each RF frequency, including geometry derivatives and the transmitter-to-receiver reference baseline. Synthetic tests compare with the direct unconditioned original-row operator. Independent band phases remain the control model. Positive brightness knots and stationary clutter are fitted on training blocks; neither supplies arbitrary time-dependent phase.</p>
<img src="01-carrier-branches.png" alt="Local carrier branches, objective penalty and numerical stationarity">
<p>Two 00002 carrier branches are strictly converged local minima. An exact profiled-residual Jacobian also obtains one converged 02002 branch; two lower-cost 02002 searches remain incomplete. The new residual Jacobian includes the derivative of the fitted real amplitudes and agrees with the original likelihood and numerical derivatives. The objective, data weights and priors are unchanged. These are local searches, not an exhaustive integer-cycle proof or a likelihood-ratio significance test under an established scattering model.</p>
<p>The left plot subtracts a different independently phased baseline for each site assignment. It compares the price of imposing common phase within each assignment, not their absolute fit quality. The absolute penalized training objectives of the converged branches are:</p>
<table><tr><th>Assignment</th><th>Carrier branch</th><th>Absolute objective</th><th>Stationarity residual</th><th>Bound parameter indices</th></tr>{minimum_rows}</table>
<img src="02-location-alternatives.png" alt="Alternative antenna coordinate corrections for converged fits">
<p>00002 places Seven, SBS, Nine and Ten at Ornata and ABC at Ridge. 02002 instead places SBS at Ridge. Receiver latitude/longitude stay fixed at the user’s coordinates; height is bounded. The fitted displacements still depend on the carrier branch and contact nuisance/antenna bounds. No fitted positions were written into capture metadata.</p>
<h2>Freeze those antennas and predict other intervals</h2>
<p>The full-recording training search selected twelve three-second regions. The region overlapping the 163–166 s calibration was excluded, leaving eleven separate intervals. Three antenna geometries are compared: nominal 00002 and its two converged flash-derived carrier branches. Each interval gets its own physical position/velocity/acceleration, AIS nuisance, and independent constant band phases. The antennas stay fixed. Alternating 125 ms blocks train the fit; coefficients are then frozen to predict the other blocks.</p>
<img src="03-interval-transfer.png" alt="All eleven interval transfer error reductions">
<p>These are squared prediction-error changes, not image SNR gains, probabilities of the site assignments or independent ground-truth vessel tracks. The regions can contain extended scattering, a sidelobe or more than one target; the test does not declare them point reflectors merely because they were detected. Optimizer failures remain marked. The first four intervals were completed before an AIS helper rejected a later knot crossing; the remaining seven use a checked piecewise representation of the original AIS Hermite curve. Position, velocity and complex-step geometry derivatives are tested on both sides of a knot. The existing 10 µm numerical tolerance was retained.</p>
<p><a href="../full-capture-ship-arcs/candidates.png">Full-recording candidate search</a>; <a href="../full-capture-ship-arc-validation/index.html">all independent-band arc checks</a>.</p>
<img src="04-common-phase-transfer.png" alt="Common physical phase versus independent band phases on three separate intervals">
<p>The common-phase check uses three local carrier starts per geometry, selected by original training likelihood. Its best starts converge in all nine cases. F01 is the brightest earlier independent interval, F03 the next earlier main-range candidate, and F05 the strongest later candidate in the training discovery. Antennas cannot move in these fits. A good fit to one interval is insufficient to establish a transferable common scattering phase; the retained separate-band control exposes the remaining mismatch.</p>
<details><summary>All separate-interval error changes</summary><table><tr><th>Interval</th><th>Frozen calibration</th><th>Unused error reduction</th><th>Both fits converged</th></tr>{rows}</table></details>
<p><a href="summary.json">Summary and exact source hashes</a>; <a href="tests.log">{int(tests[1])} passing numerical tests</a>. All changes are support analysis. Production DSP and the captured observations are unchanged. The goal of conclusive antenna calibration and validated focused videos remains open.</p></html>'''
    (args.output/'index.html').write_text(page)
    print(json.dumps({k:summary[k] for k in ('tests_passed','separate_intervals','frozen_geometry_trials','converged_carrier_fits')},indent=2))


if __name__=='__main__':main()
