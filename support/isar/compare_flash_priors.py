"""Compare old and revised antenna priors with identical data and constraints."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES=['Seven','SBS','Nine','Ten','ABC']
LETTERS={'12013':'O','12015':'E','12014':'R'}


def code(row):
    return '/'.join(LETTERS[s] for s in row['sites'])


def case(result, with_range, acceleration, width=5):
    return next(c for c in result['cases'] if c['width_cells']==width and
                c['include_range']==with_range and c['include_acceleration']==acceleration)


def best(c):
    return min(c['records'],key=lambda r:r['penalized_squared_residual'])


def compare(old, new, output):
    if old['fit_options']!=new['fit_options']:
        raise ValueError('Receiver constraints must match')
    for previous,current in zip(old['cases'],new['cases']):
        np.testing.assert_array_equal(previous['measurements'],current['measurements'])
        np.testing.assert_array_equal(previous['covariance'],current['covariance'])
        if [r['sites'] for r in previous['records']] != [r['sites'] for r in current['records']]:
            raise ValueError('Assignment sets must match')
    output.mkdir(parents=True,exist_ok=True)
    summary=[]
    fig,axes=plt.subplots(3,1,figsize=(14,11),layout='constrained')
    for ax,(with_range,acceleration,title) in zip(axes,[(False,False,'Path rate only'),
            (False,True,'Path rate and curvature'),(True,True,'Envelope range, path rate and curvature')]):
        previous,current=case(old,with_range,acceleration),case(new,with_range,acceleration)
        order=sorted(range(len(current['records'])),key=lambda i:current['records'][i]['penalized_squared_residual'])
        for c,label,colour in [(previous,'Previous priors','#d55e00'),(current,'Revised priors','#0072b2')]:
            ax.plot(np.arange(len(order)),[c['records'][i]['penalized_squared_residual'] for i in order],
                    '.-',color=colour,label=label,lw=1)
        ax.set_yscale('log');ax.set_ylabel('Penalized squared residual');ax.set_title(title)
        ax.set_xticks(np.arange(len(order))[::2],[code(current['records'][i]) for i in order[::2]],rotation=65,fontsize=8)
        ax.legend(fontsize=9)
        before,after=best(previous),best(current)
        summary.append({'observations':title,'old':before,'new':after,
                        'new_data_rms':[float(v) for v in np.sqrt(np.mean((np.array(after['predicted_observables'])-current['measurements'])**2,axis=0))],
                        'old_data_rms':[float(v) for v in np.sqrt(np.mean((np.array(before['predicted_observables'])-previous['measurements'])**2,axis=0))]})
    fig.suptitle('Revised site priors · same 32 assignments, observations and receiver constraints\n'
                 'O Ornata · E Eyre · R Ridge; channel order Seven/SBS/Nine/Ten/ABC')
    fig.savefig(output/'01-prior-comparison.png',dpi=150);plt.close(fig)
    previous,current=case(old,True,True),case(new,True,True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for c,label,colour in [(previous,'Previous priors','#d55e00'),(current,'Revised priors','#0072b2')]:
        row=best(c);residual=np.array(row['predicted_observables'])-c['measurements']
        sigma=np.sqrt(np.diagonal(c['covariance'],axis1=1,axis2=2))
        for j,(ax,scale,title) in enumerate(zip(axes,[1.,1000.,1000.],['Range (m)','Path rate (mm/s)','Path acceleration (mm/s²)'])):
            ax.errorbar(np.arange(5),residual[:,j]*scale,yerr=sigma[:,j]*scale,fmt='o-',ms=4,
                        color=colour,label=label+': '+code(row),lw=1)
            ax.axhline(0,color='#777777',lw=.5);ax.set_xticks(np.arange(5),NAMES,rotation=25)
            ax.set_ylabel('Prediction minus measured');ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.suptitle('Best range-inclusive fits · error bars are conditional extraction standard deviations')
    fig.savefig(output/'02-observable-residuals.png',dpi=160);plt.close(fig)
    row=best(current)
    offsets=np.array(row['antenna_fit']['transmitter_offsets_enu_m'])
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for j,(ax,label) in enumerate(zip(axes,['East','North','Up'])):
        ax.bar(NAMES,offsets[:,j],color='#0072b2');ax.axhline(0,color='#777777',lw=.5)
        ax.set_ylabel('Fitted displacement from revised prior (m)');ax.set_title(label)
    fig.suptitle('Corrections demanded by the best range-inclusive solution: '+code(row)+
                 f"\n{row['bound_hits']} active bounds; receiver height {10+row['antenna_fit']['receiver_offset_enu_m'][2]:.2f} m AGL")
    fig.savefig(output/'03-demanded-corrections.png',dpi=160);plt.close(fig)
    compact={'scope':__doc__,'fit_options':new['fit_options'],'assignment_count':new['assignment_count'],
             'identical_measurements_and_covariances':True,'summary':summary,
             'maximum_remaining_local_gauss_newton_squared_cost_decrease':max(r['estimated_remaining_gauss_newton_squared_cost_decrease'] for c in new['cases'] for r in c['records']),
             'common_phase_validation':new['common_phase_validation'],
             'accepted_calibration':False}
    (output/'comparison.json').write_text(json.dumps(compact,indent=2)+'\n')
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [s['observations'],code(s['old']),
        code(s['new']),f"{s['old']['data_squared_residual']:.3f}",f"{s['new']['data_squared_residual']:.3f}",
        s['new']['bound_hits']])+'</tr>' for s in summary)
    body=f'''<!doctype html><meta charset="utf-8"><title>Revised transmitter prior fit</title>
<style>body{{font:17px/1.5 system-ui;max-width:1300px;margin:2em auto;padding:0 1em}}img{{width:100%}}td,th{{padding:.5em;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff2cc;padding:1em}}</style>
<h1>Revised transmitter priors: matched refit</h1>
<p>The new coordinates and antenna heights have little effect on this short-flash fit. The best range-inclusive RMS changes from {summary[2]['old_data_rms'][0]:.3f} to {summary[2]['new_data_rms'][0]:.3f} m. Both old and new priors are tested with the same 32 assignments, exactly the same extracted observations and covariance, fixed receiver latitude/longitude, and receiver height bounded to 0–20 m AGL around the 10 m nominal height. Transmitter prior widths remain unchanged.</p>
<p class="note">This is a completed diagnostic refit, not an accepted antenna calibration. The best range-inclusive solution still reaches transmitter-coordinate bounds. Numerical convergence is good: the largest estimated remaining local Gauss–Newton decrease over all new-prior fits is {compact['maximum_remaining_local_gauss_newton_squared_cost_decrease']:.3g} in the squared objective. This local check is separate from physical model validity and global identification.</p>
<table><tr><th>Observations</th><th>Old best</th><th>New best</th><th>Old data residual²</th><th>New data residual²</th><th>New bound contacts</th></tr>{rows}</table>
<p>O/E/R mean Ornata/Eyre/Ridge. Channel order is Seven, SBS, Nine, Ten, ABC. Scores are conditional profile-likelihood residuals, not posterior site probabilities. Different rows use different observations and cannot be ranked against one another.</p>
<img src="01-prior-comparison.png"><img src="02-observable-residuals.png"><img src="03-demanded-corrections.png">
<h2>Phase control</h2><p>The existing common-phase control fits a phase origin shared by all five bands and path rate, while excluding absolute range and curvature. Its best fitted midpoint phase RMS remains about {min(new['common_phase_control']['records'],key=lambda r:r['conditional_squared_cost'])['phase_rms_deg']:.3f}°. The unused-sample phase RMS remains approximately 5–9°. The excluded range error is material; this control alone is not a location solution.</p>
<p>The next test retains phase, envelope range, rate and curvature together, including their full covariance. The single-point extraction is also being checked against multiple-component and shorter-window controls before its summaries are used to constrain antenna locations.</p>
<p><a href="comparison.json">Numerical comparison</a> · <a href="../flash-siting-sites-updated/index.html">Complete new-prior diagnostics</a> · <a href="../flash-siting-old-priors-matched/index.html">Old priors under matching constraints</a></p>'''
    (output/'index.html').write_text(body)
    return compact


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,default=Path('docs/data/isar_20261016/flash-siting-old-priors-matched/flash-siting.json'))
    p.add_argument('--new',type=Path,default=Path('docs/data/isar_20261016/flash-siting-sites-updated/flash-siting.json'))
    p.add_argument('--output',type=Path,default=Path('docs/data/isar_20261016/transmitter-prior-refit'))
    args=p.parse_args();compare(json.loads(args.old.read_text()),json.loads(args.new.read_text()),args.output)
    print(args.output/'index.html')


if __name__=='__main__':main()
