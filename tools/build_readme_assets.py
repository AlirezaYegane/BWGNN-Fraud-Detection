"""Build README graphics from saved CSVs; optional Matplotlib, no training or API calls."""
from pathlib import Path
import base64
import csv
import hashlib
import json
from statistics import mean, stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets' / 'readme'
OUT.mkdir(parents=True, exist_ok=True)
RED, MAROON, INK, GREY = '#C8102E', '#76232F', '#242B35', '#697586'
LIGHT, BORDER = '#F7F8FA', '#E5E8ED'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titleweight': 'bold',
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': GREY, 'ytick.color': INK,
    'axes.edgecolor': BORDER, 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})


def rows(relative):
    return list(csv.DictReader((ROOT / relative).open(encoding='utf-8-sig')))


def polish(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='x', color=BORDER, linewidth=0.8)
    for side in ['left', 'right', 'top']:
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=9)


core_path = 'results/comp8240_a3_core_analysis/summary_by_condition.csv'
sens_path = 'results/comp8240_a3_sensitivity_analysis/sensitivity_summary.csv'
gaps_path = 'results/comp8240_a3_core_analysis/paired_model_gaps.csv'
core, sens, gaps = rows(core_path), rows(sens_path), rows(gaps_path)
ethereum = {r['model']: r for r in core if r['dataset'] == 'ethereum'}
matched = {r['configuration'].replace('matched_', ''): r for r in sens if r['configuration'].startswith('matched_')}
amazon = {(r['condition'], r['model']): r for r in core if r['dataset'] == 'amazon'}
names, colors = ['bwgnn', 'gcn', 'mlp'], [RED, INK, '#8A97A8']

fig, axes = plt.subplots(1, 2, figsize=(12, 4.65), sharex=True)
fig.subplots_adjust(left=0.09, right=0.975, bottom=0.20, top=0.72, wspace=0.40)
fig.text(0.04, 0.94, 'Does the result hold with a closer model budget?', fontsize=19, weight='bold')
fig.text(0.04, 0.866, 'Ethereum test Macro-F1  |  Five model seeds per setting  |  Error bars: sample SD', fontsize=11, color=GREY)
for ax, table, title in zip(axes, [ethereum, matched], ['Primary configuration', 'Near-equal parameter budgets']):
    for y, model, color in zip([2, 1, 0], names, colors):
        r = table[model]
        value, sd = float(r['macro_f1_mean_percent']), float(r['macro_f1_sd_percent'])
        ax.errorbar(value, y, xerr=sd, fmt='o', color=color, markersize=10, capsize=5, elinewidth=2.2, zorder=3)
        ax.text(value + sd + 0.35, y + 0.17, '{:.2f} ± {:.2f}'.format(value, sd), color=color, fontsize=10, weight='bold')
    ax.set_yticks([2, 1, 0], ['BWGNN', 'GCN', 'MLP'])
    ax.set_xlim(86.5, 99.5); ax.set_ylim(-0.6, 2.65)
    ax.set_xticks([88, 90, 92, 94, 96, 98])
    ax.set_xlabel('Test Macro-F1 (%)', labelpad=12)
    ax.set_title(title, loc='left', fontsize=12, pad=15)
    polish(ax)
fig.text(0.04, 0.052, 'Primary: original loop policies. Budget check: one self-loop per node for both GNNs; MLP uses features only.', fontsize=9, color=GREY)
fig.savefig(OUT / 'ethereum-comparison.png', dpi=180)
plt.close(fig)

conditions = ['camo_c0', 'camo_c0p2', 'camo_c0p4', 'camo_c0p6', 'camo_c0p8']
strengths = np.array([0, 0.2, 0.4, 0.6, 0.8])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), gridspec_kw={'width_ratios': [1.25, 1]})
fig.subplots_adjust(left=0.075, right=0.965, bottom=0.22, top=0.72, wspace=0.46)
fig.text(0.04, 0.94, 'The model gap grows under oracle camouflage', fontsize=19, weight='bold')
fig.text(0.04, 0.867, 'Amazon stress test  |  Five paired construction/model seeds  |  Error bars: sample SD', fontsize=11, color=GREY)
ax = axes[0]
for model, color, marker in [('bwgnn', RED, 'o'), ('gcn', INK, 's')]:
    values = np.array([float(amazon[(c, model)]['macro_f1_mean_percent']) for c in conditions])
    sds = np.array([float(amazon[(c, model)]['macro_f1_sd_percent']) for c in conditions])
    ax.errorbar(strengths, values, yerr=sds, color=color, marker=marker, linewidth=2.3, markersize=6, capsize=4, label=model.upper())
    ax.annotate('{:.2f}'.format(values[-1]), (strengths[-1], values[-1]), xytext=(7, 4), textcoords='offset points', color=color, fontsize=10, weight='bold')
ax.set_xlim(-0.035, 0.94); ax.set_ylim(62, 100); ax.set_yticks([65, 75, 85, 95])
ax.set_xticks(strengths, ['0', '0.2', '0.4', '0.6', '0.8'])
ax.set_xlabel('Oracle camouflage strength', labelpad=12); ax.set_ylabel('Test Macro-F1 (%)', labelpad=9)
ax.legend(loc='center left', frameon=False, fontsize=10)
ax.set_axisbelow(True); ax.grid(axis='y', color=BORDER)
for side in ['top', 'right']: ax.spines[side].set_visible(False)
ax.tick_params(length=0, pad=8)
ax = axes[1]
gap_data = {}
for y, condition, label, color in [(2, 'camo_c0', 'Baseline', GREY), (1, 'camo_c0p8', 'Oracle 0.8', RED), (0, 'random_control', 'Random control', INK)]:
    values = [float(r['bwgnn_minus_gcn_macro_f1_pp']) for r in gaps if r['condition'] == condition]
    assert len(values) == 5
    value, sd = mean(values), stdev(values)
    gap_data[condition] = {'n': len(values), 'mean_pp': value, 'sample_sd_pp': sd}
    ax.errorbar(value, y, xerr=sd, fmt='o', color=color, markersize=9, elinewidth=2.2, capsize=5)
    ax.text(value + sd + 0.15, y + 0.15, '{:.2f} ± {:.2f}'.format(value, sd), fontsize=9.5, color=color, weight='bold')
ax.set_yticks([2, 1, 0], ['Baseline', 'Oracle 0.8', 'Random control'])
ax.set_xlim(21, 31.5); ax.set_ylim(-0.5, 2.6); ax.set_xticks([22, 24, 26, 28, 30])
ax.set_xlabel('Paired BWGNN − GCN gap (pp)', labelpad=12)
polish(ax)
fig.text(0.04, 0.082, 'Oracle construction uses eligible validation/test labels. Random control matches only the largest swap budget.', fontsize=9, color=GREY)
fig.text(0.04, 0.037, 'This is a controlled label-aware stress test on a fixed dataset; it does not establish robustness to real-world attacks.', fontsize=9, color=GREY)
fig.savefig(OUT / 'amazon-camouflage.png', dpi=180)
plt.close(fig)

# Place the supplied logo unchanged inside a white SVG document panel, readable in either GitHub theme.
logo = (ROOT / 'logos' / 'macquarie_logo.png').read_bytes()
logo64 = base64.b64encode(logo).decode('ascii')
header = '''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1280" height="205" viewBox="0 0 1280 205">
<rect width="1280" height="205" rx="14" fill="#FFFFFF"/>
<image x="48" y="30" width="430" height="129" xlink:href="data:image/png;base64,LOGO"/>
<rect x="1224" width="56" height="199" fill="#76232F"/>
<path d="M1190 0H1280V115Z" fill="#C8102E"/>
<text x="1170" y="77" text-anchor="end" font-family="Arial,sans-serif" font-weight="700" font-size="28" fill="#76232F">COMP8240</text>
<text x="1170" y="113" text-anchor="end" font-family="Arial,sans-serif" font-size="18" fill="#4D5868">Assessment 3 · Research project</text>
<text x="1170" y="144" text-anchor="end" font-family="Arial,sans-serif" font-size="16" fill="#697586">Alireza Yegane · 60957107</text>
<rect y="199" width="1280" height="6" fill="#C8102E"/>
</svg>'''.replace('LOGO', logo64)
(OUT / 'university-header.svg').write_text(header, encoding='utf-8')
cards = ['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="135" viewBox="0 0 1280 135"><rect width="1280" height="135" rx="12" fill="#F7F8FA"/>']
for n, (value, label) in enumerate([('128', 'Verified extension fits'), ('5', 'Seeds per primary setting'), ('3', 'Ethereum constructions'), ('72', 'Explanation ratings')]):
    x = 160 + 320 * n
    if n: cards.append('<path d="M{} 27V108" stroke="#E0E4EB"/>'.format(320 * n))
    cards.append('<text x="{}" y="63" text-anchor="middle" font-family="Arial,sans-serif" font-weight="700" font-size="38" fill="#76232F">{}</text>'.format(x, value))
    cards.append('<text x="{}" y="98" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" fill="#4D5868">{}</text>'.format(x, label))
cards.append('</svg>')
(OUT / 'study-snapshot.svg').write_text(''.join(cards), encoding='utf-8')
data = {'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in [core_path, sens_path, gaps_path]},
        'ethereum_primary': ethereum, 'ethereum_matched': matched, 'amazon': list(amazon.values()), 'paired_gaps': gap_data,
        'uncertainty': 'Sample SD (ddof=1), not confidence intervals', 'logo_source_sha256': hashlib.sha256(logo).hexdigest()}
(OUT / 'figure-data.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
print('README graphics rebuilt from audited CSVs. No model fitting or API calls.')
