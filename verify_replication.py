"""Audit included historical reproduction records; no training or network access."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, pstdev


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def close(a, b, message):
    require(math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-10), message)


def verify_records(root):
    folder = Path(root) / 'results/replication'
    manifest = json.loads((folder / 'source_manifest.json').read_text())
    records = manifest['files']
    expected = {'raw/amazon_original_10runs.log'}
    for group, count in [('amazon_diagnostic', 5), ('yelp_replication', 10)]:
        for seed in range(count):
            expected.update('raw/{}/seed_{}/{}'.format(group, seed, name)
                            for name in ['epochs.csv', 'summary.json'])
    require(len(records) == 31 and {r['path'] for r in records} == expected,
            'Replication source inventory differs.')
    for record in records:
        path = folder / record['path']
        content = path.read_bytes()
        require(len(content) == record['bytes'] and hashlib.sha256(content).hexdigest() == record['sha256'],
                'Historical source bytes differ: ' + record['path'])
    log = (folder / 'raw/amazon_original_10runs.log').read_text(encoding='utf-16')
    epochs = re.findall(r'^Epoch (\d+),', log, re.MULTILINE)
    require(epochs == [str(i) for _ in range(10) for i in range(100)],
            'Original Amazon ten-run epoch inventory differs.')
    final = re.findall(r'Test: REC ([\d.]+) PRE ([\d.]+) MF1 ([\d.]+) AUC ([\d.]+)', log)
    require(len(final) == 10, 'Expected ten original final test lines.')
    derived = {}
    for index, line in enumerate(final):
        derived[('amazon_original_unseeded', str(index), 'val_macro_f1')] = {
            'test_macro_f1': float(line[2])/100, 'test_auc_prob32': float(line[3])/100,
            'test_auc_margin64': '', 'seed': 'not_recorded',
            'epoch_zero_based': 'not_recoverable_from_final_test_line', 'threshold': 'not_recorded'}
    for group, count in [('amazon_diagnostic', 5), ('yelp_replication', 10)]:
        split_hashes = None
        for seed in range(count):
            directory = folder / 'raw' / group / ('seed_' + str(seed))
            summary = json.loads((directory / 'summary.json').read_text())
            rows = read_csv(directory / 'epochs.csv')
            require(summary['seed'] == seed and summary['observed_epochs'] == 100 and
                    [int(r['epoch_zero_based']) for r in rows] == list(range(100)),
                    'Seed/epoch inventory differs: ' + str(directory))
            require(set(summary['selections']) == {'val_macro_f1', 'val_auc_prob32', 'val_auc_margin64'},
                    'Validation selector inventory differs.')
            if split_hashes is None:
                split_hashes = summary['split_hashes']
            require(summary['split_hashes'] == split_hashes, 'Split changes within reproduction setting.')
            for selector, selected in summary['selections'].items():
                # Python max retains the first tied maximum, matching the strict > rule.
                best = max(rows, key=lambda row: float(row[selector]))
                require(int(selected['epoch_zero_based']) == int(best['epoch_zero_based']),
                        'Selection is not the first validation maximum: ' + group + '/' + str(seed) + '/' + selector)
                for name in ['val_macro_f1', 'val_auc_prob32', 'val_auc_margin64', 'threshold',
                             'test_macro_f1', 'test_auc_prob32', 'test_auc_margin64']:
                    close(selected[name], best[name], 'Selected epoch field differs: ' + name)
                derived[(group, str(seed), selector)] = dict(selected, seed=seed)
    published = read_csv(folder / 'per_run_results.csv')
    require(len(published) == len(derived) == 55 and
            len({(r['setting'], r['run_id'], r['selector']) for r in published}) == 55,
            'Per-run inventory differs.')
    for row in published:
        key = (row['setting'], row['run_id'], row['selector'])
        require(key in derived, 'Unknown per-run record.')
        for name in ['seed', 'epoch_zero_based', 'threshold', 'test_macro_f1', 'test_auc_prob32', 'test_auc_margin64']:
            expected_value = derived[key][name]
            try:
                numeric = float(expected_value)
            except (TypeError, ValueError):
                require(row[name] == str(expected_value), 'Non-numeric per-run field differs.')
            else:
                close(row[name], numeric, 'Per-run field differs: ' + name)
    summaries = json.loads((folder / 'summary.json').read_text())
    keys = {('amazon_original_unseeded', 'val_macro_f1'), ('yelp_replication', 'val_macro_f1'),
            ('amazon_diagnostic', 'val_macro_f1'), ('amazon_diagnostic', 'val_auc_margin64')}
    require(len(summaries) == 4 and {(s['setting'], s['selector']) for s in summaries} == keys,
            'Aggregate inventory differs.')
    for summary in summaries:
        rows = [r for r in published if (r['setting'], r['selector']) == (summary['setting'], summary['selector'])]
        require(summary['n_trajectories'] == len(rows) and summary['sd_convention'] == 'population_ddof0',
                'Aggregate count or SD convention differs.')
        for metric in ['test_macro_f1', 'test_auc_prob32', 'test_auc_margin64']:
            values = [100*float(r[metric]) for r in rows if r[metric] != '']
            for suffix, value in [('mean_percent', mean(values) if values else None),
                                  ('population_sd_percent', pstdev(values) if values else None)]:
                actual = summary[metric + '_' + suffix]
                if value is None:
                    require(actual is None, 'Missing original margin scores must remain missing.')
                else:
                    close(actual, value, 'Replication aggregate differs: ' + metric)
    return {'status': 'passed', 'source_files': 31, 'original_unseeded_runs': 10,
            'seeded_trajectories': 15, 'selection_rows': 55, 'summary_rows': 4,
            'scope': 'Historical log, validation selection and aggregate consistency; no private prediction or checkpoint execution.'}


if __name__ == '__main__':
    print(json.dumps(verify_records(Path(__file__).resolve().parent), indent=2))
    print('REPLICATION RECORD CHECK: PASS')
