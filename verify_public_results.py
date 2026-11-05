"""Check published result consistency using Python's standard library only.

No network, model training, API calls, or upstream bootstrap is required.
This checks published tables and raw judge evidence, not saved model predictions.
Use verify_submission.py / verify_sampling.py for the separate prediction audits.
"""
import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
from statistics import mean, stdev
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
METRICS = ('macro_f1', 'roc_auc_margin64', 'average_precision_margin64', 'pr_auc_trapezoid_margin64')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def grouped(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[key] for key in keys)].append(row)
    return result


def check_summaries(rows, summaries, keys, n_field, sd_suffix='sd_percent'):
    groups = grouped(rows, keys)
    require(len(summaries) == len(groups), 'Summary group count differs.')
    require(len({tuple(r[k] for k in keys) for r in summaries}) == len(summaries), 'Duplicate summary group.')
    for row in summaries:
        key = tuple(row[k] for k in keys)
        require(key in groups, 'Unknown summary group: ' + str(key))
        fits = groups[key]
        require(int(row[n_field]) == len(fits), 'Summary sample count differs: ' + str(key))
        for metric in METRICS:
            values = [100 * float(fit[metric]) for fit in fits]
            for suffix, expected in [('mean_percent', mean(values)), (sd_suffix, stdev(values))]:
                require(math.isclose(float(row[metric + '_' + suffix]), expected, abs_tol=1e-9),
                        'Summary differs from per-fit scores: ' + str(key) + '/' + metric + '_' + suffix)


def verify(root):
    checks = []
    for path in sorted(root.rglob('*.py')):
        if not any(part.startswith('.') or part == '__pycache__' for part in path.relative_to(root).parts):
            ast.parse(path.read_text(encoding='utf-8-sig'), feature_version=(3, 8))
    checks.append('Python 3.8 syntax')
    from verify_replication import verify_records
    replication = verify_records(root)
    checks.append('Original replication: ten-run Amazon log, 15 seeded trajectories, validation-only selection and four population-SD summaries')
    core_dir = root / 'results/comp8240_a3_core_analysis'
    sens_dir = root / 'results/comp8240_a3_sensitivity_analysis'
    sample_dir = root / 'results/comp8240_a3_sampling_analysis'
    core = read_csv(core_dir / 'all_model_results.csv')
    sens = read_csv(sens_dir / 'sensitivity_results.csv')
    sampling = read_csv(sample_dir / 'sampling_results.csv')
    require((len(core), len(sens), len(sampling)) == (75, 35, 27), 'Per-fit inventory differs.')
    inventories = [(core, ('dataset', 'condition', 'model', 'model_seed'), 5, 'model_seed'),
                   (sens, ('configuration', 'seed'), 5, 'seed'),
                   (sampling, ('construction_seed', 'model', 'model_seed'), 3, 'model_seed')]
    for rows, identity, n_seeds, seed_field in inventories:
        require(len({tuple(r[k] for k in identity) for r in rows}) == len(rows), 'Duplicate fit identity.')
        group_keys = tuple(k for k in identity if k != seed_field)
        for fits in grouped(rows, group_keys).values():
            require({int(r[seed_field]) for r in fits} == set(range(n_seeds)), 'Incomplete seed set.')
        for row in rows:
            for metric in METRICS:
                value = float(row[metric])
                require(math.isfinite(value) and 0 <= value <= 1, 'Invalid metric in per-fit table.')
    check_summaries(core, read_csv(core_dir / 'summary_by_condition.csv'), ('dataset', 'condition', 'model'), 'n')
    check_summaries(sens, read_csv(sens_dir / 'sensitivity_summary.csv'), ('configuration',), 'n')
    check_summaries(sampling, read_csv(sample_dir / 'sampling_summary.csv'), ('construction_seed', 'model'), 'n_model_seeds', 'sample_sd_percent')
    baseline = {(r['model'], r['model_seed']): r for r in core if r['dataset'] == 'ethereum'}
    for row in sampling:
        if row['construction_seed'] == '0':
            reference = baseline[(row['model'], row['model_seed'])]
            require(all(row[k] == reference[k] for k in METRICS), 'Reused construction-0 reference differs.')
    require(sum(r['construction_seed'] != '0' for r in sampling) == 18, 'New construction-fit count differs.')
    require(len({r['scored_population_hash'] for r in sens}) == 1, 'Sensitivity scored populations differ.')
    checks.append('75 core + 35 sensitivity + 18 new construction fits; nine reused references are not double counted')
    checks.append('All four metric means and sample SDs recomputed from published per-fit tables')
    core_index = {(r['dataset'], r['condition'], r['model'], r['model_seed']): r for r in core}
    gaps = read_csv(core_dir / 'paired_model_gaps.csv')
    require(len(gaps) == 30, 'Expected 30 Amazon paired gaps.')
    require(len({(r['condition'], r['pair_id']) for r in gaps}) == 30, 'Duplicate paired gap.')
    for row in gaps:
        key = ('amazon', row['condition'])
        a, b = [core_index[key + (model, row['pair_id'])] for model in ['bwgnn', 'gcn']]
        require(math.isclose(float(row['bwgnn_minus_gcn_macro_f1_pp']),
                            100 * (float(a['macro_f1']) - float(b['macro_f1'])), abs_tol=1e-9), 'Paired gap differs.')
    checks.append('Amazon paired gaps recomputed from per-seed scores')
    judge_dir = root / 'results/explanation_judge'
    spec = importlib.util.spec_from_file_location('public_judge_audit', judge_dir / 'audit_judge_evidence.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix='comp8240_public_check_') as temporary:
        temporary = Path(temporary)
        archive = temporary / 'published_judge_evidence.zip'
        with ZipFile(archive, 'w', ZIP_DEFLATED) as handle:
            for path in sorted((judge_dir / 'evidence').rglob('*')):
                if path.is_file():
                    handle.write(path, path.relative_to(judge_dir / 'evidence').as_posix())
        report = module.audit(archive, temporary / 'audit')
        require(report['audit_status'] == 'PASS' and report['total_ratings'] == 72, 'Judge audit failed.')
        for name in ['all_ratings_audited.csv', 'construction_map_after_judging.csv', 'ratings_by_construction.csv', 'pairwise_judge_agreement.csv']:
            require(read_csv(judge_dir / name) == read_csv(temporary / 'audit' / name), 'Published judge table differs: ' + name)
    checks.append('Three raw Gemini responses, request/input hashes, 72 ratings, citations and agreement re-audited')
    checked_paths = [root / 'verify_public_results.py', root / 'verify_replication.py',
                     root / 'results/replication/source_manifest.json', root / 'results/replication/per_run_results.csv',
                     core_dir / 'all_model_results.csv',
                     sens_dir / 'sensitivity_results.csv', sample_dir / 'sampling_results.csv']
    return {'status': 'passed', 'python': platform.python_version(), 'platform': platform.system(),
            'checks': checks, 'distinct_extension_fits_in_tables': 128,
            'replication_records': replication,
            'scientific_retraining': False, 'network_requests': 0,
            'checked_sources_sha256': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in checked_paths},
            'scope': 'Published-table consistency and raw judge evidence. No re-execution of model checkpoints or recomputation from private prediction arrays.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional local JSON receipt.')
    args = parser.parse_args()
    receipt = verify(ROOT)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))
    print('PUBLIC RESULT CHECK: PASS')


if __name__ == '__main__':
    main()
