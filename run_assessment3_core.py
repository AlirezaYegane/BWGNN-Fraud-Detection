"""Run the audited COMP8240 core experiments without editing the old scripts.

Python 3.8 compatible. Sequential CPU workers; fresh output directories; explicit
joint graph/model seeds; validation-only selection retained from the old trainer.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
SEEDS = [0, 1, 2, 3, 4]
LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
CONDITIONS = ['camo_c0', 'camo_c0p2', 'camo_c0p4', 'camo_c0p6', 'camo_c0p8', 'random_control']
METRICS = ['macro_f1', 'roc_auc_margin64', 'average_precision_margin64', 'pr_auc_trapezoid_margin64']

def sha(path, normalize=False):
    h = hashlib.sha256()
    if normalize:
        h.update(Path(path).read_bytes().replace(b'\r\n', b'\n'))
    else:
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
    return h.hexdigest()

def write_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(str(temp), str(path))

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def validate_inputs(project, specification, check_environment=True):
    details = {'source_hashes_normalized_lf': {}, 'datasets': {}}
    for name, expected in specification['sources_sha256_normalized_lf'].items():
        path = project / name
        require(path.is_file(), 'Project file missing: ' + str(path))
        actual = sha(path, normalize=True)
        require(actual == expected, 'Source differs from audited intake: ' + name)
        details['source_hashes_normalized_lf'][name] = actual
    for name, expected in specification['extension_sources_sha256_raw'].items():
        require(sha(project / name) == expected, 'Extension script bytes differ from the audited version: ' + name)
    for kind, record in specification['datasets'].items():
        path = project / record['relative_path']
        require(path.is_file(), 'Prepared dataset missing: ' + str(path))
        require(sha(path) == record['sha256'], 'Prepared dataset fingerprint differs: ' + kind)
        companion = path.parent / record['companion']
        require(companion.is_file(), 'Dataset companion report missing: ' + str(companion))
        require(sha(companion) == record['companion_sha256'], 'Dataset companion report differs: ' + kind)
        details['datasets'][kind] = {'path': str(path), 'sha256': record['sha256']}
    if check_environment:
        require(platform.python_version() == '3.8.20', 'Use the audited bwgnn Python 3.8.20 environment.')
        details['environment'] = {'python': platform.python_version(), 'executable': sys.executable}
        for name, expected in specification['packages'].items():
            actual = importlib.metadata.version(name)
            require(actual.split('+')[0] == expected, 'Package differs from audited environment: ' + name + '=' + actual)
            details['environment'][name] = actual
    return details

def jobs(project, specification, threads):
    eth = str(project / specification['datasets']['ethereum']['relative_path'])
    amazon = str(project / specification['datasets']['amazon']['relative_path'])
    result = []
    for seed in SEEDS:
        result.append({'id': 'ethereum_seed' + str(seed), 'kind': 'ethereum', 'model_seed': seed,
                       'graph_seed': None, 'fits': 3,
                       'arguments': ['train_ethereum.py', '--dataset', eth, '--models', 'bwgnn', 'gcn', 'mlp',
                                     '--seeds', str(seed), '--order', '2', '--hidden', '64', '--epochs', '100',
                                     '--lr', '0.01', '--threads', str(threads)]})
    for seed in SEEDS:
        result.append({'id': 'amazon_pair' + str(seed), 'kind': 'amazon', 'model_seed': seed,
                       'graph_seed': seed, 'fits': 12,
                       'arguments': ['run_amazon_camouflage.py', '--compact', amazon, '--levels',
                                     '0', '0.2', '0.4', '0.6', '0.8', '--graph-seed', str(seed),
                                     '--training-seed', str(seed), '--epochs', '100', '--threads', str(threads)]})
    return result

def run_worker(arguments, project, log_path):
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    process = None
    with Path(log_path).open('a', encoding='utf-8') as log:
        try:
            process = subprocess.Popen([sys.executable] + arguments, cwd=str(project), env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace', bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line); log.flush()
            code = process.wait()
            require(code == 0, 'Worker failed with exit code ' + str(code) + '; see ' + str(log_path))
        except BaseException:
            if process is not None and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise

def verify_job(run_root, job):
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_recall_curve, auc
    run_root = Path(run_root)
    filename = 'training_manifest.json' if job['kind'] == 'ethereum' else 'camouflage_manifest.json'
    manifest = read_json(run_root / filename)
    require(manifest['completed_runs'] == job['fits'], 'Incorrect completed-run count: ' + job['id'])
    require(manifest.get('epochs', 100) == 100, 'Epoch configuration differs.')
    if job['kind'] == 'ethereum':
        require(manifest['seeds'] == [job['model_seed']], 'Ethereum seed mismatch.')
        require(manifest['models'] == ['bwgnn', 'gcn', 'mlp'] and manifest['bwgnn_order'] == 2, 'Ethereum model configuration differs.')
    else:
        require(manifest['levels'] == LEVELS, 'Amazon level configuration differs.')
        require(manifest['graph_seed'] == job['graph_seed'] and manifest['training_seed'] == job['model_seed'], 'Amazon seed mismatch.')
    paths = sorted(run_root.rglob('result.json'))
    require(len(paths) == job['fits'], 'Expected model result files missing.')
    rows, identities = [], set()
    common_ids = {}
    for path in paths:
        record = read_json(path)
        condition = record.get('condition', 'fixed_sample')
        require(record['seed'] == job['model_seed'], 'Result seed mismatch.')
        require(record['epochs'] == 100 and record['hidden'] == 64 and record['learning_rate'] == 0.01, 'Training configuration differs.')
        require(record['order'] == (2 if record['model'] == 'bwgnn' else None), 'Filter order differs.')
        identities.add((condition, record['model']))
        score_file = path.parent / 'selected_scores.npz'
        require(sha(score_file) == record['scores_sha256'], 'Prediction fingerprint mismatch: ' + str(path))
        require(sha(path.parent / 'selected_parameters.pt') == record['checkpoint_sha256'], 'Checkpoint fingerprint mismatch.')
        with np.load(str(score_file), allow_pickle=False) as score:
            threshold = float(score['threshold'])
            require(abs(threshold - record['selected_threshold']) < 1e-14, 'Threshold mismatch.')
            require(np.any(np.isclose(threshold, np.linspace(0.05, 0.95, 19), rtol=0, atol=1e-14)), 'Threshold outside original grid.')
            require(int(score['selected_epoch']) == record['selected_epoch'], 'Selected epoch mismatch.')
            for split in ['val', 'test']:
                ids = score[split + '_global_id']
                y, prob, margin = score[split + '_labels'], score[split + '_probability32'], score[split + '_margin64']
                require(set(np.unique(y)) == {0, 1}, 'Invalid scored labels.')
                require(len(np.unique(ids)) == len(ids), 'Duplicate scored node IDs.')
                require(np.isfinite(prob).all() and np.isfinite(margin).all(), 'Non-finite prediction scores.')
                if split not in common_ids: common_ids[split] = (ids.copy(), y.copy())
                require(np.array_equal(ids, common_ids[split][0]) and np.array_equal(y, common_ids[split][1]), 'Scored population differs between conditions/models.')
                precision, recall, _ = precision_recall_curve(y, margin)
                actual = {'macro_f1': float(f1_score(y, prob > threshold, average='macro', labels=[0, 1], zero_division=0)),
                          'roc_auc_margin64': float(roc_auc_score(y, margin)),
                          'average_precision_margin64': float(average_precision_score(y, margin)),
                          'pr_auc_trapezoid_margin64': float(auc(recall, precision))}
                expected = record['validation' if split == 'val' else 'test']
                for metric, value in actual.items():
                    require(abs(value - expected[metric]) < 1e-10, 'Recomputed metric differs: ' + metric)
                if split == 'test':
                    row = {'job': job['id'], 'dataset': job['kind'], 'condition': condition,
                           'graph_seed': job['graph_seed'], 'model_seed': job['model_seed'], 'model': record['model'],
                           'parameters': record['trainable_parameter_count'], 'selected_epoch': record['selected_epoch'],
                           'threshold': threshold, 'test_nodes': len(y), 'positive_prevalence': float(np.mean(y)), **actual}
            require(not np.intersect1d(score['val_global_id'], score['test_global_id']).size, 'Validation/test overlap.')
        with (path.parent / 'epochs.csv').open(newline='', encoding='utf-8-sig') as stream:
            epochs = list(csv.DictReader(stream))
        require(len(epochs) == 100, 'Incomplete epoch history.')
        first_best = max(range(len(epochs)), key=lambda i: float(epochs[i]['val_macro_f1']))
        require(int(epochs[first_best]['epoch']) == record['selected_epoch'], 'Checkpoint was not selected by first-best validation Macro-F1.')
        rows.append(row)
    expected_ids = ({('fixed_sample', model) for model in ['bwgnn', 'gcn', 'mlp']} if job['kind'] == 'ethereum'
                    else {(condition, model) for condition in CONDITIONS for model in ['bwgnn', 'gcn']})
    require(identities == expected_ids, 'A condition/model combination is missing or duplicated.')
    return rows

def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def summaries(root, all_rows):
    import numpy as np
    write_csv(root / 'all_model_results.csv', all_rows)
    groups = {}
    for row in all_rows:
        groups.setdefault((row['dataset'], row['condition'], row['model']), []).append(row)
    aggregate = []
    for (dataset, condition, model), rows in sorted(groups.items()):
        item = {'dataset': dataset, 'condition': condition, 'model': model, 'n': len(rows), 'sd_definition': 'sample SD, ddof=1'}
        for metric in METRICS:
            values = np.asarray([row[metric] * 100 for row in rows])
            item[metric + '_mean_percent'] = float(values.mean())
            item[metric + '_sd_percent'] = float(values.std(ddof=1)) if len(values) > 1 else None
        aggregate.append(item)
    write_csv(root / 'summary_by_condition.csv', aggregate)
    amazon = {(r['model_seed'], r['condition'], r['model']): r for r in all_rows if r['dataset'] == 'amazon'}
    gaps = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            if all((seed, condition, model) in amazon for model in ['bwgnn', 'gcn']):
                gap = lambda c: 100 * (amazon[(seed, c, 'bwgnn')]['macro_f1'] - amazon[(seed, c, 'gcn')]['macro_f1'])
                item = {'pair_id': seed, 'condition': condition, 'bwgnn_minus_gcn_macro_f1_pp': gap(condition)}
                item['gap_change_from_baseline_pp'] = gap(condition) - gap('camo_c0') if (seed, 'camo_c0', 'bwgnn') in amazon else None
                item['gap_change_from_max_random_control_pp'] = (gap(condition) - gap('random_control')) if condition == 'camo_c0p8' and (seed, 'random_control', 'bwgnn') in amazon else None
                gaps.append(item)
    write_csv(root / 'paired_model_gaps.csv', gaps)

def bundle(root, filename='COMP8240_A3_Core_Evidence.zip'):
    path = root / filename
    temporary = root / 'core_evidence.tmp'
    with ZipFile(str(temporary), 'w', ZIP_DEFLATED) as archive:
        for file in sorted(root.rglob('*')):
            if not file.is_file() or file in [path, temporary] or file.suffix.lower() in {'.zip', '.tmp', '.lock'}:
                continue
            archive.write(str(file), file.relative_to(root).as_posix())
    os.replace(str(temporary), str(path))
    print('EVIDENCE ZIP: ' + str(path), flush=True)
    return path

def execute(args):
    project = Path(args.project).resolve()
    specification = read_json(HERE / 'audited_inputs.json')
    actual = validate_inputs(project, specification)
    work = jobs(project, specification, args.threads)
    protocol = {'version': 1, 'seeds': SEEDS, 'levels': LEVELS, 'models': {'ethereum': ['bwgnn', 'gcn', 'mlp'], 'amazon': ['bwgnn', 'gcn']},
                'threads': args.threads, 'epochs': 100, 'hidden': 64, 'order': 2, 'learning_rate': 0.01,
                'audited_specification': specification,
                'scope': 'Ethereum: five initialisations on one fixed sampled graph. Amazon: five paired graph/model seeds, not a crossed design; baseline graph fixed. Random control matches maximum intervention only. Original self-loop policies and validation-only selection preserved. Oracle construction uses labels across all splits. Capacity matching and filter-order sensitivity are separate subsequent analyses.',
                'planned_model_fits': 75}
    print('PREFLIGHT PASS: audited sources, dataset fingerprints and environment agree.', flush=True)
    if args.preflight_only:
        print(json.dumps({'planned_model_fits': 75, 'jobs': [j['id'] for j in work], 'inputs': actual}, indent=2))
        return
    root = Path(args.resume).resolve() if args.resume else project / 'assessment3_runs' / ('core_' + stamp())
    root.mkdir(parents=True, exist_ok=bool(args.resume))
    if args.resume:
        require(read_json(root / 'protocol.json') == protocol, 'Resume protocol/input configuration differs from this run kit.')
        state = read_json(root / 'batch_state.json')
    else:
        write_json(root / 'protocol.json', protocol)
        write_json(root / 'preflight.json', actual)
        state = {'status': 'created', 'jobs': {}, 'planned_model_fits': 75, 'verified_model_fits': 0, 'created_utc': stamp()}
        snapshot = root / 'source_snapshot'; snapshot.mkdir()
        for name in specification['sources_sha256_normalized_lf']: shutil.copy2(str(project / name), str(snapshot / name))
        shutil.copy2(str(Path(__file__).resolve()), str(snapshot / Path(__file__).name))
        shutil.copy2(str(HERE / 'audited_inputs.json'), str(snapshot / 'audited_inputs.json'))
        write_json(root / 'batch_state.json', state)
    lock = root / 'running.lock'
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('This batch has a running.lock. Do not start a second process for the same batch. If the old process was force-closed, confirm it has stopped before removing that lock.')
    os.write(descriptor, str(os.getpid()).encode('ascii')); os.close(descriptor)
    print('BATCH DIRECTORY: ' + str(root), flush=True)
    print('RESUME COMMAND: "{}" "{}" --project "{}" --resume "{}" --threads {}'.format(sys.executable, Path(__file__).resolve(), project, root, args.threads), flush=True)
    all_rows = []
    try:
        if not state.get('self_checks_passed'):
            for command in [['train_ethereum.py', '--metric-self-check'], ['train_ethereum.py', '--self-check'],
                            ['run_amazon_camouflage.py', '--self-check'], ['run_amazon_camouflage.py', '--model-self-check']]:
                run_worker(command, project, root / 'self_checks.log')
            state['self_checks_passed'] = True; write_json(root / 'batch_state.json', state)
        for index, job in enumerate(work):
            previous = state['jobs'].get(job['id'], {})
            if previous.get('status') == 'verified':
                rows = verify_job(Path(previous['run_directory']), job)
                all_rows.extend(rows)
                print('RESUME: verified completed ' + job['id'], flush=True)
                continue
            job_dir = root / job['id']; job_dir.mkdir(exist_ok=True)
            state['status'] = 'running'
            state['jobs'][job['id']] = {'status': 'running', 'started_utc': stamp()}
            write_json(root / 'batch_state.json', state)
            print('\n=== JOB {}/{}: {} ({} model fits) ==='.format(index + 1, len(work), job['id'], job['fits']), flush=True)
            started = time.monotonic()
            run_worker(job['arguments'] + ['--output', str(job_dir)], project, job_dir / 'driver_console.log')
            manifest_name = 'training_manifest.json' if job['kind'] == 'ethereum' else 'camouflage_manifest.json'
            candidates = list(job_dir.rglob(manifest_name))
            require(bool(candidates), 'Completed worker did not produce a manifest.')
            run_directory = max(candidates, key=lambda p: p.stat().st_mtime).parent
            rows = verify_job(run_directory, job)
            all_rows.extend(rows)
            state['jobs'][job['id']] = {'status': 'verified', 'run_directory': str(run_directory),
                                       'model_fits': len(rows), 'elapsed_seconds': time.monotonic() - started}
            state['verified_model_fits'] = len(all_rows)
            summaries(root, all_rows)
            write_json(root / 'batch_state.json', state)
            print('VERIFIED FITS: {}/75'.format(len(all_rows)), flush=True)
        require(len(all_rows) == 75, 'Batch has missing or duplicate model fits.')
        state['status'] = 'complete'; state['verified_model_fits'] = len(all_rows); state['completed_utc'] = stamp()
        summaries(root, all_rows); write_json(root / 'batch_state.json', state)
        bundle(root)
        print('CORE BATCH COMPLETE: 75 model fits verified. This is the core repeated-seed stage, not completion of the entire assessment.', flush=True)
    except BaseException as error:
        state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        for value in state['jobs'].values():
            if value.get('status') == 'running':
                value['status'] = state['status']
        state['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        write_json(root / 'batch_state.json', state)
        summaries(root, all_rows); bundle(root)
        raise
    finally:
        if lock.exists(): lock.unlink()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    require(args.threads > 0, 'Thread count must be positive.')
    execute(args)

if __name__ == '__main__':
    main()
