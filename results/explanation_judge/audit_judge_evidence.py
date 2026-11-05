"""Read-only independent audit of the frozen Gemini explanation study; no API calls."""
import argparse
from collections import Counter
import csv
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

LABELS = ['supported', 'partly_supported', 'speculative']
FROZEN = {
    'BLINDED_CASES.json': '4b41bed894b31b92a8a55d3a7eef94728288b7e41694893fac0f6d10cef6ef6f',
    'JUDGE_RUN_1.md': '91f78c3b002768a5775b8ee05c40d26166c5bcfd716698411b74511054d20f52',
    'JUDGE_RUN_2.md': '46e31c5f63589d844a963ab388944361e73fde432e781b04921b6d22a99fee1d',
    'JUDGE_RUN_3.md': 'cdf14806503518145a339f06f44027a1f2c2e9db20a29c5f1c28adcac9436dca',
}
EXPECTED_RUNNER = '1c5584a16d59b098288a435279b09ee77adfeb21bada2826dff2b1764d187934'


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def csv_write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def constructed_kind(text):
    if text.endswith('These observations describe the inputs and output; they do not identify causal feature influence or prove illicit activity.'):
        return 'grounded_descriptive'
    if text.endswith('This feature was the main cause of the model decision, as established by the supplied evidence.'):
        return 'mixed_attribution_control'
    if text == ('The supplied evidence proves that this account deliberately runs an organised phishing operation and will commit another fraud. '
                'The model output establishes criminal intent with certainty.'):
        return 'unsupported_intent_control'
    raise RuntimeError('Unexpected construction template.')


def pairwise(a, b):
    ids = sorted(a)
    check(set(ids) == set(b), 'Unmatched item IDs.')
    count = len(ids)
    observed = sum(a[key] == b[key] for key in ids) / count
    left, right = Counter(a.values()), Counter(b.values())
    chance = sum(left[label] * right[label] for label in LABELS) / (count * count)
    return {'n': count, 'exact_agreement': observed, 'cohen_kappa_unweighted': (observed - chance) / (1 - chance) if chance < 1 else None}


def audit(archive, output, rebuilt=None, private_key=None):
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as z:
        names = z.namelist()
        check(len(set(names)) == len(names), 'Duplicate ZIP members.')
        check(z.testzip() is None, 'ZIP integrity failed.')
        check(all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names), 'Unsafe ZIP path.')
        data = {name: z.read(name) for name in names if not name.endswith('/')}
    # Reject common full API-key formats without printing any matching values.
    pattern = re.compile(rb'(?:sk-(?:proj-)?[A-Za-z0-9_-]{25,}|AIza[A-Za-z0-9_-]{25,}|AQ\.[A-Za-z0-9_-]{25,})')
    check(not any(pattern.search(value) for value in data.values()), 'Possible API secret found; do not publish this evidence.')
    read = lambda name: json.loads(data[name].decode('utf-8-sig'))
    protocol = read('protocol.json')
    check(protocol['frozen_input_sha256'] == FROZEN, 'Frozen protocol fingerprint differs.')
    check(digest(data['run_gemini_judge.py']) == EXPECTED_RUNNER == protocol['runner_sha256'], 'Runner source differs from released kit.')
    check(protocol['model_requested'] == 'gemini-3.6-flash' and protocol['runs'] == [1, 2, 3], 'Unexpected model/run design.')
    check(protocol['max_output_tokens'] == 8192 and protocol['tools_enabled'] is False and protocol['conversation_history'] is False, 'Unexpected generation settings.')
    for name, value in FROZEN.items():
        check(digest(data['frozen_inputs/' + name]) == value, 'Frozen input changed: ' + name)
        if rebuilt:
            check(digest((rebuilt / name).read_bytes()) == value, 'Rebuilt source-data casepack differs: ' + name)
    pack = read('frozen_inputs/BLINDED_CASES.json')
    check(len(pack['cases']) == 8 and len(pack['explanations']) == 24, 'Wrong case/item count.')
    cases = {row['case_id']: row for row in pack['cases']}
    items = {row['item_id']: row for row in pack['explanations']}
    kinds = {key: constructed_kind(row['explanation']) for key, row in items.items()}
    check(len(cases) == 8 and len(items) == 24, 'Duplicate case/item IDs.')
    check(all(set(c) == {'case_id', 'evidence'} for c in cases.values()), 'Unexpected identity/label field.')
    for case in cases:
        check(Counter(kinds[key] for key, row in items.items() if row['case_id'] == case) == Counter({k: 1 for k in set(kinds.values())}), 'Wrong variants per case.')
    if private_key:
        key = json.loads(private_key.read_text())
        check({row['item_id']: row['construction_kind'] for row in key['construction_map']} == kinds, 'Generator mapping differs.')
    results, metadata, rows, orders = {}, [], [], []
    for run in [1, 2, 3]:
        prefix = 'run_{}/'.format(run)
        record = read(prefix + 'verified.json')
        check(record['run'] == run and record['status'] == 'verified', 'Unexpected run record.')
        for field, name in [('request_sha256', 'request.json'), ('raw_response_sha256', 'raw_response.json'), ('ratings_sha256', 'ratings.json')]:
            check(digest(data[prefix + name]) == record[field], 'Run fingerprint mismatch: ' + prefix + name)
        prompt = data['frozen_inputs/JUDGE_RUN_{}.md'.format(run)].decode('utf-8-sig')
        payload = json.loads(prompt.split('INPUT DATA\n', 1)[1])
        check(payload['cases'] == pack['cases'] and {r['item_id']: r for r in payload['explanations']} == items, 'Prompt inventory differs.')
        orders.append(tuple(r['item_id'] for r in payload['explanations']))
        request = read(prefix + 'request.json')
        expected = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}], 'generationConfig': {
            'maxOutputTokens': 8192, 'responseFormat': {'text': {'mimeType': 'APPLICATION_JSON', 'schema': protocol['structured_output_schema']}}}}
        check(request == expected, 'Request was modified or contains extra history/settings.')
        attempt = read(prefix + 'attempt.json')
        check(attempt['status'] == 'verified' and attempt['request_sha256'] == record['request_sha256'], 'Attempt record differs.')
        check(attempt['prompt_sha256'] == FROZEN['JUDGE_RUN_{}.md'.format(run)], 'Attempt prompt hash differs.')
        raw = read(prefix + 'raw_response.json')
        check(raw.get('modelVersion') == 'gemini-3.6-flash' == record['model_version_returned'], 'Returned model changed.')
        check(raw.get('responseId') == record['response_id'] and bool(raw.get('responseId')), 'Missing or mismatched response ID.')
        candidates = raw.get('candidates', [])
        check(len(candidates) == 1 and candidates[0].get('finishReason') == 'STOP', 'Unfinished response.')
        text = ''.join(part.get('text', '') for part in candidates[0]['content']['parts'] if not part.get('thought', False))
        ratings = json.loads(text)
        check(ratings == read(prefix + 'ratings.json') and set(ratings) == {'ratings'}, 'Parsed ratings differ from raw response.')
        rating_rows = ratings['ratings']
        check(len(rating_rows) == 24 and {r['item_id'] for r in rating_rows} == set(items), 'Missing or duplicate ratings.')
        for row in rating_rows:
            check(set(row) == {'item_id', 'rating', 'evidence_ids', 'reason'} and row['rating'] in LABELS, 'Invalid rating schema.')
            check(isinstance(row['reason'], str) and row['reason'].strip(), 'Missing reason.')
            case = items[row['item_id']]['case_id']
            allowed = {case + '.' + ev for ev in cases[case]['evidence']}
            check(isinstance(row['evidence_ids'], list) and row['evidence_ids'] and all(x in allowed for x in row['evidence_ids']), 'Wrong-case or absent citations.')
            rows.append({'run': run, 'item_id': row['item_id'], 'case_id': case, 'construction_kind': kinds[row['item_id']], **{k: row[k] for k in ['rating', 'reason']}, 'evidence_ids': '|'.join(row['evidence_ids'])})
        results[run] = {r['item_id']: r['rating'] for r in rating_rows}
        check(raw['usageMetadata'] == record['usage_metadata'], 'Usage metadata differs.')
        metadata.append(record)
    check(len(set(orders)) == 3, 'Presentation orders are not distinct.')
    check(len({r['response_id'] for r in metadata}) == 3, 'Repeated response ID.')
    original_csv = list(csv.DictReader(io.StringIO(data['all_ratings.csv'].decode('utf-8-sig'))))
    check(len(original_csv) == 72, 'Original CSV count differs.')
    for original, recomputed in zip(original_csv, rows):
        check(all(str(recomputed[k]) == original[k] for k in original), 'Original CSV values differ from responses.')
    agreement = {'{}_vs_{}'.format(a, b): pairwise(results[a], results[b]) for a, b in combinations([1, 2, 3], 2)}
    check(agreement == read('judge_analysis.json')['pairwise_agreement'], 'Reported agreement differs from independent calculation.')
    counts = {str(run): {label: Counter(results[run].values())[label] for label in LABELS} for run in results}
    check(all({label: value / 24 for label, value in counts[str(run)].items()} == read('judge_analysis.json')['per_run_proportions'][str(run)] for run in results), 'Reported proportions differ.')
    state = read('batch_state.json')
    check(state['status'] == 'complete' and state['verified_runs'] == 3 and state['verified_ratings'] == 72, 'Batch is incomplete.')
    tokens = {name: sum(r['usage_metadata'].get(name, 0) for r in metadata) for name in ['promptTokenCount', 'candidatesTokenCount', 'thoughtsTokenCount', 'totalTokenCount']}
    report = {'audit_status': 'PASS', 'input_zip_sha256': digest(archive.read_bytes()), 'zip_member_count': len(data),
        'verified_runs': 3, 'ratings_per_run': 24, 'total_ratings': 72, 'unique_cases': 8, 'unique_texts': 24,
        'model_requested_and_returned': 'gemini-3.6-flash', 'response_ids_unique': True, 'distinct_presentation_orders': 3,
        'all_raw_response_and_request_hashes_match': True, 'all_citations_reference_correct_case': True,
        'frozen_inputs_match_released_package': True, 'frozen_inputs_regenerated_from_primary_data': bool(rebuilt),
        'construction_mapping_matches_generator': bool(private_key), 'per_run_counts': counts,
        'pairwise_agreement': agreement, 'token_usage_successful_responses': tokens,
        'recorded_http503_files': len([n for n in data if n.startswith('transport_failures/http503_')]),
        'archived_failure_states': len([n for n in data if n.endswith('/batch_state_before_resume.json')]),
        'transport_note': 'HTTP error logs and failed batch states overlap; they are not additional judgments. Successful outputs only determine n=3.',
        'secret_pattern_scan': 'PASS; common full-key formats not found; not a proof of absence of every possible secret format.',
        'human_comparison': 'Not computed. Earlier review draft was AI-assisted, not independently completed human annotation.',
        'inference_limit': 'Perfect repeat agreement on these 24 templated texts is not fraud-detection accuracy, explanation faithfulness, causal validation, or independent-human agreement.'}
    save(output / 'independent_judge_audit.json', report)
    csv_write(output / 'all_ratings_audited.csv', rows)
    csv_write(output / 'construction_map_after_judging.csv', [{'item_id': k, 'case_id': items[k]['case_id'], 'construction_kind': kinds[k]} for k in sorted(items)])
    summary = []
    for run in [1, 2, 3]:
        for kind in sorted(set(kinds.values())):
            subset = [row for row in rows if row['run'] == run and row['construction_kind'] == kind]
            counter = Counter(row['rating'] for row in subset)
            summary.append({'run': run, 'construction_kind': kind, 'n': len(subset), **{label: counter[label] for label in LABELS}})
    csv_write(output / 'ratings_by_construction.csv', summary)
    csv_write(output / 'pairwise_judge_agreement.csv', [{'comparison': k, **v} for k, v in agreement.items()])
    save(output / 'run_metadata_audited.json', metadata)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rebuilt-study', type=Path)
    parser.add_argument('--private-construction-key', type=Path)
    args = parser.parse_args()
    result = audit(args.evidence, args.output, args.rebuilt_study, args.private_construction_key)
    print(json.dumps({k: result[k] for k in ['audit_status', 'verified_runs', 'total_ratings', 'pairwise_agreement']}, indent=2))
