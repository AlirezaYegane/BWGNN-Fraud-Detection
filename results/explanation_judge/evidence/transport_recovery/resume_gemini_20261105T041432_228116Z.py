"""Resume the unchanged study after an explicit Gemini HTTP 503; Python 3.8+."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def unavailable(body):
    try:
        data = json.loads(body)
        return (set(data) == {'error'} and data['error']['code'] == 503
                and data['error']['status'] == 'UNAVAILABLE')
    except (ValueError, TypeError, KeyError):
        return False


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def archive_known_failure(batch):
    pending = [p for p in batch.glob('run_*/attempt.json')
               if not (p.parent / 'verified.json').exists()]
    if not pending:
        return
    state = json.loads((batch / 'batch_state.json').read_text(encoding='utf-8-sig'))
    detail = state.get('error', '')
    body = detail.split(' | ', 1)[-1]
    if (len(pending) != 1 or state.get('status') != 'failed'
            or not detail.startswith('HTTP Error 503:') or not unavailable(body)):
        raise RuntimeError('Prior attempt is not a documented HTTP 503. Send its evidence; no retry made.')
    attempt = pending[0]
    if attempt.parent.name not in {'run_1', 'run_2', 'run_3'}:
        raise RuntimeError('Unexpected run directory.')
    record = json.loads(attempt.read_text(encoding='utf-8-sig'))
    if record.get('status') != 'submitted' or record.get('run') != int(attempt.parent.name[-1]):
        raise RuntimeError('Unexpected prior attempt state.')
    if any((attempt.parent / n).exists() for n in ['raw_response.json', 'ratings.json', 'verified.json']):
        raise RuntimeError('A model response already exists. No replacement judgment will be requested.')
    archive = batch / 'transport_failures' / ('prior_503_' + stamp())
    archive.mkdir(parents=True)
    shutil.copy2(batch / 'batch_state.json', archive / 'batch_state_before_resume.json')
    shutil.move(str(attempt.parent), str(archive / attempt.parent.name))
    print('Previous explicit HTTP 503 archived; no judgment was received.', flush=True)


def retrying_http(original, batch, sleep=time.sleep, jitter=None):
    jitter = jitter or (lambda: random.SystemRandom().uniform(0, 3))

    def http(method, path, key, payload=None, timeout=180):
        # At most 3 attempts per request: only explicit 503 UNAVAILABLE without output.
        # Unknown outcomes, timeouts, rejected answers and other status codes are never retried.
        for number in range(1, 4):
            try:
                return original(method, path, key, payload, timeout)
            except urllib.error.HTTPError as error:
                if error.code != 503:
                    raise
                raw = error.read()
                text = raw.decode('utf-8', errors='replace').replace(key, '[REDACTED_API_KEY]') if key else raw.decode('utf-8', errors='replace')
                folder = batch / 'transport_failures'
                folder.mkdir(exist_ok=True)
                eligible = method == 'POST' and unavailable(text)
                delay = (20 * (2 ** (number - 1)) + jitter()) if eligible and number < 3 else None
                write(folder / ('http503_' + stamp() + '.json'), {
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                    'method': method, 'path': path, 'attempt_in_this_call': number,
                    'http_status': 503, 'response_body': text,
                    'retry_delay_seconds': delay, 'judgment_received': False if eligible else None
                })
                if delay is None:
                    raise urllib.error.HTTPError(error.url, error.code, error.reason,
                                                 error.headers, io.BytesIO(raw)) from None
                print('HTTP 503: retry {}/2 in {:.0f}s (same model and prompt).'.format(number, delay), flush=True)
                sleep(delay)
    return http


def verified_tls():
    candidates = []
    for name in ['certifi', 'pip._vendor.certifi']:
        try:
            module = importlib.import_module(name)
            candidates.append(Path(module.where()))
        except ImportError:
            pass
    candidates.append(Path(sys.prefix) / 'Library' / 'ssl' / 'cacert.pem')
    ca = next((p for p in candidates if p.is_file()), None)
    if ca is None:
        raise RuntimeError('No PEM CA bundle found. Certificate verification must remain enabled.')
    context = ssl.create_default_context(cafile=str(ca))
    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        raise RuntimeError('TLS verification is not enabled.')
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=context)))
    return ca


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args()
    runner, batch = args.runner.resolve(), args.batch.resolve()
    if not runner.is_file() or not (batch / 'protocol.json').is_file():
        raise RuntimeError('Original runner or batch protocol not found.')
    if (batch / 'running.lock').exists():
        raise RuntimeError('Batch lock exists. Do not launch concurrent judging.')
    if hashlib.sha256(runner.read_bytes()).hexdigest() != json.loads((batch / 'protocol.json').read_text())['runner_sha256']:
        raise RuntimeError('Original runner changed. No request made.')
    ca = verified_tls()
    spec = importlib.util.spec_from_file_location('frozen_gemini_runner', str(runner))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Verify frozen study and all existing completed outputs before any archival action.
    items, evidence, _, _ = module.load_study(runner.parent / 'study')
    module.summarize(batch, items, evidence)
    archive_known_failure(batch)
    folder = batch / 'transport_recovery'
    folder.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), folder / ('resume_gemini_' + stamp() + '.py'))
    write(folder / ('policy_' + stamp() + '.json'), {
        'policy': 'Up to 2 backoff retries per logical request, only explicit HTTP 503 UNAVAILABLE with no output. Preserve every failure. Never retry accepted, invalid or uncertain responses.',
        'model_and_frozen_prompts': 'unchanged', 'ca_file': str(ca),
        'ca_sha256': hashlib.sha256(ca.read_bytes()).hexdigest(),
        'openssl': ssl.OPENSSL_VERSION, 'certificate_and_hostname_verification': True,
        'earlier_failures': 'preserved in transport_failures',
        'protocol_addendum': 'Transport retry policy introduced after observed service overload, before any accepted judgment.'
    })
    module.http = retrying_http(module.http, batch)
    print('SSL PREFLIGHT PASS. Resuming the original frozen batch.', flush=True)
    sys.argv = [str(runner), '--resume', str(batch)]
    module.main()


if __name__ == '__main__':
    main()
