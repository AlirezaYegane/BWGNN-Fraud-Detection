"""Offline fixtures; not real study judgments."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import resume_gemini as recovery


def overload():
    body = json.dumps({'error': {'code': 503, 'status': 'UNAVAILABLE', 'message': 'Synthetic overload'}}).encode()
    return urllib.error.HTTPError('https://example.invalid', 503, 'Service Unavailable', {}, io.BytesIO(body))


class RecoveryTests(unittest.TestCase):
    def test_retry_only_explicit_503_then_return_first_success(self):
        calls, sleeps = [], []
        def original(*args):
            calls.append(args)
            if len(calls) < 3:
                raise overload()
            return {'fixture': 'first success'}, {}
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            fn = recovery.retrying_http(original, Path(td), sleep=sleeps.append, jitter=lambda: 0)
            self.assertEqual(fn('POST', '/fixed', 'SYNTHETIC_KEY'), ({'fixture': 'first success'}, {}))
            self.assertEqual(len(calls), 3)
            self.assertEqual(sleeps, [20, 40])
            self.assertTrue(all(c == calls[0] for c in calls))
            files = list(Path(td).rglob('http503_*.json'))
            self.assertEqual(len(files), 2)
            self.assertTrue(all('SYNTHETIC_KEY' not in p.read_text() for p in files))

    def test_retry_limit_preserves_last_error_body(self):
        calls = []
        def original(*args):
            calls.append(1)
            raise overload()
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            fn = recovery.retrying_http(original, Path(td), sleep=lambda _: None, jitter=lambda: 0)
            with self.assertRaises(urllib.error.HTTPError) as caught:
                fn('POST', '/fixed', '')
            self.assertTrue(recovery.unavailable(caught.exception.read()))
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(list(Path(td).rglob('http503_*.json'))), 3)

    def test_no_retry_for_timeout_or_other_http_status(self):
        for error in [TimeoutError('Synthetic uncertain outcome'),
                      urllib.error.HTTPError('https://example.invalid', 400, 'Bad Request', {}, io.BytesIO(b'{}'))]:
            calls = []
            def original(*args):
                calls.append(1)
                raise error
            with tempfile.TemporaryDirectory() as td:
                fn = recovery.retrying_http(original, Path(td), sleep=lambda _: self.fail('Unexpected retry'))
                with self.assertRaises(type(error)):
                    fn('POST', '/fixed', '')
            self.assertEqual(len(calls), 1)

    def test_archive_preserves_attempt_and_rejects_existing_response(self):
        for with_response in [False, True]:
            with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
                batch = Path(td); run = batch / 'run_1'; run.mkdir()
                recovery.write(run / 'attempt.json', {'status': 'submitted', 'run': 1})
                recovery.write(run / 'request.json', {'synthetic': True})
                recovery.write(batch / 'batch_state.json', {
                    'status': 'failed', 'error': 'HTTP Error 503: Service Unavailable | ' + overload().read().decode()})
                if with_response:
                    recovery.write(run / 'raw_response.json', {'synthetic': True})
                    with self.assertRaises(RuntimeError):
                        recovery.archive_known_failure(batch)
                    self.assertTrue((run / 'attempt.json').exists())
                else:
                    recovery.archive_known_failure(batch)
                    self.assertFalse(run.exists())
                    self.assertEqual(len(list(batch.glob('transport_failures/*/run_1/attempt.json'))), 1)
                    self.assertEqual(len(list(batch.glob('transport_failures/*/batch_state_before_resume.json'))), 1)

    def test_uncertain_previous_attempt_rejected_and_python38(self):
        ast.parse((ROOT / 'resume_gemini.py').read_text(), feature_version=(3, 8))
        with tempfile.TemporaryDirectory() as td:
            batch = Path(td); run = batch / 'run_1'; run.mkdir()
            recovery.write(run / 'attempt.json', {'status': 'submitted', 'run': 1})
            recovery.write(batch / 'batch_state.json', {'status': 'failed', 'error': 'Synthetic timeout'})
            with self.assertRaises(RuntimeError):
                recovery.archive_known_failure(batch)
            self.assertTrue(run.exists())


if __name__ == '__main__':
    unittest.main()
