# Resume the frozen Gemini study after explicit service overload

Run `Resume-Gemini.ps1` using Windows PowerShell with the existing bwgnn Python environment. Default paths refer to the student's observed failed batch. No SDK installation, environment upgrades or graph training.

The original runner and protocol are unchanged. A previously documented HTTP 503 UNAVAILABLE is archived only when no model output exists. Completed judgments are verified and retained. The new transport policy is recorded before resuming: up to two retries per logical request with 20/40 seconds plus up to 3 seconds jitter, only for explicit HTTP 503 UNAVAILABLE error bodies without output. Every failure is saved. There is no retry for timeouts, unknown outcomes, other HTTP statuses, invalid answers, or undesirable ratings. The maximum number of accepted judge responses remains three; HTTP attempts may exceed three.

TLS uses an existing explicit PEM CA bundle with certificate and hostname verification enabled. The key is entered at a hidden prompt, passed through the process environment, and cleared on exit. It is never written to a request body or evidence file.

Send the printed COMP8240_A3_Gemini_Judge_Evidence.zip on completion or failure. If overload persists after the bounded retries, stop and retain all evidence. No automatic model fallback or new study batch is created.

Offline tests use synthetic transport fixtures only. No live request or real judge result has been generated in this package.

References checked 5 November 2026:
- https://aistudio.google.com/docs/troubleshooting
- https://github.com/python/cpython/issues/151504
- https://github.com/certifi/python-certifi
