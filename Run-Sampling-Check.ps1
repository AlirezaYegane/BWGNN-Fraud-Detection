param(
    [string]$ProjectPath = "C:\Users\Asus\Documents\COMP8240\Rethinking-Anomaly-Detection",
    [string]$PythonPath = (Join-Path $env:USERPROFILE "miniconda3\envs\bwgnn\python.exe"),
    [string]$CoreDirectory = "",
    [string]$AuditReport = "",
    [string]$Resume = "",
    [int]$Threads = 16,
    [switch]$PreflightOnly
)
$ErrorActionPreference = "Stop"
if (-not $CoreDirectory) { $CoreDirectory = Join-Path $ProjectPath "assessment3_runs\core_20261103T072708_683589Z" }
if (-not $AuditReport) { $AuditReport = Join-Path $ProjectPath "audit_results\ethereum_stage_20261008T114845426Z\ethereum_audit_20261008T114845_547232Z\graph_audit.json" }
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw "Python not found: $PythonPath" }
$arguments = @((Join-Path $PSScriptRoot "run_sampling_check.py"), "--project", $ProjectPath, "--core", $CoreDirectory, "--audit", $AuditReport, "--threads", [string]$Threads)
if ($Resume) { $arguments += @("--resume", $Resume) }
if ($PreflightOnly) { $arguments += "--preflight-only" }
& $PythonPath @arguments
if ($LASTEXITCODE -ne 0) { throw "Sampling check stopped. Send the complete output and sampling evidence ZIP." }
