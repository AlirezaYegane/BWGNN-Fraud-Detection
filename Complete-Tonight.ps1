param(
    [string]$ProjectPath = "C:\Users\Asus\Documents\COMP8240\Rethinking-Anomaly-Detection",
    [string]$PythonPath = (Join-Path $env:USERPROFILE "miniconda3\envs\bwgnn\python.exe"),
    [string]$SamplingDirectory = "",
    [string]$JudgeEvidencePath = (Join-Path $env:USERPROFILE "Downloads\COMP8240_Gemini_20261105_140752\gemini_judge_runs\gemini_20261105T030806_050508Z\COMP8240_A3_Gemini_Judge_Evidence.zip"),
    [switch]$RunSampling,
    [switch]$ChecksOnly
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw "Existing bwgnn Python not found." }
Push-Location -LiteralPath $PSScriptRoot
try {
    & $PythonPath .\bootstrap_upstream.py --from-project $ProjectPath
    if ($LASTEXITCODE -ne 0) { throw "Original source check failed." }
    $core = Join-Path $ProjectPath "assessment3_runs\core_20261103T072708_683589Z"
    $sensitivity = Join-Path $ProjectPath "assessment3_runs\sensitivity_20261104T024804_638492Z"
    & $PythonPath .\verify_submission.py --runtime --core $core --sensitivity $sensitivity
    if ($LASTEXITCODE -ne 0) { throw "Repository checks failed. Send the output." }
    Write-Output "REPOSITORY CHECKS: PASS"
    if ($RunSampling -and (-not $ChecksOnly)) {
        & (Join-Path $PSScriptRoot "Run-Sampling-Check.ps1") -ProjectPath $ProjectPath -PythonPath $PythonPath
    } else {
        if (-not $SamplingDirectory) { $SamplingDirectory = Join-Path $ProjectPath "assessment3_runs\sampling_20261104T080325_330519Z" }
        if (Test-Path -LiteralPath (Join-Path $SamplingDirectory "batch_state.json") -PathType Leaf) {
            $spec = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot "audited_inputs.json") | ConvertFrom-Json
            $baseline = Join-Path $ProjectPath $spec.datasets.ethereum.relative_path
            & $PythonPath .\verify_sampling.py --sampling $SamplingDirectory --core $core --baseline-dataset $baseline --output (Join-Path $PSScriptRoot "LOCAL_SAMPLING_CHECKS")
            if ($LASTEXITCODE -ne 0) { throw "Saved sampling verification failed. Send the output." }
        } else { Write-Output "Sampling batch not found at the specified path. No training was started." }
    }
    if (Test-Path -LiteralPath $JudgeEvidencePath -PathType Leaf) {
        & $PythonPath .\results\explanation_judge\audit_judge_evidence.py --evidence $JudgeEvidencePath --output (Join-Path $PSScriptRoot "LOCAL_JUDGE_CHECKS")
        if ($LASTEXITCODE -ne 0) { throw "Judge evidence verification failed. Send the output." }
        Write-Output "JUDGE EVIDENCE CHECKS: PASS"
    } else { Write-Output "Judge evidence ZIP not found at the specified path. No API request was made." }
} finally { Pop-Location }
