param(
    [string]$OriginalFolder = (Join-Path $env:USERPROFILE "Downloads\COMP8240_Gemini_20261105_140752"),
    [string]$BatchName = "gemini_20261105T030806_050508Z"
)
$ErrorActionPreference = "Stop"
$pythonPath = Join-Path $env:USERPROFILE "miniconda3\envs\bwgnn\python.exe"
$runnerPath = Join-Path $OriginalFolder "run_gemini_judge.py"
$batchPath = Join-Path $OriginalFolder ("gemini_judge_runs\" + $BatchName)
foreach ($file in @($pythonPath, $runnerPath, (Join-Path $batchPath "protocol.json"))) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "File not found: $file" }
}
$keyPointer = [IntPtr]::Zero
$secureKey = $null
try {
    $secureKey = Read-Host "Paste Gemini API key (hidden)" -AsSecureString
    $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $env:GEMINI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    if ([string]::IsNullOrWhiteSpace($env:GEMINI_API_KEY)) { throw "API key is empty." }
    & $pythonPath (Join-Path $PSScriptRoot "resume_gemini.py") --runner $runnerPath --batch $batchPath
    if ($LASTEXITCODE -ne 0) { throw "Stopped. Send the printed evidence ZIP and error output." }
} finally {
    Remove-Item Env:GEMINI_API_KEY -ErrorAction SilentlyContinue
    if ($keyPointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer) }
    if ($secureKey) { $secureKey.Dispose() }
}
