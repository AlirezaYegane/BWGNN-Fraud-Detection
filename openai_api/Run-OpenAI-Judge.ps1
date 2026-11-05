param(
    [string]$PythonPath = (Join-Path $env:USERPROFILE "miniconda3\envs\bwgnn\python.exe"),
    [string]$Model = "gpt-5.4-2026-03-05",
    [string]$Resume = "",
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw "Existing Python not found: $PythonPath" }
$runArguments = @((Join-Path $PSScriptRoot "run_openai_judge.py"), "--model", $Model)
if ($Resume) { $runArguments += @("--resume", $Resume) }
if ($DryRun) {
    & $PythonPath @runArguments --dry-run
    if ($LASTEXITCODE -ne 0) { throw "Offline preflight failed." }
    return
}
$keyPointer = [IntPtr]::Zero
try {
    $secureKey = Read-Host "Paste your OpenAI API key (hidden input)" -AsSecureString
    $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $env:OPENAI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) { throw "API key is empty." }
    & $PythonPath @runArguments
    if ($LASTEXITCODE -ne 0) { throw "OpenAI run stopped. Send the printed evidence ZIP and error output." }
} finally {
    Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
    if ($keyPointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer) }
    if ($secureKey) { $secureKey.Dispose() }
}
