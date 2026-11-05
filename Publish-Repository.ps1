param(
    [Parameter(Mandatory=$true)][string]$RepositoryUrl,
    [switch]$Push
)
$ErrorActionPreference = "Stop"
if ($RepositoryUrl -notmatch '^https://github\.com/[A-Za-z0-9-]+/[A-Za-z0-9_.-]+(?:\.git)?/?$') {
    throw "Use the HTTPS URL of your own newly created GitHub repository, with no credentials in the URL."
}
function Invoke-ProjectGit {
    param([string[]]$GitArguments)
    & git -C $PSScriptRoot @GitArguments
    if ($LASTEXITCODE -ne 0) { throw "Git command failed. Keep the output and fix the reported issue before retrying." }
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Git is not available. Use GitHub Desktop or install Git, then retry." }
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot ".git"))) { Invoke-ProjectGit -GitArguments @("init") }
$remoteNames = @(& git -C $PSScriptRoot remote)
if ($LASTEXITCODE -ne 0) { throw "Unable to list repository remotes." }
if ($remoteNames -contains "origin") {
    $existingOrigin = & git -C $PSScriptRoot remote get-url origin
    if ($LASTEXITCODE -ne 0) { throw "Unable to read existing origin." }
    if ($existingOrigin.TrimEnd('/').Replace('.git','') -ne $RepositoryUrl.TrimEnd('/').Replace('.git','')) {
        throw "This folder already has another origin. It was not changed. Use a fresh extracted repository folder."
    }
} else { Invoke-ProjectGit -GitArguments @("remote", "add", "origin", $RepositoryUrl) }
Invoke-ProjectGit -GitArguments @("add", "--all")
Invoke-ProjectGit -GitArguments @("diff", "--cached", "--stat")
& git -C $PSScriptRoot diff --cached --quiet
if ($LASTEXITCODE -eq 1) { Invoke-ProjectGit -GitArguments @("commit", "-m", "Document and verify COMP8240 reproduction and extension workflow") }
elseif ($LASTEXITCODE -ne 0) { throw "Unable to inspect staged changes." }
Invoke-ProjectGit -GitArguments @("branch", "-M", "main")
if ($Push) {
    Invoke-ProjectGit -GitArguments @("push", "-u", "origin", "main")
    Write-Output "PUSH COMPLETE. Check the repository URL while signed out; a push does not establish public visibility."
} else {
    Write-Output "LOCAL COMMIT PREPARED. To publish to your chosen repository, rerun this script with -Push."
}
