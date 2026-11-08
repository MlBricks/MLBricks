# Run from PowerShell after checking out a NEW feature branch in your MLBricks clone.
# Usage: .\patch_feature_branch.ps1 -Repo 'C:\path\to\MLBricks'
param([Parameter(Mandatory=$true)][string]$Repo)
$ErrorActionPreference = 'Stop'
$Python = if (Get-Command py -ErrorAction SilentlyContinue) { 'py' } else { 'python' }
$Runner = Join-Path $PSScriptRoot 'patch_existing_feature_branch.py'
if ($Python -eq 'py') { & py -3 $Runner --repo $Repo }
else { & python $Runner --repo $Repo }
if ($LASTEXITCODE -ne 0) { throw 'MLBricks patch failed. No GitHub push was performed.' }
Write-Host 'Local patch passed. Check git diff before any commit.'
