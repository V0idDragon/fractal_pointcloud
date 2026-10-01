param(
    [ValidateSet('objbg','objonly','modelnet40')][string]$Dataset = 'objbg',
    [int]$Epochs = 100,
    [int]$NPoints = 1024,
    [int]$BatchSize = 16,
    [int]$Centers = 32,
    [int]$Seed = 42,
    [string]$Device = 'cuda',
    [string]$CacheDir = '',
    [string]$OutputDir = '',
    [string]$ModelNetRoot = '.\data\modelnet40_h5\modelnet40_ply_hdf5_2048',
    [switch]$Amp,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$localPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $localPython) { $pythonExe = $localPython }
else { $pythonExe = (Get-Command python -ErrorAction Stop).Source }

$runner = Join-Path $projectRoot 'scripts\run_minimal_final.py'
if (-not (Test-Path -LiteralPath $runner)) {
    throw 'Missing scripts\run_minimal_final.py. Extract the whole minimal_final_runs.zip into the project.'
}
$runArgs = @($runner,'--dataset',$Dataset,'--epochs',"$Epochs",'--n-points',"$NPoints",
    '--batch-size',"$BatchSize",'--centers',"$Centers",'--seed',"$Seed",'--device',$Device)
if ($Dataset -eq 'modelnet40') { $runArgs += @('--root',$ModelNetRoot) }
if ($CacheDir) { $runArgs += @('--cache-dir',$CacheDir) }
if ($OutputDir) { $runArgs += @('--output-dir',$OutputDir) }
if ($Amp) { $runArgs += '--amp' }
if ($DryRun) { $runArgs += '--dry-run' }

Push-Location -LiteralPath $projectRoot
try {
    & $pythonExe @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Minimal final experiments failed (exit code $LASTEXITCODE)." }
}
finally { Pop-Location }
