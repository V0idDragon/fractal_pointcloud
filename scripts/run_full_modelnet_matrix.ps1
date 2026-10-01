param(
  [ValidateSet('pointnet2','dgcnn')][string]$Model = 'pointnet2',
  [string]$Root = '.\data\modelnet40_h5\modelnet40_ply_hdf5_2048',
  [string]$OutputRoot = '',
  [string]$CacheDir = '',
  [ValidateSet('cuda','cpu','auto')][string]$Device = 'cuda',
  [switch]$Amp,
  [int]$Epochs = 100,
  [int]$NPoints = 1024,
  [int]$BatchSize = 16,
  [int]$Centers = 32,
  [int]$Seed = 42,
  [int]$GraphK = 20
)

$ErrorActionPreference = 'Stop'
if ($OutputRoot -eq '') { $OutputRoot = ".\results\modelnet40_full\$Model" }
if ($CacheDir -eq '') { $CacheDir = ".\results\modelnet40_full\fractal_cache_n$NPoints" }
$ks = @(32,64,128)
$features = @('xyz','xyz_dc','xyz_db','xyz_dc_db')

Write-Host "Checking ModelNet40 HDF5 files..."
$h5 = @(Get-ChildItem -Path $Root -Recurse -Filter *.h5)
if (-not ($h5 | Where-Object { $_.Name -match 'train' })) { throw "No train HDF5 found under $Root" }
if (-not ($h5 | Where-Object { $_.Name -match 'test' })) { throw "No test HDF5 found under $Root" }

Write-Host "Precomputing all fractal descriptors once: k=$($ks -join ',')"
python -m scripts.precompute_fractal `
  --dataset modelnet40_h5 --root $Root --n-points $NPoints `
  --k $ks --centers $Centers --seed $Seed --cache-dir $CacheDir
if ($LASTEXITCODE -ne 0) { throw 'Fractal precomputation failed' }

foreach ($feature in $features) {
  foreach ($k in $ks) {
    $out = Join-Path $OutputRoot (Join-Path $feature "k$k")
    New-Item -ItemType Directory -Force $out | Out-Null
    Write-Host "RUN Model=$Model Feature=$feature k=$k n=$NPoints epochs=$Epochs"
    $args = @(
      '-m','src.train','--dataset','modelnet40_h5','--root',$Root,
      '--model',$Model,'--feature-set',$feature,'--fractal-k',$k,
      '--fractal-centers',$Centers,'--n-points',$NPoints,'--epochs',$Epochs,
      '--batch-size',$BatchSize,'--workers','0','--lr','0.001',
      '--weight-decay','0.0001','--graph-k',$GraphK,'--val-fraction','0.10',
      '--seed',$Seed,'--output-dir',$out,'--cache-dir',$CacheDir,
      '--device',$Device,'--require-fractal-cache'
    )
    if ($Amp) { $args += '--amp' }
    python @args
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $Model / $feature / k=$k" }
  }
}
Write-Host "Completed 12 runs for $Model. Results: $OutputRoot"
