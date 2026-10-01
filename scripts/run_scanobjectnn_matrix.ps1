param(
  [ValidateSet('objbg','objonly')][string]$Variant = 'objbg',
  [ValidateSet('pointnet2','dgcnn')][string[]]$Models = @('pointnet2','dgcnn'),
  [int[]]$Ks = @(16,32,64,128),
  [int]$Epochs = 100,
  [int]$BatchSize = 16,
  [int]$NPoints = 1024,
  [int]$Centers = 32,
  [string]$OutputRoot = '.\results\scanobjectnn',
  [string]$Device = 'cuda',
  [switch]$Amp,
  [switch]$ExactFps
)

$featureSets = @('xyz','xyz_dc','xyz_db','xyz_dc_db')
if ($Variant -eq 'objbg') { $dataRoot = '.\data\scanobjectnn\main_split' }
else { $dataRoot = '.\data\scanobjectnn\main_split_nobg' }
$trainH5 = Join-Path $dataRoot 'training_objectdataset.h5'
$testH5 = Join-Path $dataRoot 'test_objectdataset.h5'
$cache = Join-Path $OutputRoot 'fractal_cache'

# Populate every k before training. This uses exactly the training cache layout.
$precomputeArgs = @('-m','scripts.precompute_fractal','--dataset','scanobjectnn',
  '--train-h5',$trainH5,'--test-h5',$testH5,'--n-points',$NPoints,
  '--centers',$Centers,'--seed','42','--cache-dir',$cache,'--k') + $Ks
python @precomputeArgs
if ($LASTEXITCODE -ne 0) { throw 'Fractal precompute failed; training was not started.' }


foreach ($model in $Models) {
  foreach ($feature in $featureSets) {
    foreach ($k in $Ks) {
      $out = Join-Path $OutputRoot "$Variant\$model\$feature\k$k"
      New-Item -ItemType Directory -Force $out | Out-Null
      $args = @('.\run.py','--dataset','scanobjectnn','--train-h5',$trainH5,'--test-h5',$testH5,
        '--model',$model,'--feature-set',$feature,'--fractal-k',$k,'--fractal-centers',$Centers,
        '--n-points',$NPoints,'--epochs',$Epochs,'--batch-size',$BatchSize,'--workers','0',
        '--lr','0.001','--weight-decay','0.0001','--val-fraction','0.10','--seed','42',
        '--device',$Device,'--cache-dir',$cache,'--require-fractal-cache','--output-dir',$out)
      if ($Amp) { $args += '--amp' }
      if ($ExactFps) { $args += '--exact-fps' }
      Write-Host "Running $Variant / $model / $feature / k=$k"
      python @args
      if ($LASTEXITCODE -ne 0) { throw "Failed: $Variant / $model / $feature / k=$k" }
    }
  }
}
