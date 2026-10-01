param(
  [string]$Root = '.\data\modelnet40_h5\modelnet40_ply_hdf5_2048',
  [string]$Device = 'cuda',
  [switch]$Amp,
  [int]$Epochs = 20,
  [int]$NPoints = 512,
  [int]$K = 64,
  [string]$OutputDir = '.\results\modelnet40_screen'
)
$ErrorActionPreference = 'Stop'
$args = @('-m','scripts.run_modelnet_screening','--root',$Root,'--device',$Device,
  '--epochs',$Epochs,'--n-points',$NPoints,'--k',$K,'--output-dir',$OutputDir)
if ($Amp) { $args += '--amp' }
python @args
