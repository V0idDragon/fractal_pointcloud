# fractal_pointcloud_deep

Практический pipeline для исследования интеграции локальных фрактальных признаков в нейросетевые модели обработки 3D point clouds.

## Что внутри

- PointNet++-style hierarchical classifier (pure PyTorch, без pointnet2_ops/custom CUDA extensions).
- DGCNN/EdgeConv classifier (pure PyTorch).
- Прямые point-wise каналы `Dc` и `Db` для конфигураций `XYZ`, `XYZ+Dc`, `XYZ+Db`, `XYZ+Dc+Db`.
- Локальные Dc/Db считаются на reproducible FPS-anchor centers и распространяются на остальные точки по inverse-distance interpolation. Это снижает стоимость и сохраняет покоординатную привязку канала к облаку.
- HDF5 loaders для ModelNet40 2048 и ScanObjectNN; loader подготовленного subset в `.npy` для smoke-test.
- Кэширование фрактальных признаков, CSV history/metrics/confusion matrix и best checkpoint.
- Общий кэш признаков с манифестом отпечатка данных: один набор `k/centers` переиспользуется всеми моделями и feature-set.
- Быстрый детерминированный sampler PointNet++ по умолчанию; канонический медленный FPS доступен через `--exact-fps`.

## Установка

`pip install -r requirements.txt`

Для NVIDIA GPU ставьте PyTorch с подходящим CUDA wheel вместо CPU-only сборки.

## Smoke test на уже подготовленном частичном ModelNet40

Этот набор лежит в старом verified archive и соответствует 11 классам/малому subset, полученному при предыдущей подготовке частичного ModelNet40.

`python -m src.train --dataset prepared_modelnet --root fixtures/modelnet_partial --class-mapping fixtures/modelnet_partial/class_mapping.json --model pointnet2 --feature-set xyz_dc_db --fractal-k 32 --fractal-centers 8 --epochs 2 --batch-size 2 --n-points 256 --output-dir results/smoke --device cpu --smoke`

DGCNN check:

`python -m src.train --dataset prepared_modelnet --root fixtures/modelnet_partial --class-mapping fixtures/modelnet_partial/class_mapping.json --model dgcnn --feature-set xyz --epochs 2 --batch-size 2 --n-points 256 --output-dir results/smoke_dgcnn --device cpu --smoke`

## Full ModelNet40 HDF5

`python -m src.train --dataset modelnet40_h5 --root data/modelnet40_h5/modelnet40_ply_hdf5_2048 --model pointnet2 --feature-set xyz --epochs 100 --batch-size 16 --n-points 1024 --output-dir results/modelnet40_pointnet2`

Fractal variants use the same command with `--feature-set xyz_dc`, `xyz_db`, `xyz_dc_db` and `--fractal-k 32` (repeat later for 16/64/128).

## ScanObjectNN

`python -m src.train --dataset scanobjectnn --train-h5 data/scanobjectnn/main_split/training_objectdataset.h5 --test-h5 data/scanobjectnn/main_split/test_objectdataset.h5 --model pointnet2 --feature-set xyz_dc_db --fractal-k 32 --epochs 100 --batch-size 16 --n-points 1024 --output-dir results/scanobjectnn_objbg`

Для серии запусков задавайте один и тот же `--cache-dir results/scanobjectnn_cache`. Кэш привязан к точным нормализованным массивам, `n_points`, seed и числу центров, поэтому несовместимые признаки не будут приняты молча.

## Result layout

Each experiment has `config.json`, `metrics.json`, `history.csv`, `confusion_matrix_test.csv`, `class_mapping.json`, and `model_best.pt`. The root output also gets an appendable `summary.csv`.

## Reproducibility

Seed is set for Python/NumPy/PyTorch; dataset sampling is deterministic for a fixed seed. No PCA alignment is applied. PointNet++ and DGCNN use the same XYZ/fractal inputs, making the ablation directly comparable within an architecture.

## Automated ablations

For one architecture and one scale:

`python -m scripts.run_ablation --dataset prepared_modelnet --root fixtures/modelnet_partial --class-mapping fixtures/modelnet_partial/class_mapping.json --model pointnet2 --fractal-k 32 --fractal-centers 32 --epochs 100 --batch-size 16 --output-dir results/modelnet_partial`

For the four sensitivity scales, pass `--fractal-k 16 32 64 128`. Each run uses the same seed/data split and reuses the fractal cache.

`scripts/run_scanobjectnn_matrix.ps1` запускает полную матрицу для OBJ_BG или OBJ_ONLY и складывает общий `summary.csv`. Перед первым обучением скрипт автоматически запускает precompute для всех `k`. Для ручного precompute используйте:

```powershell
python -m scripts.precompute_fractal `
  --dataset scanobjectnn `
  --train-h5 ".\data\scanobjectnn\main_split\training_objectdataset.h5" `
  --test-h5 ".\data\scanobjectnn\main_split\test_objectdataset.h5" `
  --n-points 1024 --k 16 32 64 128 --centers 32 --seed 42 `
  --cache-dir ".\results\scanobjectnn_objbg\fractal_cache"
```

Для обязательного использования уже рассчитанного кэша добавьте к обучению `--require-fractal-cache`.

## Robustness variants

`python scripts/create_robustness_variant.py --input fixtures/modelnet_partial --output results/robust_noise005 --mode noise --value 0.005`

`python scripts/create_robustness_variant.py --input fixtures/modelnet_partial --output results/robust_n512 --mode downsample --value 512`

Then train using `--dataset prepared_modelnet --root <variant_dir> --class-mapping <variant_dir>/class_mapping.json`.
