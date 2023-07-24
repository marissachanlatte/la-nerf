# La NeRF

Install

Make sure nerfstudio is installed, then install this package with:
```bash
pip install -e .
```

How to train:
```bash
ns-train la_nerf --data data/nerfstudio/poster  --pipeline.datamanager.train-num-rays-per-batch 2048 --pipeline.datamanager.eval-num-rays-per-batch 2048 --viewer.websocket-port 44041  --pipeline.model.laplace-backend "nnj" --pipeline.model.laplace_method "laplace"
```


Train linearized laplace:

```bash
ns-train la_nerf --data ../nerfbusters/data/nerfbusters-dataset-renamed/aloe/ --pipeline.datamanager.train-num-rays-per-batch 1024 --pipeline.datamanager.eval-num-rays-per-batch 512 --pipeline.model.eval_num_rays_per_chunk 2000 --viewer.websocket-port 44041  --pipeline.model.laplace-backend "nnj" --pipeline.model.laplace_method "linearized-laplace" nerfstudio-data --eval-mode eval-frame-index
```
