# La NeRF

Install

Make sure nerfstudio is installed, then install this package with:
```bash
pip install -e .
```

How to train:
```bash
ns-train la_nerf --data data/nerfstudio/poster  --pipeline.datamanager.train-num-rays-per-batch 2048 --pipeline.datamanager.eval-num-rays-per-batch 2048 --viewer.websocket-port 44041  --pipeline.model.laplace-backend "pytorch-laplace" --pipeline.model.laplace_method "laplace"
```
