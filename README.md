# NVAE from scratch

Pytorch implementation (using Distributed Data Parallel and Automatic Mixed Precision) of **Nouveau-VAE**:
[paper link](https://arxiv.org/abs/2007.03898)

The implementation uses [FFCV](https://github.com/libffcv/ffcv) for fast data loading and 
[WandB](https://wandb.ai/site) for logging.

### Note on the implementation and useful resources

The repository is based on the official github repo of [NVAE](https://github.com/NVlabs/NVAE), implementing all the main
features and details, while trying to keep the code well organized and clean.  

Further resources that have been useful inf the implementation include: 

- Unofficial NVAE implementations by [mgp123](https://github.com/mgp123/nvae) 
and [GlassyWing](https://github.com/GlassyWing/nvae).
- The amazing Blog Post by [M.G. Portnoy](https://www.matiasgrynbergportnoy.com/posts/nvae/), which well explains the 
main architecture and some tricks.

## Repository Organization


## Install and Run


## Evaluate and some Results

how to reproduce (run two scripts)

### CIFAR 10 (Unconditioned)

| Run Name                      | Temperature | L2 (1e-5) | FID   | IS           | # (trainable) params |  
|-------------------------------|-------------|-----------|-------|--------------|---------------------:|
| NVAE 8x3                      | 0.6         | 10.097    | 19.73 | 6.00 +- 0.24 |            705.678 M |
| NVAE 8x3                      | 0.8         | 03.918    | 23.63 | 6.36 +- 0.16 |            705.678 M |
| NVAE 8x3                      | 1.0         | 01.243    | 32.50 | 5.77 +- 0.19 |            705.678 M |
