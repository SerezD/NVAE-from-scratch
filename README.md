# NVAE from scratch

pytorch (Distributed Data Parallel and AMP)
ffcv for data loading
kornia for internal normalization
wandb for logging

resources list (git repos, paper, blog)

critical parts of the code

## RESULTS

how to reproduce (run two scripts)

### CIFAR 10 (Unconditioned)

| Run Name                      | Temperature | L2 (1e-5) | FID   | IS           | # (trainable) params |  
|-------------------------------|-------------|-----------|-------|--------------|---------------------:|
| NVAE 8x3                      | 0.6         | 10.097    | 19.73 | 6.00 +- 0.24 |            705.678 M |
| NVAE 8x3                      | 0.8         | 03.918    | 23.63 | 6.36 +- 0.16 |            705.678 M |
| NVAE 8x3                      | 1.0         | 01.243    | 32.50 | 5.77 +- 0.19 |            705.678 M |


# TODO

refactor scripts
