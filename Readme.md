# SX-means (Spherical X-means)

Estimate the number of clusters on unit-sphere data using a von Mises–Fisher mixture + BIC.

## Overview

- **SX-means** extends X-means for spherical data.
- Assumes a mixture of von Mises–Fisher distributions to model unit-norm data.
- Automatically estimates **k** using BIC-based cluster splitting.  
- Supports optional fixed concentration parameter (κ) for stability.

## Files
- skmeans.py – spherical k-means helper
- sxmeans.py – main implementation of SX‑means
- sxmeans.ipynb – Jupyter Notebook with code, results, and visualization
- LICENSE – MIT license

## Note
- Make sure input vectors are normalized to length 1.
- Use fixed=True to fix κ and improve stability in some cases.
- The code for clipping to avoid math errors (e.g. zero division) caused by large kappa values or zero assigned points is commented out, but can be enabled if needed.

## Reference
- [Kazuhisa Fujita. Estimation of the number of clusters on d-dimensional sphere](https://arxiv.org/abs/2011.07530)
