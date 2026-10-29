# lartpc-supernova-pointing
Physics-informed electron direction reconstruction for supernova neutrino pointing. Running on raw Liquid Argon Time Projection Chamber (LArTPC) readouts, with muon removal, two-point DSNT coordinate prediction, and multi-view detector geometry.

<p align="center">
  Hilary Utaegbulam
</p>

<p align="center">
  <a href="docs/coordinate_transforms.md">Coordinate transformations</a> ·
  <a href="docs/gallery.md">Gallery</a> ·
  <a href="docs/citing.md">Cite</a>
</p>

![Processing overview: raw wire-plane readout, segmentation, muon and Bragg-peak suppression, and two-point direction prediction.](assets/hero.png)

## Overview

**This project investigates how the initial direction of a low-energy electron can be inferred from raw liquid argon time projection chamber (LArTPC) readout images, in an attempt to resolve head-tail ambiguity.** The goal is to support supernova neutrino pointing through improved electron head–tail reconstruction.

The project investigates its approach using simulated Integrated Cryostat and Electronics Built for Experimental Research Goals (ICEBERG) events of Michel electrons from stopping cosmic muons. Learned segmentation and local charge suppression are used to remove the parent muon's track before a physics-informed model estimates the electron direction in each wire-plane view.

## Demonstration

The overview figure shows simulated ICEBERG readout processing and an example of two-point direction prediction. The upper sequence shows particle segmentation and muon/Bragg-peak suppression. The lower panels show a different event, including the predicted direction and the DSNT spatial distributions. More two-point outputs are collected in the [gallery](docs/gallery.md).

## Method

1. **Sparse signal selection.** Robust MAD whitening, a multiscale Difference-of-Gaussians score, and hysteresis growth select candidate signal pixels from the raw ADC image, without hit finding or clustering. [`main/segmentation/`](main/segmentation)
2. **Particle segmentation.** A sparse U-Net (WarpConvNet) labels each selected pixel as muon, Michel + bremsstrahlung, or other. [`main/segmentation/sparse_unet.py`](main/segmentation/sparse_unet.py)
3. **Vertex finding and muon removal.** A DSNT vertex model locates the muon–Michel decay point in each view. All non-electron activity is removed, and the residual Bragg peak around the vertex is suppressed. [`main/vertex/`](main/vertex) · [`main/preprocessing/`](main/preprocessing)
4. **Two-point direction prediction.** In each view, a UNet++ predicts Point A (the vertex) and Point B, whose displacement from A gives the initial electron direction. Training adds ray-geometry, bremsstrahlung, and cross-view tick constraints. [`main/headtail/`](main/headtail)
5. **Multi-view voting.** A tick-sign vote and a round-robin consistency test reconcile the U, V, and Z predictions into one 3D direction. [Coordinate transformations](docs/coordinate_transforms.md#5-the-voting-system) · [`main/pipeline/end_to_end.py`](main/pipeline/end_to_end.py)

The detector geometry behind steps 4–5 is derived in [docs/coordinate_transforms.md](docs/coordinate_transforms.md) and implemented in [`main/geometry.py`](main/geometry.py).

## Repository layout

| Path | Contents |
|---|---|
| [`main/geometry.py`](main/geometry.py) | 3D ⇄ 2D wire-plane transforms, U/V/Z consistency condition, and voting (NumPy; derived in [coordinate transformations](docs/coordinate_transforms.md)) |
| [`main/segmentation/`](main/segmentation) | Sparse input mask (MAD → DoG → hysteresis) and WarpConvNet sparse U-Net |
| [`main/vertex/`](main/vertex) | DSNT muon–Michel vertex finder |
| [`main/preprocessing/`](main/preprocessing) | Muon removal and Bragg-peak suppression |
| [`main/headtail/`](main/headtail) | Two-point DSNT model, physics-informed losses, training, and PyTorch voting |
| [`main/pipeline/`](main/pipeline) | End-to-end inference from raw U/V/Z images to a 3D direction |
| [`deployment/`](deployment) | Poisson-stream study of the pipeline as an online trigger |
| [`tests/`](tests) | Geometry tests, including NumPy ↔ PyTorch agreement |
| [`tools/`](tools) | Script that draws the geometry figures |
| [`docs/`](docs) | Coordinate transformations, gallery, and citation |

## Scope

This repository presents the research method, implementation, coordinate transformations, and selected simulation results.

“Raw readout” refers to the ADC image representation used by the analysis. The demonstrated sample consists of simulated ICEBERG events, and the work focuses on electron directional reconstruction relevant to supernova pointing.

The research modules under `main/` and `deployment/` are published for reference. Because of DUNE collaboration restrictions, the simulation samples, trained weights, and site-specific configuration (such as data and checkpoint paths) are not included, so the training and inference code is not runnable as published. The detector-geometry module and its tests are self-contained.

## Running the geometry

The geometry module needs only NumPy. From the repository root:

```bash
pip install numpy pytest
python -m pytest -q
```

## Citation

If you use this project, its coordinate derivations, methods, or figures, please cite it. See [docs/citing.md](docs/citing.md), or use GitHub's **Cite this repository** button.

## License

Code is available under the [MIT License](LICENSE).
