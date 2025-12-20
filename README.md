A small testbed for binary classification of procedurally generated point clouds. The project is split into modules for data generation, modeling, and training, with hyperparameters driven by a YAML config file.

## Repository layout
- `configs/`: YAML configuration files with `data`, `model`, and `training` sections.
- `gnn_testbed/data/`: Point-cloud datasets (`PairFieldDataset`, `TriangleFieldDataset`) and collate function.
- `gnn_testbed/models/`: Model definitions and builders (default `ChiralLSSClassifier` with an LSS-GNN backbone, optional `ChiralEGNNClassifier`, `ChiralTriangleLSSClassifier`, and the older `SimplePointMLP`).
- `gnn_testbed/training/`: Training loop, metrics, and early stopping.
- `train.py`: Entry point that wires components together from a config file.
- `run_comparison.py`: Compare multiple GNN model variants from a single config file.

## Running training
Use the provided default configuration:

```bash
python train.py --config configs/default.yaml
```

Adjust the YAML file to change dataset parameters (e.g., task type, signal/noise points, or split sizes), model width, or training hyperparameters (e.g., learning rate, scheduler choice). The trainer writes checkpoints and TensorBoard logs to the `work_dir` specified in the `training` section.

### Data tasks
The point-cloud datasets can emit two binary tasks:
- `data.task: triangle` — right-handed vs. parity-flipped right triangles mixed with noise.
- `data.task: pair` — unordered point pairs plus noise (non-parity control task).

Normalization is controlled via `data.normalize` (`box` to divide by the periodic box size or `none`). Dataset options include `signal_points`, `noise_points`, `pair_distance`, and jitter settings.

> Note: The GNN models build graphs on the fly with SciPy (`scipy.spatial.Delaunay` / `cKDTree`). Install SciPy to use `lss_gnn`, `standard_egnn`, or `triangle_lss`. Switch `model.type` to `simple_point_mlp` in the config to avoid this dependency entirely.

### Switching models
- Default: `model.type: lss_gnn` (spin-aware local-frame message passing).
- EGNN variant: `model.type: standard_egnn` (distance-based equivariant updates); see `configs/standard_egnn.yaml` for a starting point.
- Triangle-local-frame variant: `model.type: triangle_lss` (triangle-centroid graph with spin-1 frames).
- MLP baseline: `model.type: simple_point_mlp`.

### Comparing models
Run the comparison script to train and evaluate multiple GNN variants in one go:

```bash
python run_comparison.py --config configs/comparison.yaml
```
