A small testbed for binary classification of procedurally generated chains. The project is split into modules for data generation, modeling, and training, with hyperparameters driven by a YAML config file.

## Repository layout
- `configs/`: YAML configuration files with `data`, `model`, and `training` sections.
- `gnn_testbed/data/`: Chain generators (`ChiralChainGenerator`, `DistancePreferenceChainGenerator`), dataset, and collate function.
- `gnn_testbed/models/`: Model definitions and builders (default `ChiralLSSClassifier` with an LSS-GNN backbone, optional `ChiralEGNNClassifier`, and the older `SimplePointMLP`).
- `gnn_testbed/training/`: Training loop, metrics, and early stopping.
- `train.py`: Entry point that wires components together from a config file.

## Running training
Use the provided default configuration:

```bash
python train.py --config configs/default.yaml
```

Adjust the YAML file to change dataset parameters (e.g., chain length, task type, or split sizes), model width, or training hyperparameters (e.g., learning rate, scheduler choice). The trainer writes checkpoints and TensorBoard logs to the `work_dir` specified in the `training` section.

### Data tasks
The unified `ChainDataset` can emit two binary tasks:
- `data.task: chiral` — left vs. right turning walks (classes `L`/`R`).
- `data.task: distance` — short- vs. long-step walks (classes `S`/`G`) with configurable `short_range` and `long_range` step lengths.

Normalization is controlled via `data.normalize` (`box` to divide by the periodic box size or `none`). The shared chain hyperparameters live under `data.chain`.

> Note: The GNN models build graphs on the fly with SciPy (`scipy.spatial.Delaunay` / `cKDTree`). Install SciPy to use the default `lss_gnn` or the alternative `standard_egnn`. Switch `model.type` to `simple_point_mlp` in the config to avoid this dependency entirely.

### Switching models
- Default: `model.type: lss_gnn` (spin-aware local-frame message passing).
- EGNN variant: `model.type: standard_egnn` (distance-based equivariant updates); see `configs/standard_egnn.yaml` for a starting point.
- MLP baseline: `model.type: simple_point_mlp`.
