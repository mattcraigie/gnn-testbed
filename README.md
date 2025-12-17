# gnn-testbed

A small testbed for binary classification of procedurally generated chiral chains. The project is split into modules for data generation, modeling, and training, with hyperparameters driven by a YAML config file.

## Repository layout
- `configs/`: YAML configuration files with `data`, `model`, and `training` sections.
- `src/gnn_testbed/data/`: Data generation utilities (`ChiralChainGenerator`, dataset, and collate function).
- `src/gnn_testbed/models/`: Model definitions and builders (currently `SimplePointMLP`).
- `src/gnn_testbed/training/`: Training loop, metrics, and early stopping.
- `train.py`: Entry point that wires components together from a config file.

## Running training
Use the provided default configuration:

```bash
python train.py --config configs/default.yaml
```

Adjust the YAML file to change dataset parameters (e.g., chain length, split sizes), model width, or training hyperparameters (e.g., learning rate, scheduler choice). The trainer writes checkpoints and TensorBoard logs to the `work_dir` specified in the `training` section.
