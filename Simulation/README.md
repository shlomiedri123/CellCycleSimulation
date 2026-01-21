# Simulation

Stochastic single-cell lineage simulator for RNAP-limited transcription across
the bacterial cell cycle. The simulator advances mRNA counts with time-dependent
birth-death dynamics and tracks cell division with binomial partitioning.

## Install

Recommended dependencies:
- numpy
- scipy
- pandas
- matplotlib
- pyyaml

Build the C++ tau-leaping extension once per environment:

```
python simulation/kernels/setup.py build_ext --inplace
```

## Physical model

Per gene, the mean-field dynamics are:

```
dm/dt = Gamma * g(t) * O(t) - gamma * m
O(t) = 1 / (1 + (k_off + Gamma) / (Nf(t) * k_on))
```

The stochastic implementation uses Poisson tau-leaping per timestep `dt`:

```
births ~ Poisson(Gamma * g(t) * O(t) * dt)
deaths ~ Poisson(gamma * m * dt)
```

`Nf(t)` is a deterministic, externally supplied time series. All rates are
time-independent; time dependence enters only via `g(t)` (gene dosage after
replication) and `Nf(t)`.

## Generate random inputs

```
python -m simulation.tools.random_gene_data \
  --n_genes 1000 \
  --n_samples 100000 \
  --out_dir simulation/examples/sim_layout \
  --seed 1
```

Outputs `random_genes.csv`, `nf_vector.npy`, `random_sim_config.yaml`, and
`random_hidden_params.json`. If you want to keep the entrypoint name
`sim_config.yaml`, rename the generated config.

## Run simulation

```
python -m simulation.run_simulation --config simulation/examples/sim_layout/sim_config.yaml
```

The YAML file defines input paths (`genes_path`, `nf_vector_path`) and output
paths (`out_path`, optional `parsed_out_path`), along with numeric simulation
parameters. If `measured_dist_path` is provided, the simulator applies lognormal
snapping. If `measured_s_vector_path` is provided, the simulator uses the
supplied S-vector for sampling (provide only one of the two). When `sparse: true`,
it writes a CSR matrix (`.npz`) plus `.cells.txt` and `.genes.txt` metadata;
otherwise it writes parsed snapshots CSV to `parsed_out_path` (or
`<out_path>_parsed.csv` by default).

## Nf(t) vector input

Nf(t) is a vector file (e.g. `nf_vector.npy` or `.csv`). The vector length must
equal `T_div / dt` (one cell-cycle), and the simulator indexes
`Nf_vec[k % (T_div/dt)]` at time `t = k * dt` so Nf(t) repeats each cycle.

## Division time sampling

Per-cell division times are drawn from a normal distribution with mean `T_div`
and standard deviation `division_time_cv * T_div`. Bounding is controlled by:

- `division_time_method: clip` (draw once, clamp to `[division_time_min, division_time_max]`)
- `division_time_method: reject` or `truncated_normal` (resample until in-bounds;
  abort after `division_time_max_attempts`)

## Out of scope

Inference, fitting, and plotting utilities are intentionally not part of the
tracked simulation package. Local tools can live outside this folder.
