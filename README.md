# TRIP / scRNA-seq Bacterial Simulator
## Install
Recommended dependencies:

- numpy
- scipy
- pandas
- matplotlib
- pyyaml

For the stochastic kernel, build the C++ extension once per environment:

```bash
python simulation/kernels/setup.py build_ext --inplace
```

## Generate Random Config + Genes

```bash
python -m tools.random_gene_data   --n_genes 1000   --n_samples 100000   --out_dir out/configs   --seed 1
```

Outputs `random_genes.csv`, `nf_vector.npy`, `random_sim_config.yaml`, and `random_hidden_params.json` in `out/configs`.

## Run Simulation

```bash
python3 run_simulation.py --config sim_config.yaml
```

The YAML file defines input paths (`genes_path`, `nf_vector_path`) and output
paths (`out_path`, optional `parsed_out_path`), along with all numeric
simulation parameters. If `measured_dist_path` is provided, the simulator
applies lognormal snapping. If `measured_s_vector_path` is provided, the
simulator uses the supplied S-vector for sampling (provide only one of the two).
When `sparse: true`, it writes a CSR matrix (`.npz`) plus `.cells.txt` and
`.genes.txt` metadata; otherwise it writes parsed snapshots CSV to
`parsed_out_path` (or `<out_path>_parsed.csv` by default).

## Plot Age Distribution

```bash
python -m tools.plots age --snapshots snapshots.csv --config sim_config.yaml --out out/age_distribution.png
```

## Plot TRIP Profiles (Grid)

```bash
python -m tools.plots trip --snapshots snapshots.csv --config sim_config.yaml --out out/trip_profiles_grid.png --max_genes 100
```

## Plot Nf(t)

```bash
python -m tools.plots nf --config sim_config.yaml --out out/nf_profile.png
```

## Nf(t) Vector Input

Nf(t) is a vector file (e.g. `nf_vector.npy` or `.csv`). The
vector length must equal `T_div / dt` (one cell-cycle), and the simulator
indexes `Nf_vec[k % (T_div/dt)]` at time `t = k * dt` so Nf(t) repeats each cycle.

## Division Time Sampling

Per-cell division times are drawn from a normal distribution with mean `T_div`
and standard deviation `division_time_cv * T_div`. Bounding is controlled by:

- `division_time_method: clip` (draw once, clamp to `[division_time_min, division_time_max]`)
- `division_time_method: reject` or `truncated_normal` (resample until in-bounds;
  abort after `division_time_max_attempts`)
