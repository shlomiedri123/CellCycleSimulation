# Simulation

This package implements a stochastic single-cell lineage simulator for
RNAP-limited transcription across the bacterial cell cycle. The simulator
advances mRNA counts with time-dependent birth-death dynamics and tracks
cell division with binomial partitioning.

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

## Minimal run

Build the tau-leaping extension and run the simulator:

```
python simulation/kernels/setup.py build_ext --inplace
python run_simulation.py --config data/random_sim_config.yaml
```

Ensure the time grid is consistent: `len(nf_vec) == T_div / dt` and
`T_total / dt` is an integer.

