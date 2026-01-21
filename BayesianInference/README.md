# BaysienInference

Statistical inference and validation pipeline for single-cell bacterial scRNA-seq.

This pipeline **does not generate data**. It consumes simulator outputs:
- `snapshot.csv` — long format with `cell_id,gene_id,count`
- `params.csv` — ground-truth parameters for validation only

## What it does

1) Infers cell age posteriors `P_c(t)` and MAP ages.
2) Computes gene expression profiles `m_g(t)` from biophysical parameters.
3) Selects the best model per gene by comparing log-likelihoods across three candidate equations.
4) Validates inferred results against simulator truth (PoC gate + recovery report).

## Model assumptions and identifiability

- Three candidate equations for `m_g(t)` are compared per gene: full Eq. (2), Regime I, Regime II (Eq. 4).
- No regime is assumed a priori; regime selection is likelihood-based.
- Promoter occupancy uses the quasi-steady form `O_g(t)` from Eqs. (5–6).
- The adiabatic limit `m_g(t) = g_g(t) Γ_g O_g(t) / γ_g` is not used for inference.
- The exponential age prior `p(t) = 2 ln 2 · 2^{-t}` is assumed (balanced growth), not inferred.
- `ψ_g(t)` is derived from `m_g(t)` via normalization; it is not an independent parameter.
- Only parameter ratios (e.g., `Γ_g/γ_g` and combinations involving `M_0`) are identifiable from steady-state data.

## Input formats

### snapshot.csv (long format)

```
cell_id,gene_id,count
cell_0,gene_0,12
cell_0,gene_1,4
...
```

Wide format with one row per cell and gene columns is also accepted.

### params.csv

Single CSV with optional fields per row (blank allowed):

Columns:
- `cell_id`, `t_true`
- `gene_id`, `gamma_g`, `Gamma_g`, `k_on_g`, `k_off_g`, `regime`
- `t`, `Nf_t`
- `gene_id`, `t`, `G_true` (for validation)
- `mu`, `sigma`

## Run PoC

```
python -m BaysienInference.cli \
  --snapshot BaysienInference/tests/fixtures/snapshot_small.csv \
  --params BaysienInference/tests/fixtures/params_small.csv \
  --out out/poc \
  --nt 20 \
  --max-iters 15
```

The pipeline will:
- run the likelihood sanity gate (`LL_true > LL_perturbed`)
- infer ages and profiles
- write outputs and validation plots

Expected runtime for the PoC fixture is under 1 minute on a laptop.

## Outputs

- `inferred_ages.csv`
- `inferred_profiles.csv` (columns: `gene_id`, `t`, `m`, `psi`)
- `inferred_params.csv`
- `loglikelihood_trace.csv`
- `recovery_report.md`
- `age_recovery.png`, `profile_similarity.png` (and `nf_recovery.png` if available)

## Scaling notes

This PoC is tuned for small datasets (≈100 cells, 10 genes). For larger runs, increase `--nt` and `--max-iters`, and consider optimizing sparse operations.
