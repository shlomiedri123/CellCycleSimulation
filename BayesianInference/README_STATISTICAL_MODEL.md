# Bayesian Inference of Cell Ages and Gene Expression Profiles
## Step 0 - Inputs and notation

Inputs and definitions:
- Observed counts n_{cg}
- Total counts per cell S_c = sum_g n_{cg}
- Cell age t in [0,1]
- Gene dosage g_g(t) from Cooper-Helmstetter
- Biophysical parameters (Gamma_g, gamma_g, k_on,g, k_off,g)
- Free RNAP trajectory N_f(t)

See `BaysienInference/io.py` (data loading) and `BaysienInference/infer.py` (main loop).

## Step 1 - Zero-iteration age posterior from total counts

P(t | S_c) propto P(S_c | t) P(t)

with:
- P(t) = 2 ln 2 * 2^{-t} (balanced growth)
- P(S_c | t) derived from log-normal sequencing depth

Reference: Eq. (15)-(21) in CellSizeNonlinearScaling-3.
See `BaysienInference/likelihood.py` (`compute_log_p0`) and `BaysienInference/infer.py` (initialization).

## Step 2 - Iteration 0: empirical gene profiles

Given P_0(t | S_c), compute an empirical estimate:

```
G_g^{(0)}(t) =
  [sum_c n_{cg} P_0(t | S_c)] /
  [sum_c P_0(t | S_c)]
```

Reference: Eq. (22) in CellSizeNonlinearScaling-3 .
See `BaysienInference/infer.py` (`update_gene_profiles`).

## Step 3 - Biophysical model fitting (core constraint)

IMPORTANT:
G_g(t) is NOT a free function.
It is constructed from biophysical parameters via:

d m_g / dt = g_g(t) Gamma_g O_g(t) - gamma_g m_g

Three candidate solutions are considered:
1. Full time-dependent solution (Eq. 2)
2. Regime I (upper Eq. 4)
3. Regime II (lower Eq. 4)

For each gene:
- All three models are evaluated
- Likelihoods are compared
- The best regime is selected

See `BaysienInference/fit_biophys.py` (`compute_m_g`, `compute_m_all`, regime selection).

## Step 4 - Constructing G_g(t) from biophysics

Once biophysical parameters and regime are chosen, the expected gene profile is:

G_g(t) = m_g(t) (from the selected equation)

See `BaysienInference/fit_biophys.py` and `BaysienInference/infer.py` (rebuilding G_gt).

## Step 5 - Posterior update for cell ages

P(t | n_g, S) propto P(t | S) * product_g (G_g(t))^{n_{cg}}


Reference: Eq. (23) in CellSizeNonlinearScaling-3.
See `BaysienInference/infer.py` (`update_posteriors`) and `BaysienInference/likelihood.py`.

## Step 6 - Iteration loop and convergence

Iteration k:
1. Use P_k(t) to estimate empirical G_g^{(k)}
2. Fit biophysical parameters and regimes
3. Construct G_g^{(k)}(t) from the biophysical model
4. Update P_{k+1}(t)
5. Repeat until convergence of log-likelihood

See `BaysienInference/infer.py` (main inference loop) and `BaysienInference/likelihood.py` (log-likelihood tracking).

## Step 7 - Outputs

Outputs written by the pipeline:
- Posterior age distributions
- Mechanistic gene expression profiles m_g(t)
- Selected regime per gene
- Biophysical parameters (if enabled)

See `BaysienInference/cli.py` (output writing).

## Roadmap (function-level)

1) Inputs: counts n_{cg}, depths S_c
   - `BaysienInference/io.py` (loaders)

2) Initialize ages: P(t | S_c)
   - `BaysienInference/likelihood.py` (`compute_log_p0`)
   - `BaysienInference/infer.py` (initialization)

3) Empirical profiles (no S_c/2^t correction)
   - `BaysienInference/infer.py` (`update_gene_profiles`)

4) Biophysical construction
   - `BaysienInference/fit_biophys.py` (`compute_m_g`, `compute_m_all`, regime selection)

5) Derived ψ and posterior update
   - `BaysienInference/infer.py` (`update_posteriors`)
   - `BaysienInference/likelihood.py` (`log_likelihood_bayes`)

## IMPORTANT NOTES

- No adiabatic assumption is made by default
- Regime selection is likelihood-based
- Likelihoods are evaluated up to t-independent constants
- psi_g(t) = m_g(t) / sum_j m_j(t) is derived AFTER inference
- Only effective ratios are inferred
