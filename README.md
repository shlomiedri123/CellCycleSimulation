# Bacterial scRNA-seq Simulation & Inference

This project provides tools for simulating bacterial single-cell RNA sequencing data and performing Bayesian/ML inference on cell ages.

## Modules

- **Simulation/** - Cell cycle simulation with tau-leaping stochastic transcription
- **BayesianInference/** - Bayesian inference for gene expression parameters and cell age
- **MLComparison/** - Machine learning models for age prediction (MLP, Attention-based)
- **RealDataAnalysis/** - Tools to download and analyze real E. coli scRNA-seq data (TRIPs)

## Install

Dependencies:
- numpy, scipy, pandas, matplotlib, pyyaml
- scikit-learn, torch (for ML models)
- pybind11 (for C++ kernel)

Build the C++ tau-leaping kernel:

```bash
python Simulation/kernels/setup.py build_ext --inplace
```

## Quick Start

### 1. Generate Random Config + Genes

```bash
python Simulation/tools/random_gene_data.py \
    --n_genes 1000 \
    --n_samples 100000 \
    --out_dir out/configs \
    --seed 1
```

### 2. Run Simulation

```bash
python -m Simulation.run --config sim_config.yaml
```

### 3. Run Full Analysis Pipeline

```bash
python final_analysis.py
```

This runs:
- Simulation with specified parameters
- Bayesian inference (NMF fitting, age inference)
- ML model comparison
- Data retention analysis (100%, 50%, 20%)
- Generates all plots and metrics in `final_results/`

## Configuration

See example configs in:
- `Simulation/examples/` - Simulation configuration
- `BayesianInference/examples/` - Bayesian inference configuration

## Key Parameters

- `T_div`: Cell division time
- `dt`: Time step for tau-leaping
- `n_cells`: Number of cells to sample
- `n_genes`: Number of genes to simulate
- `k_on`, `k_off`: Transcription burst kinetics
- `k_m`: mRNA degradation rate

## Output

The analysis pipeline produces:
- Gene expression profiles m_g(t)
- Bayesian age inference with posterior distributions
- ML model comparison (SimpleMLP, DeepMLP, GeneAttention)
- Data retention analysis at different capture rates
