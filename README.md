# Copula-Based Imputation Example

This R project demonstrates missing value imputation using a Gibbs sampling-based copula method.

## Structure

- `00_load_packages.R`: Install/load required libraries and C++ source
- `01_data_generation.R`: Simulate mixed-type data and inject MCAR missingness
- `02_copula_imputation.R`: Define copula-based imputation function
- `03_evaluation.R`: Evaluate imputation using R² and scatter plot
- `04_main.R`: Entry script to run the pipeline

## Requirements

- R packages: Rcpp, ggplot2, broom, mice, missForest, VIM, scales
- C++ source: Place your `copula_sampling.cpp` in `src/`
