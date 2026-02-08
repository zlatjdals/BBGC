# Main Execution for Copula Imputation

source("00_load_packages.R")
source("01_data_generation.R")
source("02_copula_imputation.R")
source("03_evaluation.R")

n <- 1000
p <- 15
sparsity <- 0.2

data <- generate_mixed_data(n, p)
data_nonna <- data
data_mis <- get_MAR_testset(data, prop = sparsity, prop2 = sparsity, miss_force = TRUE)
index_na <- which(is.na(data_mis))

data_imputed <- copula_imputation(data_mis , iter = 10000, burnin = 5000, thin = 10 , cont_indices = 6:15 )

results <- evaluate_copula(data_imputed, data_nonna, index_na)
print(results$r_squared)
print(results$coefficients)
print(results$nrmse)

results$plot
