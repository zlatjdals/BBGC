# Copula Imputation Function

copula_imputation <- function(data, rep = 10, iter = 1000, burnin = 500, thin = 10) {
  num_samples <- (iter - burnin) / thin
  X_sample <- vector("list", rep * num_samples)
  index <- 1
  for (i in 1:rep) {
    result <- gibbs_copula(data, iter = iter, burnin = burnin, thin = thin, alpha = ncol(data) + 5)
    X_sample[index:(index + num_samples - 1)] <- result$X_sample
    index <- index + num_samples
  }
  X_imputed <- Reduce("+", X_sample) / length(X_sample)
  return(X_imputed)
}
