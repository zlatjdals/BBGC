# Generate example dataset and introduce missingness
generate_correlation_matrix <- function(p) {
  outer(1:p, 1:p, function(i, j) (abs(i - j) + 1)^(-2))
}

generate_data <- function(n = 100, p = 5, missing_rate = 0.1, seed = 123) {
  set.seed(seed)
  data <- matrix(rnorm(n * p), nrow = n)
  colnames(data) <- paste0("V", 1:p)
  index_na <- sample(length(data), size = round(length(data) * missing_rate))
  data[index_na] <- NA
  return(list(data_mis = data, index_na = index_na))
}

generate_mixed_data <- function(n, p) {
  require(mvtnorm)
  R <- generate_correlation_matrix(p)
  data <- mvtnorm::rmvnorm(n, rep(0, p), R)
  data[, 1:5] <- round(data[, 1:5])
  data[, 6:10] <- qbeta(pnorm(data[, 6:10]), 1, 1)
  data[, 11:15] <- qexp(pnorm(data[, 11:15]), 1)
  colnames(data) <- paste0("V", 1:p)
  return(data)
}

# Functions to insert missing values under MAR mechanism

get_MAR_testset <- function(data, prop = 0.1, prop2 = NULL, miss_force = FALSE) {
  N <- nrow(data)
  p <- ncol(data)
  if (miss_force) {
    suffle_idx <- sample(1:p, p)
    rev_idx <- order(suffle_idx)
    data <- data[, suffle_idx]
    pm <- c(rep(prop2, (p - 1)) + (prop2 - 0.05)/(p - 1), 0.05)
  } else {
    pm <- rep(prop, p)
  }
  
  k <- p - 1
  n_miss <- p
  pct_miss <- prop
  iter <- 1
  
  while (any(n_miss == p) | (prop > round(pct_miss, 3))) {
    M <- matrix(0, N, p)
    w <- runif(p - 1)
    b <- runif(p - 1)
    M[, 1] <- rbinom(N, 1, pm[1])
    simulated_data <- data
    tmp <- 0
    
    for (j in 1:k) {
      tmp <- tmp + exp(-(w[j] * M[, j] * data[, j] + b[j] * (1 - M[, j])))
      prob <- (tmp * N * pm[j + 1]) / sum(tmp)
      prob[prob > 1] <- 1
      M[, (j + 1)] <- rbinom(N, 1, prob)
    }
    
    M <- apply(M, 2, as.logical)
    simulated_data[M] <- NA
    n_miss <- apply(simulated_data, 1, function(x) sum(is.na(x)))
    pct_miss <- sum(M) / (N * p)
    iter <- iter + 1
  }
  
  if (any(n_miss == p)) stop("Missing Data Generation Failed!")
  if (miss_force) simulated_data <- simulated_data[, rev_idx]
  
  message("Missing rate: ", round(pct_miss * 100, 1), "%")
  return(simulated_data)
}