# Load Required Packages
required_packages <- c("Rcpp", "missForest", "VIM", "mice", "ggplot2", "scales", "broom", "gridExtra")

installed <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!(pkg %in% installed)) {
    install.packages(pkg, dependencies = TRUE)
  }
  library(pkg, character.only = TRUE)
}

library(Rcpp)
sourceCpp("src/copula_sampling.cpp")
