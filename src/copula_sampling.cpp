#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
#include <Rmath.h> // For random number generation

// [[Rcpp::depends(RcppArmadillo)]]

// Convert covariance matrix to correlation matrix
// [[Rcpp::export]]
arma::mat cov2cor(const arma::mat& cov) {
  arma::vec d = 1 / sqrt(cov.diag());
  arma::mat cor = diagmat(d) * cov * diagmat(d);
  cor = 0.5 * (cor + cor.t()); // Ensure symmetry
  return cor;
}

// Get unique values excluding NAs
// [[Rcpp::export]]
arma::vec unique_non_na(const arma::vec& y) {
  arma::vec filtered_y = y.elem(arma::find_finite(y));
  return arma::unique(filtered_y);
}

// Detect ordinal columns based on threshold for unique values
// [[Rcpp::export]]
Rcpp::LogicalVector detect_ordinal_columns(const arma::mat& Y, int threshold) {
  int p = Y.n_cols;
  Rcpp::LogicalVector is_ordinal(p, false);
  for (int j = 0; j < p; ++j) {
    arma::vec unique_vals = unique_non_na(Y.col(j));
    if (unique_vals.n_elem <= threshold) {
      is_ordinal[j] = true;
    }
  }
  return is_ordinal;
}

// Sample from Dirichlet(1,...,1) distribution
// [[Rcpp::export]]
arma::vec sample_dirichlet(int n) {
  arma::vec g = arma::randg(n, arma::distr_param(1.0, 1.0));
  return g / arma::sum(g);
}

// Bayesian bootstrap ECDF (retaining NAs)
// [[Rcpp::export]]
arma::vec bayesian_bootstrap_ecdf(const arma::vec& col) {
  int n = col.n_elem;
  arma::uvec finite_indices = arma::find_finite(col);
  arma::vec col_finite = col.elem(finite_indices);
  arma::vec ecdf_vals(n, arma::fill::value(arma::datum::nan));

  if (col_finite.n_elem == 0) return ecdf_vals;

  arma::vec weights = sample_dirichlet(col_finite.n_elem);
  arma::uvec sorted_indices = arma::sort_index(col_finite);
  arma::vec sorted_col = col_finite(sorted_indices);
  arma::vec ecdf_finite(col_finite.n_elem, arma::fill::zeros);

  double cumulative_weight = 0.0;
  for (int i = 0; i < col_finite.n_elem; ++i) {
    cumulative_weight += weights(i);
    ecdf_finite(i) = cumulative_weight;
  }

  ecdf_finite = n * ecdf_finite / (n + 1);
  arma::vec ecdf_finite_reordered = arma::zeros(col_finite.n_elem);
  ecdf_finite_reordered(sorted_indices) = ecdf_finite;
  ecdf_vals(finite_indices) = ecdf_finite_reordered;

  return ecdf_vals;
}

// Compute ECDF intervals for ordinal variables
// [[Rcpp::export]]
List compute_ordinal_cdf(const arma::mat& X, const arma::mat& BB_ecdf, const LogicalVector is_ordinal) {
  int p = X.n_cols, n = X.n_rows;
  arma::mat lower_cdf(n, p), upper_cdf(n, p);

  for (int j = 0; j < p; ++j) {
    if (is_ordinal[j]) {
      arma::uvec valid_indices = arma::find_finite(X.col(j));
      for (int i : valid_indices) {
        double lower_val = 0.0, upper_val = 0.0;
        for (int k : valid_indices) {
          if (X(k, j) < X(i, j)) lower_val = std::max(lower_val, BB_ecdf(k, j));
          if (X(k, j) == X(i, j)) upper_val = std::max(upper_val, BB_ecdf(k, j));
        }
        if (upper_val == (double)n / (n + 1)) upper_val = 1.0;
        lower_cdf(i, j) = lower_val;
        upper_cdf(i, j) = upper_val;
      }
    }
  }

  return List::create(Named("lower_cdf") = lower_cdf, Named("upper_cdf") = upper_cdf);
}

// Standardize columns using ECDF (handling ordinal and continuous variables)
// [[Rcpp::export]]
arma::mat standardize_columns(const arma::mat& X, const arma::mat& BB_ecdf, const LogicalVector is_ordinal,
                              const arma::mat& lower_cdf, const arma::mat& upper_cdf) {
  arma::mat X_std = X;
  int p = X.n_cols, n = X.n_rows;

  for (int j = 0; j < p; ++j) {
    arma::vec ecdf_vals = BB_ecdf.col(j);
    arma::uvec valid = arma::find_finite(ecdf_vals);
    for (int i : valid) {
      if (is_ordinal[j]) {
        double U = R::runif(0.0, 1.0);
        X_std(i, j) = R::qnorm(lower_cdf(i,j) + U * (upper_cdf(i,j) - lower_cdf(i,j)), 0.0, 1.0, true, false);
      } else {
        X_std(i, j) = R::qnorm(ecdf_vals(i), 0.0, 1.0, true, false);
      }
    }
    arma::uvec na_indices = arma::find_nonfinite(ecdf_vals);
    for (int i : na_indices) {
      X_std(i, j) = R::rnorm(0.0, 1.0);
    }
  }

  return X_std;
}

// Inverse transform from Z-space to X-space using ECDF
// [[Rcpp::export]]
arma::mat inverse_transform(const arma::mat& Z, const arma::mat& BB_ecdf, const arma::mat& Y) {
  int p = Z.n_cols, n = Z.n_rows;
  arma::mat X(n, p);

  for (int j = 0; j < p; ++j) {
    arma::vec ecdf_vals = BB_ecdf.col(j);
    arma::vec x_col = Y.col(j);
    arma::uvec finite = arma::find_finite(x_col);
    arma::vec x_finite = x_col.elem(finite);

    if (x_finite.n_elem == 0) {
      X.col(j).fill(arma::datum::nan);
      continue;
    }

    for (int i = 0; i < n; ++i) {
      double q = R::pnorm(Z(i, j), 0.0, 1.0, true, false);
      arma::uword idx = arma::index_min(arma::abs(ecdf_vals - q));
      X(i, j) = x_col(idx);
    }
  }

  return X;
}

// Sample covariance matrix Sigma ~ Inverse-Wishart
// [[Rcpp::export]]
arma::mat gibbs_Sigma(const arma::mat& Z, double alpha) {
  int n = Z.n_rows, p = Z.n_cols;
  arma::mat S = Z.t() * Z;
  return iwishrnd(S + (alpha - p - 1) * arma::eye(p, p), n + alpha);
}

// Gibbs update for Z
// [[Rcpp::export]]
arma::mat gibbs_Z(arma::mat Z, arma::mat Y, const arma::mat& R, const Rcpp::LogicalVector is_ordinal,
                  const arma::mat& lower_cdf, const arma::mat& upper_cdf) {
  int n = Z.n_rows, p = Z.n_cols;

  for (int j = 0; j < p; ++j) {
    if (is_ordinal[j]) {
      arma::uvec valid = arma::find_finite(Y.col(j));
      for (int i : valid) {
        double U = R::runif(0.0, 1.0);
        Z(i, j) = R::qnorm(lower_cdf(i,j) + U * (upper_cdf(i,j) - lower_cdf(i,j)), 0.0, 1.0, true, false);
      }
    }
  }

  for (int j = 0; j < p; ++j) {
    arma::uvec ir_NA = arma::find_nonfinite(Y.col(j));
    if (!ir_NA.is_empty()) {
      arma::rowvec Rj_ex = R.row(j); Rj_ex.shed_col(j);
      arma::mat R_ex = R; R_ex.shed_row(j); R_ex.shed_col(j);
      arma::mat mu_j = Z.rows(ir_NA); mu_j.shed_col(j);
      mu_j = mu_j * arma::solve(R_ex, Rj_ex.t());
      double var_j = R(j, j) - arma::as_scalar(Rj_ex * arma::solve(R_ex, Rj_ex.t()));
      for (arma::uword idx = 0; idx < ir_NA.n_elem; ++idx) {
        Z(ir_NA[idx], j) = R::rnorm(mu_j(idx, 0), std::sqrt(var_j));
      }
    }
  }

  return Z;
}

// Main Gibbs sampler function
// [[Rcpp::export]]
Rcpp::List gibbs_copula(const arma::mat& Y, int iter, int burnin, int thin, double alpha,
                        int threshold = 10, Rcpp::Nullable<Rcpp::IntegerVector> ordinal_idx = R_NilValue) {
  auto start_time = std::chrono::steady_clock::now();
  int n = Y.n_rows, p = Y.n_cols;

  // Determine ordinal columns
  Rcpp::LogicalVector is_ordinal(p, false);
  if (ordinal_idx.isNull()) {
    is_ordinal = detect_ordinal_columns(Y, threshold);
  } else {
    Rcpp::IntegerVector idx_vec(ordinal_idx);
    for (int i = 0; i < idx_vec.size(); ++i) {
      if (idx_vec[i] >= 1 && idx_vec[i] <= p) is_ordinal[idx_vec[i] - 1] = true;
    }
  }

  arma::mat Z, Sigma, R, BB_ecdf(n, p);
  for (int j = 0; j < p; ++j) {
    BB_ecdf.col(j) = bayesian_bootstrap_ecdf(Y.col(j));
  }

  List ordinal_cdf = compute_ordinal_cdf(Y, BB_ecdf, is_ordinal);
  arma::mat lower_cdf = ordinal_cdf["lower_cdf"];
  arma::mat upper_cdf = ordinal_cdf["upper_cdf"];

  Z = standardize_columns(Y, BB_ecdf, is_ordinal, lower_cdf, upper_cdf);
  Sigma = gibbs_Sigma(Z, alpha);
  R = cov2cor(Sigma);

  Rcpp::List R_sample, X_sample;

  for (int i = 0; i < iter; ++i) {
    Z = gibbs_Z(Z, Y, R, is_ordinal, lower_cdf, upper_cdf);
    Sigma = gibbs_Sigma(Z, alpha);
    R = cov2cor(Sigma);

    if (i >= burnin && i % thin == 0) {
      X_sample.push_back(inverse_transform(Z, BB_ecdf, Y));
      R_sample.push_back(R);
    }

    if (i % 100 == 0) {
      Rcpp::Rcout << "Iteration: " << i << std::endl;
      auto now = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
      Rcpp::Rcout << "Elapsed time: " << elapsed << " seconds" << std::endl;
    }

    Rcpp::checkUserInterrupt();
  }

  auto end_time = std::chrono::steady_clock::now();
  auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
  Rcpp::Rcout << "Iteration: " << iter << "\nElapsed time: " << total_time << " seconds" << std::endl;

  return List::create(Named("R_sample") = R_sample, Named("X_sample") = X_sample, Named("BB_ecdf") = BB_ecdf);
}
