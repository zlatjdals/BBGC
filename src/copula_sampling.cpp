/**
 * Mixed Data Imputation using Bayesian Bootstrap and Gaussian Copula
 * * Description:
 * This code implements a Gibbs sampler for fully Bayesian estimation of a 
 * Gaussian Copula model dealing with mixed continuous and discrete/ordinal data.
 * It utilizes a Bayesian Bootstrap approach for marginal distributions.
 *
 * Dependencies: 
 * Rcpp, RcppArmadillo
 */

#include <RcppArmadillo.h>
#include <Rmath.h>
#include <algorithm>
#include <vector>
#include <chrono>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// Helper: Covariance to Correlation
arma::mat cov2cor_safe(const arma::mat& cov) {
  arma::vec d = 1.0 / sqrt(cov.diag());
  d.elem(find_nonfinite(d)).zeros(); 
  arma::mat cor = diagmat(d) * cov * diagmat(d);
  return 0.5 * (cor + cor.t());
}

// -----------------------------------------------------------------------------
// 1. Update Bounds & Direct Z for Continuous (BB Weights)
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
List sample_bb_moments(const arma::mat& X, const arma::uvec& is_continuous) {
  int n = X.n_rows;
  int p = X.n_cols;
  
  arma::mat L(n, p, arma::fill::zeros); 
  arma::mat U(n, p, arma::fill::zeros);
  arma::mat Z_direct(n, p, arma::fill::zeros); // Store deterministic Z values for continuous variables
  
  std::vector<arma::vec> cutpoints_list(p);
  
  for (int j = 0; j < p; ++j) {
    arma::vec x_col = X.col(j);
    
    // Generate sort indices (assuming NaNs are handled upstream)
    arma::uvec sort_idx = sort_index(x_col); 
    arma::vec x_sorted = x_col.elem(sort_idx);
    
    // 1. Generate Bayesian Bootstrap Weights (Common)
    arma::vec C(n + 1);
    C(0) = 0.0;
    C(n) = 1.0;
    
    if (n > 1) {
      arma::vec rand_vals = runif(n - 1); 
      rand_vals = sort(rand_vals);
      for(int k=0; k < n-1; ++k) {
        C(k+1) = rand_vals(k);
      }
    }
    
    // 2. Branch processing based on variable type
    if (is_continuous(j) == 1) {
      // [Continuous Case]
      // Deterministic transformation: Z = Phi^-1( F(x) )
      // F(x_(i)) = C[i+1] (Cumulative Weights)
      // Apply n/(n+1) scaling for stability (prevents Inf at C[n]=1.0)
      
      double scale_factor = (double)n / (double)(n + 1);
      
      for (int k = 0; k < n; ++k) {
        int original_idx = sort_idx(k);
        
        // Cumulative probability of the k-th smallest value = C(k+1)
        double cdf_val = C(k+1) * scale_factor; 
        
        // Clamp if too close to 0 or 1
        if(cdf_val < 1e-9) cdf_val = 1e-9;
        if(cdf_val > 1.0 - 1e-9) cdf_val = 1.0 - 1e-9;
        
        Z_direct(original_idx, j) = R::qnorm(cdf_val, 0, 1, 1, 0);
      }
      
    } else {
      // [Discrete/Ordinal Case]
      // Logic for discrete/ordinal: Calculate intervals [L, U] considering ties
      int i = 0;
      while (i < n) {
        int start = i;
        while (i < n - 1 && x_sorted(i) == x_sorted(i+1)) {
          i++;
        }
        int end = i;
        
        double lower_bound = C(start);
        double upper_bound = C(end + 1);
        
        if(lower_bound < 1e-9) lower_bound = 1e-9;
        if(upper_bound > 1.0 - 1e-9) upper_bound = 1.0 - 1e-9;
        
        for (int k = start; k <= end; ++k) {
          int original_idx = sort_idx(k);
          L(original_idx, j) = lower_bound;
          U(original_idx, j) = upper_bound;
        }
        i++;
      }
    }
    
    cutpoints_list[j] = C; 
  }
  
  return List::create(
    Named("L") = L, 
    Named("U") = U, 
    Named("Z_direct") = Z_direct,
    Named("cutpoints") = cutpoints_list
  );
}

// -----------------------------------------------------------------------------
// 2. Update Z (Hybrid: Deterministic for Cont, Sampling for Ord)
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
arma::mat update_Z_final(arma::mat Z, const arma::mat& R, 
                         const arma::mat& L, const arma::mat& U, 
                         const arma::mat& Z_direct,
                         const arma::uvec& is_continuous,
                         const std::vector<arma::uvec>& miss_idx_list) {
  int n = Z.n_rows;
  int p = Z.n_cols;
  
  arma::mat is_missing(n, p, arma::fill::zeros); 
  for(int j=0; j<p; ++j) {
    for(auto idx : miss_idx_list[j]) is_missing(idx, j) = 1.0;
  }
  
  for (int j = 0; j < p; ++j) {
    arma::rowvec r_j = R.row(j); r_j.shed_col(j);
    arma::mat R_neg = R; R_neg.shed_row(j); R_neg.shed_col(j);
    
    // Ridge Correction
    arma::mat R_neg_inv;
    try {
      R_neg_inv = inv_sympd(R_neg);
    } catch (...) {
      R_neg_inv = inv(R_neg + arma::eye(p-1, p-1) * 1e-6);
    }
    
    arma::vec beta = R_neg_inv * r_j.t();
    double sigma2 = R(j,j) - arma::as_scalar(r_j * beta);
    double sigma = sqrt(std::max(1e-10, sigma2));
    
    arma::mat Z_neg = Z; Z_neg.shed_col(j);
    arma::vec mu_vec = Z_neg * beta;
    
    for(int i=0; i<n; ++i) {
      double z_new;
      
      // 1. Handling Missing Data: Always use Conditional Normal Sampling
      if (is_missing(i, j) == 1.0) {
        z_new = R::rnorm(mu_vec(i), sigma);
      } 
      // 2. Handling Observed Data
      else {
        if (is_continuous(j) == 1) {
          // [Continuous] Deterministic Update
          // Use values already calculated in sample_bb_moments
          z_new = Z_direct(i, j);
        } else {
          // [Discrete/Ordinal] Truncated Normal Sampling
          double lp = L(i,j);
          double up = U(i,j);
          
          double l_z = R::qnorm(lp, 0, 1, 1, 0);
          double u_z = R::qnorm(up, 0, 1, 1, 0);
          
          if(!std::isfinite(l_z)) l_z = -8.0;
          if(!std::isfinite(u_z)) u_z = 8.0;
          
          double pl = R::pnorm(l_z, mu_vec(i), sigma, 1, 0);
          double pu = R::pnorm(u_z, mu_vec(i), sigma, 1, 0);
          
          if(pl >= pu) {
            z_new = mu_vec(i); 
          } else {
            double u_rand = R::runif(pl, pu);
            z_new = R::qnorm(u_rand, mu_vec(i), sigma, 1, 0);
          }
        }
      }
      
      // Safety Clamp
      if(z_new > 8.0) z_new = 8.0; 
      if(z_new < -8.0) z_new = -8.0;
      if(!std::isfinite(z_new)) z_new = 0.0;
      
      Z(i, j) = z_new;
    }
  }
  return Z;
}

// -----------------------------------------------------------------------------
// 3. Update X (Imputation) - (Continuous variables can share this logic)
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
arma::mat update_X_impute(arma::mat X, const arma::mat& Z, 
                          const std::vector<arma::vec>& cutpoints_list,
                          const std::vector<arma::uvec>& miss_idx_list) {
  int p = X.n_cols;
  for (int j = 0; j < p; ++j) {
    if (miss_idx_list[j].n_elem == 0) continue;
    
    arma::vec C = cutpoints_list[j];
    arma::vec x_col = X.col(j);
    
    arma::uvec valid_idx = find_finite(x_col);
    if(valid_idx.n_elem == 0) continue;
    
    arma::vec x_sorted = sort(x_col.elem(valid_idx));
    
    for (arma::uword idx : miss_idx_list[j]) {
      double z_val = Z(idx, j);
      double u_val = R::pnorm(z_val, 0, 1, 1, 0);
      
      auto it = std::lower_bound(C.begin(), C.end(), u_val);
      int k = std::distance(C.begin(), it);
      
      if (k <= 0) k = 1;
      if (k > (int)x_sorted.n_elem) k = x_sorted.n_elem;
      
      X(idx, j) = x_sorted(k - 1);
    }
  }
  return X;
}

// -----------------------------------------------------------------------------
// 4. Main Gibbs Sampler
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List gibbs_copula_fully_bayesian(arma::mat Y, arma::uvec cont_indices, 
                                       int iter, int burnin, int thin) {
  
  auto start_time = std::chrono::steady_clock::now();
  
  int n = Y.n_rows;
  int p = Y.n_cols;
  
  // Create Boolean Vector for Continuous variables (needs 0-based index handling)
  // Convert 1-based R indices to 0-based C++ indices.
  arma::uvec is_continuous(p, arma::fill::zeros);
  for(unsigned int k=0; k < cont_indices.n_elem; ++k) {
    int idx = cont_indices(k) - 1; // R -> C++ Index conversion
    if(idx >= 0 && idx < p) {
      is_continuous(idx) = 1;
    }
  }
  
  // Initialization: Handle All-NA columns strictly
  arma::mat X = Y;
  std::vector<arma::uvec> miss_idx_list(p);
  
  for (int j = 0; j < p; ++j) {
    arma::vec col = Y.col(j);
    arma::uvec miss_idx = arma::find_nonfinite(col);
    miss_idx_list[j] = miss_idx;
    
    if (miss_idx.n_elem > 0) {
      arma::uvec obs_idx = arma::find_finite(col);
      if (obs_idx.n_elem > 0) {
        double col_mean = arma::mean(col.elem(obs_idx));
        for(arma::uword i : miss_idx) {
          X(i, j) = col_mean;
        }
      }
    }
  }
  
  // Initialize Z
  arma::mat Z(n, p, arma::fill::zeros);
  for (int j=0; j<p; ++j) {
    arma::uvec s_idx = arma::sort_index(X.col(j));
    for(int i=0; i<n; ++i) {
      Z(s_idx(i), j) = R::qnorm((i + 1.0) / (n + 1.0), 0, 1, 1, 0);
    }
  }
  
  arma::mat R_mat = arma::cor(Z);
  if(!R_mat.is_finite()) R_mat = arma::eye(p, p);
  
  Rcpp::List R_sample, X_sample, C_sample;
  Rcpp::Environment stats("package:stats");
  Rcpp::Function rWishart = stats["rWishart"];
  
  Rcpp::Rcout << "Starting BBGC (Hybrid Continuous/Discrete)..." << std::endl;
  
  for (int it = 0; it < iter; ++it) {
    
    // Step 1: Update Bounds & Direct Z (New Function)
    List bounds = sample_bb_moments(X, is_continuous);
    
    // Step 2: Update Z (New Function)
    Z = update_Z_final(Z, R_mat, 
                       bounds["L"], bounds["U"], bounds["Z_direct"], 
                       is_continuous, miss_idx_list);
    
    // Step 3: Update R (Standard)
    arma::mat S = Z.t() * Z;
    double df = n + p + 1;
    Rcpp::NumericVector res = rWishart(1, df, arma::inv_sympd(S + arma::eye(p,p)));
    arma::cube W_res(res.begin(), p, p, 1, false);
    arma::mat Sigma = arma::inv_sympd(W_res.slice(0));
    R_mat = cov2cor_safe(Sigma);
    
    // Step 4: Update X (Imputation)
    X = update_X_impute(X, Z, bounds["cutpoints"], miss_idx_list);
    
    if (it >= burnin && (it - burnin) % thin == 0) {
      R_sample.push_back(R_mat);
      X_sample.push_back(X);
      C_sample.push_back(bounds["cutpoints"]); // cutpoints are generated in both cases
    }
    
    if (it % 1000 == 0) {
      Rcpp::Rcout << "Iteration: " << it << std::endl;
      auto end_time = std::chrono::steady_clock::now();
      auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
      Rcpp::Rcout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;
      Rcpp::checkUserInterrupt();
    }
  }
  
  Rcpp::Rcout << "Iteration : " << iter << std::endl;
  auto end_time = std::chrono::steady_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
  Rcpp::Rcout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;
  
  return List::create(
    Named("R_sample") = R_sample, 
    Named("X_sample") = X_sample,
    Named("C_sample") = C_sample
  );
}
