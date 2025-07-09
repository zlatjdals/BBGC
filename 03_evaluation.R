# Evaluation Function

evaluate_copula <- function(imputed, original, idx_na) {
  true_vals <- original[idx_na]
  imp_vals <- imputed[idx_na]

  df <- data.frame(True = true_vals, Imputed = imp_vals)
  model <- lm(True ~ Imputed, data = df)
  summary_lm <- summary(model)

  # NRMSE 
  mse <- mean((true_vals - imp_vals)^2)
  rmse <- sqrt(mse)
  nrmse <- rmse / sd(true_vals)

  list(
    r_squared = summary_lm$r.squared,
    coefficients = coef(model),
    nrmse = nrmse,
    plot = ggplot(df, aes(x = True, y = Imputed)) +
      geom_point(color = "steelblue", alpha = 0.6) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
      theme_minimal() +
      coord_equal() +
      labs(title = "Copula Imputation: Imputed vs True", x = "True Value", y = "Imputed Value")
  )
}
