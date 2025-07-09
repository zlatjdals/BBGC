# Evaluation Function

evaluate_copula <- function(imputed, original, idx_na) {
  df <- data.frame(True = original[idx_na], Imputed = imputed[idx_na])
  model <- lm(True ~ Imputed, data = df)
  summary_lm <- summary(model)

  list(
    r_squared = summary_lm$r.squared,
    coefficients = coef(model),
    plot = ggplot(df, aes(x = True, y = Imputed)) +
      geom_point(color = "steelblue", alpha = 0.6) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
      theme_minimal() +
      coord_equal() +
      labs(title = "Copula Imputation: Imputed vs True", x = "True Value", y = "Imputed Value")
  )
}
