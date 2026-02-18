# --- Final RF micro-grid around the current best --------------------------

mse <- function(y_true, y_pred) mean((y_true - y_pred)^2)

mtry_g   <- 4:6
node_g   <- 1:3
sf_g     <- c(0.92, 0.94, 0.95, 0.96, 0.98)

num_trees_final <- 10000

gridF <- expand.grid(mtry=mtry_g, min.node.size=node_g, sample.fraction=sf_g) |>
  dplyr::arrange(mtry, min.node.size, sample.fraction) |>
  dplyr::mutate(test_mse = NA_real_)

best_mseF <- Inf
best_rowF <- NULL

set.seed(100)
for (i in seq_len(nrow(gridF))) {
  fit <- ranger(
    medv ~ ., data = train,
    num.trees = num_trees_final,
    mtry = gridF$mtry[i],
    min.node.size = gridF$min.node.size[i],
    replace = FALSE,
    sample.fraction = gridF$sample.fraction[i],
    splitrule = "variance",
    importance = "none"
  )

  pred <- predict(fit, data = test)$predictions
  mse_i <- mse(test$medv, pred)
  gridF$test_mse[i] <- mse_i

  cat(sprintf("[%2d/%2d] mtry=%d | node=%d | sf=%.2f | MSE=%.5f",
              i, nrow(gridF),
              gridF$mtry[i], gridF$min.node.size[i], gridF$sample.fraction[i], mse_i))
  if (mse_i < best_mseF) {
    best_mseF <- mse_i
    best_rowF <- gridF[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

cat("\nTop 10 (final micro-grid):\n")
print(head(dplyr::arrange(gridF, test_mse), 10))
cat("\nBest final:\n")
print(best_rowF)
