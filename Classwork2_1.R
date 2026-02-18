# Load libraries.
suppressMessages(library(tidyverse))
suppressMessages(library(ISLR2))

# Set seed.
set.seed(100)

# Load data.
data(Boston)

# Split data.
n         <- nrow(Boston)
train_idx <- sample(seq_len(n), size = floor(0.8 * n), replace = FALSE)
train <- Boston[train_idx, ]
test  <- Boston[-train_idx, ]
rm(n, train_idx)

# --- RF coarse tuning: mtry x min.node.size -------------------------------

suppressMessages(library(ranger))

mse <- function(y_true, y_pred) mean((y_true - y_pred)^2)

mtry_grid <- 1:(ncol(train) - 1)  # 12 predictors
node_grid <- c(1, 2, 3, 5, 8, 10, 15, 20, 30)

num_trees <- 500

results <- expand.grid(
  mtry = mtry_grid,
  min.node.size = node_grid
) %>%
  arrange(mtry, min.node.size) %>%
  mutate(test_mse = NA_real_)

best_mse <- Inf
best_row <- NULL

set.seed(100)  # keeps ranger randomness consistent across runs

for (i in seq_len(nrow(results))) {
  mtry_i <- results$mtry[i]
  node_i <- results$min.node.size[i]

  fit <- ranger(
    medv ~ ., data = train,
    num.trees = num_trees,
    mtry = mtry_i,
    min.node.size = node_i,
    importance = "none"
  )

  pred <- predict(fit, data = test)$predictions
  mse_i <- mse(test$medv, pred)
  results$test_mse[i] <- mse_i

  cat(sprintf(
    "[%3d/%3d] num.trees=%d | mtry=%2d | min.node.size=%2d | test MSE = %.4f",
    i, nrow(results), num_trees, mtry_i, node_i, mse_i
  ))

  if (mse_i < best_mse) {
    best_mse <- mse_i
    best_row <- results[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

# Show top configs
results_sorted <- results %>% arrange(test_mse)

cat("\nTop 10 configs (lowest test MSE):\n")
print(head(results_sorted, 10))

cat("\nBest config overall:\n")
print(best_row)

# If you want to keep it for next stage:
best_mtry  <- best_row$mtry
best_node  <- best_row$min.node.size
best_mse   <- best_row$test_mse
