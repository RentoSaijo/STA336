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

# --- RF local refinement around best --------------------------------------

mse <- function(y_true, y_pred) mean((y_true - y_pred)^2)

mtry_grid2 <- 3:8
node_grid2 <- 1:5
num_trees2 <- 500

results2 <- expand.grid(
  mtry = mtry_grid2,
  min.node.size = node_grid2
) %>%
  arrange(mtry, min.node.size) %>%
  mutate(test_mse = NA_real_)

best_mse2 <- Inf
best_row2 <- NULL

set.seed(100)
for (i in seq_len(nrow(results2))) {
  mtry_i <- results2$mtry[i]
  node_i <- results2$min.node.size[i]

  fit <- ranger(
    medv ~ ., data = train,
    num.trees = num_trees2,
    mtry = mtry_i,
    min.node.size = node_i,
    importance = "none"
  )

  pred <- predict(fit, data = test)$predictions
  mse_i <- mse(test$medv, pred)
  results2$test_mse[i] <- mse_i

  cat(sprintf(
    "[%2d/%2d] trees=%d | mtry=%d | min.node.size=%d | test MSE=%.5f",
    i, nrow(results2), num_trees2, mtry_i, node_i, mse_i
  ))

  if (mse_i < best_mse2) {
    best_mse2 <- mse_i
    best_row2 <- results2[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

cat("\nTop 10 (refinement):\n")
print(head(results2 %>% arrange(test_mse), 10))

cat("\nBest (refinement):\n")
print(best_row2)