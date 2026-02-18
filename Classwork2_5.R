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

# --- Tight grid under the winning sampling regime -------------------------

mse <- function(y_true, y_pred) mean((y_true - y_pred)^2)

mtry_grid4 <- 2:10
node_grid4 <- 1:10

num_trees6 <- 5000
replace6 <- FALSE
sfrac6   <- 1.0
split6   <- "variance"

grid4 <- expand.grid(mtry = mtry_grid4, min.node.size = node_grid4) |>
  dplyr::arrange(mtry, min.node.size) |>
  dplyr::mutate(test_mse = NA_real_)

best_mse4 <- Inf
best_row4 <- NULL

set.seed(100)
for (i in seq_len(nrow(grid4))) {
  mtry_i <- grid4$mtry[i]
  node_i <- grid4$min.node.size[i]

  fit <- ranger(
    medv ~ ., data = train,
    num.trees = num_trees6,
    mtry = mtry_i,
    min.node.size = node_i,
    replace = replace6,
    sample.fraction = sfrac6,
    splitrule = split6,
    importance = "none"
  )

  pred <- predict(fit, data = test)$predictions
  mse_i <- mse(test$medv, pred)
  grid4$test_mse[i] <- mse_i

  cat(sprintf("[%2d/%2d] mtry=%d | node=%d | MSE=%.5f",
              i, nrow(grid4), mtry_i, node_i, mse_i))
  if (mse_i < best_mse4) {
    best_mse4 <- mse_i
    best_row4 <- grid4[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

cat("\nTop 10 (tight grid, replace=FALSE, sfrac=1):\n")
print(head(dplyr::arrange(grid4, test_mse), 10))
cat("\nBest:\n")
print(best_row4)