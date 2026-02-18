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

# --- Micro-sweep sample.fraction near 1.0 --------------------------------

mse <- function(y_true, y_pred) mean((y_true - y_pred)^2)

mtry_star <- best_row4$mtry
node_star <- best_row4$min.node.size

num_trees7 <- 7000  # stabilize the very top
split7 <- "variance"

sf_grid <- c(0.85, 0.90, 0.95, 1.00)

out_sf <- data.frame(sample.fraction = sf_grid, test_mse = NA_real_)

set.seed(100)
for (i in seq_along(sf_grid)) {
  sf <- sf_grid[i]
  fit <- ranger(
    medv ~ ., data = train,
    num.trees = num_trees7,
    mtry = mtry_star,
    min.node.size = node_star,
    replace = FALSE,
    sample.fraction = sf,
    splitrule = split7,
    importance = "none"
  )
  pred <- predict(fit, data = test)$predictions
  out_sf$test_mse[i] <- mse(test$medv, pred)

  cat(sprintf("[%d/%d] sample.fraction=%.2f | MSE=%.5f\n",
              i, length(sf_grid), sf, out_sf$test_mse[i]))
}

out_sf[order(out_sf$test_mse), ]