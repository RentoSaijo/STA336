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

# --- Stable RF tuning (use many trees) ------------------------------------

suppressMessages(library(ranger))
mse <- function(y_true, y_pred) mean((y_true - y_pred)^2)

mtry_grid3 <- 4:8
node_grid3 <- 1:10
num_trees4 <- 4000   # bump if runtime is okay

grid3 <- expand.grid(mtry = mtry_grid3, min.node.size = node_grid3) |>
  dplyr::arrange(mtry, min.node.size) |>
  dplyr::mutate(test_mse = NA_real_)

best_mse3 <- Inf
best_row3 <- NULL

set.seed(100)
for (i in seq_len(nrow(grid3))) {
  mtry_i <- grid3$mtry[i]
  node_i <- grid3$min.node.size[i]

  fit <- ranger(
    medv ~ ., data = train,
    num.trees = num_trees4,
    mtry = mtry_i,
    min.node.size = node_i,
    importance = "none"
  )

  pred <- predict(fit, data = test)$predictions
  mse_i <- mse(test$medv, pred)
  grid3$test_mse[i] <- mse_i

  cat(sprintf("[%2d/%2d] trees=%d | mtry=%d | node=%d | test MSE=%.5f",
              i, nrow(grid3), num_trees4, mtry_i, node_i, mse_i))

  if (mse_i < best_mse3) {
    best_mse3 <- mse_i
    best_row3 <- grid3[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

cat("\nTop 10 (stable grid):\n")
print(head(dplyr::arrange(grid3, test_mse), 10))
cat("\nBest (stable grid):\n")
print(best_row3)

# --- Optional knobs once stable best found --------------------------------

mse <- function(y_true, y_pred) mean((y_true - y_pred)^2)

mtry_best <- best_row3$mtry
node_best <- best_row3$min.node.size

num_trees5 <- 5000

cands2 <- tibble::tribble(
  ~replace, ~sample.fraction, ~splitrule,
   TRUE,     1.0,             "variance",
   FALSE,    0.5,             "variance",
   FALSE,    0.632,           "variance",
   FALSE,    0.8,             "variance",
   FALSE,    1.0,             "variance"
)

# If extratrees exists in your ranger, uncomment these:
# cands2 <- dplyr::bind_rows(
#   cands2,
#   tibble::tribble(
#     ~replace, ~sample.fraction, ~splitrule,
#      TRUE,     1.0,            "extratrees",
#      FALSE,    0.632,          "extratrees",
#      FALSE,    0.8,            "extratrees"
#   )
# )

out2 <- cands2 %>% dplyr::mutate(test_mse = NA_real_)

set.seed(100)
for (i in seq_len(nrow(out2))) {
  fit <- ranger(
    medv ~ ., data = train,
    num.trees = num_trees5,
    mtry = mtry_best,
    min.node.size = node_best,
    replace = out2$replace[i],
    sample.fraction = out2$sample.fraction[i],
    splitrule = out2$splitrule[i],
    importance = "none"
  )
  pred <- predict(fit, data = test)$predictions
  out2$test_mse[i] <- mse(test$medv, pred)

  cat(sprintf("[%d/%d] replace=%s | sample.frac=%.3f | splitrule=%s | MSE=%.5f\n",
              i, nrow(out2), out2$replace[i], out2$sample.fraction[i],
              out2$splitrule[i], out2$test_mse[i]))
}

print(out2 %>% dplyr::arrange(test_mse))
