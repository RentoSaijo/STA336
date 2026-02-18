# ============================================================
# Boston (ISLR2) medv prediction competition
# Final model: Random Forest (ranger)
# Split: 80/20 with set.seed(100)
# Hyperparams: num.trees=10000, mtry=5, min.node.size=1,
#              replace=FALSE, sample.fraction=0.95, splitrule="variance"
# ============================================================

suppressMessages(library(tidyverse))
suppressMessages(library(ISLR2))
suppressMessages(library(ranger))

mse <- function(y_true, y_pred) mean((y_true - y_pred)^2)

# Data + split
set.seed(100)
data(Boston)

n <- nrow(Boston)
train_idx <- sample(seq_len(n), size = floor(0.8 * n), replace = FALSE)
train <- Boston[train_idx, ]
test  <- Boston[-train_idx, ]
rm(n, train_idx)

# Final fit (toggle importance as you like)
mtry_final  <- 5
node_final  <- 1
sf_final    <- 0.95
trees_final <- 10000

set.seed(100)
rf_final <- ranger(
  medv ~ ., data = train,
  num.trees = trees_final,
  mtry = mtry_final,
  min.node.size = node_final,
  replace = FALSE,
  sample.fraction = sf_final,
  splitrule = "variance",
  importance = "permutation"  # change to "none" if you want
)

# Test MSE
pred_test <- predict(rf_final, data = test)$predictions
test_mse  <- mse(test$medv, pred_test)
cat(sprintf("FINAL Random Forest test MSE: %.5f\n", test_mse))

# Optional: importance (only meaningful if importance="permutation")
if (!is.null(rf_final$variable.importance)) {
  imp <- sort(rf_final$variable.importance, decreasing = TRUE)
  cat("\nPermutation importance:\n")
  print(imp)
}
