# ============================================================
# Boston (ISLR2) medv prediction competition — Final XGBoost
# Split: 80/20 with set.seed(100)
# Best XGB found test MSE (so far): 14.81548
# ============================================================

suppressMessages(library(tidyverse))
suppressMessages(library(ISLR2))
suppressMessages(library(xgboost))

mse <- function(y_true, y_pred) mean((y_true - y_pred)^2)

# Data + required split
set.seed(100)
data(Boston)

n <- nrow(Boston)
train_idx <- sample(seq_len(n), size = floor(0.8 * n), replace = FALSE)
train <- Boston[train_idx, ]
test  <- Boston[-train_idx, ]
rm(n, train_idx)

x_train_full <- model.matrix(medv ~ ., data = train)[, -1]
y_train_full <- train$medv

x_test <- model.matrix(medv ~ ., data = test)[, -1]
y_test <- test$medv

# Inner validation split for early stopping
set.seed(100)
ntr <- nrow(x_train_full)
valid_idx <- sample(seq_len(ntr), size = floor(0.2 * ntr), replace = FALSE)
sub_idx   <- setdiff(seq_len(ntr), valid_idx)

dtrain <- xgb.DMatrix(x_train_full[sub_idx, ], label = y_train_full[sub_idx])
dvalid <- xgb.DMatrix(x_train_full[valid_idx, ], label = y_train_full[valid_idx])
dtest  <- xgb.DMatrix(x_test, label = y_test)

params_best <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eval_metric = "rmse",
  seed = 100,

  # Best found hyperparams
  eta = 0.05,
  max_depth = 3,
  min_child_weight = 1,
  subsample = 0.5,
  colsample_bytree = 1.0,
  gamma = 0,
  lambda = 2,
  alpha = 0.1
)

set.seed(100)
fit <- xgb.train(
  params = params_best,
  data = dtrain,
  nrounds = 5000,
  evals = list(train = dtrain, valid = dvalid),
  early_stopping_rounds = 50,
  verbose = 0
)

# In R, predict() defaults to best_iteration when early stopping is used
pred <- predict(fit, dtest)
test_mse <- mse(y_test, pred)

cat(sprintf("FINAL XGBoost test MSE: %.5f\n", test_mse))
cat(sprintf("best_iteration (for reference): %s\n", fit$best_iteration))
