# ============================================================
# Boston (ISLR2) medv prediction competition — XGBoost tuning
# Uses test set as leaderboard (allowed). Prints test MSE each run.
# Split: 80/20 with set.seed(100)
# ============================================================

suppressMessages(library(tidyverse))
suppressMessages(library(ISLR2))
suppressMessages(library(xgboost))

mse <- function(y_true, y_pred) mean((y_true - y_pred)^2)

# ----------------------------
# Data + required split
# ----------------------------
set.seed(100)
data(Boston)

n <- nrow(Boston)
train_idx <- sample(seq_len(n), size = floor(0.8 * n), replace = FALSE)
train <- Boston[train_idx, ]
test  <- Boston[-train_idx, ]
rm(n, train_idx)

# X matrices (drop intercept)
x_train_full <- model.matrix(medv ~ ., data = train)[, -1]
y_train_full <- train$medv

x_test <- model.matrix(medv ~ ., data = test)[, -1]
y_test <- test$medv

# ----------------------------
# Inner split for early stopping (train -> (subtrain, valid))
# ----------------------------
set.seed(100)
ntr <- nrow(x_train_full)
valid_idx <- sample(seq_len(ntr), size = floor(0.2 * ntr), replace = FALSE)
sub_idx   <- setdiff(seq_len(ntr), valid_idx)

dtrain <- xgb.DMatrix(data = x_train_full[sub_idx, ], label = y_train_full[sub_idx])
dvalid <- xgb.DMatrix(data = x_train_full[valid_idx, ], label = y_train_full[valid_idx])
dtest  <- xgb.DMatrix(data = x_test, label = y_test)

watchlist <- list(train = dtrain, valid = dvalid)

# ----------------------------
# Utility: run one configuration + print progress
# ----------------------------
run_xgb <- function(params,
                    nrounds = 5000,
                    early_stopping_rounds = 50,
                    verbose_eval = 0,
                    seed = 100) {

  params2 <- modifyList(
    list(
      booster = "gbtree",
      objective = "reg:squarederror",
      eval_metric = "rmse",
      seed = seed
    ),
    params
  )

  fit <- xgb.train(
    params = params2,
    data = dtrain,
    nrounds = nrounds,
    evals = list(train = dtrain, valid = dvalid),   # new name (was watchlist)
    early_stopping_rounds = early_stopping_rounds,
    verbose = verbose_eval
  )

  # Robust best-iteration retrieval:
  # - fit$best_iteration is base-1 (R attribute) when present
  # - xgb.attr(fit, "best_iteration") is base-0 when present -> add 1
  best_iter <- fit$best_iteration
  if (is.null(best_iter)) {
    bi0 <- xgb.attr(fit, "best_iteration")
    if (!is.null(bi0)) best_iter <- as.integer(bi0) + 1L
  }
  if (is.null(best_iter)) best_iter <- as.integer(nrounds)

  # In R, predict() will by default use best_iteration when early stopping is used
  pred <- predict(fit, dtest)
  test_mse <- mse(y_test, pred)

  list(fit = fit, best_iter = as.integer(best_iter), test_mse = test_mse)
}

# ============================================================
# STAGE B2: re-tune subsample x colsample with best (lambda, alpha)
# Fix: depth=3, mcw=1, eta=0.05, gamma=0, lambda=2, alpha=0.1
# ============================================================

best_md  <- 3
best_mcw <- 1
best_eta <- 0.05
best_gam <- 0
best_lam <- 2
best_alp <- 0.1

sub_grid <- c(0.4, 0.5, 0.6, 0.7)
col_grid <- c(0.8, 0.9, 1.0)

gridB2 <- expand.grid(subsample = sub_grid,
                      colsample_bytree = col_grid) |>
  dplyr::arrange(subsample, colsample_bytree) |>
  dplyr::mutate(test_mse = NA_real_, best_iter = NA_integer_)

best_mse_B2 <- Inf
best_cfg_B2 <- NULL

set.seed(100)
for (i in seq_len(nrow(gridB2))) {
  ss <- gridB2$subsample[i]
  cs <- gridB2$colsample_bytree[i]

  params_i <- base_params
  params_i$max_depth <- best_md
  params_i$min_child_weight <- best_mcw
  params_i$eta <- best_eta
  params_i$gamma <- best_gam
  params_i$lambda <- best_lam
  params_i$alpha  <- best_alp
  params_i$subsample <- ss
  params_i$colsample_bytree <- cs

  out <- run_xgb(params_i, seed = 100)

  gridB2$test_mse[i]  <- out$test_mse
  gridB2$best_iter[i] <- out$best_iter

  cat(sprintf("[%2d/%2d] subsample=%.1f | colsample=%.1f | best_iter=%d | test MSE=%.5f",
              i, nrow(gridB2), ss, cs, out$best_iter, out$test_mse))

  if (out$test_mse < best_mse_B2) {
    best_mse_B2 <- out$test_mse
    best_cfg_B2 <- gridB2[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

cat("\nTop 10 (Stage B2):\n")
print(head(dplyr::arrange(gridB2, test_mse), 10))
cat("\nBest (Stage B2):\n")
print(best_cfg_B2)