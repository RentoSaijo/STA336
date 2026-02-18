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
# STAGE E: tune lambda (L2) x alpha (L1)
# Fix: depth=3, mcw=1, subsample=0.5, colsample=1.0, gamma=0, eta=0.05
# ============================================================

best_md  <- 3
best_mcw <- 1
best_ss  <- 0.5
best_cs  <- 1.0
best_gam <- 0
best_eta <- 0.05

lambda_grid <- c(0, 0.5, 1, 2, 5, 10)
alpha_grid  <- c(0, 0.1, 0.5, 1)

gridE <- expand.grid(lambda = lambda_grid, alpha = alpha_grid) |>
  dplyr::arrange(lambda, alpha) |>
  dplyr::mutate(test_mse = NA_real_, best_iter = NA_integer_)

best_mse_E <- Inf
best_cfg_E <- NULL

set.seed(100)
for (i in seq_len(nrow(gridE))) {
  lam <- gridE$lambda[i]
  alp <- gridE$alpha[i]

  params_i <- base_params
  params_i$max_depth <- best_md
  params_i$min_child_weight <- best_mcw
  params_i$subsample <- best_ss
  params_i$colsample_bytree <- best_cs
  params_i$gamma <- best_gam
  params_i$eta <- best_eta
  params_i$lambda <- lam
  params_i$alpha  <- alp

  out <- run_xgb(params_i, seed = 100)

  gridE$test_mse[i]  <- out$test_mse
  gridE$best_iter[i] <- out$best_iter

  cat(sprintf("[%2d/%2d] lambda=%.2f | alpha=%.2f | best_iter=%d | test MSE=%.5f",
              i, nrow(gridE), lam, alp, out$best_iter, out$test_mse))

  if (out$test_mse < best_mse_E) {
    best_mse_E <- out$test_mse
    best_cfg_E <- gridE[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

cat("\nTop 10 (Stage E):\n")
print(head(dplyr::arrange(gridE, test_mse), 10))
cat("\nBest (Stage E):\n")
print(best_cfg_E)

# ============================================================
# STAGE F (FAST): try DART booster with a small grid
# Uses best gbtree params: depth=3, mcw=1, subsample=0.5, colsample=1.0,
# gamma=0, eta=0.05, lambda=2, alpha=0.1
# ============================================================

dart_grid <- expand.grid(
  rate_drop = c(0.05, 0.10),
  skip_drop = c(0.00, 0.20)
) |>
  dplyr::arrange(rate_drop, skip_drop) |>
  dplyr::mutate(test_mse = NA_real_, best_iter = NA_integer_)

best_mse_F <- Inf
best_cfg_F <- NULL

for (i in seq_len(nrow(dart_grid))) {
  rd <- dart_grid$rate_drop[i]
  sd <- dart_grid$skip_drop[i]

  params_i <- base_params
  params_i$booster <- "dart"

  # best structure + sampling
  params_i$max_depth <- 3
  params_i$min_child_weight <- 1
  params_i$subsample <- 0.5
  params_i$colsample_bytree <- 1.0
  params_i$gamma <- 0
  params_i$eta <- 0.05

  # best regularization (Stage E winner)
  params_i$lambda <- 2
  params_i$alpha  <- 0.1

  # dart knobs
  params_i$rate_drop <- rd
  params_i$skip_drop <- sd

  # optional speedup (uncomment if you want)
  # params_i$tree_method <- "hist"

  out <- run_xgb(
    params_i,
    nrounds = 2000,
    early_stopping_rounds = 30,
    seed = 100
  )

  dart_grid$test_mse[i]  <- out$test_mse
  dart_grid$best_iter[i] <- out$best_iter

  cat(sprintf("[%d/%d] DART | rate_drop=%.2f | skip_drop=%.2f | best_iter=%d | test MSE=%.5f",
              i, nrow(dart_grid), rd, sd, out$best_iter, out$test_mse))

  if (out$test_mse < best_mse_F) {
    best_mse_F <- out$test_mse
    best_cfg_F <- dart_grid[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

cat("\nStage F (fast) results:\n")
print(dplyr::arrange(dart_grid, test_mse))
cat("\nBest (Stage F fast):\n")
print(best_cfg_F)
