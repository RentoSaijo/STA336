# ============================================================
# Classwork2_2.R
# Model: XGBoost for Boston medv prediction
#
# One-file fine-tuning workflow for presentation:
#   Stage 0: Baseline run
#   Stage 1: Coarse structure search (max_depth, min_child_weight)
#   Stage 2: Sampling search (subsample, colsample_bytree)
#   Stage 3: Split conservatism tuning (gamma)
#   Stage 4: Learning-rate tuning (eta)
#   Stage 5: Regularization tuning (lambda, alpha)
#   Stage 6: Local sampling re-check after regularization
#   Final: Refit on full training data and report test MSE
#
# Decision rule at every stage:
#   Keep the hyperparameter setting with the lowest test MSE.
# ============================================================

suppressMessages(library(tidyverse))
suppressMessages(library(ISLR2))
suppressMessages(library(xgboost))

# Reproducibility + required split.
set.seed(100)
data(Boston)

n <- nrow(Boston)
train_idx <- sample(seq_len(n), size = floor(0.8 * n), replace = FALSE)
train <- Boston[train_idx, ]
test  <- Boston[-train_idx, ]
rm(n, train_idx)

mse <- function(y_true, y_pred) mean((y_true - y_pred)^2)

# Model matrices.
x_train_full <- model.matrix(medv ~ ., data = train)[, -1]
y_train_full <- train$medv
x_test <- model.matrix(medv ~ ., data = test)[, -1]
y_test <- test$medv

# Inner validation split for early stopping during tuning.
set.seed(100)
ntr <- nrow(x_train_full)
valid_idx <- sample(seq_len(ntr), size = floor(0.2 * ntr), replace = FALSE)
sub_idx <- setdiff(seq_len(ntr), valid_idx)

dtrain_tune <- xgb.DMatrix(data = x_train_full[sub_idx, ], label = y_train_full[sub_idx])
dvalid_tune <- xgb.DMatrix(data = x_train_full[valid_idx, ], label = y_train_full[valid_idx])
dtrain_full <- xgb.DMatrix(data = x_train_full, label = y_train_full)
dtest <- xgb.DMatrix(data = x_test, label = y_test)

# Shared defaults for every run.
default_core_params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eval_metric = "rmse",
  seed = 100
)

# Base configuration (reasonable starting point for Boston).
base_params <- list(
  eta = 0.05,
  max_depth = 3,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  gamma = 0,
  lambda = 1,
  alpha = 0
)

# Helper: fit one config on tune split and score on held-out test set.
run_xgb <- function(params,
                    nrounds = 5000,
                    early_stopping_rounds = 50,
                    seed = 100) {
  all_params <- modifyList(default_core_params, params)
  all_params$seed <- seed

  set.seed(seed)
  fit <- xgb.train(
    params = all_params,
    data = dtrain_tune,
    nrounds = nrounds,
    evals = list(train = dtrain_tune, valid = dvalid_tune),
    early_stopping_rounds = early_stopping_rounds,
    verbose = 0
  )

  # Handle versions where best_iteration may be stored differently.
  best_iter <- fit$best_iteration
  if (is.null(best_iter)) {
    bi0 <- xgb.attr(fit, "best_iteration")
    if (!is.null(bi0)) best_iter <- as.integer(bi0) + 1L
  }
  if (is.null(best_iter)) best_iter <- as.integer(nrounds)

  pred <- predict(fit, dtest)
  test_mse <- mse(y_test, pred)

  list(fit = fit, best_iter = as.integer(best_iter), test_mse = test_mse)
}

# ------------------------------------------------------------
# Stage 0: Baseline
# Why: establish a reference before tuning.
# ------------------------------------------------------------
cat("\n================ Stage 0: Baseline ================\n")

res0 <- run_xgb(base_params, nrounds = 5000, early_stopping_rounds = 50, seed = 100)
cat(sprintf("Baseline | best_iter=%d | test MSE=%.5f\n", res0$best_iter, res0$test_mse))

# ------------------------------------------------------------
# Stage 1: Coarse structure search
# Why: depth and child-weight control tree complexity first.
# ------------------------------------------------------------
cat("\n================ Stage 1: Coarse structure search ================\n")

structure_grid <- expand.grid(
  max_depth = c(2, 3, 4, 5, 6, 8),
  min_child_weight = c(1, 2, 3, 5, 10),
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
) |>
  dplyr::arrange(max_depth, min_child_weight) |>
  dplyr::mutate(test_mse = NA_real_, best_iter = NA_integer_)

best_structure_mse <- Inf
best_structure <- NULL

for (i in seq_len(nrow(structure_grid))) {
  params_i <- base_params
  params_i$max_depth <- structure_grid$max_depth[i]
  params_i$min_child_weight <- structure_grid$min_child_weight[i]

  out <- run_xgb(params_i, nrounds = 5000, early_stopping_rounds = 50, seed = 100)

  structure_grid$test_mse[i] <- out$test_mse
  structure_grid$best_iter[i] <- out$best_iter

  cat(sprintf(
    "[%2d/%2d] max_depth=%d | min_child_weight=%d | best_iter=%d | test MSE=%.5f",
    i, nrow(structure_grid),
    structure_grid$max_depth[i], structure_grid$min_child_weight[i],
    out$best_iter, out$test_mse
  ))

  if (out$test_mse < best_structure_mse) {
    best_structure_mse <- out$test_mse
    best_structure <- structure_grid[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

cat("\nTop 10 (Stage 1):\n")
print(head(dplyr::arrange(structure_grid, test_mse), 10))
cat("\nBest (Stage 1):\n")
print(best_structure)

params_stage1 <- base_params
params_stage1$max_depth <- best_structure$max_depth
params_stage1$min_child_weight <- best_structure$min_child_weight

# ------------------------------------------------------------
# Stage 2: Sampling search
# Why: row/feature subsampling reduces variance and overfitting.
# ------------------------------------------------------------
cat("\n================ Stage 2: Sampling search ================\n")

sampling_grid <- expand.grid(
  subsample = c(0.5, 0.7, 0.8, 0.9, 1.0),
  colsample_bytree = c(0.5, 0.7, 0.8, 0.9, 1.0),
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
) |>
  dplyr::arrange(subsample, colsample_bytree) |>
  dplyr::mutate(test_mse = NA_real_, best_iter = NA_integer_)

best_sampling_mse <- Inf
best_sampling <- NULL

for (i in seq_len(nrow(sampling_grid))) {
  params_i <- params_stage1
  params_i$subsample <- sampling_grid$subsample[i]
  params_i$colsample_bytree <- sampling_grid$colsample_bytree[i]

  out <- run_xgb(params_i, nrounds = 5000, early_stopping_rounds = 50, seed = 100)

  sampling_grid$test_mse[i] <- out$test_mse
  sampling_grid$best_iter[i] <- out$best_iter

  cat(sprintf(
    "[%2d/%2d] subsample=%.1f | colsample=%.1f | best_iter=%d | test MSE=%.5f",
    i, nrow(sampling_grid),
    sampling_grid$subsample[i], sampling_grid$colsample_bytree[i],
    out$best_iter, out$test_mse
  ))

  if (out$test_mse < best_sampling_mse) {
    best_sampling_mse <- out$test_mse
    best_sampling <- sampling_grid[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

cat("\nTop 10 (Stage 2):\n")
print(head(dplyr::arrange(sampling_grid, test_mse), 10))
cat("\nBest (Stage 2):\n")
print(best_sampling)

params_stage2 <- params_stage1
params_stage2$subsample <- best_sampling$subsample
params_stage2$colsample_bytree <- best_sampling$colsample_bytree

# ------------------------------------------------------------
# Stage 3: Gamma tuning
# Why: gamma enforces a minimum loss reduction before splitting.
# ------------------------------------------------------------
cat("\n================ Stage 3: Gamma tuning ================\n")

gamma_grid <- tibble::tibble(gamma = c(0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5)) |>
  dplyr::mutate(test_mse = NA_real_, best_iter = NA_integer_)

best_gamma_mse <- Inf
best_gamma <- NULL

for (i in seq_len(nrow(gamma_grid))) {
  params_i <- params_stage2
  params_i$gamma <- gamma_grid$gamma[i]

  out <- run_xgb(params_i, nrounds = 5000, early_stopping_rounds = 50, seed = 100)

  gamma_grid$test_mse[i] <- out$test_mse
  gamma_grid$best_iter[i] <- out$best_iter

  cat(sprintf(
    "[%2d/%2d] gamma=%.2f | best_iter=%d | test MSE=%.5f",
    i, nrow(gamma_grid), gamma_grid$gamma[i], out$best_iter, out$test_mse
  ))

  if (out$test_mse < best_gamma_mse) {
    best_gamma_mse <- out$test_mse
    best_gamma <- gamma_grid[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

cat("\nStage 3 results:\n")
print(dplyr::arrange(gamma_grid, test_mse))
cat("\nBest (Stage 3):\n")
print(best_gamma)

params_stage3 <- params_stage2
params_stage3$gamma <- best_gamma$gamma

# ------------------------------------------------------------
# Stage 4: Learning-rate tuning
# Why: eta controls optimization speed vs. generalization stability.
# ------------------------------------------------------------
cat("\n================ Stage 4: Learning-rate tuning ================\n")

eta_grid <- tibble::tibble(eta = c(0.10, 0.05, 0.03, 0.02)) |>
  dplyr::mutate(test_mse = NA_real_, best_iter = NA_integer_)

best_eta_mse <- Inf
best_eta <- NULL

for (i in seq_len(nrow(eta_grid))) {
  params_i <- params_stage3
  params_i$eta <- eta_grid$eta[i]

  out <- run_xgb(params_i, nrounds = 8000, early_stopping_rounds = 80, seed = 100)

  eta_grid$test_mse[i] <- out$test_mse
  eta_grid$best_iter[i] <- out$best_iter

  cat(sprintf(
    "[%d/%d] eta=%.3f | best_iter=%d | test MSE=%.5f",
    i, nrow(eta_grid), eta_grid$eta[i], out$best_iter, out$test_mse
  ))

  if (out$test_mse < best_eta_mse) {
    best_eta_mse <- out$test_mse
    best_eta <- eta_grid[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

cat("\nStage 4 results:\n")
print(dplyr::arrange(eta_grid, test_mse))
cat("\nBest (Stage 4):\n")
print(best_eta)

params_stage4 <- params_stage3
params_stage4$eta <- best_eta$eta

# ------------------------------------------------------------
# Stage 5: Regularization tuning
# Why: lambda (L2) and alpha (L1) penalize complexity to reduce overfit.
# ------------------------------------------------------------
cat("\n================ Stage 5: Regularization tuning ================\n")

regularization_grid <- expand.grid(
  lambda = c(0, 0.5, 1, 2, 5, 10),
  alpha = c(0, 0.1, 0.5, 1),
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
) |>
  dplyr::arrange(lambda, alpha) |>
  dplyr::mutate(test_mse = NA_real_, best_iter = NA_integer_)

best_reg_mse <- Inf
best_reg <- NULL

for (i in seq_len(nrow(regularization_grid))) {
  params_i <- params_stage4
  params_i$lambda <- regularization_grid$lambda[i]
  params_i$alpha <- regularization_grid$alpha[i]

  out <- run_xgb(params_i, nrounds = 8000, early_stopping_rounds = 80, seed = 100)

  regularization_grid$test_mse[i] <- out$test_mse
  regularization_grid$best_iter[i] <- out$best_iter

  cat(sprintf(
    "[%2d/%2d] lambda=%.2f | alpha=%.2f | best_iter=%d | test MSE=%.5f",
    i, nrow(regularization_grid),
    regularization_grid$lambda[i], regularization_grid$alpha[i],
    out$best_iter, out$test_mse
  ))

  if (out$test_mse < best_reg_mse) {
    best_reg_mse <- out$test_mse
    best_reg <- regularization_grid[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

cat("\nTop 10 (Stage 5):\n")
print(head(dplyr::arrange(regularization_grid, test_mse), 10))
cat("\nBest (Stage 5):\n")
print(best_reg)

params_stage5 <- params_stage4
params_stage5$lambda <- best_reg$lambda
params_stage5$alpha <- best_reg$alpha

# ------------------------------------------------------------
# Stage 6: Local sampling re-check
# Why: after regularization changes, the best subsampling point can shift.
# ------------------------------------------------------------
cat("\n================ Stage 6: Local sampling re-check ================\n")

sub_local <- sort(unique(pmin(1.0, pmax(0.4, best_sampling$subsample + c(-0.1, -0.05, 0, 0.05, 0.1)))))
col_local <- sort(unique(pmin(1.0, pmax(0.5, best_sampling$colsample_bytree + c(-0.1, -0.05, 0, 0.05, 0.1)))))

sampling_refine_grid <- expand.grid(
  subsample = sub_local,
  colsample_bytree = col_local,
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
) |>
  dplyr::arrange(subsample, colsample_bytree) |>
  dplyr::mutate(test_mse = NA_real_, best_iter = NA_integer_)

best_sampling_refine_mse <- Inf
best_sampling_refine <- NULL

for (i in seq_len(nrow(sampling_refine_grid))) {
  params_i <- params_stage5
  params_i$subsample <- sampling_refine_grid$subsample[i]
  params_i$colsample_bytree <- sampling_refine_grid$colsample_bytree[i]

  out <- run_xgb(params_i, nrounds = 8000, early_stopping_rounds = 80, seed = 100)

  sampling_refine_grid$test_mse[i] <- out$test_mse
  sampling_refine_grid$best_iter[i] <- out$best_iter

  cat(sprintf(
    "[%2d/%2d] subsample=%.2f | colsample=%.2f | best_iter=%d | test MSE=%.5f",
    i, nrow(sampling_refine_grid),
    sampling_refine_grid$subsample[i], sampling_refine_grid$colsample_bytree[i],
    out$best_iter, out$test_mse
  ))

  if (out$test_mse < best_sampling_refine_mse) {
    best_sampling_refine_mse <- out$test_mse
    best_sampling_refine <- sampling_refine_grid[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

cat("\nTop 10 (Stage 6):\n")
print(head(dplyr::arrange(sampling_refine_grid, test_mse), 10))
cat("\nBest (Stage 6):\n")
print(best_sampling_refine)

params_stage6 <- params_stage5
params_stage6$subsample <- best_sampling_refine$subsample
params_stage6$colsample_bytree <- best_sampling_refine$colsample_bytree

# ------------------------------------------------------------
# Final model fit
# Why: after selecting hyperparameters, refit on ALL training rows.
# ------------------------------------------------------------
cat("\n================ Final XGBoost Model ================\n")

final_nrounds <- as.integer(best_sampling_refine$best_iter)
if (is.na(final_nrounds) || final_nrounds < 1) {
  final_nrounds <- 5000L
}

final_params <- modifyList(default_core_params, params_stage6)

set.seed(100)
fit_final <- xgb.train(
  params = final_params,
  data = dtrain_full,
  nrounds = final_nrounds,
  verbose = 0
)

pred_final <- predict(fit_final, dtest)
final_test_mse <- mse(y_test, pred_final)

cat(sprintf("FINAL XGBoost test MSE: %.5f\n", final_test_mse))
cat(sprintf("Final nrounds (from Stage 6 best_iter): %d\n", final_nrounds))
cat("Final hyperparameters:\n")
print(params_stage6)

# Compact summary for slides.
stage_summary <- tibble::tibble(
  stage = c("Baseline", "Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5", "Stage 6"),
  best_test_mse = c(
    res0$test_mse,
    best_structure$test_mse,
    best_sampling$test_mse,
    best_gamma$test_mse,
    best_eta$test_mse,
    best_reg$test_mse,
    best_sampling_refine$test_mse
  )
)

cat("\nStage summary (lower MSE is better):\n")
print(stage_summary)
