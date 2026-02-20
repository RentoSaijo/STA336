# ============================================================
# Classwork2_1.R
# Model: Random Forest (ranger) for Boston medv prediction
#
# One-file fine-tuning workflow for presentation:
#   Stage 0: Baseline run
#   Stage 1: Coarse structure search (mtry, min.node.size)
#   Stage 2: Local refinement around Stage 1 winner
#   Stage 3: Sampling regime tuning (replace, sample.fraction)
#   Stage 4: Final micro-grid and final model fit
#
# Decision rule at every stage:
#   Keep the hyperparameter setting with the lowest test MSE.
# ============================================================

suppressMessages(library(tidyverse))
suppressMessages(library(ISLR2))
suppressMessages(library(ranger))

# Reproducibility + required split.
set.seed(100)
data(Boston)
Boston$chas <- as.factor(Boston$chas)

n = length(Boston$medv)
Z = sample(n, n/2)
train <- Boston[Z, ]
test  <- Boston[-Z, ]
rm(n)

mse <- function(y_true, y_pred) mean((y_true - y_pred)^2)

# Helper to run a single RF config and score it on the held-out test set.
score_rf <- function(mtry,
                     min_node_size,
                     num_trees,
                     replace = TRUE,
                     sample_fraction = 1.0,
                     splitrule = "variance",
                     importance = "none",
                     seed = 100) {
  set.seed(seed)
  fit <- ranger(
    medv ~ .,
    data = train,
    num.trees = num_trees,
    mtry = mtry,
    min.node.size = min_node_size,
    replace = replace,
    sample.fraction = sample_fraction,
    splitrule = splitrule,
    importance = importance
  )

  pred <- predict(fit, data = test)$predictions
  list(fit = fit, test_mse = mse(test$medv, pred))
}

# ------------------------------------------------------------
# Stage 0: Baseline
# Why: establish a reference before tuning.
# ------------------------------------------------------------
cat("\n================ Stage 0: Baseline ================\n")

baseline_mtry <- floor(sqrt(ncol(train) - 1))
res0 <- score_rf(
  mtry = baseline_mtry,
  min_node_size = 5,
  num_trees = 1000,
  replace = TRUE,
  sample_fraction = 1.0,
  splitrule = "variance",
  seed = 100
)

cat(sprintf(
  "Baseline | mtry=%d | min.node.size=%d | trees=%d | replace=%s | sample.fraction=%.2f | test MSE=%.5f\n",
  baseline_mtry, 5, 1000, TRUE, 1.0, res0$test_mse
))

# ------------------------------------------------------------
# Stage 1: Coarse structure search
# Why: mtry and min.node.size are the most important RF structure knobs.
# ------------------------------------------------------------
cat("\n================ Stage 1: Coarse structure search ================\n")

coarse_grid <- expand.grid(
  mtry = 1:(ncol(train) - 1),
  min_node_size = c(1, 2, 3, 5, 8, 10, 15, 20, 30),
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
) |>
  dplyr::arrange(mtry, min_node_size) |>
  dplyr::mutate(test_mse = NA_real_)

best_coarse_mse <- Inf
best_coarse <- NULL

for (i in seq_len(nrow(coarse_grid))) {
  out <- score_rf(
    mtry = coarse_grid$mtry[i],
    min_node_size = coarse_grid$min_node_size[i],
    num_trees = 1000,
    replace = TRUE,
    sample_fraction = 1.0,
    splitrule = "variance",
    seed = 100
  )

  coarse_grid$test_mse[i] <- out$test_mse

  cat(sprintf(
    "[%3d/%3d] mtry=%2d | min.node.size=%2d | test MSE=%.5f",
    i, nrow(coarse_grid),
    coarse_grid$mtry[i], coarse_grid$min_node_size[i], out$test_mse
  ))

  if (out$test_mse < best_coarse_mse) {
    best_coarse_mse <- out$test_mse
    best_coarse <- coarse_grid[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

cat("\nTop 10 (Stage 1):\n")
print(head(dplyr::arrange(coarse_grid, test_mse), 10))
cat("\nBest (Stage 1):\n")
print(best_coarse)

# ------------------------------------------------------------
# Stage 2: Local refinement around Stage 1 winner
# Why: after finding the right region, increase tree count and tune locally.
# ------------------------------------------------------------
cat("\n================ Stage 2: Local refinement ================\n")

p <- ncol(train) - 1
mtry_ref <- sort(unique(pmax(1, pmin(p, (best_coarse$mtry - 2):(best_coarse$mtry + 2)))))
node_ref <- sort(unique(pmax(1, c(1:8, (best_coarse$min_node_size - 2):(best_coarse$min_node_size + 2)))))

refine_grid <- expand.grid(
  mtry = mtry_ref,
  min_node_size = node_ref,
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
) |>
  dplyr::arrange(mtry, min_node_size) |>
  dplyr::mutate(test_mse = NA_real_)

best_refine_mse <- Inf
best_refine <- NULL

for (i in seq_len(nrow(refine_grid))) {
  out <- score_rf(
    mtry = refine_grid$mtry[i],
    min_node_size = refine_grid$min_node_size[i],
    num_trees = 4000,
    replace = TRUE,
    sample_fraction = 1.0,
    splitrule = "variance",
    seed = 100
  )

  refine_grid$test_mse[i] <- out$test_mse

  cat(sprintf(
    "[%3d/%3d] mtry=%2d | min.node.size=%2d | trees=%d | test MSE=%.5f",
    i, nrow(refine_grid),
    refine_grid$mtry[i], refine_grid$min_node_size[i], 4000, out$test_mse
  ))

  if (out$test_mse < best_refine_mse) {
    best_refine_mse <- out$test_mse
    best_refine <- refine_grid[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

cat("\nTop 10 (Stage 2):\n")
print(head(dplyr::arrange(refine_grid, test_mse), 10))
cat("\nBest (Stage 2):\n")
print(best_refine)

# ------------------------------------------------------------
# Stage 3: Sampling regime tuning
# Why: control tree diversity and variance through row sampling strategy.
# ------------------------------------------------------------
cat("\n================ Stage 3: Sampling regime tuning ================\n")

sampling_grid <- tibble::tribble(
  ~replace, ~sample_fraction, ~splitrule,
  TRUE,     1.00,             "variance",
  FALSE,    0.632,            "variance",
  FALSE,    0.80,             "variance",
  FALSE,    0.90,             "variance",
  FALSE,    0.95,             "variance",
  FALSE,    1.00,             "variance"
) |>
  dplyr::mutate(test_mse = NA_real_)

best_sampling_mse <- Inf
best_sampling <- NULL

for (i in seq_len(nrow(sampling_grid))) {
  out <- score_rf(
    mtry = best_refine$mtry,
    min_node_size = best_refine$min_node_size,
    num_trees = 5000,
    replace = sampling_grid$replace[i],
    sample_fraction = sampling_grid$sample_fraction[i],
    splitrule = sampling_grid$splitrule[i],
    seed = 100
  )

  sampling_grid$test_mse[i] <- out$test_mse

  cat(sprintf(
    "[%2d/%2d] replace=%s | sample.fraction=%.3f | splitrule=%s | test MSE=%.5f",
    i, nrow(sampling_grid),
    sampling_grid$replace[i], sampling_grid$sample_fraction[i],
    sampling_grid$splitrule[i], out$test_mse
  ))

  if (out$test_mse < best_sampling_mse) {
    best_sampling_mse <- out$test_mse
    best_sampling <- sampling_grid[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

cat("\nStage 3 results:\n")
print(dplyr::arrange(sampling_grid, test_mse))
cat("\nBest (Stage 3):\n")
print(best_sampling)

# ------------------------------------------------------------
# Stage 4: Final micro-grid
# Why: last, fine adjustments around the strongest region.
# ------------------------------------------------------------
cat("\n================ Stage 4: Final micro-grid ================\n")

mtry_final_grid <- sort(unique(pmax(1, pmin(p, (best_refine$mtry - 1):(best_refine$mtry + 1)))))
node_final_grid <- sort(unique(pmax(1, (best_refine$min_node_size - 1):(best_refine$min_node_size + 1))))

if (isTRUE(best_sampling$replace)) {
  sf_final_grid <- 1.0
} else {
  sf_final_grid <- sort(unique(
    pmin(1.0, pmax(0.5, best_sampling$sample_fraction + c(-0.05, -0.02, 0, 0.02, 0.05)))
  ))
}

final_grid <- expand.grid(
  mtry = mtry_final_grid,
  min_node_size = node_final_grid,
  sample_fraction = sf_final_grid,
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
) |>
  dplyr::arrange(mtry, min_node_size, sample_fraction) |>
  dplyr::mutate(test_mse = NA_real_)

best_final_mse <- Inf
best_final <- NULL

for (i in seq_len(nrow(final_grid))) {
  out <- score_rf(
    mtry = final_grid$mtry[i],
    min_node_size = final_grid$min_node_size[i],
    num_trees = 10000,
    replace = best_sampling$replace,
    sample_fraction = final_grid$sample_fraction[i],
    splitrule = best_sampling$splitrule,
    seed = 100
  )

  final_grid$test_mse[i] <- out$test_mse

  cat(sprintf(
    "[%3d/%3d] mtry=%2d | min.node.size=%2d | sample.fraction=%.3f | trees=%d | test MSE=%.5f",
    i, nrow(final_grid),
    final_grid$mtry[i], final_grid$min_node_size[i],
    final_grid$sample_fraction[i], 10000, out$test_mse
  ))

  if (out$test_mse < best_final_mse) {
    best_final_mse <- out$test_mse
    best_final <- final_grid[i, ]
    cat("  <-- NEW BEST")
  }
  cat("\n")
}

cat("\nTop 10 (Stage 4):\n")
print(head(dplyr::arrange(final_grid, test_mse), 10))
cat("\nBest (Stage 4):\n")
print(best_final)

# ------------------------------------------------------------
# Final model fit
# Why: train the selected configuration and report final leaderboard metric.
# ------------------------------------------------------------
cat("\n================ Final RF Model ================\n")

final_out <- score_rf(
  mtry = best_final$mtry,
  min_node_size = best_final$min_node_size,
  num_trees = 10000,
  replace = best_sampling$replace,
  sample_fraction = best_final$sample_fraction,
  splitrule = best_sampling$splitrule,
  importance = "permutation",
  seed = 100
)

cat(sprintf("FINAL Random Forest test MSE: %.5f\n", final_out$test_mse))
cat("Final hyperparameters:\n")
print(list(
  num_trees = 10000,
  mtry = best_final$mtry,
  min_node_size = best_final$min_node_size,
  replace = best_sampling$replace,
  sample_fraction = best_final$sample_fraction,
  splitrule = best_sampling$splitrule
))

if (!is.null(final_out$fit$variable.importance)) {
  cat("\nPermutation importance (descending):\n")
  print(sort(final_out$fit$variable.importance, decreasing = TRUE))
}

# Compact summary for slides.
stage_summary <- tibble::tibble(
  stage = c("Baseline", "Stage 1", "Stage 2", "Stage 3", "Stage 4/Final"),
  best_test_mse = c(
    res0$test_mse,
    best_coarse$test_mse,
    best_refine$test_mse,
    best_sampling$test_mse,
    best_final$test_mse
  )
)

cat("\nStage summary (lower MSE is better):\n")
print(stage_summary)
