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

# --- Stabilize top configs with more trees --------------------------------

mse <- function(y_true, y_pred) mean((y_true - y_pred)^2)

num_trees3 <- 3000

candidates <- tibble::tribble(
  ~mtry, ~min.node.size,
     5,            1,
     6,            1,
     5,            2,
     4,            3,
     6,            5,
     7,            3
)

stab <- candidates %>%
  mutate(test_mse = NA_real_)

set.seed(100)
for (i in seq_len(nrow(stab))) {
  mtry_i <- stab$mtry[i]
  node_i <- stab$min.node.size[i]

  fit <- ranger(
    medv ~ ., data = train,
    num.trees = num_trees3,
    mtry = mtry_i,
    min.node.size = node_i,
    importance = "none"
  )
  pred <- predict(fit, data = test)$predictions
  stab$test_mse[i] <- mse(test$medv, pred)

  cat(sprintf(
    "[%d/%d] trees=%d | mtry=%d | min.node.size=%d | test MSE=%.5f\n",
    i, nrow(stab), num_trees3, mtry_i, node_i, stab$test_mse[i]
  ))
}

cat("\nStabilized ranking:\n")
print(stab %>% arrange(test_mse))
