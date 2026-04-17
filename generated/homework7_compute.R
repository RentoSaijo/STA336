library(ISLR2)
library(tree)
library(randomForest)

cat("=== PART 1: AUTO CLASSIFICATION TREE ===\n")

auto_df <- ISLR2::Auto
auto_median_mpg <- median(auto_df$mpg)
auto_df$ECO <- factor(
  ifelse(auto_df$mpg > auto_median_mpg, "Economy", "Consuming"),
  levels = c("Consuming", "Economy")
)

cat("Median mpg:", auto_median_mpg, "\n")
cat("ECO counts:\n")
print(table(auto_df$ECO))
cat("\n")

auto_tree_full <- tree::tree(ECO ~ . - name, data = auto_df)
cat("Auto full tree:\n")
print(auto_tree_full)
cat("\nAuto full tree summary:\n")
print(summary(auto_tree_full))
cat("\n")

auto_tree_reduced <- tree::tree(ECO ~ horsepower + weight + acceleration, data = auto_df)
cat("Auto reduced tree:\n")
print(auto_tree_reduced)
cat("\nAuto reduced tree summary:\n")
print(summary(auto_tree_reduced))
cat("\n")

set.seed(1)
n_auto <- length(auto_df$ECO)
Z_auto <- sample(n_auto, n_auto / 2)
auto_train <- auto_df[Z_auto, ]
auto_test <- auto_df[-Z_auto, ]

auto_tree_train <- tree::tree(ECO ~ horsepower + weight + acceleration, data = auto_train)
auto_test_pred <- predict(auto_tree_train, newdata = auto_test, type = "class")

cat("Auto training tree:\n")
print(auto_tree_train)
cat("\nAuto test confusion matrix:\n")
print(table(pred = auto_test_pred, truth = auto_test$ECO))
cat("Auto test error:", mean(auto_test_pred != auto_test$ECO), "\n\n")

auto_cv <- tree::cv.tree(auto_tree_train, FUN = prune.misclass)
cat("Auto CV tree output:\n")
print(auto_cv)
cat("\n")

auto_best_size <- auto_cv$size[which.min(auto_cv$dev)]
cat("Auto best size:", auto_best_size, "\n")

auto_pruned <- tree::prune.misclass(auto_tree_train, best = auto_best_size)
auto_pruned_pred <- predict(auto_pruned, newdata = auto_test, type = "class")

cat("Auto pruned tree:\n")
print(auto_pruned)
cat("\nAuto pruned test confusion matrix:\n")
print(table(pred = auto_pruned_pred, truth = auto_test$ECO))
cat("Auto pruned test error:", mean(auto_pruned_pred != auto_test$ECO), "\n\n")

cat("=== PART 1: CARSEATS REGRESSION TREE ===\n")

carseats_df <- ISLR2::Carseats

set.seed(1)
n_carseats <- nrow(carseats_df)
Z_carseats <- sample(n_carseats, n_carseats / 2)
carseats_train <- carseats_df[Z_carseats, ]
carseats_test <- carseats_df[-Z_carseats, ]

carseats_tree <- tree::tree(Sales ~ ., data = carseats_train)
carseats_pred <- predict(carseats_tree, newdata = carseats_test)
carseats_mse <- mean((carseats_pred - carseats_test$Sales)^2)

cat("Carseats tree:\n")
print(carseats_tree)
cat("\nCarseats tree summary:\n")
print(summary(carseats_tree))
cat("Carseats test MSE:", carseats_mse, "\n\n")

carseats_cv <- tree::cv.tree(carseats_tree)
cat("Carseats CV tree output:\n")
print(carseats_cv)
cat("\n")

carseats_best_size <- carseats_cv$size[which.min(carseats_cv$dev)]
cat("Carseats best size:", carseats_best_size, "\n")

carseats_pruned <- tree::prune.tree(carseats_tree, best = carseats_best_size)
carseats_pruned_pred <- predict(carseats_pruned, newdata = carseats_test)
carseats_pruned_mse <- mean((carseats_pruned_pred - carseats_test$Sales)^2)

cat("Carseats pruned tree:\n")
print(carseats_pruned)
cat("\nCarseats pruned test MSE:", carseats_pruned_mse, "\n\n")

set.seed(1)
carseats_bag <- randomForest::randomForest(
  Sales ~ .,
  data = carseats_train,
  mtry = ncol(carseats_train) - 1,
  importance = TRUE
)
carseats_bag_pred <- predict(carseats_bag, newdata = carseats_test)
carseats_bag_mse <- mean((carseats_bag_pred - carseats_test$Sales)^2)

cat("Carseats bagging MSE:", carseats_bag_mse, "\n")
cat("Carseats variable importance:\n")
print(randomForest::importance(carseats_bag))
cat("\n")

cat("=== PART 3: OJ CLASSIFICATION TREE ===\n")

oj_df <- ISLR2::OJ

set.seed(1)
oj_train_idx <- sample(seq_len(nrow(oj_df)), 800)
oj_train <- oj_df[oj_train_idx, ]
oj_test <- oj_df[-oj_train_idx, ]

oj_tree <- tree::tree(Purchase ~ ., data = oj_train)
oj_train_pred <- predict(oj_tree, newdata = oj_train, type = "class")
oj_test_pred <- predict(oj_tree, newdata = oj_test, type = "class")

cat("OJ tree:\n")
print(oj_tree)
cat("\nOJ tree summary:\n")
print(summary(oj_tree))
cat("\nOJ train confusion matrix:\n")
print(table(pred = oj_train_pred, truth = oj_train$Purchase))
cat("OJ training error:", mean(oj_train_pred != oj_train$Purchase), "\n\n")

cat("OJ test confusion matrix:\n")
print(table(pred = oj_test_pred, truth = oj_test$Purchase))
cat("OJ test error:", mean(oj_test_pred != oj_test$Purchase), "\n\n")

oj_cv <- tree::cv.tree(oj_tree, FUN = prune.misclass)
cat("OJ CV tree output:\n")
print(oj_cv)
cat("\n")

oj_best_size <- oj_cv$size[which.min(oj_cv$dev)]
cat("OJ best size:", oj_best_size, "\n")

oj_pruned <- tree::prune.misclass(oj_tree, best = oj_best_size)
oj_pruned_train_pred <- predict(oj_pruned, newdata = oj_train, type = "class")
oj_pruned_test_pred <- predict(oj_pruned, newdata = oj_test, type = "class")

cat("OJ pruned tree:\n")
print(oj_pruned)
cat("\nOJ pruned train confusion matrix:\n")
print(table(pred = oj_pruned_train_pred, truth = oj_train$Purchase))
cat("OJ pruned training error:", mean(oj_pruned_train_pred != oj_train$Purchase), "\n\n")

cat("OJ pruned test confusion matrix:\n")
print(table(pred = oj_pruned_test_pred, truth = oj_test$Purchase))
cat("OJ pruned test error:", mean(oj_pruned_test_pred != oj_test$Purchase), "\n")
