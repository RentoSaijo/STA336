# Load libraries.
suppressMessages(library(tidyverse))
suppressMessages(library(GGally))
suppressMessages(library(ISLR2))
suppressMessages(library(caret))

# Load data.
data(Auto)

# Fit SLR.
attach(Auto)
reg <- lm(mpg ~ weight)
reg
plot(weight, mpg)
abline(reg, col = 'red', lwd = 3)

# Fit spline with 2 df.
spline2 <- smooth.spline(weight, mpg, df = 3)
lines(spline2, col = 'blue', lwd = 3)

# Fit spline with 3 df.
spline3 <- smooth.spline(weight, mpg, df = 3)
lines(spline3, col = 'green', lwd = 3)


# Fit spline with 20 df.
spline20 <- smooth.spline(weight, mpg, df = 20)
lines(spline20, col = 'yellow', lwd = 3)

# Fit spline with 200 df.
spline200 <- smooth.spline(weight, mpg, df = 200)
lines(spline200, col = 'gray', lwd = 3)
detach(Auto)

# Set seed.
set.seed(1/30/2026)

# Split train and test data.
split_data <- caret::createDataPartition(Auto$mpg, p = 0.8, list = FALSE)
train <- Auto[split_data, ]
test  <- Auto[-split_data, ]
rm(split_data)
dplyr::glimpse(train)
dplyr::glimpse(test)

# Smooth spline on the training.
ss5 <- smooth.spline(train$weight, train$mpg, df = 5)

# Prediction on the test.
y_hat <- predict(ss5, x = test$weight)
predicted <- y_hat$y
actual    <- test$mpg
data <- data.frame(actual = actual, predicted = predicted)

# Calculate MSE using postResample function from caret
res <- caret::postResample(pred = data$predicted, obs = data$actual)
mse <- as.numeric(res["RMSE"])^2
mse

# Cross validate.
dfs  <- c()
mses <- c()
for (i in seq(from = 1.1, to = 50.0, by = 0.1)) {
  ss    <- smooth.spline(train$weight, train$mpg, df = i)
  y_hat <- predict(ss, x = test$weight)
  predicted <- y_hat$y
  actual    <- test$mpg
  data      <- data.frame(actual = actual, predicted = predicted)
  res  <- caret::postResample(pred = data$predicted, obs = data$actual)
  mse  <- as.numeric(res["RMSE"])^2
  dfs  <- append(dfs, i)
  mses <- append(mses, mse)
}
mses_dfs <- data.frame(dfs, mses)
View(mses_dfs)

# Load data.
data(Auto)

# View data.
tibble::glimpse(Auto)

# Fix data.
Auto <- Auto %>% 
  dplyr::mutate(origin = factor(
    origin, 
    levels = 1:3, 
    labels = c("US", "EU", "JP")
  ))

# EDA
GGally::ggpairs(
  Auto,
  columns   = setdiff(colnames(Auto), c('origin', 'name')),
  aes(color = origin, alpha = 0.5)
)

# Models
m1 <- lm(mpg ~ origin, data = Auto) # SLR
summary(m1)
m2 <- lm(mpg ~ weight + origin, data = Auto) # MLR
summary(m2)
m3 <- lm(mpg ~ weight*origin, data = Auto) # interaction
summary(m3)
m4 <- lm(mpg ~ weight + I(weight^2), data = Auto) # polynomial
summary(m4)
m44 <- lm(mpg ~ weight, data = Auto) # linear
summary(m4)

# Vizualize models.
ggplot(Auto, aes(x = weight, y = mpg)) +
  geom_point() +
  geom_line(aes(y = fitted.values(m44), col = 'red')) +
  geom_line(aes(y = fitted.values(m4), col = 'blue'))


# Compare models.
# 1. MSE on test data; 2. adjusted R^2; 3. significance of added predictor(s); 4. partial F test (i.e., anova).
anova(m4, m44)

# Conditions for the model.
par(mfrow = c(2, 2))
plot(m4)
