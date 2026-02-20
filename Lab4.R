# CV and LOOCV
# Regression

# Load libraries.
suppressMessages(library(tidyverse))
suppressMessages(library(boot))
suppressMessages(library(ISLR2))

# Load data.
data(Auto)

# Split data.
set.seed(2/20/2026)
n = length(Auto$mpg)
Z = sample(n, n/2)
Auto_train <- Auto[Z, ]
Auto_test  <- Auto[-Z, ]

# Model 1
names(Auto)
model1 <- lm(mpg ~ weight + horsepower + acceleration, data = Auto_train)

# Predict
Auto_test$pred_Y <- predict(model1, Auto_test)
plot(Auto_test$mpg, Auto_test$pred_Y)
abline(0, 1, col = 'red')

# MSE
mean((Auto_test$mpg - Auto_test$pred_Y)^2)

# LOOCV
glm1 <- glm(mpg ~ weight + horsepower + acceleration, data = Auto_train)
cv.error <- cv.glm(Auto_train, glm1)
cv.error$delta[1]
plot(Auto_train$horsepower, Auto_train$mpg)

ps        <- 1:10
cv.errors <- c()
for (p in ps) {
  glm.model <- glm(mpg ~ weight + poly(horsepower, p) + acceleration, data = Auto_train)
  cv.errors <- c(cv.errors, cv.glm(Auto_train, glm.model)$delta[1])
}
ps_cv.errors <- data.frame(p = ps, cv.error = cv.errors)
plot(ps_cv.errors$p, ps_cv.errors$cv.error)
lines(cv.errors)

glm2 <- glm(mpg ~ weight + poly(horsepower, 2) + acceleration, data = Auto_train)
Auto_test$pred_Y <- predict(glm2, Auto_test)
plot(Auto_test$mpg, Auto_test$pred_Y)
abline(0, 1, col = 'red')
# MSE
mean((Auto_test$mpg - Auto_test$pred_Y)^2)

# K-fold CV
ks        <- 1:10
cv.errors <- c()
for (k in ks) {
  glm.model <- glm(mpg ~ weight + poly(horsepower, k) + acceleration, data = Auto_train)
  cv.errors <- c(cv.errors, cv.glm(Auto_train, glm.model, K = 10)$delta[1])
}
ks_cv.errors <- data.frame(k = ks, cv.error = cv.errors)
plot(cv.errors)
lines(cv.errors)

# Log-loss function

# Load libraries.
suppressMessages(library(tidyverse))
suppressMessages(library(boot))

# Read data,
Dep <- readr::read_csv('data/Depression.csv', show_col_types = FALSE) %>% 
  dplyr::filter(!is.na(Diagnosis))

# Split data.
set.seed(2/20/2026)
n = length(Dep$Diagnosis)
Z = sample(n, n/2)
train <- Dep[Z, ]
test  <- Dep[-Z, ]

# Define loss.
loss <- function(Y, p) {
  mean((Y == 1 & p < 0.5) | (Y == 0 & p > 0.5))
}

# GLM
model1 <- glm(Diagnosis ~ Gender + Guardian_status + Cohesion_score, data = Dep, family = 'binomial')
summary(model1)
cv.error.model1 <- boot::cv.glm(Dep, model1, loss) # 16.2% of the time, our prediction is incorrect.

model2 <- glm(Diagnosis ~ Gender + Cohesion_score, data = Dep, family = 'binomial')
summary(model2)
cv.error.model2 <- boot::cv.glm(Dep, model2, loss) # 15.7% of the time, our prediction is incorrect.

# Partial F-test
anova(model2, model1, test = 'Chisq')
