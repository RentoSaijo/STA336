# Load libraries.
suppressMessages(library(tidyverse))
suppressMessages(library(leaps))
suppressMessages(library(ISLR2))
suppressMessages(library(MASS))

# Load data.
data(Auto)
data(Boston)
data(Caravan)

# Best subset.
m1 <- leaps::regsubsets(mpg ~ . - name - origin, data = Auto)
test <- leaps::regsubsets(Purchase ~ ., data = Caravan[30:86], really.big = TRUE)
plot(summary(m1)$rsq)
lines(summary(m1)$rsq)
plot(summary(m1)$bic)
lines(summary(m1)$bic)

# Stepwise.
Boston$chas <- as.factor(Boston$chas)
full <- lm(medv ~ ., data = Boston)
null <- lm(medv ~ 1, data = Boston)
m2 <- stats::step(null, scope = list(lower = null, upper = full), direction = 'forward')
summary(m2)
m3 <- stats::step(full, direction = 'backward')
summary(m3)
m4 <- stats::step(full, direction = 'both')
summary(m4)
