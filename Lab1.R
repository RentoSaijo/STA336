# Lab 1: Play Around

x <- c(2, 3, 5)
y <- c('A', 'A', 'B', 'D')
z <- c(1, 5, 8, 10, 12)

x + z
x * z

length(x)
length(y)

# rm(list = ls())

m1 <- matrix(data = z, nrow = 2, byrow = TRUE)

sqrt(z)
z ^ 2

rand_norm <- rnorm(100)
hist(rnorm(100, mean = 5, sd = 2))

set.seed(Sys.Date())
x <- rnorm(100)
y <- rgamma(100, shape = 1)
cor(x, y)

var(y)
sd(y)
sqrt(var(y))

library(ISLR2)
data(Auto)
head(Auto)
summary(Auto)
dim(Auto)
Auto[1:100, ]
Auto_smaller <- Auto[, c(1, 3, 4, 5, 8)]
head(Auto_smaller)

library(tidyverse)
Auto_smaller <- Auto_smaller %>% 
  mutate(origin = factor(origin, labels = c('U.S.', 'Europe', 'Japan')))

g1 <- ggplot(Auto_smaller, aes(x = displacement, y = mpg, col = origin)) + 
  geom_point() +
  scale_color_manual(values = c('U.S.' = 'red', 'Europe' = 'blue', 'Japan' = 'yellow')) +
  theme_classic()

# par(mfrow = c(2, 2))
