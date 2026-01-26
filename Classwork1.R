# Load packages.
suppressMessages(library(tidyverse))
suppressMessages(library(GGally))

# Read from CSV.
INCOME <- read_csv('data/Income2.csv', show_col_types = FALSE) %>% 
  select(education = Education, seniority = Seniority, income = Income)

# Summary.
summary(INCOME)

# Make scatterplot matrix.
ggpairs(INCOME)

# Fit linear regressions.
m1 <- lm(income ~ education, data = INCOME)
summary(m1)

m2 <- lm(income ~ education + seniority, data = INCOME)
summary(m2)

# Create scatterplot.
ggplot(data = INCOME, aes(x = education, y = income)) +
  geom_point(data = INCOME, aes(col = seniority)) +
  geom_smooth()
