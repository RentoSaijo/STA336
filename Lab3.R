# Load libraries.
suppressMessages(library(tidyverse))
suppressMessages(library(GGally))
suppressMessages(library(ISLR2))
suppressMessages(library(caret))

# Set seed.
set.seed(20060527)

# Read from CSV.
depression_raw <- readr::read_csv('data/Depression.csv', show_col_types = FALSE)

# Check missing values.
colSums(is.na(depression_raw))

# Check summary.
summary(depression_raw)

# Clean data.
dpsn <- depression_raw %>% 
  dplyr::select(
    id         = ID,
    gender     = Gender,
    guardian   = Guardian_status,
    cohesion   = Cohesion_score,
    depression = Depression_score,
    diagnosis  = Diagnosis,
  ) %>% 
  dplyr::filter(!is.na(diagnosis)) %>% 
  dplyr::mutate(
    gender    = factor(gender),
    guardian  = factor(guardian),
    diagnosis = factor(diagnosis)
  )

# Check summary again.
summary(dpsn)

# Split data into train and test.
split_data <- caret::createDataPartition(dpsn$diagnosis, p = 0.5, list = FALSE)
train <- dpsn[split_data, ]
test  <- dpsn[-split_data, ]
rm(split_data)

# Build logistic regression.
m1 <- glm(diagnosis ~ gender + guardian + cohesion, data = train, family = binomial)
summary(m1)

# Predict probabilities.
train$pDiagnosis <- fitted.values(m1)
train$xDiagnosis <- if_else(train$pDiagnosis > 0.5, 1, 0)
train$correct    <- if_else(train$xDiagnosis == train$diagnosis, TRUE, FALSE)
(308 + 0) / (308 + 0 + 58 + 1)

# Predict probabilities 2.
train$pDiagnosis2 <- fitted.values(m1)
train$xDiagnosis2 <- if_else(train$pDiagnosis2 > 0.15804, 1, 0)
train$correct2    <- if_else(train$xDiagnosis2 == train$diagnosis, TRUE, FALSE)
table(train$diagnosis, train$xDiagnosis2)
(196 + 40) / (196 + 40 + 113 + 18)

# Find best cutoff.
aDiagnosis <- factor(test$diagnosis, levels = c(0, 1))
pDiagnosis <- predict(m1, newdata = test, type = "response")
is <- seq(0, summary(pDiagnosis)['Max.'], by = 0.0001)
accuracies <- numeric(length(is))
for (k in seq_along(is)) {
  i <- is[k]
  xDiagnosis <- factor(if_else(pDiagnosis > i, 1, 0), levels = c(0, 1))
  cross_tab  <- table(aDiagnosis, xDiagnosis)  # always 2x2 now
  accuracies[k] <- sum(diag(cross_tab)) / sum(cross_tab)
}
is_accuracies <- data.frame(is, accuracies)

# Find ROC.
install.packages('Epi')
library(Epi)
Epi::ROC(form = diagnosis ~ gender + guardian + cohesion, data = train, plot = 'ROC', MX = TRUE)
