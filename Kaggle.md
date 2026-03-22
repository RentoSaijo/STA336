# Kaggle Spring Notes: Spaceship Titanic

Research date: March 21, 2026

## Slide 1: Competition Goal and Context

Spaceship Titanic is a binary classification competition. The task is to predict whether a passenger was `Transported` to an alternate dimension after the ship collided with a spacetime anomaly. The prediction target is therefore a Bernoulli random variable, not a continuous response, so this is fundamentally a classification problem rather than a regression problem. The official competition setup uses a hidden test set and requires a submission with one `True` or `False` value for each `PassengerId` in `test.csv`. The class prompt also notes an important piece of context that should appear on the slide: this is an ongoing getting-started Kaggle competition with a rolling leaderboard, so there is no single fixed “winner” whose solution can be treated as the final answer.

That point matters for how we should talk about the competition. In a closed Kaggle competition, it is normal to summarize “top-team solutions.” Here, that language is weaker, because rankings age out and the leaderboard changes over time. A better framing is that we are studying strong public workflows: public notebooks, GitHub repos, and exported writeups from participants who documented what worked and why.

The evaluation metric is classification accuracy. If we denote true positives, true negatives, false positives, and false negatives by `TP`, `TN`, `FP`, and `FN`, then

`Accuracy = (TP + TN) / (TP + TN + FP + FN)`.

Because the training target is almost perfectly balanced in the attached `train.csv`, accuracy is actually a reasonable metric here. The training file has 8,693 rows, of which 4,378 are `True` and 4,315 are `False`. That means the majority-class baseline is only

`4378 / 8693 = 0.5036`,

so a classifier that simply predicts the more common class would achieve only 50.36% accuracy. This immediately tells us two things. First, the competition is not trivial. Second, any workflow reaching roughly 80% accuracy is not just making a small cosmetic improvement; it is making a very large improvement over the naive baseline.

One quantitative way to say that in the presentation is to compare error rates. The majority baseline has error rate `1 - 0.5036 = 0.4964`. Maria Aguilera’s tuned LightGBM workflow reports a cross-validation score of about `0.806627`, so its error rate is `1 - 0.806627 = 0.193373`. The proportional reduction in error is therefore

`1 - 0.193373 / 0.496376 ≈ 0.610`,

or about a 61.0% error-rate reduction relative to the majority-class rule. That is a stronger and more informative statement than saying only that “boosting gets around 80%.”

## Slide 2: Dataset Structure

The attached local files confirm the scale of the competition. `train.csv` contains 8,693 rows and 14 columns; `test.csv` contains 4,277 rows and 13 columns. The difference is exactly the absence of the response variable `Transported` from the test set. The columns fall into four broad groups.

The first group is identifier-like variables: `PassengerId` and `Name`. The second is demographic or travel metadata: `HomePlanet`, `CryoSleep`, `Destination`, `Age`, and `VIP`. The third is location information, concentrated in the single string field `Cabin`. The fourth is spending behavior: `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, and `VRDeck`.

The important point is that several of these variables are only superficially “raw.” `PassengerId` looks like a useless ID until one notices that it has the form `gggg_pp`, where the prefix identifies a travel group and the suffix identifies a position within that group. `Cabin` looks like a single categorical string until one notices it has the form `deck/number/side`. `Name` looks like free text until one realizes the surname may reveal families traveling together. In other words, a large fraction of the predictive structure is hidden inside string parsing, not just in the obviously numeric columns.

The missingness pattern is also more interesting than it first appears. In the attached training file, every feature other than `PassengerId` and `Transported` has about 2% to 2.5% missingness, but the missing values are not concentrated in one disposable variable. They occur in fields that are potentially important: `CryoSleep`, `VIP`, `HomePlanet`, `Cabin`, `Age`, and the spending variables. The exact blank counts in `train.csv` are:

| Column | Missing count | Missing percent |
| --- | ---: | ---: |
| `CryoSleep` | 217 | 2.50% |
| `ShoppingMall` | 208 | 2.39% |
| `VIP` | 203 | 2.34% |
| `HomePlanet` | 201 | 2.31% |
| `Name` | 200 | 2.30% |
| `Cabin` | 199 | 2.29% |
| `VRDeck` | 188 | 2.16% |
| `FoodCourt` | 183 | 2.11% |
| `Spa` | 183 | 2.11% |
| `Destination` | 182 | 2.09% |
| `RoomService` | 181 | 2.08% |
| `Age` | 179 | 2.06% |

The corresponding test file has a very similar pattern, with 80 to 106 missing values in most non-ID columns. That similarity is useful because it suggests the imputation logic learned from training data will likely matter on the hidden test data as well.

Another reason the missingness should be handled carefully rather than by aggressive row deletion is that most rows are almost complete already. In the training set, 6,606 rows, or 75.99%, have no missing values at all. Another 1,867 rows, or 21.48%, are missing only one value. Only 203 rows are missing two values, and only 17 rows are missing three. There are no rows missing four or more values. So the issue is not “garbage rows”; the issue is small numbers of strategically missing fields in otherwise useful rows. That strongly favors imputation over deletion.

The balance of the target is another structural fact that should be presented numerically rather than vaguely. The training file contains 4,378 `True` outcomes and 4,315 `False` outcomes, so the response is almost exactly balanced: 50.36% versus 49.64%. That is why accuracy is acceptable here in a way that it would not be in a class-imbalanced medical or fraud-detection setting.

## Slide 3: Validation Strategy and Evaluation Metric

The official metric is accuracy, but a strong presentation should distinguish between the competition metric and the validation strategy. Kaggle scores a submitted file once against hidden labels. Participants, however, need a way to compare models before they submit. That is where validation enters.

From an ISLR perspective, this is exactly a model-selection problem. If we partition the training data into `K` folds, then a generic cross-validation estimate of classification error can be written as

`Err_CV = (1/K) * sum_{k=1}^K [ (1 / |V_k|) * sum_{i in V_k} I(y_i != f^(-k)(x_i)) ]`,

where `V_k` is the validation fold in split `k` and `f^(-k)` is the model trained on the remaining folds. Cross-validated accuracy is then simply `1 - Err_CV`, or equivalently the mean of the foldwise accuracies.

What makes the public Spaceship Titanic notebooks interesting is that the stronger ones actually use this logic rather than treating the public leaderboard as the only score that matters. PatrickSVM’s notebook reports two concrete validation devices. For a Random Forest with 1,100 trees, `criterion='entropy'`, `max_depth=9`, and `max_features='sqrt'`, the notebook reports `RF.oob_score_ = 0.8060508455`, using the out-of-bag estimate as a proxy for generalization error. In the same notebook, a HistGradientBoostingClassifier with `max_iter=120`, `learning_rate=0.075`, `max_depth=10`, `max_leaf_nodes=16`, and `min_samples_leaf=6` was evaluated by 10-fold cross-validation and achieved the fold scores

`[0.8023, 0.7770, 0.7977, 0.8067, 0.8021, 0.8228, 0.8262, 0.8216, 0.8239, 0.7975]`

with mean accuracy `0.8077817018`. [3][4]

Maria Aguilera’s notebook provides an even richer model-comparison view. Before tuning, she compares several classifiers under a common validation setup and reports mean scores and standard deviations on the training data. In the exported notebook, the reported mean CV scores include approximately:

| Model | Mean score | Std. dev. |
| --- | ---: | ---: |
| XGBoost | 0.8020 | 0.0061 |
| SVC | 0.8005 | 0.0035 |
| Random Forest | 0.7997 | 0.0058 |
| LightGBM | 0.7985 | 0.0069 |
| CatBoost | 0.7955 | 0.0022 |
| AdaBoost | 0.7952 | 0.0074 |
| Logistic Regression | 0.7951 | 0.0055 |

After tuning, her notebook reports LightGBM as the top model among those compared, with best cross-validation score `0.8066274343` and the following best-parameter pattern: `colsample_bytree=0.8`, `max_depth=20`, `min_split_gain=0.4`, `n_estimators=400`, `num_leaves=100`, `reg_alpha=1.3`, `reg_lambda=1.1`, `subsample=0.9`, and `subsample_freq=20`. [5]

These numbers are useful for two reasons. First, they show that the competition is not dominated by one bizarre model. Many competent models cluster between about 0.795 and 0.803 even before tuning. Second, they show that tuning does help, but not infinitely. Maria’s untuned logistic regression is at about `0.7951`, while tuned LightGBM rises to about `0.8066`. That is only about `0.0115`, or 1.15 percentage points, higher. The large jump comes earlier, when one moves from a weak baseline to a well-engineered feature set; the final model class still matters, but less than people often assume.

This is where it is useful to mention a very simple baseline. The Flatiron starter notebook for the competition fits a one-feature logistic regression using only `Spa`, with median imputation after the train/test split. Its reported test accuracy is `0.6266427718`. [7] That is already better than the 50.36% majority baseline, but it is far below the ~0.80 level reached by the stronger public pipelines. Quantitatively, the jump from the one-feature logistic baseline `0.6266` to Maria’s cross-validated engineered logistic baseline `0.7951` is about `0.1685`, or 16.85 percentage points. That is a very concrete way to show that feature engineering and preprocessing are doing much of the real work.

A final methodological issue deserves its own sentence because it shows actual critical thinking rather than summary. This is my inference from the data, not a statement I found written explicitly in the public notebooks: because `PassengerId` encodes travel groups and group members are strongly correlated, a completely random validation split may be mildly optimistic if members of the same group are split across train and validation folds. That does not make stratified CV “wrong,” but it means a group-aware split would be an even stricter test. Mentioning this is valuable because it shows you understand validation as a design choice, not just a buzzword.

## Slide 4: Feature Engineering and Workflow

The single most important lesson from studying Spaceship Titanic public work is that the good notebooks do not jump straight to the model. They begin by asking what the columns actually mean and which relationships are structurally plausible. The recurring workflow looks like this: inspect distributions and missingness, decode structured variables, build a small number of interpretable engineered features, perform logic-based imputation, and only then compare models.

The most consistent engineered features come from `PassengerId`, `Cabin`, `Name`, and aggregated spending.

Start with `PassengerId`. PatrickSVM explicitly parses it into `GroupID`, `GroupPos`, and `GroupSize`. Maria Aguilera also derives passenger-group features and uses them during imputation. Samyak Raj Bayar’s public `code.py` computes `GroupSize` from the `PassengerId` prefix. [4][5][8] This is not arbitrary feature inflation. In the attached training data, the group structure is genuinely informative. If we infer group membership from the `PassengerId` prefix, then 87.18% of groups have a uniform `Transported` label in the observed training set. Solo passengers are transported only 45.24% of the time, whereas grouped passengers are transported 56.69% of the time. Transport probability also changes by group size:

| Group size | Passenger count | Transport rate |
| --- | ---: | ---: |
| 1 | 4,805 | 45.24% |
| 2 | 1,682 | 53.80% |
| 3 | 1,020 | 59.31% |
| 4 | 412 | 64.08% |
| 5 | 265 | 59.25% |
| 6 | 174 | 61.49% |
| 7 | 231 | 54.11% |
| 8 | 104 | 39.42% |

These are not tiny effects. The difference between a solo traveler and a typical group-of-four traveler is almost nineteen percentage points.

The second major feature source is `Cabin`. PatrickSVM splits it into `Deck`, `CabinNum`, and `Side`; Amir Fares splits it into `Deck`, `Num`, and `Side`; Samyak Raj Bayar parses it as `Deck`, `CabinNumber`, and `Side`. Maria Aguilera goes even further, constructing cabin-region features from the cabin number. [4][5][6][8] Again, this is strongly supported by the attached data. Transport rates differ sharply by deck:

| Deck | Count | Transport rate |
| --- | ---: | ---: |
| A | 256 | 49.61% |
| B | 779 | 73.43% |
| C | 747 | 68.01% |
| D | 478 | 43.31% |
| E | 876 | 35.73% |
| F | 2,794 | 43.99% |
| G | 2,559 | 51.62% |
| T | 5 | 20.00% |
| Missing | 199 | 50.25% |

Even the side of the ship carries signal: passengers on side `S` are transported 55.50% of the time, versus 45.13% on side `P`. More importantly, the side effect is not constant across decks. On deck `C`, side `S` has a transport rate of 76.35% while side `P` has 58.06%, a gap of over 18 percentage points. On deck `E`, the gap is much smaller: 37.14% versus 34.27%. This is one of the clearest examples in the local data of an interaction structure that tree models can exploit naturally and linear models can only capture if we explicitly include interaction terms.

The deck information is also useful for imputation. PatrickSVM notes that deck and home planet are related, and Maria Aguilera imputes `HomePlanet` from `CabinDeck`. [4][5] The attached training data supports this strongly. Among rows with both values observed:

- deck `A` is 252/252 Europa
- deck `B` is 766/766 Europa
- deck `C` is 734/734 Europa
- deck `G` is 2498/2498 Earth
- deck `F` contains Earth and Mars, but no Europa

So if a row has a missing `HomePlanet` but a known deck `B`, imputing Europa is not a wild guess; it is empirically very well justified.

The third major feature source is spending behavior. Several public notebooks create total-spend or no-spend features. PatrickSVM constructs total expenses and log-transforms the spending variables for modeling. Maria Aguilera creates a `No_Expenditure` indicator and uses it directly for imputation and feature analysis. [4][5] The attached data shows why. If total spend is defined as the sum of `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, and `VRDeck`, then passengers with zero total spend are transported 78.65% of the time, while passengers with positive total spend are transported only 29.86% of the time. That is a difference of 48.79 percentage points.

This no-spend pattern connects directly to `CryoSleep`, which is perhaps the single strongest raw feature in the dataset. In the attached training file, passengers with `CryoSleep=True` are transported 81.76% of the time, while passengers with `CryoSleep=False` are transported only 32.89% of the time. The interaction with spending is almost structural. Among `CryoSleep=True` passengers, the zero-spend share is 100%. Among `CryoSleep=False` passengers, the zero-spend share is only 9.52%. Looking from the other direction, among zero-spend passengers, 83.14% are in `CryoSleep=True`, 14.18% are in `CryoSleep=False`, and only 2.68% have missing cryosleep status. Among passengers with positive spend, the share with `CryoSleep=True` is literally 0%. This is exactly why the better notebooks do not just fill `CryoSleep` or the spending fields with blind global modes or means: the variables constrain one another.

Age is another variable that becomes more useful once connected to the rest of the structure. Children aged 12 or below are transported 69.98% of the time in the observed training rows, compared with 48.31% for adults above 12. The transported class is also younger on average: mean age 27.75 versus 29.92 for the non-transported class. Maria Aguilera uses subgroup medians involving `HomePlanet`, no-spend status, solo/group status, and `CabinDeck` to impute age; PatrickSVM also emphasizes the importance of detailed case-based imputation rather than a single blunt rule. [4][5]

An important nuance that should appear in the presentation is that not every spending category behaves identically. In the attached data, passengers who were not transported spent much more on `RoomService`, `Spa`, and `VRDeck`, but `FoodCourt` is actually higher on average for transported passengers. The class means are:

| Spending category | Mean if `Transported=False` | Mean if `Transported=True` |
| --- | ---: | ---: |
| `RoomService` | 380.4 | 61.9 |
| `FoodCourt` | 375.2 | 520.6 |
| `ShoppingMall` | 163.9 | 175.1 |
| `Spa` | 552.3 | 60.4 |
| `VRDeck` | 532.3 | 67.6 |

This nuance helps explain why stronger notebooks look at both total spend and the individual categories, rather than collapsing everything too aggressively.

Finally, public workflows repeatedly use group or family consistency to fill missing cabin information. Maria Aguilera’s notebook shows this explicitly, and the local data supports it. In groups with more than one observed value, `HomePlanet` is identical within the group 100% of the time, and `CabinSide` is also identical within the group 100% of the time. `CabinDeck` is identical within the group 69.16% of the time. Those are very strong signals for imputation. [5]

## Slide 5: Modeling Approaches Used by Strong Public Submissions

The model story becomes much more informative once we distinguish among three levels: very simple baselines, engineered linear baselines, and nonlinear tree ensembles.

At the very low end, the Flatiron starter notebook uses a one-feature logistic regression with `Spa` as the only predictor and median imputation after the split. Its reported test accuracy is `0.6266427718`. [7] This is useful as a pedagogical baseline because it shows that even one sensible variable can beat the majority rule, but it is far from competitive.

At the next level, Maria Aguilera’s notebook shows what happens when the feature engineering is strong even if the model is still relatively classical. Her logistic regression reaches about `0.81` accuracy on a holdout report in the exported notebook and about `0.7951 ± 0.0055` in cross-validation. [5] This is a very important nuance for the course connection: linear models are not useless here. Once the features are properly engineered and encoded, logistic regression becomes a serious baseline. The main limitation is not that logistic regression is “bad”; the limitation is that it only models additive effects on the log-odds scale unless we hand-engineer interactions.

If we write the logistic model as

`P(Y=1 | X=x) = exp(beta_0 + x^T beta) / (1 + exp(beta_0 + x^T beta))`,

then each feature contributes linearly to the log-odds `log(p/(1-p))`. That works well if the signal is roughly additive. But Spaceship Titanic contains interaction-heavy structure. The effect of side depends on deck. The usefulness of no-spend depends on cryosleep. Group structure changes how we interpret cabin fields. A logistic model can represent these relationships only if we add terms like

`beta_3 * (CryoSleep * NoSpend)` or `beta_4 * (Deck_C * Side_S)`.

Tree-based ensembles, by contrast, can learn such threshold and interaction rules automatically.

That is why the strongest public pipelines tend to center on ensemble trees. PatrickSVM’s public notebook presents two ensemble models after a long preprocessing pipeline: a Random Forest and a HistGradientBoostingClassifier. The Random Forest uses bagging and reports `oob_score_ = 0.80605`. The HistGradientBoosting model, with tuned hyperparameters and 10-fold cross-validation, reports mean accuracy `0.80778`. [4] Patrick’s README states that his best submission was produced with a GradientBoostingClassifier-based workflow and achieved over 80.8% with a top-6% ranking at that time. [3]

Maria Aguilera’s notebook evaluates a wide family of models on the same engineered feature set: Naive Bayes, Logistic Regression, SGD Classifier, Decision Tree, SVC, Random Forest, AdaBoost, CatBoost, LightGBM, XGBoost, and KNN. On a holdout classification report, several of them cluster around 0.80 to 0.81 accuracy, with Logistic Regression, AdaBoost, LightGBM, and XGBoost each hitting about 0.81 in that specific report. [5] Her cross-validated ranking is more informative for model selection, and there the top untuned model is XGBoost at about `0.8020 ± 0.0061`, with Random Forest and LightGBM just behind. After hyperparameter tuning, LightGBM becomes the best of the compared models at `0.806627`. [5]

Amir Fares’s workflow is different in style but consistent in conclusion. He tries a broad set of models including Random Forest, Logistic Regression, XGBoost, LightGBM, CatBoost, AdaBoost, KNN, Decision Tree, and a DNN, then combines the stronger ones in a weighted ensemble. His README reports a Kaggle score of `0.80` and a top-28% ranking at the time he recorded it. [6] Even here, though, the story is still not “deep learning wins.” The deep neural network is just one model among many inside a broader ensemble workflow.

Samyak Raj Bayar’s public baseline is simpler but again points in the same direction. His `code.py` uses parsed cabin features, group size, title extraction from `Name`, one-hot encoding, a HistGradientBoostingClassifier, and 5-fold `StratifiedKFold` cross-validation. [8] This is a useful source because it shows that even a compact, reproducible public baseline still chooses feature engineering plus gradient boosting rather than a raw black-box neural network.

If I had to summarize the public model evidence in one sentence, it would be this: once preprocessing is done well, many competent models live in the high-0.79 to low-0.80 range, but the most consistently strong public results come from boosted-tree methods and occasionally from ensembles built on top of them.

## Slide 6: Strengths, Limitations, Constraints, and Trade-offs

The main strength of the strongest public workflows is that they use domain logic to reduce noise before model fitting. This is easiest to see in the imputation strategy. Filling `CryoSleep` or total spend with a global mode or mean throws away structure that is plainly visible in the data. Using the facts that `CryoSleep=True` implies zero spend, that group members almost always share home planet and cabin side, and that some decks are essentially associated with one planet recovers information instead of merely smoothing over blanks.

The second strength is methodological discipline. PatrickSVM does not present a single lucky submission; he reports out-of-bag and cross-validation numbers. Maria Aguilera does not just say “LightGBM was best”; she shows mean CV scores, standard deviations, and tuned best scores under a common validation scheme. [4][5] This is a better scientific workflow than comparing isolated leaderboard submissions because it allows like-for-like comparison of competing models.

The third strength is interpretability at the feature level even when the final model is nonlinear. Many of the most important engineered variables are easy to explain to a non-technical audience: group size, cabin deck, cabin side, no-spend status, cryosleep status, and family surname. That makes the workflow pedagogically strong for a class presentation.

The main limitations come from the same places as the strengths. First, many of the imputations are assumption-driven. If we infer a missing `HomePlanet` from deck `B`, that is empirically sensible in the training data, but it still relies on the assumption that the hidden test data follows the same pattern. Second, some categories are very small. Deck `T`, for example, has only 5 observed training passengers, so its transport rate of 20% is unstable and should not be over-interpreted.

Third, the nonlinear models are not automatically superior by a huge margin. Maria Aguilera’s cross-validation results are useful here. Untuned Logistic Regression at `0.7951` is already within about 0.69 percentage points of untuned XGBoost at `0.8020`, and tuned LightGBM at `0.8066` is only about 1.15 percentage points above the logistic baseline built on the same engineered features. [5] That means the competition is a good example of a broader principle: preprocessing and feature construction often determine most of the performance, while the final model family refines the last few points.

This is exactly where the bias-variance trade-off from ISLR becomes concrete. Random Forest is a relatively stable high-variance-control method because it averages many decorrelated trees; it gives PatrickSVM an OOB estimate of `0.80605` without requiring the more delicate boosting dynamics. HistGradientBoosting and LightGBM are more flexible and can squeeze out slightly more performance, but they require more hyperparameter care. The gains are real, but they are not infinite, and that is why one should not overstate the role of the final algorithm.

Another real limitation is validation design. As noted earlier, this is my inference from the local data rather than a direct source quote: because passenger groups are correlated, a purely random cross-validation split may slightly overstate performance if related passengers appear on both sides of the split. A stricter strategy would be group-aware cross-validation. This is a worthwhile criticism to include because it shows that “best practice” is not the same as “beyond criticism.”

Finally, the competition being ongoing creates a presentation constraint. Public leaderboard positions recorded in README files are snapshots from a particular date, not permanent historical rankings. So the safest thing to emphasize is not the rank itself but the workflow and the validation evidence that supports it.

## Slide 7: Best Practices and Conclusion

The most transferable best practice from Spaceship Titanic is that the strongest Kaggle work is really a workflow discipline rather than a model trick. Good participants begin with a baseline, audit the dataset carefully, and ask which variables contain hidden structure. Then they validate locally, compare models under a common scheme, and only after that worry about the public leaderboard.

The second transferable best practice is to engineer features that reflect how the data were generated. `PassengerId` is useful because passengers travel in groups; `Cabin` is useful because physical location on the ship matters; no-spend indicators are useful because cryosleep passengers are confined and do not spend money; surnames can be useful because families often share travel behavior. These are not random feature ideas. They are modelable consequences of the story behind the data.

The third best practice is to separate the role of feature engineering from the role of model choice. A very weak baseline logistic model using only `Spa` gets roughly `0.6266`. A much richer engineered logistic pipeline gets about `0.7951`. A tuned boosted-tree pipeline pushes that into the `0.806` to `0.808` range. [4][5][7] That decomposition is valuable because it shows where the gains really come from.

The fourth best practice is to be honest about trade-offs. Ensembles and boosted trees usually score well on tabular competitions, but they are harder to explain and tune. Penalized linear models are easier to interpret and connect directly to ISLR, but they need much more manual feature construction if the real signal is interaction-heavy. Neither approach is universally “correct”; the point is to match the tool to the structure of the problem.

My overall conclusion is that Spaceship Titanic is an unusually good class example because it sits right at the boundary between classical statistical learning and practical machine learning. The data are small enough that one can still reason carefully about individual variables, but structured enough that naive modeling leaves a lot of performance on the table. The public work on the competition shows that the biggest gains come from understanding the dataset, encoding that understanding in features and imputations, and then using cross-validated ensemble methods to polish the result.

## Explicit Course Connections to ISLR

This section is separate because the connection should be explicit and technical, not a vague appendix.

Cross-validation is central because the test labels are hidden and the leaderboard is noisy. In ISLR terms, we are trying to estimate test error and select among competing models. PatrickSVM’s 10-fold cross-validation mean of `0.80778` for HistGradientBoosting and Maria Aguilera’s model-comparison table are concrete examples of the `K`-fold CV logic taught in the course. [4][5] This is not just theory; it is the main mechanism by which participants decide whether a preprocessing change or a new model actually helps.

Logistic regression is a useful reference model here because it gives a clean probability model and a direct connection to the classification material in ISLR. The logistic form

`P(Y=1 | X=x) = exp(beta_0 + x^T beta) / (1 + exp(beta_0 + x^T beta))`

is appropriate for a binary response, and Maria’s notebook shows that after careful feature engineering and encoding, logistic regression can reach about `0.7951 ± 0.0055` in CV and about `0.81` on a holdout report. [5] So the right lesson is not “logistic regression fails.” The right lesson is that linear decision boundaries become much more competitive when the features are well designed.

Ridge, lasso, and elastic net are relevant because a serious linear baseline for this competition would almost certainly involve a large one-hot-encoded feature matrix. Once we split `Cabin` into deck, number, and side, encode `HomePlanet`, `Destination`, possibly surname or title features, and potentially include interaction terms, the design matrix can become wide and collinear. In that setting the ISLR penalized objectives are directly applicable:

Ridge logistic regression minimizes

`-log L(beta) + lambda * sum_{j=1}^p beta_j^2`,

lasso logistic regression minimizes

`-log L(beta) + lambda * sum_{j=1}^p |beta_j|`,

and elastic net mixes the two penalties. A penalized linear model would therefore be a principled course-aligned baseline for this competition. The reason public notebooks often move beyond it is not that the idea is wrong; it is that tree ensembles automatically recover nonlinear interactions and threshold effects that linear models only see if we write them down ourselves.

The bias-variance trade-off is visible in the clustering of Maria Aguilera’s CV results. Logistic Regression, CatBoost, LightGBM, Random Forest, SVC, and XGBoost all land in a fairly tight band between roughly `0.795` and `0.802` before tuning. [5] That tells us two things. First, once the feature set is strong, variance in algorithm choice is smaller than people might expect. Second, the more flexible models can still eke out gains, but the marginal improvement over a good engineered linear baseline is measured in one or two percentage points, not in miracles. That is exactly the sort of trade-off ISLR tries to teach: more flexible models can reduce bias, but only at the cost of more tuning complexity and potentially more variance.

Finally, model selection in the ISLR sense is all over this competition. The public notebooks compare many models under the same validation framework, inspect whether gains are real, and then pick the simplest workflow that maintains strong performance. That is much closer to the statistical-learning mindset than to the stereotype of Kaggle as random trial-and-error.

## Sources

1. Official competition page: [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
2. Class prompt / competition description provided with the assignment, including the rolling-leaderboard note and dataset field descriptions
3. PatrickSVM README: [Spaceship-Titanic-Kaggle-Challenge](https://github.com/PatrickSVM/Spaceship-Titanic-Kaggle-Challenge)
4. PatrickSVM notebook: [Spaceship_Titanic_Kaggle_Competition.ipynb](https://github.com/PatrickSVM/Spaceship-Titanic-Kaggle-Challenge/blob/main/Spaceship_Titanic_Kaggle_Competition.ipynb)
5. Maria Aguilera exported notebook: [Spaceship Titanic](https://maria-aguilera.github.io/projects/spaceship-titanic.html)
6. Amir Fares repository and Kaggle notebook link: [Kaggle-Spaceship-Titanic](https://github.com/AmirFARES/Kaggle-Spaceship-Titanic)
7. Flatiron starter notebook: [BSC-DS-2022-spaceship-titanic](https://github.com/flatiron-school/BSC-DS-2022-spaceship-titanic)
8. Samyak Raj Bayar public baseline code: [code.py](https://github.com/samyakrajbayar/Kaggle-Spaceship-Titanic/blob/main/code.py)

## Evidence Notes

All dataset counts, percentages, and conditional transport rates in this document were computed from the attached local `train.csv` and `test.csv` files. Public-workflow numbers such as OOB scores, CV means, holdout accuracies, tuned best scores, and hyperparameters come from the linked public notebooks and repositories. The concern about possible group leakage in random cross-validation is my own methodological inference from the structure of `PassengerId` and should be presented as an analytical critique, not as a sourced claim from those notebooks.
