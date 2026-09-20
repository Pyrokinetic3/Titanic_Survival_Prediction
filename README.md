# Titanic Survival Prediction

A Random Forest classifier built for Kaggle's [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition, predicting whether a passenger survived based on features like class, sex, age, fare, and family size.

## Project Overview

The goal was to predict passenger survival using the classic Titanic dataset. Rather than jumping straight to a "final" model, this project is built around a process of iterative tuning — testing one hyperparameter at a time, reflecting on the results, and adjusting course when something didn't add up.

## Approach

**1. Baseline setup**
Started with a Random Forest Classifier trained on `Pclass`, `Sex`, `SibSp`, `Parch`, `Age`, and `Fare`, using a single train/test split (`train_test_split`, `random_state=1`) to evaluate changes.

**2. Manual hyperparameter tuning**
Wrote a separate function for each hyperparameter (number of trees, split criterion, max depth, minimum leaf samples, max features), looping through candidate values and keeping whichever produced the highest accuracy on the held-out split. Also tested two hyperparameters jointly, since tuning them one at a time meant later ones were being optimized against an already-constrained model (e.g., testing minimum leaf samples after already fixing a shallow max depth).

**3. Feature engineering**
Tried adding `Embarked` — no improvement, so it was dropped to keep the model simpler and faster. Then extracted passenger titles (Mr., Mrs., Miss., Master., etc.) from the `Name` column into a new `Title` feature, one-hot encoded it, and re-ran the tuning process. This produced a meaningful accuracy bump, landing on a final manual configuration of `n_estimators=10`, `criterion='entropy'`, `max_depth=8`, `min_samples_leaf=1`, `max_features=1`.

**4. Cross-validation**
The single train/test split used throughout the manual tuning process doesn't account for overfitting to that one particular split — every hyperparameter was chosen by checking its score against the exact same held-out passengers, which biases the results toward that specific slice of data. To address this, `GridSearchCV` with 5-fold stratified cross-validation was used to search the same hyperparameter space more rigorously, averaging performance across five different splits instead of relying on one.

## Results

| Model | Accuracy |
|---|---|
| Manually tuned (no CV) | 79.2% |
| Cross-validated (GridSearchCV) | 77.3% |

At first glance the manually tuned model looks better, but that comparison is a bit misleading. The manual model's hyperparameters were selected by repeatedly checking performance against the same 418-passenger test set — so its score is partly a reflection of how well it happens to fit that one specific group of passengers, not necessarily how well it would perform on new data. With only 418 test passengers, an 8-prediction swing is well within the range of variation you'd expect just from which specific people ended up in the test set.

The cross-validated model's lower score is a more honest estimate of how the model is likely to generalize, since it wasn't tuned against a single fixed test set. In other words: the manual model probably got a bit lucky here, while the CV model is the more trustworthy approach going forward, even though its reported number is lower.

## Limitations

- **Small dataset.** With only ~891 training rows and 418 test rows, accuracy estimates carry meaningful statistical noise — differences of a few percentage points between models aren't necessarily meaningful.
- **Test-set reuse during manual tuning.** The step-by-step hyperparameter search evaluated every candidate against the same held-out split, which can overstate how well the resulting configuration generalizes.
- **Limited feature set.** Features like `Cabin` and `Ticket` were left out; they're sparse and messy in their raw form but likely contain some extractable signal (e.g., deck level from `Cabin`) that wasn't explored here.
- **No comparison against other model types.** Only Random Forest was tested — a gradient boosting model or logistic regression baseline might perform differently on a dataset this size.
- **Hardcoded file paths.** The notebook was written and run inside a Kaggle notebook environment, so it expects the Titanic dataset at `/kaggle/input/competitions/titanic/`. Running it outside Kaggle requires downloading `train.csv` and `test.csv` from the competition page and updating the file paths.

## Running This Notebook

1. Download `train.csv` and `test.csv` from the [Kaggle Titanic competition page](https://www.kaggle.com/competitions/titanic/data).
2. Either run the notebook directly on Kaggle (where the paths already resolve correctly), or update `titanic_train_file_path` and `titanic_validation_file_path` to point to wherever you saved the files locally.
3. Run all cells in order. The notebook produces two submission files — one from the manually tuned model and one from the cross-validated model.

## Next Steps

- Expand the cross-validation grid further and compare fold-level variance, not just the average, to get a better sense of how stable each candidate configuration really is.
- Try engineering a few more features (family size, whether a passenger was traveling alone, deck extracted from `Cabin`).
- Test other model families (gradient boosting, logistic regression) as a point of comparison against the Random Forest results.
