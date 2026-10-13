# WanBattedBall

A notebook that predicts the outcome of a batted ball (out, single, double, triple, home run) from pitch and swing data, built for the [NWDS batted-ball competition on Kaggle](https://www.kaggle.com/competitions/nwds-batted-balls) run by Nick Wan. It finished around top 25. That placement is also stated in the [baseball-analytics](https://github.com/tdub29/baseball-analytics) README, which is where the rest of my baseball modeling lives.

## What it does

- **Task:** multiclass classification. For each row in `test.csv` it outputs a probability for each of `out`, `single`, `double`, `triple`, `home_run`, written to `submission.csv` keyed by `uid`.
- **Data:** Kaggle's `train.csv`, `test.csv` and `sample_submission.csv`. They are not in this repo; the notebook reads them from `/kaggle/input/nwds-batted-balls/`.
- **Features:** hundreds of derived columns in three groups.
  - Handedness-adjusted pitch movement, spin axis and plate location, plus physics-style ratios such as pitch speed difference, movement complexity and approach angle. These come from the Michael Semenoff and Thiago Mantuani notebooks, as credited in the notebook.
  - Stuff-vs-expected residuals from Nick Wan's code: CatBoost regressors trained with 3-fold cross-validation predict spin axis, horizontal and vertical break, and bat speed and swing length, and the residuals become features.
  - My own interactions: squared terms, baserunner and late-inning pressure, collision angle, and pitch-type flags crossed with pitch and swing attributes.
- **Model:** one CatBoost classifier (`loss_function='MultiClass'`, 100 iterations, depth 6, learning rate 0.1) trained on every column except `pitch_type`, `game_type`, `game_year`, `pitch_number`, `pitch_name`, `outcome` and `xbh`.
- **Metric:** multiclass ROC AUC (one-vs-one) of **0.709** on a random 80/20 split of the training data, as printed in the notebook. This is a local check, not the leaderboard score.

## Run it

1. Open `batted-ball-prediction-notebook.ipynb` as a Kaggle notebook and add the `nwds-batted-balls` competition data.
2. Run all cells. The last cell writes `submission.csv` and a saved model file, `fast_catboost_model_with_test_split.cbm`.

The notebook uses pandas, scikit-learn and CatBoost for the submitted model. It also imports LightGBM, XGBoost, Keras and Optuna.

## Known limits

- The notebook says it was rushed. The model was not tuned and feature scales were not standardized.
- The split is random, not by game or date, so the 0.709 may be optimistic.
- A KMeans fielder-positioning idea built on Statcast sprint speed and baserunning data is left in the notebook as commented-out code. It was not used because that file was not available inside Kaggle.
