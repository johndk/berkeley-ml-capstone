# Appendix F

[Back to the main report](README.md)

This appendix records the experiments that have been completed. [Appendix E](Appendix-E.md) describes the plan;
Appendix F shows what was actually run and the results. Separate tables record the model setup, the quality of its rankings and probabilities,
and its results at specific decision thresholds. This makes it clear when a result changes because the threshold changed
rather than because the fitted model changed. Exact source-field allowlists are documented in
[Appendix D](Appendix-D.md).

## Results recording rules

- Add a result only after its notebook completes the full model search and the outside validation check.
- Record the training and validation years, row counts, target, and feature-set version. Scores should be compared only
  when they describe the same flight population and outcome.
- Use mean average precision from the forward-moving 2019 folds to explain choices made during training. Use the 2023
  validation results to compare completed experiments for the same model.
- Average precision, ROC AUC, and Brier score do not depend on a classification threshold. Accuracy, balanced accuracy,
  precision, recall, F1, and MCC do, so each of those results must state which threshold was used.
- Choose a *training-selected* threshold only from predictions made for held-out parts of the training year. Do not use
  the 2023 validation outcomes to choose it.
- Do not place Models 1A, 2A, 2B, and 2C on one combined leaderboard because their targets or flight groups differ.
  Compare classifiers within the same model. Compare 2A, 2B, and 2C only when they use exactly the same arrival rows.
- Keep 2024 out of these development tables until the final model design is fixed. Record the eventual 2024 evaluation
  as the final test, not as another development-year check.

## Experiment configurations

| Model | Classifier | Experiment | Notebook | Target | Feature set | Training data | External validation | Selected configuration |
|---|---|---:|---|---|---|---|---|---|
| 1A | Logistic regression | 01 | [logistic_regression_1a_01.ipynb](models/logistic_regression_1a_01.ipynb) | `DepDel15` | 20 pre-pushback source fields; 99 columns after training-based missing-value handling and category encoding | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | L1 logistic regression; `C=10`; no class weighting; uses all 99 prepared columns because separate `SelectFromModel` selection did not improve validation |
| 1A | Logistic regression | 02 | [logistic_regression_1a_02.ipynb](models/logistic_regression_1a_02.ipynb) | `DepDel15` | 27 compact source and calculated fields; 235 columns after preparation; 52 selected | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | L1 feature selection at 1.5 times mean importance, followed by L2 logistic regression; `C=0.01`; no class weighting |
| 1A | Logistic regression | 03 | [logistic_regression_1a_03.ipynb](models/logistic_regression_1a_03.ipynb) | `DepDel15` | 54 broad source and calculated fields; 1,826 columns after preparation; 1,824 nonconstant columns | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | Remove constant columns, then fit L1 logistic regression; `C=0.01`; no class weighting; uses all 1,824 remaining columns because the top-50, top-100, and top-200 choices did not improve validation |
| 1A | Logistic regression | 04 | [logistic_regression_1a_04.ipynb](models/logistic_regression_1a_04.ipynb) | `DepDel15` | Experiment 01's 20-field raw base plus three causal 30-minute backlog fields; 103 columns after preparation; 40 nonzero coefficients | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | Compact backlog variant (`PENDING_COUNT`, `COMPLETED_COUNT`, and `MEAN_DEP_DELAY`); L1 logistic regression; `C=0.01`; no class weighting; the compact and full variants both round to 0.3852 training CV AP, and the compact variant is selected |
| 1A | Logistic regression | 05 | [logistic_regression_1a_05.ipynb](models/logistic_regression_1a_05.ipynb) | `DepDel15` | Experiment 01's 20-field raw base plus all 13 causal rotation fields; 172 nonconstant columns after preparation and all retained | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | Full rotation manifest; L2 logistic regression; `C=1`; no class weighting; passthrough selected. Top-200 is equivalent because the fitted design has only 172 columns; top-50 and top-100 do not improve temporal AP. Final BTS tail assignment makes this a retrospective upper-bound experiment. |
| 1A | Logistic regression | 06 | [logistic_regression_1a_06.ipynb](models/logistic_regression_1a_06.ipynb) | `DepDel15` | Experiment 01's 20-field raw base plus all 13 rotation fields reconstructed from full raw BTS airport history; turns over 24 hours are masked without dropping rows; 197 nonconstant prepared columns retained | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | Fixed Experiment 05 L2 logistic configuration (`C=1`, no class weighting, no top-N selection). Six predeclared history/feature variants are compared; full history with the 24-hour long-turn mask is selected by 2019 temporal AP and also performs best in 2023. Final BTS tail assignment retains the retrospective upper-bound limitation. |
| 1A | Logistic regression | 07 | [logistic_regression_1a_07.ipynb](models/logistic_regression_1a_07.ipynb) | `DepDel15` | Experiment 06's selected 20-field raw base and 13 full-history rotation fields plus Experiment 04's three compact 30-minute backlog fields; 201 nonconstant prepared columns retained | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | Fixed L2 logistic regression (`C=1`, no class weighting or top-N selection) with the 24-hour rotation mask. Rotation-only is the control; rotation plus pending count, completed count, and mean signed recent departure delay is selected. Existing feature files are paired in memory after identity validation. |
| 1A | Logistic regression | 08 | [logistic_regression_1a_08.ipynb](models/logistic_regression_1a_08.ipynb) | `DepDel15` | Experiment 06's selected 20-field raw base and 13 full-history rotation fields plus three compact 60-minute backlog fields; 201 nonconstant prepared columns retained | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | Fixed L2 logistic regression (`C=1`, no class weighting or top-N selection) with the 24-hour rotation mask. Rotation plus W60 pending count, completed count, and mean signed departure delay improves both the rotation-only control and the corresponding W30 Experiment 07. |
| 1A | Decision tree | 01 | [decision_tree_1a_01.ipynb](models/decision_tree_1a_01.ipynb) | `DepDel15` | 20 pre-pushback source fields; 99 columns after training-based missing-value handling and category encoding | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | Entropy tree with class weighting; depth 15; minimum leaf 500; minimum split 250; 157 fitted leaves |
| 1A | Decision tree | 02 | [decision_tree_1a_02.ipynb](models/decision_tree_1a_02.ipynb) | `DepDel15` | 34 compact source and calculated fields for tree models; 265 columns after preparation | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | Gini tree without class weighting; depth 15; minimum leaf 500; minimum split 250; 161 fitted leaves |
| 1A | Random forest | 01 | [random_forest_1a_01.ipynb](models/random_forest_1a_01.ipynb) | `DepDel15` | Exact 34-field compact non-backlog tree manifest from Decision Tree Experiment 02; 265 columns after preparation | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | 400 bootstrap trees; Gini criterion; depth 15; minimum leaf 100; square-root feature sampling; no class weighting; out-of-bag accuracy diagnostic 0.8162 |
| 1A | Random forest | 02 | [random_forest_1a_02.ipynb](models/random_forest_1a_02.ipynb) | `DepDel15` | CatBoost 04's exact 41 source fields: raw pre-pushback base, full-history rotation with the 24-hour mask, compact airport-wide W60 backlog, and five same-airline W60 fields; 210 prepared fields after fold-local imputation, missing indicators, and one-hot encoding | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | 300 bootstrap trees; Gini criterion; depth 18; minimum leaf 25; 50% feature sampling; no class weighting; selected from 16 configurations by five-fold temporal AP. OOB accuracy is 0.8840; the 80-fit search and all-2019 refit take 1,251.2 seconds. Final BTS tail assignment retains the retrospective upper-bound limitation. |
| 1A | Linear discriminant analysis | 01 | [lda_1a_01.ipynb](models/lda_1a_01.ipynb) | `DepDel15` | CatBoost 04's exact 41 source fields and 24-hour rotation mask; six categorical and 35 numeric fields become 210 dense prepared predictors after fold-local imputation, missing indicators, one-hot encoding, and scaling | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | LSQR solver with fixed shrinkage 0.01, selected from ten ordinary and shrinkage LDA configurations using five chronological folds. The operating threshold is selected from held-out 2019 predictions. Final BTS tail assignment retains the retrospective upper-bound limitation. |
| 1A | CatBoost | 01 | [catboost_1a_01.ipynb](models/catboost_1a_01.ipynb) | `DepDel15` | Exact 34-field compact non-backlog tree manifest; six categorical fields handled natively and 28 numeric fields retained without scaling | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | CatBoost 1.2.10; 45 trees selected as the median fold-best count; depth 6; learning rate 0.10; L2 leaf regularization 10; no class weighting |
| 1A | CatBoost | 02 | [catboost_1a_02.ipynb](models/catboost_1a_02.ipynb) | `DepDel15` | Logistic Regression Experiment 08's exact 36 source fields: 20-field raw base, all 13 full-history rotation fields with the 24-hour mask, and three compact W60 backlog fields; six categorical fields handled natively and 30 numeric fields retained without scaling | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | CatBoost 1.2.10; 683 trees selected as the median fold-best count; depth 6; learning rate 0.03; L2 leaf regularization 3; no class weighting. Selected by average precision from a 16-configuration, five-fold temporal search with fold-local early stopping. Final BTS tail assignment retains the retrospective upper-bound limitation. |
| 1A | CatBoost | 03 | [catboost_1a_03.ipynb](models/catboost_1a_03.ipynb) | `DepDel15` | Controlled comparison of CatBoost 02's 36-field compact W60 manifest with a 41-field manifest containing all eight existing W60 backlog fields | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | Fixed CatBoost 02 classifier: 683 trees, depth 6, learning rate 0.03, L2 leaf regularization 3, and no class weighting. Compact W60 is selected by mean 2019 temporal AP, 0.7010 versus 0.7006 for full W60. The selected result therefore reproduces CatBoost 02 rather than replacing it. |
| 1A | CatBoost | 04 | [catboost_1a_04.ipynb](models/catboost_1a_04.ipynb) | `DepDel15` | CatBoost 02's exact 36-field control versus a 41-field variant adding five same-airline W60 fields to the unchanged raw, full-history rotation, and compact airport-wide W60 base | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | Fixed CatBoost 02 classifier: 683 trees, depth 6, learning rate 0.03, L2 leaf regularization 3, and no class weighting. The same-airline variant is selected by mean 2019 temporal AP, 0.7018 versus 0.7010 for the control. Existing rotation and both backlog files are paired in memory after three-way identity validation. |
| 1A | CatBoost | 05 | [catboost_1a_05.ipynb](models/catboost_1a_05.ipynb) | `DepDel15` | CatBoost 04's 41-field control versus schedule-cycle, compact-weather, and combined additions from the existing departure feature dataset; the largest candidate has 51 fields | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | Three predeclared guardrails reject all changes. The 51-field representation gains only 0.0014 mean temporal AP; the best depth-8 candidate gains 0.0021; and `has_time=True` gains 0.0010. The selected result therefore retains CatBoost 04's 41 fields, 683 trees, depth 6, learning rate 0.03, L2 leaf regularization 3, random strength 1, default row permutations, and no class weighting. |
| 1A | Multilayer perceptron | 02 | [mlp_1a_02.ipynb](models/mlp_1a_02.ipynb) | `DepDel15` | The same 36 source fields as Logistic Regression Experiment 08 and CatBoost Experiment 02; six categorical fields one-hot encoded, 30 numeric fields median-imputed with missing indicators and standardized; 202 final prepared inputs | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | TensorFlow 2.21 Keras MLP; hidden layers `(64, 32)` with ReLU, batch normalization, dropout 0.25/0.20, and L2 `0.0001`; Adam learning rate `0.001`; batch size 512; 16 epochs selected as the median fold-best count; 15,489 trainable parameters. Selected from four configurations with five chronological folds and fold-local PR-AUC early stopping. Final BTS tail assignment retains the retrospective upper-bound limitation. |
| 1A | CatBoost / MLP blend | 01 | [ensemble_1a_01.ipynb](models/ensemble_1a_01.ipynb) | `DepDel15` | Aligned probabilities from CatBoost 04's 41-field raw, rotation, airport-wide W60, and same-airline W60 manifest and MLP 02's 36-field raw, rotation, and compact airport-wide W60 manifest | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | Fixed component configurations and identical five chronological folds. A weighted arithmetic mean of 0.75 CatBoost and 0.25 MLP is selected from five predeclared weights by mean 2019 fold AP; the 0.31 threshold is selected from aggregated 2019 OOF probabilities. No component is retuned, no combined dataset is written, and the final-tail assignment limitation remains. |
| 1A | CatBoost calibration | 01 | [calibration_1a_01.ipynb](models/calibration_1a_01.ipynb) | `DepDel15` | CatBoost 04's unchanged 41-field manifest and fixed classifier; probability-only comparison of uncalibrated, sigmoid, and isotonic outputs | 2019: 107,430 JFK departures; calibration assessed forward on held-out folds 2–5 | 2023: 109,983 JFK departures; delay rate 0.2364 | Mean forward Brier score selects the unchanged probabilities: 0.0918 uncalibrated versus 0.0922 sigmoid and 0.0923 isotonic. The rejected corrections are not evaluated on 2023. CatBoost 04 and its 0.31 threshold therefore remain unchanged. |
| 1A | CatBoost audit | 01 | [catboost_audit_1a_01.ipynb](models/catboost_audit_1a_01.ipynb) | `DepDel15` | CatBoost 04's unchanged 41-field manifest; six categorical and 35 numeric fields; subgroup diagnostics plus CatBoost SHAP values on a fixed 5,000-row 2023 sample | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | Frozen CatBoost 04 classifier and 0.31 threshold. Traffic quartiles are defined from 2019; all other groups use transparent fixed rules. The audit reproduces CatBoost 04, explains four representative outcome cases, and makes no model, calibration, feature, or threshold selection from 2023. |
| 2A | Logistic regression | 01 | [logistic_regression_2a_01.ipynb](models/logistic_regression_2a_01.ipynb) | `ArrDel15` | 27 compact pre-pushback fields; 98 columns after training-based missing-value handling and category encoding | 2019: 107,354 JFK arrivals; delay rate 0.2029 | 2023: 109,947 JFK arrivals; delay rate 0.2463 | L1 logistic regression; `C=0.1`; no class weighting |
| 2A | CatBoost | 01 | [catboost_2a_01.ipynb](models/catboost_2a_01.ipynb) | `ArrDel15` | Exact BL-A-27 allowlist: two native categorical and 25 numeric fields. No new feature dataset is created. | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | CatBoost 1.2.10; 103 trees; depth 6; learning rate 0.10; L2 leaf regularization 3; balanced class weighting. Selected from 16 configurations using five chronological folds and fold-local early stopping. |
| 2A | CatBoost | 02 | [catboost_2a_02.ipynb](models/catboost_2a_02.ipynb) | `ArrDel15` | BL-A-27 plus saved `SCHED_DEP_HOUR`, `Month`, and `DayOfWeek` and model-side `AIRLINE_ORIGIN`: 30 saved fields and 31 CatBoost inputs; six categorical and 25 numeric. No new feature dataset is created. | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | CatBoost 1.2.10; 40 trees; depth 6; learning rate 0.10; L2 leaf regularization 3; balanced class weighting. The expansion is selected in a fixed-classifier screen and then tuned over 16 configurations using five chronological folds. |
| 2B | Logistic regression | 01 | [logistic_regression_2b_01.ipynb](models/logistic_regression_2b_01.ipynb) | `ArrDel15` | Exact 27-field 2A base plus signed `DepDelay`; 99 prepared columns | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | L1 logistic regression; `C=0.01`; no class weighting |
| 2B | Logistic regression | 02 | [logistic_regression_2b_02.ipynb](models/logistic_regression_2b_02.ipynb) | `ArrDel15` | BL-A-27 plus signed `DepDelay`: 28 saved fields. `MINUTES_TO_SCHEDULED_ARRIVAL_AT_PUSHBACK = CRSElapsedTime - DepDelay` is calculated in the notebook and represented by a degree-3 spline with five fold-fitted quantile knots; 105 prepared predictors. No new feature dataset is created. | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | Four representations are screened with fixed L1 logistic regression (`C=0.01`, no class weighting). The selected pushback-margin spline is tuned over 16 configurations and retains L1, `C=0.01`, and no class weighting. |
| 2B | CatBoost | 01 | [catboost_2b_01.ipynb](models/catboost_2b_01.ipynb) | `ArrDel15` | Logistic Regression 2B-02's 28 saved fields plus numeric model-side `MINUTES_TO_SCHEDULED_ARRIVAL_AT_PUSHBACK`: 29 CatBoost inputs; two categorical and 27 numeric. No new feature dataset is created. | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | CatBoost 1.2.10; 62 trees; depth 6; learning rate 0.10; L2 leaf regularization 10; balanced class weighting. Selected from 16 configurations using five chronological folds and fold-local early stopping. |
| 2B | Linear discriminant analysis | 01 | [lda_2b_01.ipynb](models/lda_2b_01.ipynb) | `ArrDel15` | Logistic Regression 2B-02's selected 28 source fields plus the same fold-fitted degree-3 pushback-margin spline; 105 dense prepared predictors. No new feature dataset is created. | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | LSQR solver with fixed shrinkage 0.01, selected from ten ordinary and shrinkage LDA configurations using five chronological folds. The operating threshold is selected from held-out 2019 predictions. |
| 2C | Logistic regression | 01 | [logistic_regression_2c_01.ipynb](models/logistic_regression_2c_01.ipynb) | `ArrDel15` | Exact 28-field 2B base plus log taxi-out and two cyclical takeoff-time fields; 102 prepared columns | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | L1 logistic regression; `C=0.1`; no class weighting |
| 2C | Logistic regression | 02 | [logistic_regression_2c_02.ipynb](models/logistic_regression_2c_02.ipynb) | `ArrDel15` | The 27-field 2A base plus `DepDelay`, raw `TaxiOut`, and two cyclical takeoff-time fields: 31 source fields. `MINUTES_TO_SCHEDULED_ARRIVAL = CRSElapsedTime - DepDelay - TaxiOut` is calculated in the notebook and represented by a degree-3 spline with five fold-fitted quantile knots; 108 prepared predictors. No new feature dataset is created. | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | The three representations are screened with fixed L1 logistic regression (`C=0.1`, no class weighting). The selected raw-taxi spline variant is then tuned over 12 practical configurations and retains L1, `C=0.1`, and no class weighting. `C=10` is omitted after the initial run produced impractically long fits and did not win Experiment 01. |
| 2C | Logistic regression | 03 | [logistic_regression_2c_03.ipynb](models/logistic_regression_2c_03.ipynb) | `ArrDel15` | Experiment 02's selected 31-field raw-taxi source allowlist plus the same fold-fitted degree-3 schedule-margin spline; 108 prepared predictors. No new feature dataset is created. | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | Ridge L2 logistic regression with the `lbfgs` solver; `C=0.01`; no class weighting. Selected from ten configurations comprising five regularization strengths with and without balanced class weights using five chronological folds. |
| 2C | Gaussian Naive Bayes | 01 | [gaussian_naive_bayes_2c_01.ipynb](models/gaussian_naive_bayes_2c_01.ipynb) | `ArrDel15` | Logistic Regression 2C-02's selected 31-field raw-taxi source allowlist plus the same fold-fitted degree-3 schedule-margin spline; 108 dense prepared predictors. No new feature dataset is created. | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | Gaussian Naive Bayes with empirical class priors and `var_smoothing=0.01`. The smoothing value is selected from 12 values from `1e-12` through `1e-1` using five chronological 2019 folds. |
| 2C | Support vector machine | 01 | [svm_2c_01.ipynb](models/svm_2c_01.ipynb) | `ArrDel15` | Logistic Regression 2C-02's selected 31-field raw-taxi source allowlist plus the same fold-fitted degree-3 schedule-margin spline; 108 dense prepared predictors. Approximate RBF candidates use 256 temporary random Fourier components. No new feature dataset is created. | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | Linear SVM with squared-hinge loss, L2 penalty, `C=0.001`, and no class weighting. Selected over the linear and approximate RBF candidates using five chronological folds. A sigmoid map fitted to held-out 2019 decision scores supplies probabilities. |
| 2C | Linear discriminant analysis | 01 | [lda_2c_01.ipynb](models/lda_2c_01.ipynb) | `ArrDel15` | Logistic Regression 2C-02's selected 31-field raw-taxi source allowlist plus the same fold-fitted degree-3 schedule-margin spline; 108 dense prepared predictors. No new feature dataset is created. | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | LSQR solver with fixed shrinkage 0.01, selected from ten ordinary and shrinkage LDA configurations using five chronological folds. The operating threshold is selected from held-out 2019 predictions. |
| 2C | CatBoost | 01 | [catboost_2c_01.ipynb](models/catboost_2c_01.ipynb) | `ArrDel15` | Logistic Regression 2C-01's exact 31-field allowlist; two categorical and 29 numeric fields handled without one-hot encoding or scaling | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | CatBoost 1.2.10; 406 trees selected as the median fold-best count; depth 6; learning rate 0.03; L2 leaf regularization 10; no class weighting. Selected from 16 configurations with five chronological folds and fold-local early stopping. |
| 2C | CatBoost | 02 | [catboost_2c_02.ipynb](models/catboost_2c_02.ipynb) | `ArrDel15` | The same 31 source fields as Logistic Regression 2C-02's selected raw-taxi allowlist, plus model-side `MINUTES_TO_SCHEDULED_ARRIVAL = CRSElapsedTime - DepDelay - TaxiOut`: 32 CatBoost inputs. No spline and no new feature dataset are used. | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | CatBoost 1.2.10; 364 trees selected as the median fold-best count; depth 6; learning rate 0.03; L2 leaf regularization 10; no class weighting. The two representations are screened with the fixed Experiment 01 classifier. The selected raw-taxi schedule-margin representation is then tuned over eight configurations with five chronological folds and fold-local early stopping. |
| 2C | Multilayer perceptron | 01 | [mlp_2c_01.ipynb](models/mlp_2c_01.ipynb) | `ArrDel15` | Logistic Regression 2C-02's selected 31-field raw-taxi source allowlist plus a fold-fitted degree-3 schedule-margin spline with five quantile knots; 108 prepared inputs after imputation, missing indicators, one-hot encoding, and scaling. No new feature dataset is used. | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | TensorFlow 2.21 Keras MLP; hidden layers `(64, 32)` with ReLU, batch normalization, dropout 0.25/0.20, and L2 `0.0001`; Adam learning rate `0.001`; batch size 512; 11 epochs selected as the median fold-best count; 9,473 trainable parameters. Selected from six configurations with five chronological folds and fold-local PR-AUC early stopping. |
| 2C | CatBoost / MLP blend | 01 | [ensemble_2c_01.ipynb](models/ensemble_2c_01.ipynb) | `ArrDel15` | Aligned probabilities from CatBoost 2C-02's 32 inputs and MLP 2C-01's 108 prepared inputs. Both fixed components begin with the same 31 raw-taxi source fields. No combined feature or prediction dataset is written. | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | Fixed component configurations and identical five chronological folds. A weighted arithmetic mean of 0.50 CatBoost and 0.50 MLP is selected from five predeclared weights by mean 2019 fold AP. The 0.38 threshold is selected from aligned 2019 out-of-fold probabilities. Neither component is retuned. |

## Ranking and calibration results

The training CV column is the mean average precision from five time-ordered 2019 folds. All other measures come from
the complete 2023 validation dataset, which is used to compare the completed experiments. Higher average
precision and ROC AUC are better; a lower Brier score means the predicted probabilities are better.

| Model | Classifier | Experiment | Training CV AP | Validation prevalence | Validation AP | Validation ROC AUC | Validation Brier score |
|---|---|---:|---:|---:|---:|---:|---:|
| 1A | Logistic regression | 01 | 0.3068 | 0.2364 | 0.3959 | 0.6825 | 0.1692 |
| 1A | Logistic regression | 02 | 0.3108 | 0.2364 | 0.3869 | 0.6762 | 0.1707 |
| 1A | Logistic regression | 03 | 0.3080 | 0.2364 | 0.3920 | 0.6762 | 0.1702 |
| 1A | Logistic regression | 04 | 0.3852 | 0.2364 | 0.4184 | 0.6903 | 0.1653 |
| 1A | Logistic regression | 05 | 0.5930 | 0.2364 | 0.6515 | 0.7932 | 0.1316 |
| 1A | Logistic regression | 06 | 0.6375 | 0.2364 | 0.6960 | 0.8188 | 0.1222 |
| 1A | Logistic regression | 07 | 0.6523 | 0.2364 | 0.7020 | 0.8242 | 0.1202 |
| 1A | Logistic regression | 08 | 0.6577 | 0.2364 | 0.7053 | 0.8267 | 0.1192 |
| 1A | Decision tree | 01 | 0.2961 | 0.2364 | 0.3683 | 0.6603 | 0.2160 |
| 1A | Decision tree | 02 | 0.3027 | 0.2364 | 0.3689 | 0.6638 | 0.1726 |
| 1A | Random forest | 01 | 0.3327 | 0.2364 | 0.3970 | 0.6812 | 0.1700 |
| 1A | Random forest | 02 | 0.6996 | 0.2364 | 0.7409 | 0.8477 | 0.1111 |
| 1A | Linear discriminant analysis | 01 | 0.6566 | 0.2364 | 0.7032 | 0.8261 | 0.1239 |
| 1A | CatBoost | 01 | 0.3459 | 0.2364 | 0.4104 | 0.6886 | 0.1692 |
| 1A | CatBoost | 02 | 0.7025 | 0.2364 | 0.7473 | 0.8511 | 0.1100 |
| 1A | CatBoost | 03 | 0.7010 | 0.2364 | 0.7473 | 0.8511 | 0.1100 |
| 1A | CatBoost | 04 | 0.7018 | 0.2364 | 0.7526 | 0.8546 | 0.1086 |
| 1A | CatBoost | 05 | 0.7018 | 0.2364 | 0.7526 | 0.8546 | 0.1086 |
| 1A | Multilayer perceptron | 02 | 0.6963 | 0.2364 | 0.7405 | 0.8446 | 0.1112 |
| 1A | CatBoost / MLP blend | 01 | 0.7047 | 0.2364 | 0.7532 | 0.8544 | 0.1085 |
| 2A | Logistic regression | 01 | 0.3128 | 0.2463 | 0.4019 | 0.6744 | 0.1741 |
| 2A | CatBoost | 01 | 0.3283 | 0.2463 | 0.3954 | 0.6714 | 0.2174 |
| 2A | CatBoost | 02 | 0.3194 | 0.2463 | 0.3975 | 0.6715 | 0.2206 |
| 2B | Logistic regression | 01 | 0.8644 | 0.2463 | 0.8682 | 0.9126 | 0.0755 |
| 2B | Logistic regression | 02 | 0.8653 | 0.2463 | 0.8704 | 0.9150 | 0.0748 |
| 2B | CatBoost | 01 | 0.8714 | 0.2463 | 0.8715 | 0.9178 | 0.0869 |
| 2B | Linear discriminant analysis | 01 | 0.8507 | 0.2463 | 0.8661 | 0.9119 | 0.1010 |
| 2C | Logistic regression | 01 | 0.9013 | 0.2463 | 0.9090 | 0.9457 | 0.0616 |
| 2C | Logistic regression | 02 | 0.9081 | 0.2463 | 0.9145 | 0.9491 | 0.0594 |
| 2C | Logistic regression | 03 | 0.9066 | 0.2463 | 0.9117 | 0.9477 | 0.0612 |
| 2C | Gaussian Naive Bayes | 01 | 0.7937 | 0.2463 | 0.8282 | 0.8867 | 0.2226 |
| 2C | Support vector machine | 01 | 0.9077 | 0.2463 | 0.9116 | 0.9473 | 0.0604 |
| 2C | Linear discriminant analysis | 01 | 0.8889 | 0.2463 | 0.8979 | 0.9394 | 0.0884 |
| 2C | CatBoost | 01 | 0.9102 | 0.2463 | 0.9080 | 0.9461 | 0.0626 |
| 2C | CatBoost | 02 | 0.9112 | 0.2463 | 0.9092 | 0.9466 | 0.0620 |
| 2C | Multilayer perceptron | 01 | 0.9079 | 0.2463 | 0.9149 | 0.9501 | 0.0605 |
| 2C | CatBoost / MLP blend | 01 | 0.9131 | 0.2463 | 0.9143 | 0.9500 | 0.0603 |

Calibration Experiment 01 uses a separate forward-chained comparison because its selection objective is probability
reliability rather than ranking. Fold 1 supplies the first independent calibration rows; methods are assessed on folds
2–5 using calibrators fitted only to earlier held-out predictions. Only the training-selected method is carried into
2023. Because the uncalibrated probabilities win, the existing CatBoost 04 ranking and threshold rows remain the final
result and are not duplicated as a nominally different model.

| Calibration method | 2019 forward mean Brier | 2019 forward mean log loss | 2019 forward mean ECE | 2023 Brier | 2023 log loss | 2023 ECE |
|---|---:|---:|---:|---:|---:|---:|
| Uncalibrated — selected | **0.0918** | **0.3124** | **0.0125** | 0.1086 | 0.3567 | 0.0282 |
| Sigmoid | 0.0922 | 0.3137 | 0.0156 | — | — | — |
| Isotonic | 0.0923 | 0.3152 | 0.0160 | — | — | — |

## Operating-threshold results

Each experiment is shown twice: once with the standard 0.50 threshold and once with the threshold that produced the
best F1 score on held-out 2019 training folds. The 2023 outcomes are never used to choose a threshold. The two rows for
an experiment use the same fitted model and the same 2023 probabilities; only the cutoff used to issue a delay
prediction changes.

| Model | Classifier | Experiment | Evaluation data | Threshold policy | Threshold | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1A | Logistic regression | 01 | 2023 external validation | Default | 0.50 | 0.7657 | 0.5114 | 0.5892 | 0.0291 | 0.0554 | 0.0902 |
| 1A | Logistic regression | 01 | 2023 external validation | Training-selected F1 | 0.17 | 0.6152 | 0.6323 | 0.3396 | 0.6649 | 0.4496 | 0.2255 |
| 1A | Logistic regression | 02 | 2023 external validation | Default | 0.50 | 0.7641 | 0.5025 | 0.5854 | 0.0065 | 0.0128 | 0.0420 |
| 1A | Logistic regression | 02 | 2023 external validation | Training-selected F1 | 0.22 | 0.6803 | 0.6250 | 0.3734 | 0.5201 | 0.4347 | 0.2260 |
| 1A | Logistic regression | 03 | 2023 external validation | Default | 0.50 | 0.7644 | 0.5039 | 0.6038 | 0.0097 | 0.0192 | 0.0535 |
| 1A | Logistic regression | 03 | 2023 external validation | Training-selected F1 | 0.22 | 0.6740 | 0.6243 | 0.3683 | 0.5300 | 0.4346 | 0.2230 |
| 1A | Logistic regression | 04 | 2023 external validation | Default | 0.50 | 0.7714 | 0.5371 | 0.6075 | 0.0926 | 0.1608 | 0.1689 |
| 1A | Logistic regression | 04 | 2023 external validation | Training-selected F1 | 0.21 | 0.6587 | 0.6355 | 0.3635 | 0.5916 | 0.4503 | 0.2367 |
| 1A | Logistic regression | 05 | 2023 external validation | Default | 0.50 | 0.8251 | 0.6486 | 0.8539 | 0.3138 | 0.4589 | 0.4483 |
| 1A | Logistic regression | 05 | 2023 external validation | Training-selected F1 | 0.26 | 0.8053 | 0.7115 | 0.5988 | 0.5336 | 0.5644 | 0.4407 |
| 1A | Logistic regression | 06 | 2023 external validation | Default | 0.50 | 0.8401 | 0.6825 | 0.8648 | 0.3836 | 0.5315 | 0.5062 |
| 1A | Logistic regression | 06 | 2023 external validation | Training-selected F1 | 0.31 | 0.8319 | 0.7236 | 0.6932 | 0.5181 | 0.5930 | 0.4981 |
| 1A | Logistic regression | 07 | 2023 external validation | Default | 0.50 | 0.8415 | 0.6902 | 0.8451 | 0.4033 | 0.5460 | 0.5109 |
| 1A | Logistic regression | 07 | 2023 external validation | Training-selected F1 | 0.31 | 0.8303 | 0.7324 | 0.6739 | 0.5466 | 0.6036 | 0.5016 |
| 1A | Logistic regression | 08 | 2023 external validation | Default | 0.50 | 0.8430 | 0.6953 | 0.8395 | 0.4151 | 0.5555 | 0.5164 |
| 1A | Logistic regression | 08 | 2023 external validation | Training-selected F1 | 0.29 | 0.8259 | 0.7392 | 0.6487 | 0.5746 | 0.6094 | 0.4994 |
| 1A | Decision tree | 01 | 2023 external validation | Default | 0.50 | 0.6690 | 0.6165 | 0.3605 | 0.5169 | 0.4247 | 0.2092 |
| 1A | Decision tree | 01 | 2023 external validation | Training-selected F1 | 0.49 | 0.6546 | 0.6210 | 0.3536 | 0.5572 | 0.4326 | 0.2126 |
| 1A | Decision tree | 02 | 2023 external validation | Default | 0.50 | 0.7649 | 0.5142 | 0.5376 | 0.0387 | 0.0723 | 0.0933 |
| 1A | Decision tree | 02 | 2023 external validation | Training-selected F1 | 0.19 | 0.6538 | 0.6218 | 0.3536 | 0.5610 | 0.4338 | 0.2137 |
| 1A | Random forest | 01 | 2023 external validation | Default | 0.50 | 0.7643 | 0.5022 | 0.6818 | 0.0052 | 0.0103 | 0.0445 |
| 1A | Random forest | 01 | 2023 external validation | Training-selected F1 | 0.21 | 0.6594 | 0.6308 | 0.3616 | 0.5766 | 0.4445 | 0.2293 |
| 1A | Random forest | 02 | 2023 external validation | Default | 0.50 | 0.8528 | 0.7107 | 0.8733 | 0.4412 | 0.5863 | 0.5521 |
| 1A | Random forest | 02 | 2023 external validation | Training-selected F1 | 0.31 | 0.8452 | 0.7570 | 0.7069 | 0.5896 | 0.6429 | 0.5488 |
| 1A | Linear discriminant analysis | 01 | 2023 external validation | Default | 0.50 | 0.8415 | 0.7023 | 0.8009 | 0.4384 | 0.5666 | 0.5122 |
| 1A | Linear discriminant analysis | 01 | 2023 external validation | Training-selected F1 | 0.182804 | 0.8162 | 0.7438 | 0.6121 | 0.6066 | 0.6093 | 0.4892 |
| 1A | CatBoost | 01 | 2023 external validation | Default | 0.50 | 0.7655 | 0.5065 | 0.6706 | 0.0153 | 0.0299 | 0.0752 |
| 1A | CatBoost | 01 | 2023 external validation | Training-selected F1 | 0.19 | 0.6594 | 0.6349 | 0.3637 | 0.5884 | 0.4495 | 0.2358 |
| 1A | CatBoost | 02 | 2023 external validation | Default | 0.50 | 0.8548 | 0.7139 | 0.8796 | 0.4467 | 0.5925 | 0.5592 |
| 1A | CatBoost | 02 | 2023 external validation | Training-selected F1 | 0.32 | 0.8510 | 0.7516 | 0.7443 | 0.5631 | 0.6411 | 0.5579 |
| 1A | CatBoost | 03 | 2023 external validation | Default | 0.50 | 0.8548 | 0.7139 | 0.8796 | 0.4467 | 0.5925 | 0.5592 |
| 1A | CatBoost | 03 | 2023 external validation | Training-selected F1 | 0.32 | 0.8510 | 0.7516 | 0.7443 | 0.5631 | 0.6411 | 0.5579 |
| 1A | CatBoost | 04 | 2023 external validation | Default | 0.50 | 0.8567 | 0.7186 | 0.8785 | 0.4567 | 0.6010 | 0.5657 |
| 1A | CatBoost | 04 | 2023 external validation | Training-selected F1 | 0.31 | 0.8516 | 0.7598 | 0.7327 | 0.5858 | 0.6511 | 0.5640 |
| 1A | CatBoost | 05 | 2023 external validation | Default | 0.50 | 0.8567 | 0.7186 | 0.8785 | 0.4567 | 0.6010 | 0.5657 |
| 1A | CatBoost | 05 | 2023 external validation | Training-selected F1 | 0.307829 | 0.8513 | 0.7603 | 0.7306 | 0.5877 | 0.6514 | 0.5637 |
| 1A | Multilayer perceptron | 02 | 2023 external validation | Default | 0.50 | 0.8545 | 0.7132 | 0.8794 | 0.4454 | 0.5913 | 0.5581 |
| 1A | Multilayer perceptron | 02 | 2023 external validation | Training-selected F1 | 0.30 | 0.8468 | 0.7530 | 0.7204 | 0.5751 | 0.6396 | 0.5495 |
| 1A | CatBoost / MLP blend | 01 | 2023 external validation | Default | 0.50 | 0.8570 | 0.7180 | 0.8843 | 0.4544 | 0.6003 | 0.5671 |
| 1A | CatBoost / MLP blend | 01 | 2023 external validation | Training-selected F1 | 0.31 | 0.8524 | 0.7595 | 0.7372 | 0.5834 | 0.6514 | 0.5655 |
| 2A | Logistic regression | 01 | 2023 external validation | Default | 0.50 | 0.7569 | 0.5116 | 0.6488 | 0.0282 | 0.0540 | 0.0971 |
| 2A | Logistic regression | 01 | 2023 external validation | Training-selected F1 | 0.22 | 0.6515 | 0.6212 | 0.3650 | 0.5616 | 0.4425 | 0.2153 |
| 2A | CatBoost | 01 | 2023 external validation | Default | 0.50 | 0.6517 | 0.6212 | 0.3652 | 0.5611 | 0.4424 | 0.2154 |
| 2A | CatBoost | 01 | 2023 external validation | Training-selected F1 | 0.41 | 0.5481 | 0.6185 | 0.3223 | 0.7572 | 0.4522 | 0.2068 |
| 2A | CatBoost | 02 | 2023 external validation | Default | 0.50 | 0.6411 | 0.6228 | 0.3598 | 0.5868 | 0.4461 | 0.2159 |
| 2A | CatBoost | 02 | 2023 external validation | Training-selected F1 | 0.45 | 0.5798 | 0.6218 | 0.3331 | 0.7047 | 0.4524 | 0.2101 |
| 2B | Logistic regression | 01 | 2023 external validation | Default | 0.50 | 0.9050 | 0.8281 | 0.9155 | 0.6766 | 0.7781 | 0.7327 |
| 2B | Logistic regression | 01 | 2023 external validation | Training-selected F1 | 0.37 | 0.9050 | 0.8423 | 0.8729 | 0.7189 | 0.7884 | 0.7336 |
| 2B | Logistic regression | 02 | 2023 external validation | Default | 0.50 | 0.9060 | 0.8297 | 0.9178 | 0.6793 | 0.7807 | 0.7359 |
| 2B | Logistic regression | 02 | 2023 external validation | Training-selected F1 | 0.39 | 0.9058 | 0.8412 | 0.8812 | 0.7138 | 0.7887 | 0.7357 |
| 2B | CatBoost | 01 | 2023 external validation | Default | 0.50 | 0.8948 | 0.8519 | 0.7978 | 0.7674 | 0.7823 | 0.7132 |
| 2B | CatBoost | 01 | 2023 external validation | Training-selected F1 | 0.50 | 0.8948 | 0.8519 | 0.7978 | 0.7674 | 0.7823 | 0.7132 |
| 2B | Linear discriminant analysis | 01 | 2023 external validation | Default | 0.50 | 0.8729 | 0.7429 | 0.9939 | 0.4869 | 0.6536 | 0.6427 |
| 2B | Linear discriminant analysis | 01 | 2023 external validation | Training-selected F1 | 0.128015 | 0.9043 | 0.8352 | 0.8888 | 0.6990 | 0.7826 | 0.7309 |
| 2C | Logistic regression | 01 | 2023 external validation | Default | 0.50 | 0.9206 | 0.8608 | 0.9190 | 0.7431 | 0.8217 | 0.7786 |
| 2C | Logistic regression | 01 | 2023 external validation | Training-selected F1 | 0.49 | 0.9206 | 0.8620 | 0.9156 | 0.7464 | 0.8224 | 0.7786 |
| 2C | Logistic regression | 02 | 2023 external validation | Default | 0.50 | 0.9241 | 0.8664 | 0.9249 | 0.7528 | 0.8300 | 0.7886 |
| 2C | Logistic regression | 02 | 2023 external validation | Training-selected F1 | 0.45 | 0.9239 | 0.8712 | 0.9098 | 0.7672 | 0.8324 | 0.7885 |
| 2C | Logistic regression | 03 | 2023 external validation | Default | 0.50 | 0.9207 | 0.8554 | 0.9370 | 0.7267 | 0.8185 | 0.7790 |
| 2C | Logistic regression | 03 | 2023 external validation | Training-selected F1 | 0.39 | 0.9222 | 0.8705 | 0.9011 | 0.7686 | 0.8296 | 0.7838 |
| 2C | Gaussian Naive Bayes | 01 | 2023 external validation | Default | 0.50 | 0.7245 | 0.7698 | 0.4677 | 0.8590 | 0.6056 | 0.4670 |
| 2C | Gaussian Naive Bayes | 01 | 2023 external validation | Training-selected F1 | 0.99999996 | 0.8700 | 0.7415 | 0.9675 | 0.4883 | 0.6491 | 0.6307 |
| 2C | Support vector machine | 01 | 2023 external validation | Default | 0.50 | 0.9228 | 0.8667 | 0.9159 | 0.7560 | 0.8283 | 0.7851 |
| 2C | Support vector machine | 01 | 2023 external validation | Training-selected F1 | 0.428646 | 0.9224 | 0.8736 | 0.8937 | 0.7774 | 0.8315 | 0.7846 |
| 2C | Linear discriminant analysis | 01 | 2023 external validation | Default | 0.50 | 0.8858 | 0.7707 | 0.9866 | 0.5439 | 0.7012 | 0.6810 |
| 2C | Linear discriminant analysis | 01 | 2023 external validation | Training-selected F1 | 0.114816 | 0.9134 | 0.8661 | 0.8616 | 0.7727 | 0.8147 | 0.7604 |
| 2C | CatBoost | 01 | 2023 external validation | Default | 0.50 | 0.9193 | 0.8580 | 0.9193 | 0.7372 | 0.8182 | 0.7749 |
| 2C | CatBoost | 01 | 2023 external validation | Training-selected F1 | 0.37 | 0.9188 | 0.8704 | 0.8810 | 0.7750 | 0.8246 | 0.7747 |
| 2C | CatBoost | 02 | 2023 external validation | Default | 0.50 | 0.9204 | 0.8591 | 0.9227 | 0.7385 | 0.8204 | 0.7779 |
| 2C | CatBoost | 02 | 2023 external validation | Training-selected F1 | 0.36 | 0.9193 | 0.8713 | 0.8817 | 0.7767 | 0.8259 | 0.7763 |
| 2C | Multilayer perceptron | 01 | 2023 external validation | Default | 0.50 | 0.9229 | 0.8599 | 0.9382 | 0.7356 | 0.8246 | 0.7856 |
| 2C | Multilayer perceptron | 01 | 2023 external validation | Training-selected F1 | 0.49 | 0.9233 | 0.8612 | 0.9364 | 0.7388 | 0.8260 | 0.7866 |
| 2C | CatBoost / MLP blend | 01 | 2023 external validation | Default | 0.50 | 0.9226 | 0.8604 | 0.9343 | 0.7377 | 0.8245 | 0.7847 |
| 2C | CatBoost / MLP blend | 01 | 2023 external validation | Training-selected F1 | 0.38 | 0.9235 | 0.8725 | 0.9037 | 0.7718 | 0.8326 | 0.7875 |

### Model 1A logistic-regression comparison

These experiments tested which operational information improved departure predictions with logistic regression.
Experiments 01–03 compared the raw baseline with two general engineered feature sets. Neither engineered version
improved the 2023 results, so Experiment 01 remained the raw baseline.

The operational features produced clearer gains. The 30-minute backlog raised 2023 average precision (AP) from 0.3959
to 0.4184. Aircraft rotation raised it to 0.6515. Full movement history and the 24-hour turn mask raised it again to
0.6960. Combining rotation with backlog produced 0.7020 with a 30-minute window and 0.7053 with a 60-minute window.
Experiment 08 was therefore the strongest Model 1A logistic-regression design.

The rotation results are retrospective upper bounds. BTS identifies the aircraft that operated the flight, but not
necessarily the aircraft assigned at the prediction time. Live use would require timestamped aircraft assignments and
timely gate and arrival events.

### Model 1A decision-tree comparison

This experiment tested whether a single tree benefited from compact engineered features. Experiment 02's 2023 AP was
almost unchanged, but its Brier score improved from 0.2160 to 0.1726. It became the preferred single-tree design,
although both trees remained weaker than the raw logistic baseline.

### Model 1A random-forest comparison

This experiment tested whether an ensemble of trees used the compact features better than one tree. Random Forest 01
reached 0.3970 AP in 2023, compared with 0.3689 for the single tree. It was a useful non-backlog baseline, but it
remained weaker than models using backlog and rotation information.

### Model 1A exact-manifest Random Forest comparison

This experiment separated the value of CatBoost 04's features from the value of the CatBoost classifier. Random Forest
02 used the same 41-field allowlist. Its 2023 AP reached 0.7409, showing that the rotation and backlog fields provide
most of the performance gain. CatBoost 04 remained stronger at 0.7526 AP and had better
probability quality, F1, and MCC. The aircraft-assignment limitation applies to both models.

### Model 1A CatBoost comparison

These experiments tested CatBoost first with the general features and then with the operational features. CatBoost 01
was the strongest non-backlog tree model, with 2023 AP of 0.4104. CatBoost 02 added rotation and the compact 60-minute
backlog; AP increased to 0.7473. CatBoost 03 added the remaining backlog fields, but did not improve the
result. The compact backlog set was retained.

CatBoost 02 was substantially better than the matching logistic model. This supports nonlinear relationships among
aircraft availability, airport pressure, schedule, airline, and route. The aircraft-assignment limitation still
applies.

### Model 1A neural-network comparison

This experiment tested whether a neural network improved predictions from the same 36 fields used by CatBoost 02. Its
2023 AP was 0.7405, compared with 0.7473 for CatBoost. The MLP confirmed the value of nonlinear modeling, but did not
replace CatBoost. The aircraft-assignment limitation remains.

### Model 1A same-airline CatBoost comparison

This experiment tested whether same-airline congestion added information beyond airport-wide congestion. CatBoost 04
added five same-airline backlog fields. Its 2023 AP increased from 0.7473 to 0.7526, Brier score improved
from 0.1100 to 0.1086, and F1 increased from 0.6411 to 0.6511. The gain was small but consistent, so CatBoost 04 became
the preferred Model 1A design. Live use still requires timely operational data and verified aircraft assignments.

### Model 1A final CatBoost search

This experiment tested whether small feature or capacity changes could materially improve CatBoost 04. CatBoost 05
tested schedule-cycle fields, compact weather fields, additional capacity, and chronological CatBoost handling. None
cleared the required 0.003 improvement in 2019 AP. CatBoost 04 remained unchanged. This provided a
practical stopping point for further tuning with the available information.

### Model 1A CatBoost/MLP blend comparison

This experiment tested whether CatBoost and the MLP made different enough errors to benefit from a blend. Training
selected 75% CatBoost and 25% MLP. In 2023, AP increased only from 0.7526 to 0.7532 and ROC AUC decreased slightly. The
gain did not justify running two models, so CatBoost 04 remained preferred.

### Model 1A CatBoost calibration comparison

This experiment tested whether a probability correction made CatBoost 04 more reliable. The original probabilities
had a better 2019 Brier score than sigmoid or isotonic calibration. The two corrections were therefore rejected before
2023 evaluation. CatBoost 04 kept its original probabilities and 0.31
threshold.

### Model 1A CatBoost subgroup and SHAP audit

The audit looked for weak groups and identified which fields drove CatBoost 04's predictions. It reproduced the
2023 result and reviewed performance across operational groups.

| Audit dimension | Groups | Smallest group | AP range | Recall range at 0.31 | Largest absolute mean-probability gap |
|---|---:|---:|---:|---:|---:|
| Airline | 8 | 362 | 0.5954–0.8115 | 0.2925–0.6432 | 0.1328 |
| Month | 12 | 8,596 | 0.6499–0.8201 | 0.4343–0.6895 | 0.0453 |
| Route | 41 | 588 | 0.5747–0.8341 | 0.3067–0.7259 | 0.1014 |
| Time of day | 4 | 682 | 0.2257–0.8429 | 0.0759–0.7144 | 0.0375 |
| Planned-traffic quartile | 4 | 18,497 | 0.5601–0.8145 | 0.3785–0.6739 | 0.0364 |
| Weather | 4 | 6,510 | 0.7308–0.8196 | 0.5563–0.6842 | 0.0364 |

Overnight departures were the clearest weak group. Performance improved as planned traffic increased. SHAP identified
turn time, inbound arrival delay, rotation status, scheduled departure time, and backlog as the leading inputs. These
are predictive relationships, not proof of cause. The audit did not change the model.

### Model 2A/2B/2C logistic-regression timing comparison

These experiments measured how much new information became available at pushback and after takeoff. They use the same
arrival rows. Before pushback, Model 2A reached 0.4019 AP. Adding actual departure delay at pushback raised AP to
0.8682. Adding taxi-out information after takeoff raised it to 0.9090. Actual departure delay
provided the largest gain; taxi-out time provided a smaller additional gain.

### Model 2A CatBoost comparisons

These experiments tested whether CatBoost improved the early arrival prediction made before pushback. Neither CatBoost
experiment improved on Logistic Regression 2A-01. The expanded CatBoost model reached 0.3975 AP,
compared with 0.4019 for logistic regression, and its probabilities were less reliable. Logistic Regression 2A-01
remained preferred.

### Model 2B pushback-margin and CatBoost comparisons

These experiments represented the remaining scheduled time more directly and tested CatBoost on the same idea.
Logistic Regression 2B-02 added a spline for the time remaining until scheduled arrival. Its 2023 AP improved from
0.8682 to 0.8704 and its Brier score improved from 0.0755 to 0.0748. CatBoost reached slightly higher AP at 0.8715, but
had a worse Brier score, F1, and MCC. Logistic Regression 2B-02 remained preferred.

### Model 2C CatBoost comparison

This experiment isolated the effect of changing the classifier while keeping the same after-takeoff fields. CatBoost
01 reached 0.9080 AP in 2023, compared with 0.9090 for logistic regression, and its Brier score was slightly worse.
CatBoost provided slightly higher recall, but did not provide a
clear overall improvement.

### Model 2C schedule-margin comparison

This experiment represented the time remaining until scheduled arrival more directly. Logistic Regression 2C-02 used
raw `TaxiOut` and a spline for that margin. Its 2023 AP increased from 0.9090 to 0.9145, Brier score improved from
0.0616 to 0.0594, and F1 increased from 0.8224 to 0.8324.
Experiment 02 became the preferred Model 2C design.

### Model 2C ridge-logistic comparison

This experiment tested whether retaining and shrinking every coefficient improved the sparse L1 model. Experiment 03
used L2 regularization. Its 2023 AP was 0.9117, below Experiment 02's 0.9145, and its probability and classification
measures were also weaker. The L1 model remained preferred.

### Model 2C Gaussian Naive Bayes comparison

This experiment tested a simple probability model on the selected Model 2C representation. Gaussian Naive Bayes was
substantially weaker. Its 2023 AP was 0.8282 and its Brier score was 0.2226. Its selected threshold was nearly 1.0,
which showed that its probabilities were not reliable for this dataset. Logistic Regression
2C-02 remained preferred.

### Model 2C support-vector-machine comparison

This experiment tested whether a maximum-margin classifier or a nonlinear boundary improved Model 2C. The linear SVM
outperformed the approximate RBF version during training. In 2023, its AP was 0.9116, below Logistic Regression 2C-02
at 0.9145. Its Brier score, F1, and MCC were also slightly weaker. Logistic regression remained
preferred.

### Linear discriminant analysis comparisons

These experiments tested a common linear alternative on the strongest representation for Models 1A, 2B, and 2C. All
three selected LSQR with shrinkage 0.01. None improved on the preferred model at its prediction time: CatBoost 04 for
Model 1A, Logistic
Regression 2B-02, or Logistic Regression 2C-02.

### Model 2C CatBoost schedule-margin comparison

This experiment tested whether the schedule-margin idea also helped CatBoost. CatBoost 02 added raw `TaxiOut` and a
numeric schedule margin. Its 2023 AP improved from 0.9080 to 0.9092, confirming that the schedule-margin idea was
useful. Logistic Regression 2C-02 remained stronger at 0.9145 AP with better
probability and classification measures.

### Model 2C neural-network comparison

This experiment tested whether a neural network improved on the selected Model 2C features. The MLP reached 0.9149 AP,
only 0.0004 above Logistic Regression 2C-02. Its Brier score, F1, and MCC were weaker. This was not a clear standalone
improvement, but the different model structure justified testing a simple blend.

### Model 2C CatBoost/MLP blend comparison

This experiment tested whether combining CatBoost and MLP probabilities improved Model 2C. Training selected an equal
blend. In 2023, its AP was 0.9143, below both the MLP and Logistic Regression 2C-02. Its F1 was only 0.0002 higher than
logistic regression, while probability quality and MCC were worse.
The added complexity was not justified, so Logistic Regression 2C-02 remained preferred.

## Final 2024 evaluation

The 2024 outcomes were opened only after one design was frozen for each distinct prediction time. The
[final evaluation notebook](models/final_evaluation_2024_01.ipynb) refits each design on all 2019 rows, predicts 2024,
and applies the previously selected threshold. It performs no model search, feature search, calibration, or threshold
selection. The 2024 results are final test estimates, not another basis for choosing a model.

| Model | Prediction time | Frozen design | Source fields | Prepared predictors | Frozen setting | Threshold | 2024 rows | 2024 delay rate |
|---|---|---|---:|---:|---|---:|---:|---:|
| 1A | Before pushback for a JFK departure | CatBoost 04 | 41 | 41 | 683 trees; depth 6; learning rate 0.03; L2 leaf regularization 3; no class weighting | 0.31 | 104,715 | 0.2038 |
| 2A | Before pushback for a JFK-bound flight | Logistic Regression 01 | 27 | 98 | L1; `C=0.1`; no class weighting | 0.22 | 104,555 | 0.2161 |
| 2B | At pushback | Logistic Regression 02 | 28 | 105 | L1; `C=0.01`; degree-3 pushback-margin spline; no class weighting | 0.39 | 104,555 | 0.2161 |
| 2C | After takeoff | Logistic Regression 02 | 31 | 108 | L1; `C=0.1`; raw `TaxiOut`; degree-3 schedule-margin spline; no class weighting | 0.45 | 104,555 | 0.2161 |

### Final ranking and probability results

| Model | 2023 AP | 2024 AP | AP change | 2024 ROC AUC | 2024 Brier score |
|---|---:|---:|---:|---:|---:|
| 1A CatBoost 04 | 0.7526 | 0.7250 | -0.0276 | 0.8517 | 0.0983 |
| 2A Logistic Regression 01 | 0.4019 | 0.3482 | -0.0537 | 0.6569 | 0.1603 |
| 2B Logistic Regression 02 | 0.8704 | 0.8757 | +0.0053 | 0.9254 | 0.0626 |
| 2C Logistic Regression 02 | 0.9145 | 0.9268 | +0.0123 | 0.9611 | 0.0465 |

### Final operating-threshold results

The default and frozen-threshold rows use the same 2024 probabilities. Changing the threshold changes only the point
at which a flight is classified as delayed.

| Model | Threshold policy | Threshold | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1A CatBoost 04 | Default | 0.50 | 0.8730 | 0.7133 | 0.8694 | 0.4437 | 0.5875 | 0.5630 |
| 1A CatBoost 04 | Frozen training-selected F1 | 0.31 | 0.8670 | 0.7530 | 0.7246 | 0.5605 | 0.6321 | 0.5593 |
| 2A Logistic Regression 01 | Default | 0.50 | 0.7854 | 0.5105 | 0.5768 | 0.0262 | 0.0502 | 0.0873 |
| 2A Logistic Regression 01 | Frozen training-selected F1 | 0.22 | 0.6520 | 0.6097 | 0.3184 | 0.5351 | 0.3993 | 0.1877 |
| 2B Logistic Regression 02 | Default | 0.50 | 0.9223 | 0.8428 | 0.9187 | 0.7029 | 0.7964 | 0.7597 |
| 2B Logistic Regression 02 | Frozen training-selected F1 | 0.39 | 0.9201 | 0.8535 | 0.8740 | 0.7362 | 0.7992 | 0.7540 |
| 2C Logistic Regression 02 | Default | 0.50 | 0.9402 | 0.8853 | 0.9238 | 0.7885 | 0.8508 | 0.8177 |
| 2C Logistic Regression 02 | Frozen training-selected F1 | 0.45 | 0.9401 | 0.8905 | 0.9091 | 0.8031 | 0.8528 | 0.8179 |

The 2024 arrival results confirm the information ladder. Average precision rises from 0.3482 before pushback to 0.8757
at pushback and 0.9268 after takeoff. Model 2A remains the weakest arrival prediction point because it has no realized
operating delay.
Models 2B and 2C generalize at least as well as they did in 2023.

Model 1A retains nearly the same ROC AUC as in 2023, but its average precision and frozen-threshold F1 are lower. Model
2A also declines. These are final-test findings; they do not authorize post-test tuning or a new model choice.
CatBoost 1A-04 and the three selected logistic-regression designs remain the reported frozen models. The rotation-based
Model 1A result also retains the retrospective aircraft-assignment limitation described above.

## Post-test combined-training sensitivity

Four append-only experiments test routine retraining after the official 2024 evaluation. Each experiment copies one
frozen selected design, combines the 2019 and 2023 training rows, and applies the existing operating threshold to 2024.
No feature, model, parameter, or threshold is selected from these results. Because the 2024 outcomes were already
examined, these are post-test sensitivity results and do not replace the final estimates above.

| Model | Experiment notebook | Training rows | Source fields | Pooled prepared predictors | Fit seconds | Fit status |
|---|---|---:|---:|---:|---:|---|
| 1A CatBoost 06 | [catboost_1a_06.ipynb](models/catboost_1a_06.ipynb) | 217,413 | 41 | 41 | 51.0 | Completed |
| 2A Logistic Regression 02 | [logistic_regression_2a_02.ipynb](models/logistic_regression_2a_02.ipynb) | 217,301 | 27 | 101 | 33.9 | Completed |
| 2B Logistic Regression 03 | [logistic_regression_2b_03.ipynb](models/logistic_regression_2b_03.ipynb) | 217,301 | 28 | 108 | 2.7 | Completed |
| 2C Logistic Regression 04 | [logistic_regression_2c_04.ipynb](models/logistic_regression_2c_04.ipynb) | 217,301 | 31 | 111 | 3,353.4 | Reached the unchanged 5,000-iteration ceiling without convergence |

The three additional prepared predictors in each pooled logistic-regression experiment come from categories learned
across both training years. The source allowlists do not change.

### Combined-training ranking and probability results

| Model | 2019-only AP | Combined AP | AP change | Combined ROC AUC | ROC AUC change | Combined Brier | Brier change |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1A CatBoost 06 | 0.7250 | 0.7313 | +0.0063 | 0.8553 | +0.0036 | 0.0965 | -0.0018 |
| 2A Logistic Regression 02 | 0.3482 | 0.3544 | +0.0062 | 0.6696 | +0.0127 | 0.1589 | -0.0014 |
| 2B Logistic Regression 03 | 0.8757 | 0.8769 | +0.0012 | 0.9261 | +0.0007 | 0.0621 | -0.0005 |
| 2C Logistic Regression 04 | 0.9268 | 0.9270 | +0.0002 | 0.9614 | +0.0003 | 0.0468 | +0.0003 |

### Combined-training operating-threshold results

| Model | Threshold policy | Threshold | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1A CatBoost 06 | Default | 0.50 | 0.8749 | 0.7259 | 0.8429 | 0.4745 | 0.6072 | 0.5711 |
| 1A CatBoost 06 | Frozen | 0.31 | 0.8622 | 0.7641 | 0.6854 | 0.5986 | 0.6391 | 0.5563 |
| 2A Logistic Regression 02 | Default | 0.50 | 0.7851 | 0.5174 | 0.5317 | 0.0460 | 0.0846 | 0.1058 |
| 2A Logistic Regression 02 | Frozen | 0.22 | 0.6266 | 0.6229 | 0.3144 | 0.6163 | 0.4164 | 0.2047 |
| 2B Logistic Regression 03 | Default | 0.50 | 0.9227 | 0.8473 | 0.9083 | 0.7145 | 0.7999 | 0.7612 |
| 2B Logistic Regression 03 | Frozen | 0.39 | 0.9192 | 0.8569 | 0.8605 | 0.7472 | 0.7998 | 0.7525 |
| 2C Logistic Regression 04 | Default | 0.50 | 0.9400 | 0.8877 | 0.9158 | 0.7955 | 0.8514 | 0.8172 |
| 2C Logistic Regression 04 | Frozen | 0.45 | 0.9393 | 0.8931 | 0.8979 | 0.8116 | 0.8525 | 0.8161 |

Combined training provides the clearest gains for Models 1A and 2A. CatBoost 1A improves AP, ROC AUC, Brier score,
and frozen-threshold F1, although MCC is 0.0030 lower. Model 2A improves every reported ranking measure and raises
frozen-threshold F1 by 0.0171 and MCC by 0.0170, but it remains much weaker than the later arrival predictions because
it still lacks realized operating information.

The changes for Models 2B and 2C are negligible. Model 2B improves ranking and Brier score slightly, while its
frozen-threshold F1 is effectively unchanged and MCC is 0.0015 lower. Model 2C gains only 0.0002 AP, worsens Brier
score by 0.0003, and slightly lowers F1 and MCC. Its unchanged solver also fails to converge after 3,353.4 seconds.
The exact pooled 2C retraining procedure is therefore not operationally justified by this result.

These findings support combined-year retraining as a possible future deployment practice for Models 1A and 2A, but
an untouched later year would be required to evaluate that practice fairly. The official 2019-trained final models and
their 2024 test results remain unchanged.
