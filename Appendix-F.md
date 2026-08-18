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
| 1A | CatBoost | 01 | [catboost_1a_01.ipynb](models/catboost_1a_01.ipynb) | `DepDel15` | Exact 34-field compact non-backlog tree manifest; six categorical fields handled natively and 28 numeric fields retained without scaling | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | CatBoost 1.2.10; 45 trees selected as the median fold-best count; depth 6; learning rate 0.10; L2 leaf regularization 10; no class weighting |
| 1A | CatBoost | 02 | [catboost_1a_02.ipynb](models/catboost_1a_02.ipynb) | `DepDel15` | Logistic Regression Experiment 08's exact 36 source fields: 20-field raw base, all 13 full-history rotation fields with the 24-hour mask, and three compact W60 backlog fields; six categorical fields handled natively and 30 numeric fields retained without scaling | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | CatBoost 1.2.10; 683 trees selected as the median fold-best count; depth 6; learning rate 0.03; L2 leaf regularization 3; no class weighting. Selected by average precision from a 16-configuration, five-fold temporal search with fold-local early stopping. Final BTS tail assignment retains the retrospective upper-bound limitation. |
| 1A | CatBoost | 03 | [catboost_1a_03.ipynb](models/catboost_1a_03.ipynb) | `DepDel15` | Controlled comparison of CatBoost 02's 36-field compact W60 manifest with a 41-field manifest containing all eight existing W60 backlog fields | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | Fixed CatBoost 02 classifier: 683 trees, depth 6, learning rate 0.03, L2 leaf regularization 3, and no class weighting. Compact W60 is selected by mean 2019 temporal AP, 0.7010 versus 0.7006 for full W60. The selected result therefore reproduces CatBoost 02 rather than replacing it. |
| 1A | CatBoost | 04 | [catboost_1a_04.ipynb](models/catboost_1a_04.ipynb) | `DepDel15` | CatBoost 02's exact 36-field control versus a 41-field variant adding five same-airline W60 fields to the unchanged raw, full-history rotation, and compact airport-wide W60 base | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | Fixed CatBoost 02 classifier: 683 trees, depth 6, learning rate 0.03, L2 leaf regularization 3, and no class weighting. The same-airline variant is selected by mean 2019 temporal AP, 0.7018 versus 0.7010 for the control. Existing rotation and both backlog files are paired in memory after three-way identity validation. |
| 1A | Multilayer perceptron | 02 | [mlp_1a_02.ipynb](models/mlp_1a_02.ipynb) | `DepDel15` | The same 36 source fields as Logistic Regression Experiment 08 and CatBoost Experiment 02; six categorical fields one-hot encoded, 30 numeric fields median-imputed with missing indicators and standardized; 202 final prepared inputs | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | TensorFlow 2.21 Keras MLP; hidden layers `(64, 32)` with ReLU, batch normalization, dropout 0.25/0.20, and L2 `0.0001`; Adam learning rate `0.001`; batch size 512; 16 epochs selected as the median fold-best count; 15,489 trainable parameters. Selected from four configurations with five chronological folds and fold-local PR-AUC early stopping. Final BTS tail assignment retains the retrospective upper-bound limitation. |
| 1A | CatBoost / MLP blend | 01 | [ensemble_1a_01.ipynb](models/ensemble_1a_01.ipynb) | `DepDel15` | Aligned probabilities from CatBoost 04's 41-field raw, rotation, airport-wide W60, and same-airline W60 manifest and MLP 02's 36-field raw, rotation, and compact airport-wide W60 manifest | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | Fixed component configurations and identical five chronological folds. A weighted arithmetic mean of 0.75 CatBoost and 0.25 MLP is selected from five predeclared weights by mean 2019 fold AP; the 0.31 threshold is selected from aggregated 2019 OOF probabilities. No component is retuned, no combined dataset is written, and the final-tail assignment limitation remains. |
| 1A | CatBoost calibration | 01 | [calibration_1a_01.ipynb](models/calibration_1a_01.ipynb) | `DepDel15` | CatBoost 04's unchanged 41-field manifest and fixed classifier; probability-only comparison of uncalibrated, sigmoid, and isotonic outputs | 2019: 107,430 JFK departures; calibration assessed forward on held-out folds 2–5 | 2023: 109,983 JFK departures; delay rate 0.2364 | Mean forward Brier score selects the unchanged probabilities: 0.0918 uncalibrated versus 0.0922 sigmoid and 0.0923 isotonic. The rejected corrections are not evaluated on 2023. CatBoost 04 and its 0.31 threshold therefore remain unchanged. |
| 1A | CatBoost audit | 01 | [catboost_audit_1a_01.ipynb](models/catboost_audit_1a_01.ipynb) | `DepDel15` | CatBoost 04's unchanged 41-field manifest; six categorical and 35 numeric fields; subgroup diagnostics plus CatBoost SHAP values on a fixed 5,000-row 2023 sample | 2019: 107,430 JFK departures; delay rate 0.1877 | 2023: 109,983 JFK departures; delay rate 0.2364 | Frozen CatBoost 04 classifier and 0.31 threshold. Traffic quartiles are defined from 2019; all other groups use transparent fixed rules. The audit reproduces CatBoost 04, explains four representative outcome cases, and makes no model, calibration, feature, or threshold selection from 2023. |
| 2A | Logistic regression | 01 | [logistic_regression_2a_01.ipynb](models/logistic_regression_2a_01.ipynb) | `ArrDel15` | 27 compact pre-pushback fields; 98 columns after training-based missing-value handling and category encoding | 2019: 107,354 JFK arrivals; delay rate 0.2029 | 2023: 109,947 JFK arrivals; delay rate 0.2463 | L1 logistic regression; `C=0.1`; no class weighting |
| 2B | Logistic regression | 01 | [logistic_regression_2b_01.ipynb](models/logistic_regression_2b_01.ipynb) | `ArrDel15` | Exact 27-field 2A base plus signed `DepDelay`; 99 prepared columns | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | L1 logistic regression; `C=0.01`; no class weighting |
| 2C | Logistic regression | 01 | [logistic_regression_2c_01.ipynb](models/logistic_regression_2c_01.ipynb) | `ArrDel15` | Exact 28-field 2B base plus log taxi-out and two cyclical takeoff-time fields; 102 prepared columns | 2019: the same 107,354 JFK arrivals; delay rate 0.2029 | 2023: the same 109,947 JFK arrivals; delay rate 0.2463 | L1 logistic regression; `C=0.1`; no class weighting |

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
| 1A | CatBoost | 01 | 0.3459 | 0.2364 | 0.4104 | 0.6886 | 0.1692 |
| 1A | CatBoost | 02 | 0.7025 | 0.2364 | 0.7473 | 0.8511 | 0.1100 |
| 1A | CatBoost | 03 | 0.7010 | 0.2364 | 0.7473 | 0.8511 | 0.1100 |
| 1A | CatBoost | 04 | 0.7018 | 0.2364 | 0.7526 | 0.8546 | 0.1086 |
| 1A | Multilayer perceptron | 02 | 0.6963 | 0.2364 | 0.7405 | 0.8446 | 0.1112 |
| 1A | CatBoost / MLP blend | 01 | 0.7047 | 0.2364 | 0.7532 | 0.8544 | 0.1085 |
| 2A | Logistic regression | 01 | 0.3128 | 0.2463 | 0.4019 | 0.6744 | 0.1741 |
| 2B | Logistic regression | 01 | 0.8644 | 0.2463 | 0.8682 | 0.9126 | 0.0755 |
| 2C | Logistic regression | 01 | 0.9013 | 0.2463 | 0.9090 | 0.9457 | 0.0616 |

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
| 1A | CatBoost | 01 | 2023 external validation | Default | 0.50 | 0.7655 | 0.5065 | 0.6706 | 0.0153 | 0.0299 | 0.0752 |
| 1A | CatBoost | 01 | 2023 external validation | Training-selected F1 | 0.19 | 0.6594 | 0.6349 | 0.3637 | 0.5884 | 0.4495 | 0.2358 |
| 1A | CatBoost | 02 | 2023 external validation | Default | 0.50 | 0.8548 | 0.7139 | 0.8796 | 0.4467 | 0.5925 | 0.5592 |
| 1A | CatBoost | 02 | 2023 external validation | Training-selected F1 | 0.32 | 0.8510 | 0.7516 | 0.7443 | 0.5631 | 0.6411 | 0.5579 |
| 1A | CatBoost | 03 | 2023 external validation | Default | 0.50 | 0.8548 | 0.7139 | 0.8796 | 0.4467 | 0.5925 | 0.5592 |
| 1A | CatBoost | 03 | 2023 external validation | Training-selected F1 | 0.32 | 0.8510 | 0.7516 | 0.7443 | 0.5631 | 0.6411 | 0.5579 |
| 1A | CatBoost | 04 | 2023 external validation | Default | 0.50 | 0.8567 | 0.7186 | 0.8785 | 0.4567 | 0.6010 | 0.5657 |
| 1A | CatBoost | 04 | 2023 external validation | Training-selected F1 | 0.31 | 0.8516 | 0.7598 | 0.7327 | 0.5858 | 0.6511 | 0.5640 |
| 1A | Multilayer perceptron | 02 | 2023 external validation | Default | 0.50 | 0.8545 | 0.7132 | 0.8794 | 0.4454 | 0.5913 | 0.5581 |
| 1A | Multilayer perceptron | 02 | 2023 external validation | Training-selected F1 | 0.30 | 0.8468 | 0.7530 | 0.7204 | 0.5751 | 0.6396 | 0.5495 |
| 1A | CatBoost / MLP blend | 01 | 2023 external validation | Default | 0.50 | 0.8570 | 0.7180 | 0.8843 | 0.4544 | 0.6003 | 0.5671 |
| 1A | CatBoost / MLP blend | 01 | 2023 external validation | Training-selected F1 | 0.31 | 0.8524 | 0.7595 | 0.7372 | 0.5834 | 0.6514 | 0.5655 |
| 2A | Logistic regression | 01 | 2023 external validation | Default | 0.50 | 0.7569 | 0.5116 | 0.6488 | 0.0282 | 0.0540 | 0.0971 |
| 2A | Logistic regression | 01 | 2023 external validation | Training-selected F1 | 0.22 | 0.6515 | 0.6212 | 0.3650 | 0.5616 | 0.4425 | 0.2153 |
| 2B | Logistic regression | 01 | 2023 external validation | Default | 0.50 | 0.9050 | 0.8281 | 0.9155 | 0.6766 | 0.7781 | 0.7327 |
| 2B | Logistic regression | 01 | 2023 external validation | Training-selected F1 | 0.37 | 0.9050 | 0.8423 | 0.8729 | 0.7189 | 0.7884 | 0.7336 |
| 2C | Logistic regression | 01 | 2023 external validation | Default | 0.50 | 0.9206 | 0.8608 | 0.9190 | 0.7431 | 0.8217 | 0.7786 |
| 2C | Logistic regression | 01 | 2023 external validation | Training-selected F1 | 0.49 | 0.9206 | 0.8620 | 0.9156 | 0.7464 | 0.8224 | 0.7786 |

### Current Model 1A logistic-regression comparison

Experiment 02 has the best mean average precision in the 2019 time-based validation (0.3108), but that advantage does
not continue in 2023. Experiment 03 improves on Experiment 02 by 0.0051 in 2023 average precision and by 0.0005 in Brier
score. However, it remains below Experiment 01 by 0.0039 in average precision and 0.0063 in ROC AUC, and its Brier score
is 0.0010 higher. In the 2019 search, none of the top-50, top-100, or top-200 feature sets wins; the best version uses all
1,824 nonconstant encoded fields with an L1 classifier. At thresholds chosen from the training data, Experiments 02 and
03 have nearly the same F1, while Experiment 01 has the best combination of F1, ranking, and probability quality.

Experiment 01 therefore remains the Model 1A raw logistic-regression baseline. The current evidence does not support
choosing either general engineered feature set, and a larger general-purpose logistic feature-selection search is not
the next priority.

Experiment 04 tests a narrower operational hypothesis and produces a clearer improvement. Its training-only search
compares the three-field compact backlog set with all eight backlog fields. Both variants round to 0.3852 mean 2019
temporal-validation average precision, and the compact version is selected. The selected L1 model uses `C=0.01`, no
class weighting, and 40 nonzero coefficients among 103 prepared columns. Relative to Experiment
01, its 2019 CV average precision increases by 0.0784. The improvement persists in 2023: average precision increases by
0.0225 to 0.4184, ROC AUC increases by 0.0078 to 0.6903, and Brier score decreases by 0.0039 to 0.1653. At the threshold
selected from 2019, F1 increases from 0.4496 to 0.4503 and MCC from 0.2255 to 0.2367.

`BACKLOG_W30_PENDING_COUNT` is the largest standardized coefficient in Experiment 04 at 0.3339. Completed count is
negative at -0.1898 and mean signed recent departure delay is positive at 0.1675; these are associations within a
correlated feature set, not causal effects. Experiment 04 was the preferred Model 1A logistic feature set before the
rotation experiment and remains the preferred backlog design, while Experiment 01 remains its raw reference. BTS can
reconstruct the backlog features for historical modeling, but a live model would need timely gate-out events to know
which earlier-scheduled flights are completed or still pending.

Experiment 05 produces the first large Model 1A step-change. The training-only search compares five compact rotation
fields with all 13 generated fields and compares all nonconstant columns with L1-ranked top-50, top-100, and top-200
sets. The full manifest wins with mean 2019 temporal-validation average precision 0.5930, 0.2078 above Experiment 04.
The selected final classifier is L2 logistic regression with `C=1`, no class weighting, and all 172 prepared columns.
The top-200 candidate is effectively identical because only 172 nonconstant columns exist; top-50 and top-100 do not
improve average precision.

The improvement persists in 2023. Relative to Experiment 04, average precision increases by 0.2331 to 0.6515, ROC AUC
increases by 0.1029 to 0.7932, and Brier score decreases by 0.0337 to 0.1316. At the threshold of 0.26 selected from
2019, F1 reaches 0.5644 and MCC 0.4407, improvements of 0.1141 and 0.2040. At the default threshold, precision is
0.8539 and MCC is 0.4483, showing that the rotation model also identifies a smaller high-confidence delayed group.

The 2,995 validation flights whose assigned inbound aircraft had not arrived by scheduled departure represent 2.72%
of 2023 rows and have a 99.63% departure-delay rate. This strong state is not the only source of discrimination: after
those rows are excluded, Experiment 05 still has average precision 0.5620, ROC AUC 0.7665, and Brier score 0.1352 on the
remaining 106,988 rows. Those subgroup scores describe a different population and are diagnostic rather than a direct
leaderboard comparison.

Before the full-history sensitivity study, Experiment 05 was the preferred retrospective Model 1A logistic design. It
was not yet a preferred deployable design: BTS identifies the aircraft that ultimately operated each departure rather
than proving which tail
was assigned at the prediction cutoff. A timestamped aircraft-assignment feed must validate that assumption. Until
then, all rotation performance—including subgroup diagnostics—must be described as an upper bound. Experiment 06 next
tests the history and feature sensitivities under the same assignment caveat.

Experiment 06 isolates the effect of rotation-history completeness and feature availability without retuning the
classifier. All six variants use Experiment 05's L2 logistic regression with `C=1`, no class weighting, and every
nonconstant prepared column. The cohort-history control exactly reproduces Experiment 05's 0.5930 training AP and
0.6515 validation AP, confirming the controlled implementation. The raw baseline reaches 0.3988 validation AP.
Full-history schedule-only features raise AP to 0.4417; live rotation state without the observed inbound outcomes
raises it to 0.5658. This demonstrates that scheduled leg order contains useful information, but knowing whether the
assigned aircraft has actually arrived provides most of the rotation improvement.

Replacing cohort-limited history with all eligible raw JFK BTS movements raises mean 2019 temporal AP from 0.5930 to
0.6292 and 2023 AP from 0.6515 to 0.6848. ROC AUC rises from 0.7932 to 0.8091 and Brier score falls from 0.1316 to
0.1245. The improvement is concentrated where history matters: among the 12,166 validation rows whose preceding
inbound reconstruction changes, AP rises from 0.3334 to 0.7203. On unchanged-history rows the two designs are
essentially equal. The result therefore supports fuller event history rather than a coincidental global model shift.

The predeclared 24-hour sensitivity performs best. It preserves every target row but replaces rotation values for
matches with scheduled turns over 24 hours with a distinct `LONG_TURN_EXCLUDED` state. Training AP reaches 0.6375;
2023 AP reaches 0.6960, ROC AUC 0.8188, and Brier score 0.1222. At the 0.31 threshold selected from 2019, F1 is 0.5930
and MCC 0.4981. After the 3,565 full-history `NOT_ARRIVED` rows are excluded, AP remains 0.6034, so the improvement is
not limited to the nearly deterministic late-aircraft group. Experiment 06 becomes the preferred retrospective Model
1A logistic design, while the final-tail assignment caveat remains unchanged. A live deployment still requires a
timestamped aircraft-assignment feed and timely arrival events.

Experiment 07 tests whether the selected compact backlog state remains useful after conditioning on Experiment 06's
flight-specific rotation information. The rotation-only control exactly reproduces Experiment 06: mean 2019 temporal
AP is 0.6375 and 2023 AP is 0.6960. Adding only 30-minute pending count, completed count, and mean signed recent
departure delay raises training AP to 0.6523 and 2023 AP to 0.7020. ROC AUC rises from 0.8188 to 0.8242 and Brier score
falls from 0.1222 to 0.1202. The improvement is modest but consistent across training and development data.

The added information is not confined to the nearly deterministic late-aircraft cases. Excluding `NOT_ARRIVED`, AP
rises from 0.6034 to 0.6104; among already-arrived rotations it rises from 0.5806 to 0.5869. Backlog contributes most
clearly when rotation is unavailable or excluded, where AP increases from 0.6305 to 0.6458 and Brier score falls from
0.2076 to 0.2014. At the unchanged 0.31 training-selected threshold, combined-model F1 reaches 0.6036 and MCC 0.5016.
The standardized backlog coefficients retain the same plausible directions as Experiment 04: pending count is
positive (0.2401), completed count negative (-0.1179), and recent signed delay positive (0.1146).

Experiment 07 therefore became the preferred retrospective Model 1A logistic design at that stage. The feature families are
operationally complementary: rotation describes whether the assigned aircraft is available, while backlog represents
airport-wide pressure on an available aircraft. This conclusion does not remove the final-tail assignment limitation;
the combined result remains an upper bound until aircraft assignment can be verified at the prediction cutoff.

Experiment 08 repeats the controlled combination with a 60-minute backlog window. The rotation-only control again
reproduces Experiment 06. W60 raises mean 2019 temporal AP to 0.6577 and 2023 AP to 0.7053, exceeding W30 by 0.0054 and
0.0033 respectively. Relative to Experiment 07, ROC AUC rises from 0.8242 to 0.8267 and Brier score falls from 0.1202
to 0.1192. At the 0.29 threshold selected from 2019, F1 reaches 0.6094; the default-threshold MCC reaches 0.5164.

The broader window improves every diagnostic subgroup. Excluding `NOT_ARRIVED`, W60 AP is 0.6146 versus 0.6104 for
W30; among already-arrived rotations it is 0.5904 versus 0.5869. Where rotation is unavailable or excluded, W60 AP is
0.6565 versus 0.6458 and its Brier score is 0.1986 versus 0.2014. The W60 coefficient signs remain consistent:
pending count is positive (0.2604), completed count negative (-0.1206), and recent signed delay positive (0.1573).
Experiment 08 therefore replaces Experiment 07 as the preferred retrospective Model 1A logistic design. The gain is
incremental rather than transformative, but it is consistent in both years and supports sustained one-hour airport
pressure as slightly more informative than the immediate 30-minute snapshot.

### Current Model 1A decision-tree comparison

Experiment 02 improves mean 2019 time-based validation average precision by 0.0066. The improvement continues in 2023
but remains small: average precision rises by 0.0006 and ROC AUC by 0.0035. The Brier score improves more clearly, from
0.2160 to 0.1726. That difference is not caused by the feature set alone, because Experiment 01 uses class weighting and
Experiment 02 does not. At their training-selected thresholds, the two trees are nearly equal; Experiment 02 improves
F1 by 0.0012 and MCC by 0.0011.

The tree uses the calculated traffic features. ASPM fields provide 0.0872 of Experiment 02's total impurity-based
feature importance, compared with 0.0425 for the six source ASPM fields in Experiment 01.
`ASPM_MAX_HOURLY_TRAFFIC` provides 0.0571 by itself and ranks fourth among the source features. This does not show that
traffic caused a delay, and related fields can share or exchange importance. It does show that the peak-traffic summary
helps the tree make splits more efficiently than the source counts alone. Scheduled departure time remains the most
important source feature.

Experiment 02 is therefore the preferred single-tree feature set and the starting point for the planned Random Forest
and CatBoost experiments. Its ranking improvement is too small to claim a clear overall gain. Both decision trees remain
below the raw logistic baseline in 2023 average precision, ROC AUC, and F1 at the training-selected threshold. The next
step is to test whether tree ensembles make better use of the traffic and weather relationships than a single tree.

### Current Model 1A random-forest comparison

Random Forest Experiment 01 carries forward Decision Tree Experiment 02's exact 34-field compact non-backlog feature
manifest. The five-fold 2019 search selects 400 trees, depth 15, minimum leaf size 100, square-root feature sampling,
and no class weighting. The selected forest has 265 prepared columns and an out-of-bag accuracy diagnostic of 0.8162;
the out-of-bag score is reported only as a diagnostic and did not select the model or its threshold.

The ensemble produces a clear improvement over its single-tree reference. Mean 2019 temporal-validation average
precision rises by 0.0300, from 0.3027 to 0.3327. In 2023, average precision rises by 0.0281 to 0.3970, ROC AUC rises by
0.0174 to 0.6812, and Brier score falls by 0.0026 to 0.1700. At the threshold selected from held-out 2019 predictions,
F1 rises by 0.0107 to 0.4445 and MCC rises by 0.0156 to 0.2293. The forest therefore uses the compact traffic, weather,
calendar, and schedule representation more effectively than one bounded tree.

The selected forest is unweighted, so its default 0.50 threshold detects very few delayed flights: 2023 recall is only
0.0052. The training-selected 0.21 threshold is essential for the intended classification trade-off, raising recall to
0.5766 with precision 0.3616. This threshold was selected exclusively from 2019 out-of-fold predictions; the 2023
labels did not influence it.

Scheduled departure time has the largest aggregated impurity importance. Time of day ranks next, followed by the
three-hour ASPM scheduled-arrival total; several other ASPM arrival, departure, trend, and peak fields also appear among
the leading predictors. The forest is approximately tied with raw Logistic Regression Experiment 01 in ranking quality:
its 2023 average precision is 0.0011 higher, while ROC AUC is 0.0013 lower and Brier score is 0.0008 higher. It remains
below the backlog-enhanced Logistic Regression Experiment 04 in average precision, ROC AUC, Brier score, and
training-selected F1. Random Forest Experiment 01 is consequently a useful non-backlog ensemble baseline, not the
current preferred Model 1A result. Random Forest Experiment 02 now supplies the separate append-only rotation and
dual-scope W60 comparison described below.

### Current Model 1A exact-manifest Random Forest comparison

Random Forest Experiment 02 supplies the missing controlled comparison. It uses CatBoost 04's exact 41 source fields,
the same full-history rotation construction and 24-hour mask, the same compact airport-wide W60 fields, and the same
five same-airline W60 fields. Its six categorical fields are imputed and one-hot encoded within each fold; the 35
numeric fields receive fold-local median imputation and missing indicators, producing 210 prepared fields in the final
fit. The 2019 search compares 16 combinations of depth, leaf size, feature sampling, and class weighting while holding
the ensemble at 300 bootstrap trees.

The selected forest uses depth 18, minimum leaf size 25, 50% feature sampling, and no class weighting. Its mean 2019
temporal AP is 0.6996, only 0.0022 below CatBoost 04's 0.7018. This is a substantial improvement over Random Forest 01's
0.3327 and demonstrates that the rotation and backlog features—not merely CatBoost—supply most of the predictive
step-change. The selected forest's out-of-bag accuracy is 0.8840, but that value is diagnostic only and does not select
the model or threshold. The 80-fit search and all-2019 refit require 1,251.2 seconds, making even this bounded search
computationally substantial.

CatBoost separates more clearly in the independent year. Random Forest 02 reaches 2023 AP 0.7409, ROC AUC 0.8477, and
Brier score 0.1111. CatBoost 04 is better by 0.0117 AP and 0.0069 ROC AUC, with Brier score lower by 0.0025. At their
common independently selected threshold of 0.31, the forest obtains precision 0.7069, recall 0.5896, F1 0.6429, and
MCC 0.5488; CatBoost obtains 0.7327, 0.5858, 0.6511, and 0.5640 respectively. CatBoost therefore retains better
ranking, probability quality, F1, and MCC, while the forest trades some precision for slightly higher recall.

Rotation fields dominate the forest's source importance: actual and log actual turn duration rank first and second,
followed by rotation status and the not-arrived indicator. Airport-wide pending count ranks eighth, mean signed delay
twelfth, same-airline mean delay eighteenth, and same-airline pending share nineteenth. These values are not causal,
but they confirm that the forest actively uses every operational feature family. Random Forest 02 is retained as a
strong independent comparator and feature-validation result; CatBoost 04 remains the preferred Model 1A classifier.
The final-tail assignment caveat remains, no combined dataset is written, and 2024 remains untouched.

### Current Model 1A CatBoost comparison

CatBoost Experiment 01 uses the same 34-field compact non-backlog manifest as the preferred single tree and the Random
Forest baseline. Unlike those scikit-learn models, CatBoost receives the six categorical fields directly and handles
its 28 numeric fields without scaling. The 2019 temporal search independently applies early stopping in every fold and
selects depth 6, learning rate 0.10, L2 leaf regularization 10, and no class weighting. The median fold-best count is 45
trees, which is then fixed while the final model is trained on all 2019 rows without using any 2023 outcome.
Balanced automatic class weighting was included in the same fold-local search but did not improve mean average
precision; no validation or test rows were reweighted or resampled.

This is the strongest of the three non-backlog tree methods tested so far. Relative to Random Forest Experiment 01,
mean 2019 temporal-validation average precision rises by 0.0132 to 0.3459. On 2023, average precision rises by 0.0134
to 0.4104, ROC AUC rises by 0.0074 to 0.6886, and Brier score falls by 0.0008 to 0.1692. At the 0.19 threshold selected
from held-out 2019 predictions, precision is 0.3637, recall is 0.5884, F1 is 0.4495, and MCC is 0.2358. Compared with
the Random Forest at its independently selected threshold, CatBoost improves F1 by 0.0050 and MCC by 0.0065.

CatBoost also improves over raw Logistic Regression Experiment 01 by 0.0145 in 2023 average precision and 0.0061 in
ROC AUC, while matching its Brier score to four decimal places. Their training-selected F1 scores are nearly identical:
0.4495 for CatBoost and 0.4496 for logistic regression. The compact non-backlog CatBoost result therefore improves
ranking without materially changing the final thresholded F1.

Scheduled departure time has the largest CatBoost prediction-value-change importance, followed by temperature, month,
and the three-hour ASPM scheduled-arrival total. Airline, airline-destination, time-of-day, and numerous ASPM traffic
fields also receive importance, supporting the value of native categorical handling and nonlinear interactions. These
importance values describe the fitted model and do not show that any feature caused a delay.

Backlog Logistic Regression Experiment 04 remains the strongest completed Model 1A result: its 2023 average precision
is 0.0080 higher, ROC AUC is 0.0017 higher, Brier score is 0.0039 lower, and training-selected F1 is 0.0008 higher than
CatBoost Experiment 01. CatBoost is now the preferred non-backlog nonlinear model and provides a clean reference for a
separate append-only backlog CatBoost experiment. The early-stopped best iteration varies substantially across the
five 2019 folds, so a future CatBoost tuning expansion should remain modest and retain chronological validation rather
than treating 45 trees as a universally stable optimum.

CatBoost Experiment 02 applies the nonlinear model to Logistic Regression Experiment 08's exact source-feature
hypothesis: the 20-field raw baseline, all 13 full-history rotation fields with the 24-hour long-turn mask, and the
three compact W60 backlog fields. The six categorical fields are handled natively and the 30 numeric fields are not
scaled. The five-fold 2019 temporal search compares 16 configurations with fold-local early stopping. It selects depth
6, learning rate 0.03, L2 leaf regularization 3, no class weighting, and 683 trees, the median best count across folds.
Its mean temporal-validation average precision is 0.7025. The existing rotation and backlog datasets are paired in
memory after strict row-identity validation; no combined feature dataset is created.

This produces a material improvement over Logistic Regression Experiment 08. On the same 2023 rows, average precision
rises by 0.0420 to 0.7473, ROC AUC rises by 0.0244 to 0.8511, and Brier score falls by 0.0092 to 0.1100. At the 0.32
threshold selected only from held-out 2019 predictions, F1 reaches 0.6411 and MCC 0.5579, improvements of 0.0317 and
0.0585 over Logistic Regression Experiment 08 at its independently selected threshold. At the default threshold,
precision is 0.8796, recall 0.4467, F1 0.5925, and MCC 0.5592.

The improvement is not confined to the nearly deterministic late-aircraft cases. After the 3,565 full-history
`NOT_ARRIVED` rows are excluded, average precision is 0.6732, compared with 0.6146 for Logistic Regression Experiment
08. Among already-arrived rotations it is 0.6568 versus 0.5904; where rotation is unavailable or excluded it is 0.6884
versus 0.6565. The leading CatBoost importance values are actual turn time, its log transform, rotation status, inbound
arrival delay, and scheduled turn time. W60 pending count ranks sixth, W60 mean signed delay twelfth, and W60 completed
count twentieth, so all three backlog fields contribute to the fitted nonlinear model. Importance describes predictive
use within this fitted model, not causation.

CatBoost Experiment 02 is therefore the preferred completed retrospective Model 1A candidate. Its gain supports the
hypothesis that nonlinear interactions among aircraft availability, turn timing, airport pressure, schedule, carrier,
and route contain information that logistic regression cannot represent. It does not remove the deployment caveat:
the final BTS tail number may not equal the aircraft assigned at the prediction cutoff. The result remains an upper
bound until a timestamped assignment feed and timely operational events are available. The 2024 final-test data remains
untouched.

CatBoost Experiment 03 tests all eight existing W60 backlog fields without retuning the classifier. It fixes Experiment
02's 683 trees, depth 6, learning rate 0.03, L2 leaf regularization 3, no class weighting, rows, temporal folds,
full-history rotation manifest, and 24-hour long-turn mask. The compact control has 36 source fields; the expanded
variant has 41 and adds scheduled count, delayed-departure count, delay rate, nonnegative mean delay, and total delay
minutes. Both feature paths are paired in memory from the established files, and no new combined dataset is created.

The compact manifest wins the predeclared 2019 selection measure. Its mean fixed-model temporal AP is 0.7010 versus
0.7006 for the full manifest. Compact is higher in three folds, full is higher in one, and the remaining fold is equal
to four decimal places. The aggregated fixed-model out-of-fold probabilities tell a slightly different but
non-selecting story—full AP is 0.7137 versus compact AP 0.7134—because fold sizes and prevalences differ. The experiment
uses the mean of the five equally weighted chronological fold scores, consistent with the project protocol, so compact
is selected before 2023 is examined.

The outside-year result confirms that there is no ranking or probability gain. On 2023, full W60 has AP 0.7472 versus
0.7473 for compact, ROC AUC 0.8508 versus 0.8511, and Brier score 0.1101 versus 0.1100. Full W60's independently selected
0.31 threshold produces F1 0.6428 versus 0.6411 for compact, but MCC is slightly lower, 0.5570 versus 0.5579, and the
threshold-dependent difference cannot override the training-only feature selection. Full W60 is also fractionally
lower after `NOT_ARRIVED` is excluded and where rotation is unavailable or excluded; already-arrived AP rounds to
0.6568 for both variants.

The five additional fields receive nonzero fitted importance in the rejected full model, led by nonnegative mean delay
at 0.8738. Scheduled count, delay rate, total delay minutes, and delayed-departure count have importance 0.4464, 0.4364,
0.3482, and 0.1117. Their presence does not improve held-out ranking, which is consistent with their substantial
algebraic overlap with pending count, completed count, and mean signed delay. Experiment 03 therefore retains CatBoost
02's compact W60 manifest and does not replace the preferred retrospective Model 1A candidate. The negative ablation is
useful evidence that simply adding every available backlog summary is not beneficial. The final-tail assignment caveat
and untouched 2024 final-test status remain unchanged.

### Current Model 1A neural-network comparison

Multilayer Perceptron Experiment 02 uses the same 36 source fields, target rows, 24-hour rotation mask, 2019 temporal
folds, and 2023 validation population as Logistic Regression Experiment 08 and CatBoost Experiment 02. Unlike CatBoost,
the Keras model receives 202 prepared inputs after one-hot encoding six categorical fields and applying fold-local
median imputation, missing-value indicators, and standardization to 30 numeric fields. It retains every target row.
The bounded four-configuration search compares two network sizes and two learning rates. The smaller `(64, 32)` network
wins with mean 2019 temporal-validation average precision 0.6963. It uses ReLU activations, batch normalization,
dropout 0.25/0.20, L2 regularization 0.0001, Adam learning rate 0.001, batch size 512, and 16 epochs, the median
fold-best count. The final network has 15,489 trainable parameters.

The MLP confirms the nonlinear improvement. Relative to Logistic Regression Experiment 08, 2023 average precision
rises by 0.0352 to 0.7405, ROC AUC rises by 0.0179 to 0.8446, and Brier score falls by 0.0080 to 0.1112. At its 0.30
training-selected threshold, F1 reaches 0.6396 and MCC 0.5495, improvements of 0.0302 and 0.0501 over logistic
regression at its independently selected threshold. At the default threshold, the MLP has 0.8794 precision, 0.4454
recall, 0.5913 F1, and 0.5581 MCC.

CatBoost Experiment 02 remains slightly stronger. Its 2023 average precision is 0.0068 higher, ROC AUC 0.0065 higher,
and Brier score 0.0012 lower. Its training-selected F1 is 0.0015 higher and MCC 0.0084 higher. The same ordering appears
in the rotation diagnostics: after `NOT_ARRIVED` rows are excluded, MLP average precision is 0.6649 versus CatBoost's
0.6732; among already-arrived rotations it is 0.6495 versus 0.6568; where rotation is unavailable or excluded it is
0.6730 versus 0.6884. These are small but consistent differences, so CatBoost remains the preferred completed
retrospective Model 1A candidate while the MLP provides a strong independent confirmation.

Grouped permutation diagnostics on a fixed 20,000-row 2023 sample identify log actual turn time as the dominant MLP
field, followed by log inbound overdue time, log scheduled turn time, airline, and inbound arrival delay. W60 pending
count ranks tenth, W60 mean signed delay fourteenth, and W60 completed count twentieth. All three backlog fields reduce
average precision when shuffled, supporting their complementary value after rotation information is present. These
permutation values describe predictive use rather than causation. The result does not yet justify a more complicated
categorical-embedding network: the small MLP already nearly matches CatBoost, the deeper candidate did not improve 2019
temporal average precision, and CatBoost remains easier to interpret and slightly better on every principal 2023
ranking and probability measure.

As with the rotation logistic-regression and CatBoost experiments, the final-tail assignment limitation remains. The
MLP result is a retrospective upper bound until the aircraft assignment known at the prediction cutoff can be verified.
The 2024 final-test data remains untouched.

### Current Model 1A same-airline CatBoost comparison

CatBoost Experiment 04 tests whether airline-specific operational pressure adds information after the model already
knows aircraft rotation and airport-wide pressure. It pairs three established files in memory after exact row-identity
validation: full-history rotation, airport-wide W60 backlog, and same-airline W60 backlog. The 36-field control exactly
reproduces CatBoost Experiment 02. The 41-field expanded variant adds same-airline pending count, completed count, mean
signed delay, delay rate, and pending share. Both use the fixed 683-tree CatBoost configuration, five chronological 2019
folds, the 24-hour rotation mask, identical target rows, and separately selected training-only thresholds.

The expanded variant wins the predeclared 2019 selection measure, although by a small margin. Mean temporal AP rises
from 0.7010 to 0.7018, mean ROC AUC from 0.8442 to 0.8448, and mean Brier score improves from 0.0917 to 0.0916. It has
higher AP in folds 1, 2, and 5 and lower AP in folds 3 and 4. Its aggregated fixed-model OOF AP is 0.7142 versus 0.7134
for the control, and its training-selected threshold is 0.31 rather than 0.32. The modest training margin warrants
caution, but all three training summaries move in the favorable direction.

The improvement becomes clearer on the untouched 2023 development rows. Average precision rises by 0.0053 to 0.7526,
ROC AUC by 0.0035 to 0.8546, and Brier score improves by 0.0014 to 0.1086. At the independently selected thresholds, F1
rises from 0.6411 to 0.6511 and MCC from 0.5579 to 0.5640. At the default threshold, recall rises from 0.4467 to 0.4567,
F1 from 0.5925 to 0.6010, and MCC from 0.5592 to 0.5657, with precision essentially preserved at 0.8785.

The gain is present in every diagnostic subgroup. Excluding the 3,565 `NOT_ARRIVED` rows, AP rises from 0.6732 to
0.6801; among already-arrived rotations it rises from 0.6568 to 0.6633. The largest gain occurs where rotation is
unavailable or excluded: AP rises from 0.6884 to 0.6995, ROC AUC from 0.7628 to 0.7711, and Brier score improves from
0.1912 to 0.1871. This supports the intended interpretation that same-airline state supplies complementary operational
context rather than merely restating the nearly deterministic late-aircraft condition.

All five added fields receive nonzero fitted importance. Same-airline mean signed delay ranks nineteenth overall with
importance 1.3604, pending share twentieth at 1.3264, and pending count twenty-third at 1.1943. Delay rate and completed
count are smaller at 0.2857 and 0.2143. Airport-wide pending count and mean delay remain more important than their
same-airline counterparts, indicating that the scopes are complementary rather than interchangeable.

CatBoost Experiment 04 therefore becomes the preferred completed retrospective Model 1A candidate. Its training gain
is small enough that the feature family should not be expanded casually, but its consistent ranking, calibration,
threshold, and subgroup improvements justify retaining the five-field same-airline extension. No permanent combined
dataset is created. Deployment requires the same timely schedule and gate-out feed as the airport-wide backlog, plus
the already-known reporting-airline code. The final-tail assignment caveat remains, and 2024 remains untouched.

### Current Model 1A CatBoost/MLP blend comparison

Ensemble Experiment 01 tests whether the selected CatBoost and neural-network candidates make sufficiently different
errors for a simple probability blend to improve Model 1A. It regenerates both components on the same five
chronological 2019 folds and identical rows. The CatBoost component uses Experiment 04's 41-field manifest and fixed
683-tree configuration; the MLP uses Experiment 02's 36-field manifest and fixed 16-epoch `(64, 32)` configuration.
The experiment compares five predeclared weighted arithmetic means ranging from pure CatBoost to pure MLP. Component
models are not retuned, and the 2023 development data plays no role in choosing the weight or operating threshold.

Training selects 0.75 CatBoost plus 0.25 MLP. Its mean fold AP is 0.7047, compared with 0.7018 for pure CatBoost and
0.6934 for the fixed-epoch MLP regenerated on these aligned folds. The blend improves on CatBoost in four of five
folds; its aggregate OOF AP is 0.7174 versus 0.7142, ROC AUC is 0.8511 versus 0.8490, and Brier score improves from
0.0916 to 0.0911. The fixed-epoch MLP's aligned-fold figure differs from its original 0.6963 model-selection result
because the earlier value came from fold-local early-stopped search fits. The ensemble uses the already selected
16-epoch configuration so that the component definition is reproducible and fixed before external validation.

The 2023 gain is negligible. Relative to CatBoost Experiment 04, AP rises only 0.0006 to 0.7532 and Brier score improves
only 0.0001 to 0.1085, while ROC AUC declines 0.0002 to 0.8544. At the common training-selected threshold of 0.31, F1
rises from 0.6511 to 0.6514 and MCC from 0.5640 to 0.5655. At the default threshold, MCC rises from 0.5657 to 0.5671,
but F1 falls from 0.6010 to 0.6003. These movements are too small to constitute a material outside-year improvement.

The subgroup evidence is also mixed. The blend raises AP by 0.0010 when `NOT_ARRIVED` rows are excluded and by 0.0014
among already-arrived rotations, with essentially unchanged ROC AUC and Brier score. Where rotation is unavailable or
excluded, however, AP falls by 0.0027, ROC AUC by 0.0018, and Brier score worsens by 0.0018. CatBoost and MLP
probabilities are highly correlated: 0.9570 on aligned 2019 OOF rows and 0.9718 in 2023. The limited error diversity
explains why blending produces little additional information.

The blend is retained as a completed negative-or-neutral result, but it does not replace CatBoost Experiment 04 as the
preferred Model 1A candidate. Running two preprocessing pipelines and two model runtimes, including TensorFlow, is not
justified by a 0.0006 AP gain with a simultaneous ROC AUC decline. Calibration Experiment 01 therefore starts with
CatBoost 04 and retains its unchanged probabilities after neither correction improves training-only reliability. No
combined feature or prediction dataset is written, the final-tail assignment caveat remains, and 2024 remains
untouched.

### Current Model 1A CatBoost calibration comparison

Calibration Experiment 01 keeps CatBoost 04's classifier and 41-field manifest fixed and asks whether its probabilities
need a post-model correction. It compares unchanged probabilities with sigmoid and isotonic mappings using only 2019
held-out predictions. Fold 1 supplies the first independent calibration sample; for assessment folds 2–5, each
calibrator is fitted only to earlier held-out folds. Mean forward Brier score is the primary selection measure, with
log loss and ten-bin expected calibration error as supporting diagnostics.

The uncalibrated probabilities win all three training-only reliability measures. Their mean Brier score is 0.0918,
compared with 0.0922 for sigmoid and 0.0923 for isotonic. Mean log loss is 0.3124 uncalibrated, 0.3137 sigmoid, and
0.3152 isotonic; mean expected calibration error is 0.0125, 0.0156, and 0.0160 respectively. Sigmoid preserves AP and
ROC AUC, as expected from a monotonic correction, but does not improve reliability. Isotonic also reduces mean AP from
0.7117 to 0.7018 because its stepwise mapping introduces probability ties.

Following the predeclared protocol, the rejected corrections are not carried into 2023. The selected result is therefore
exactly CatBoost 04: AP 0.7526, ROC AUC 0.8546, Brier score 0.1086, log loss 0.3567, and expected calibration error
0.0282. Its training-selected threshold remains 0.31, with 2023 F1 0.6511 and MCC 0.5640. This negative result is useful:
CatBoost 04's native probabilities are already better calibrated across the 2019 temporal shifts than either added
mapping, so another fitted stage would add complexity without evidence of benefit. CatBoost 04 remains unchanged and
preferred; the final-tail assignment caveat remains, and 2024 remains untouched.

### Current Model 1A CatBoost subgroup and SHAP audit

CatBoost Audit 01 freezes Experiment 04's 41-field manifest, 683-tree classifier, and 0.31 training-selected threshold.
The all-2019 refit exactly reproduces the published complete-2023 result: AP 0.7526, ROC AUC 0.8546, Brier score 0.1086,
F1 0.6511, and MCC 0.5640. The notebook makes no selection from these results. Planned-traffic quartiles use boundaries
learned only from 2019—170.25, 195, and 213 summed movements—and all other audit groups use fixed operational rules.

| Audit dimension | Groups | Smallest group | AP range | Recall range at 0.31 | Largest absolute mean-probability gap |
|---|---:|---:|---:|---:|---:|
| Airline | 8 | 362 | 0.5954–0.8115 | 0.2925–0.6432 | 0.1328 |
| Month | 12 | 8,596 | 0.6499–0.8201 | 0.4343–0.6895 | 0.0453 |
| Route | 41 | 588 | 0.5747–0.8341 | 0.3067–0.7259 | 0.1014 |
| Time of day | 4 | 682 | 0.2257–0.8429 | 0.0759–0.7144 | 0.0375 |
| Planned-traffic quartile | 4 | 18,497 | 0.5601–0.8145 | 0.3785–0.6739 | 0.0364 |
| Weather | 4 | 6,510 | 0.7308–0.8196 | 0.5563–0.6842 | 0.0364 |

AP must be read relative to each group's prevalence, and the smallest groups need caution. The clearest coverage gap is
for the 682 departures scheduled from 00:00 through 05:59: prevalence is 0.1158, AP is 0.2257, and recall at 0.31 is
0.0759. The larger 06:00–11:59 group has AP 0.5678 and recall 0.3828, while evening departures reach AP 0.8429 and
recall 0.7144. Performance also rises monotonically across the 2019-defined planned-traffic bands: AP moves from 0.5601
in Q1 to 0.8145 in Q4, and recall moves from 0.3785 to 0.6739. The model underpredicts mean risk in every month and
weather group. July has the largest monthly gap at -0.0453; low visibility has the largest weather gap at -0.0364.
Among large airline populations, JetBlue has a -0.0679 gap on 36,121 rows. The larger gaps for Hawaiian and the pooled
small-route group are descriptive warnings based on smaller or heterogeneous populations, not calibration targets.

CatBoost SHAP values on the fixed 5,000-row 2023 sample reinforce the earlier fitted-importance result. Log actual turn
time and actual turn time have the two largest mean absolute contributions, followed by inbound arrival delay, rotation
status, scheduled departure time, and airport-wide W60 pending count. Airport-wide mean signed delay ranks ninth;
same-airline mean signed delay, pending share, and pending count rank twelfth, fifteenth, and twentieth. All eight
selected backlog fields have nonzero mean absolute SHAP values. The notebook also records the five largest signed
contributions for a representative true negative, false positive, false negative, and true positive selected near the
median probability of each outcome type. SHAP values describe how the frozen model forms a prediction; they do not show
that a field caused a delay.

The audit identifies monitoring priorities rather than a new development round. CatBoost Experiment 04, its native
probabilities, and its 0.31 threshold remain preferred and unchanged. No combined feature or prediction dataset is
written, the final-tail assignment limitation remains, and 2024 remains untouched.

### Model 2A/2B/2C logistic-regression timing comparison

These three experiments use identical 2019 and 2023 arrival rows, the same target, the same five time-ordered training
folds, and the same 16 logistic-regression configurations. Model 2B changes only by adding signed `DepDelay`; Model 2C
then changes only by adding log taxi-out duration and two cyclical actual-takeoff fields. Their differences can therefore
be interpreted as the incremental predictive value available at each operational milestone.

Before pushback, Model 2A provides modest ranking skill: 2023 AP is 0.4019 and ROC AUC is 0.6744. Adding signed gate
delay in Model 2B produces the largest information gain. AP rises by 0.4662 to 0.8682, ROC AUC rises by 0.2382 to 0.9126,
and Brier score falls by 0.0987 to 0.0755. At the training-selected threshold, F1 rises from 0.4425 to 0.7884. The
standardized `DepDelay` coefficient is 6.53 and is much larger than every remaining Model 2B coefficient.

Model 2C adds a further, smaller but still material improvement after takeoff. Relative to Model 2B, AP rises by 0.0408
to 0.9090, ROC AUC rises by 0.0330 to 0.9457, Brier score falls by 0.0138 to 0.0616, and training-selected F1 rises by
0.0339 to 0.8224. In the fitted Model 2C design, standardized coefficient magnitude is 7.92 for `DepDelay` and 1.29 for
`LOG_TAXI_OUT_MINUTES`. The takeoff sine coefficient is only 0.032 and L1 regularization sets the cosine coefficient to
zero. The post-takeoff gain is therefore associated primarily with realized taxi-out duration rather than clock time.

The results support the planned information ladder: schedule, origin congestion, and weather provide a useful early
estimate; actual gate delay dominates once pushback occurs; and realized taxi time supplies an additional update after
takeoff. The operational fields are valid only at their stated prediction times and would require live gate-out and
takeoff feeds in deployment. The 2024 arrivals remain untouched for final testing.
