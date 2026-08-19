# Appendix E

[Back to the main report](README.md)

This appendix lists the experiments for the project's four core models: 1A, 2A, 2B, and 2C, followed by the frozen
2024 final evaluation. Every experiment predicts whether a flight will be at least 15 minutes late. The experiment
roadmap includes the standard classifier families tested directly in the reference papers, but not every planned
experiment was run. A method mentioned only in a paper's review of other research is not automatically included,
especially when it needs data this project does not collect.

## Experiment protocol

All experiments follow the same rules so their results can be compared fairly.

1. **Use consistent flight groups and targets.** Model 1A predicts `DepDel15` for JFK departures. Models 2A, 2B, and 2C
   predict `ArrDel15` for the same JFK arrivals. Cancelled and diverted flights are excluded.
2. **Keep the time order.** Use earlier 2019 dates to train and later 2019 dates to select features and settings. After
   the design is fixed, train on all of 2019 and validate on 2023. Keep 2024 untouched until final model selection.
3. **State the feature allowlist.** Each notebook lists its permitted fields. Raw baselines use available source fields;
   compact engineered experiments use selected features from [Appendix C](Appendix-C.md). Models 2B and 2C add only
   information available after the Model 2A prediction time. [Appendix D](Appendix-D.md) records the exact allowlists.
4. **Learn only from training data.** Fit missing-value handling, category conversion, scaling, feature selection, class
   weights, probability adjustments, and decision thresholds without using validation or test data.
5. **Use the same measures.** Average precision is the main ranking measure. Also report ROC AUC, precision, recall,
   F1, MCC, balanced accuracy, confusion matrix, Brier score, calibration, and accuracy. Report results at both the
   0.50 threshold and the threshold selected from development data.
6. **Limit model searches.** Start with a small documented parameter set and expand only when validation results support
   it. Record training and prediction time. Clearly identify any comparison that uses a fixed training-data sample.
7. **Make results repeatable.** Use fixed random seeds and save the time folds and selected features. Add one results row
   per run. Name notebooks `<classifier>_<model>_<experiment>.ipynb`, and never reuse an experiment number.
8. **Explain and review the results.** Report feature importance and individual examples for selected linear and tree
   models. Use SHAP when supported, but treat findings as relationships rather than causes. Check results by month, time
   of day, airline, route, weather, and planned traffic level.

## Primary binary-classification experiments

The table includes Model 1A departure experiments and Model 2A, 2B, and 2C arrival experiments. The Model 1A feature set
evolved separately and extends well beyond the features used by the arrival models. Each description identifies what
changed in that experiment.

The roadmap contains 55 experiments. **Completed** means the notebook was run and its result is reported in
[Appendix F](Appendix-F.md). **Not attempted** means the planned notebook was not run and no result is reported. The
reference papers informed the roadmap, but time did not permit every reference-derived experiment to be attempted.

| Experiment group | Completed | Not attempted | Total |
|---|---:|---:|---:|
| Departures | 22 | 10 | 32 |
| Arrivals | 17 | 6 | 23 |
| **Total** | **39** | **16** | **55** |

| Model | Classifier | Experiment | Planned notebook | Status | Description |
|---|---|---:|---|---|---|
| 1A | Logistic regression | 01 | `logistic_regression_1a_01.ipynb` | Completed | Fit the raw schedule, airline, route, traffic, and weather baseline. |
| 1A | Logistic regression | 02 | `logistic_regression_1a_02.ipynb` | Completed | Test compact engineered features with tuned regularization and class weights. |
| 1A | Logistic regression | 03 | `logistic_regression_1a_03.ipynb` | Completed | Compare all safe pre-pushback fields with L1-ranked subsets of 50, 100, and 200. |
| 1A | Logistic regression | 04 | `logistic_regression_1a_04.ipynb` | Completed | Add 30-minute airport backlog fields to the raw baseline. |
| 1A | Logistic regression | 05 | `logistic_regression_1a_05.ipynb` | Completed | Add rotation fields and compare compact, full, and L1-ranked versions. |
| 1A | Logistic regression | 06 | `logistic_regression_1a_06.ipynb` | Completed | Compare limited- and full-history rotation, component ablations, and a 24-hour mask. |
| 1A | Logistic regression | 07 | `logistic_regression_1a_07.ipynb` | Completed | Add three compact 30-minute backlog fields to the full-history rotation model. |
| 1A | Logistic regression | 08 | `logistic_regression_1a_08.ipynb` | Completed | Replace the 30-minute backlog with the matching 60-minute fields. |
| 1A | Decision tree | 01 | `decision_tree_1a_01.ipynb` | Completed | Fit the original single-tree baseline. |
| 1A | Decision tree | 02 | `decision_tree_1a_02.ipynb` | Completed | Compare source and compact engineered features while tuning the tree. |
| 1A | Decision tree | 03 | `decision_tree_1a_03.ipynb` | Not attempted | Test a pruned scikit-learn equivalent of RepTree. |
| 1A | K-nearest neighbors | 01 | `knn_1a_01.ipynb` | Not attempted | Tune KNN on compact scaled and encoded features. |
| 1A | Gaussian Naive Bayes | 01 | `naive_bayes_1a_01.ipynb` | Not attempted | Tune Gaussian Naive Bayes on compact encoded features. |
| 1A | RBF support-vector classifier | 01 | `svc_rbf_1a_01.ipynb` | Not attempted | Tune an RBF SVM on scaled inputs. |
| 1A | Linear discriminant analysis | 01 | `lda_1a_01.ipynb` | Completed | Test LDA solvers and shrinkage using CatBoost 04's 41-field allowlist. |
| 1A | Bagging classifier | 01 | `bagging_1a_01.ipynb` | Not attempted | Tune bagged trees and compare them with a single tree. |
| 1A | Random forest | 01 | `random_forest_1a_01.ipynb` | Completed | Tune Random Forest on the compact tree feature set. |
| 1A | Random forest | 02 | `random_forest_1a_02.ipynb` | Completed | Tune Random Forest on CatBoost 04's 41-field allowlist and compare the classifiers. |
| 1A | Extra Trees | 01 | `extra_trees_1a_01.ipynb` | Not attempted | Compare Extra Trees with Random Forest on the same features. |
| 1A | AdaBoost | 01 | `adaboost_1a_01.ipynb` | Not attempted | Tune AdaBoost with small decision trees. |
| 1A | Gradient boosting | 01 | `gradient_boosting_1a_01.ipynb` | Not attempted | Tune gradient boosting on the compact feature set. |
| 1A | CatBoost | 01 | `catboost_1a_01.ipynb` | Completed | Test CatBoost's direct handling of categorical fields. |
| 1A | CatBoost | 02 | `catboost_1a_02.ipynb` | Completed | Combine raw, full-history rotation, and compact 60-minute backlog fields. |
| 1A | CatBoost | 03 | `catboost_1a_03.ipynb` | Completed | Compare compact and full eight-field 60-minute backlog versions. |
| 1A | CatBoost | 04 | `catboost_1a_04.ipynb` | Completed | Add five same-airline backlog fields to the airport-wide backlog and rotation fields. |
| 1A | CatBoost | 05 | `catboost_1a_05.ipynb` | Completed | Test schedule-cycle and compact-weather additions, followed by a limited capacity search. |
| 1A | Multilayer perceptron | 01 | `mlp_1a_01.ipynb` | Not attempted | Tune a small regularized neural network. |
| 1A | Multilayer perceptron | 02 | `mlp_1a_02.ipynb` | Completed | Compare two MLP sizes and two learning rates on the 36-field rotation/backlog allowlist. |
| 1A | CatBoost / MLP blend | 01 | `ensemble_1a_01.ipynb` | Completed | Select a weighted blend of CatBoost 04 and MLP 02 using 2019 predictions. |
| 1A | LR / DT / RF imbalance study | 01 | `imbalance_1a_01.ipynb` | Not attempted | Considered but not attempted. |
| 1A | CatBoost calibration | 01 | `calibration_1a_01.ipynb` | Completed | Compare uncalibrated, sigmoid, and isotonic CatBoost 04 probabilities. |
| 1A | CatBoost audit | 01 | `catboost_audit_1a_01.ipynb` | Completed | Audit CatBoost 04 by key groups and explain a 5,000-row sample with SHAP. |
| 2A | Logistic regression | 01 | `logistic_regression_2a_01.ipynb` | Completed | Fit compact pre-pushback schedule, route, traffic, and weather features. |
| 2B | Logistic regression | 01 | `logistic_regression_2b_01.ipynb` | Completed | Add signed `DepDelay` to the Model 2A features. |
| 2B | Logistic regression | 02 | `logistic_regression_2b_02.ipynb` | Completed | Compare linear `DepDelay`, delay and margin splines, and departure-time cycles. |
| 2B | Linear discriminant analysis | 01 | `lda_2b_01.ipynb` | Completed | Test LDA solvers and shrinkage using the selected Model 2B-02 representation. |
| 2C | Logistic regression | 01 | `logistic_regression_2c_01.ipynb` | Completed | Add `TaxiOut` and takeoff-time features to the Model 2B base. |
| 2C | Logistic regression | 02 | `logistic_regression_2c_02.ipynb` | Completed | Compare the control, raw `TaxiOut`, and a schedule-margin spline. |
| 2C | Logistic regression | 03 | `logistic_regression_2c_03.ipynb` | Completed | Compare five L2 strengths with and without balanced class weights. |
| 2C | Gaussian Naive Bayes | 01 | `gaussian_naive_bayes_2c_01.ipynb` | Completed | Tune variance smoothing using the selected Model 2C-02 representation. |
| 2C | Support vector machine | 01 | `svm_2c_01.ipynb` | Completed | Compare a linear SVM with a 256-component approximate RBF model. |
| 2C | Linear discriminant analysis | 01 | `lda_2c_01.ipynb` | Completed | Test LDA solvers and shrinkage using the selected Model 2C-02 representation. |
| 2A | Decision tree | 01 | `decision_tree_2a_01.ipynb` | Not attempted | Apply a tuned single tree to the pre-pushback features. |
| 2B | Decision tree | 01 | `decision_tree_2b_01.ipynb` | Not attempted | Add signed `DepDelay` to the Model 2A tree. |
| 2C | Decision tree | 01 | `decision_tree_2c_01.ipynb` | Not attempted | Add taxi-out and takeoff fields to the Model 2B tree. |
| 2A | Random forest | 01 | `random_forest_2a_01.ipynb` | Not attempted | Apply Random Forest to the pre-pushback features. |
| 2B | Random forest | 01 | `random_forest_2b_01.ipynb` | Not attempted | Add signed `DepDelay` to the Model 2A forest. |
| 2C | Random forest | 01 | `random_forest_2c_01.ipynb` | Not attempted | Add taxi-out and takeoff fields to the Model 2B forest. |
| 2A | CatBoost | 01 | `catboost_2a_01.ipynb` | Completed | Tune CatBoost on the exact BL-A-27 allowlist. |
| 2A | CatBoost | 02 | `catboost_2a_02.ipynb` | Completed | Test calendar fields and a model-side `AIRLINE_ORIGIN` field. |
| 2B | CatBoost | 01 | `catboost_2b_01.ipynb` | Completed | Tune CatBoost on the selected 28-field pushback-margin representation. |
| 2C | CatBoost | 01 | `catboost_2c_01.ipynb` | Completed | Add taxi-out and takeoff fields to the Model 2B CatBoost base. |
| 2C | CatBoost | 02 | `catboost_2c_02.ipynb` | Completed | Compare the control with raw `TaxiOut` and a numeric schedule margin. |
| 2C | Multilayer perceptron | 01 | `mlp_2c_01.ipynb` | Completed | Compare three network sizes and two learning rates using the Model 2C-02 representation. |
| 2C | CatBoost / MLP blend | 01 | `ensemble_2c_01.ipynb` | Completed | Select a weighted blend of CatBoost 2C-02 and MLP 2C-01 using 2019 predictions. |

## Final 2024 evaluation

The final evaluation uses one frozen design at each distinct prediction time. The selected classifier, allowlist,
model-side transformations, settings, and operating threshold are copied from the named experiment. Each model is
refit on all 2019 rows and evaluated on 2024. No 2024 result is used to choose or revise a model.

| Model | Prediction time | Frozen design | Frozen threshold | Final notebook |
|---|---|---|---:|---|
| 1A | Before pushback for a JFK departure | CatBoost 04 | 0.31 | [final_evaluation_2024_01.ipynb](models/final_evaluation_2024_01.ipynb) |
| 2A | Before pushback for a JFK-bound flight | Logistic Regression 01 | 0.22 | [final_evaluation_2024_01.ipynb](models/final_evaluation_2024_01.ipynb) |
| 2B | At pushback | Logistic Regression 02 | 0.39 | [final_evaluation_2024_01.ipynb](models/final_evaluation_2024_01.ipynb) |
| 2C | After takeoff | Logistic Regression 02 | 0.45 | [final_evaluation_2024_01.ipynb](models/final_evaluation_2024_01.ipynb) |

## Post-test combined-training sensitivity

These append-only experiments were added after the official 2024 results were examined. Each one copies a frozen
selected design and changes only the training population from 2019 to combined 2019 and 2023 data. The existing
allowlist, model-side transformations, classifier settings, and operating threshold remain fixed. The experiments do
not replace the official final evaluation or reopen model selection.

| Model | Experiment | Notebook | Combined training rows | Fixed threshold | Controlled change |
|---|---:|---|---:|---:|---|
| 1A | CatBoost 06 | [catboost_1a_06.ipynb](models/catboost_1a_06.ipynb) | 217,413 | 0.31 | Training population only |
| 2A | Logistic Regression 02 | [logistic_regression_2a_02.ipynb](models/logistic_regression_2a_02.ipynb) | 217,301 | 0.22 | Training population only |
| 2B | Logistic Regression 03 | [logistic_regression_2b_03.ipynb](models/logistic_regression_2b_03.ipynb) | 217,301 | 0.39 | Training population only |
| 2C | Logistic Regression 04 | [logistic_regression_2c_04.ipynb](models/logistic_regression_2c_04.ipynb) | 217,301 | 0.45 | Training population only |

## Reference models and project scope

This table explains why methods were placed on the roadmap. It does not mean every listed method was run. The
**Status** column in the experiment table above is the execution record.

| Reference | Models or methods tested in the paper | How this project uses the paper |
|---|---|---|
| [Snell et al.](resources/docs/02_Snell_MLFlightDelayPrediction.pdf) | Logistic regression, KNN, bagging, decision tree, RepTree, random forest, neural network, SVM, and SMOTE. | These classifier families informed the Model 1A experiments. A cost-complexity-pruned tree was planned as the scikit-learn equivalent of RepTree. Although Snell reports scenarios using delay fields, `DepDel15` and `DepDelay` are not allowed as inputs to Model 1A or 2A. |
| [Zoutendijk and Mitici](resources/docs/03_Zoutendijk_ProbabilisticFlightDelay.pdf) | Airline, airport, weather, schedule, and traffic inputs; evaluation of flight-specific delay uncertainty. | The paper informs the feature design and the project's checks of predicted-probability quality. The core experiments still predict the binary 15-minute target. |
| [Li](resources/docs/04_Li_DelayPropagationPrediction.pdf) | Random forest, random-forest recursive feature elimination, SMOTE, and chained delay-propagation variants. | Random Forest and training-only RFECV informed the roadmap. A completed Model 1A extension tests one causally masked preceding inbound leg. Longer aircraft sequences and network-wide delay spread remain outside scope. |
| [AlBassam](resources/docs/05_AlBassam_MLDelayEval.pdf) | Decision tree, random forest, SVC, logistic regression, KNN, and Naive Bayes with random over-sampling, SMOTE, and ADASYN. | The six classifier families informed the Model 1A experiments; not all were attempted. The class-balancing methods were reviewed but not pursued. Actual arrival, delay-cause, and previous-flight fields are not used because they are unavailable at the project's early prediction times or directly describe an outcome. |
| [Chen and Li](resources/docs/06_Chen_ChainedDelayPrediction.pdf) | Random forest, recursive feature elimination, SMOTE, and chained delay-propagation variants. | Random Forest and RFECV informed the roadmap. The completed rotation extension follows only the immediately preceding known inbound; recursive or network-wide propagation remains outside scope. |
| [Pineda-Jaramillo et al.](resources/docs/15_Pineda_ExplainableDelayML.pdf) | Logistic regression, decision tree, Naive Bayes, KNN, RBF SVM, LDA, AdaBoost, Extra Trees, random forest, and gradient boosting; SMOTE-ENN; SHAP and Sobol explanations. | The ten classifiers informed the Model 1A experiments; not all were attempted. The class-balancing method was reviewed but not pursued, and SHAP was completed for the selected Model 1A CatBoost model. Weather observed at destination landing or origin takeoff is not used for 1A or 2A because it is not available at prediction time. |
| [Beltman et al.](resources/docs/16_Beltman_DepartureDelayForecast.pdf) | CatBoost and neural-network methods at several pre-departure prediction times. | CatBoost was tested as a core classifier. The paper's changing 90-to-15-minute prediction times are not reproduced because the current annual sources do not provide equivalent rolling operational snapshots. |

The excluded longer aircraft-chain and changing-time-horizon designs may still be useful, but they need information
this project does not collect or does not allow at the relevant prediction time. They should be reconsidered only if
the project formally adds those data sources and updates its prediction-time rules.
