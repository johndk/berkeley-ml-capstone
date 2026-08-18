# Appendix E

[Back to the main report](README.md)

This appendix lists the planned experiments for the project's four core models: 1A, 2A, 2B, and 2C. Every experiment
predicts whether a flight will be at least 15 minutes late. The plan includes the standard classifier families tested
directly in the reference papers. A method mentioned only in a paper's review of other research is not automatically
included, especially when it needs data this project does not collect.

Model 1A is used to compare the broad set of classifiers. The strongest and most useful approaches are then applied to
the three arrival prediction times. This keeps the plan manageable and avoids testing every classifier and feature set
at every prediction time. If another classifier clearly performs better than the main set on Model 1A, matching 2A,
2B, and 2C experiments should be added using the next available notebook numbers.

## Experiment protocol

All experiments follow the same rules. This makes it more likely that a change in results comes from the classifier,
feature set, or prediction time—not from using a different sample or evaluation process.

1. **Targets and flight groups.** Model 1A uses JFK departures and target `DepDel15`. Models 2A, 2B, and 2C use the same
   JFK arrivals and target `ArrDel15`. Only the information available at each arrival prediction time changes.
   Cancelled and diverted flights are not included.
2. **Keep the time order.** Use complete days and move forward through 2019 when selecting features and model settings:
   earlier periods train the model and later periods check it. After the experiment design is fixed, train on all of
   2019 and use 2023 as an outside check. Keep 2024 untouched until the final model and decision threshold are selected.
3. **Use clear feature lists.** Each notebook must list the fields the model is allowed to use. A *raw baseline* uses
   available source fields with only the preparation required by the classifier. A *compact engineered* experiment
   uses a smaller, nonduplicative set of calculated features from [Appendix C](Appendix-C.md). Models 2B and 2C must begin with the same
   pre-pushback fields as Model 2A before adding information that becomes available later. The exact implemented
   allowlists and their experiment-to-experiment changes are recorded in [Appendix D](Appendix-D.md).
4. **Learn from training data only.** Filling missing values, converting categories, scaling, choosing features,
   balancing classes, adjusting probabilities, and choosing a threshold must all use training folds only. Use
   `SMOTENC` when synthetic sampling includes categorical fields; ordinary SMOTE must not create fractional one-hot
   categories. Never resample validation or test data.
5. **Compare models with the same measures.** Average precision is the main ranking measure because delayed flights are
   the smaller class. Also report ROC AUC, precision, recall, F1, MCC, balanced accuracy, the confusion matrix, Brier
   score, and a calibration curve. Ordinary accuracy is supporting information. Choose the decision threshold from
   development data and report results at both 0.50 and the selected threshold.
6. **Keep model searches practical.** Start with a small, documented set of parameter choices and expand it only when
   validation results support more work. Record training and prediction time. Initial KNN, RBF SVM, and neural-network
   checks may use a fixed sample drawn only from the training data, but any reported comparison must say so.
7. **Make results repeatable.** Use fixed random seeds, save the time-fold definitions and selected feature names, and
   add one summary row per run to the shared results table. Notebook names follow
   `<classifier>_<model>_<experiment>.ipynb`. Do not reuse an experiment number after results have been reported.
8. **Explain results and check important groups.** For the selected linear and tree models, report overall feature
   importance and explanations for individual flights. Use SHAP where the model supports it, and describe relationships
   rather than claiming that a feature caused a delay. Check errors and probability quality by month, time of day,
   airline, route, weather, and planned traffic level.

## Primary binary-classification experiments

Stage I rebuilds the two starting baselines under the common rules and compares the classifier families tested directly
by Snell, AlBassam, or Pineda-Jaramillo et al. Stage II applies the project's four main classifier families to every
arrival model. Each description identifies what changes in that experiment; every row still follows the timing and
training-data rules above.

| Stage | Model | Classifier | Experiment | Planned notebook | Summary | Description |
|---|---|---|---:|---|---|---|
| I | 1A | Logistic regression | 01 | `logistic_regression_1a_01.ipynb` | Raw linear baseline | Rebuild the original baseline from schedule, airline, route, planned-traffic, and weather fields. Apply regularization, scaling, and the common time split. |
| I | 1A | Logistic regression | 02 | `logistic_regression_1a_02.ipynb` | Compact engineered linear model | Replace duplicate time and calendar fields with a selected set of cyclical, traffic, route, and weather features. Tune regularization and class weighting. |
| I | 1A | Logistic regression | 03 | `logistic_regression_1a_03.ipynb` | Broad engineered feature model | Start with all safe pre-pushback features and eligible source fields. Within each training fold, compare the top 50, 100, and 200 L1-ranked fields with using all nonconstant encoded fields. |
| I | 1A | Logistic regression | 04 | `logistic_regression_1a_04.ipynb` | Causal 30-minute departure backlog | Retain the Experiment 01 raw baseline and add operational state reconstructed at scheduled departure. Within the 2019 time folds, compare a compact pending/completed/recent-delay set with all eight backlog fields, then test the selected configuration on 2023. Historical BTS reconstructs these values; deployment would require a live gate-out event feed. |
| I | 1A | Logistic regression | 05 | `logistic_regression_1a_05.ipynb` | Causal aircraft rotation | Retain the Experiment 01 raw baseline and add rotation state from `JFK_YEAR_departures_rotation.csv`. Within the 2019 temporal folds, compare a compact five-field set with all 13 rotation fields and compare passthrough with L1-ranked top-50, top-100, and top-200 prepared columns. Validate the selected pipeline on 2023 and report both overall results and a diagnostic excluding aircraft not arrived by cutoff. Treat performance as a retrospective upper bound unless prediction-time tail assignment can be verified. |
| I | 1A | Logistic regression | 06 | `logistic_regression_1a_06.ipynb` | Full-history rotation sensitivity and ablation | Hold Experiment 05's selected classifier fixed while comparing cohort-limited and full raw BTS movement history, raw baseline, schedule-only, state-only, and all rotation fields. Predeclare a 24-hour long-turn masking sensitivity, preserve every target row, select thresholds only from 2019 temporal predictions, and validate each configuration on 2023. |
| I | 1A | Logistic regression | 07 | `logistic_regression_1a_07.ipynb` | Combined rotation and compact backlog | Hold Experiment 06's classifier, full-history rotation fields, and 24-hour mask fixed. Join the three compact 30-minute backlog fields selected in Experiment 04 only after strict row-identity validation, then compare rotation-only with rotation-plus-backlog using 2019 temporal folds and untouched 2023 validation. Do not create or replace a combined feature dataset. |
| I | 1A | Logistic regression | 08 | `logistic_regression_1a_08.ipynb` | Combined rotation and 60-minute backlog | Repeat Experiment 07 with the separately generated W60 pending count, completed count, and mean signed recent departure delay. Hold the classifier, full-history rotation fields, 24-hour mask, target rows, time folds, and validation year fixed so the effect of the broader backlog window can be compared with W30. |
| I | 1A | Decision tree | 01 | `decision_tree_1a_01.ipynb` | Raw tree baseline | Rebuild the original single-tree baseline using the common time folds and measures. |
| I | 1A | Decision tree | 02 | `decision_tree_1a_02.ipynb` | Compact engineered tree | Compare eligible source fields with the smaller calculated feature set for tree models. Tune tree depth, leaf size, split method, and class weighting. |
| I | 1A | Decision tree | 03 | `decision_tree_1a_03.ipynb` | Pruned tree / RepTree equivalent | Limit and prune the tree to provide a repeatable scikit-learn equivalent of Snell's reduced-error-pruned RepTree. |
| I | 1A | K-nearest neighbors | 01 | `knn_1a_01.ipynb` | Scaled nearest-neighbor model | Use the compact numeric and encoded feature set. Tune the number of neighbors, their weighting, and the distance measure; record memory use and prediction time. |
| I | 1A | Gaussian Naive Bayes | 01 | `naive_bayes_1a_01.ipynb` | Naive Bayes baseline | Use a compact encoded feature set, tune variance smoothing, and check the quality of both rankings and predicted probabilities. |
| I | 1A | RBF support-vector classifier | 01 | `svc_rbf_1a_01.ipynb` | Non-linear SVM | Scale numeric inputs, tune `C` and `gamma`, use class weights, and produce probability estimates within the training process. |
| I | 1A | Linear discriminant analysis | 01 | `lda_1a_01.ipynb` | Linear discriminant model | Compare supported LDA solvers and shrinkage settings on a compact, scaled feature set. Check that the encoded inputs remain numerically stable. |
| I | 1A | Bagging classifier | 01 | `bagging_1a_01.ipynb` | Bagged tree model | Tune the number of trees, the rows and features sampled, and the size of each tree. Compare the result with a single decision tree. |
| I | 1A | Random forest | 01 | `random_forest_1a_01.ipynb` | Compact engineered forest | Use the compact tree feature set and tune the number and size of trees, features sampled, and class weighting. |
| I | 1A | Random forest | 02 | `random_forest_1a_02.ipynb` | Rotation and dual-scope W60 backlog | Give Random Forest CatBoost 04's exact 41-field manifest and 24-hour rotation mask. Tune depth, leaf regularization, feature sampling, and class weighting with 2019 temporal folds, select the threshold using 2019 only, and validate once on 2023. Compare directly with CatBoost 04 to separate feature value from classifier-family value. |
| I | 1A | Extra Trees | 01 | `extra_trees_1a_01.ipynb` | More-randomized tree ensemble | Compare Extra Trees with Random Forest using the same tree feature set and a similar amount of model tuning. |
| I | 1A | AdaBoost | 01 | `adaboost_1a_01.ipynb` | AdaBoost model | Tune the number of models, learning rate, and size of the small base trees. Check sensitivity to unusual or possibly mislabeled delays. |
| I | 1A | Gradient boosting | 01 | `gradient_boosting_1a_01.ipynb` | Gradient-boosted trees | Tune the number of trees, learning rate, tree depth, and row sampling on the compact feature set. |
| I | 1A | CatBoost | 01 | `catboost_1a_01.ipynb` | CatBoost with categorical fields | Let CatBoost handle eligible categorical fields directly, use class weighting, and stop training based on time-ordered validation. Compare it with one-hot-encoded tree models. |
| I | 1A | CatBoost | 02 | `catboost_1a_02.ipynb` | Full-history rotation and 60-minute backlog | Apply CatBoost to the exact source-feature hypothesis selected by Logistic Regression Experiment 08: the 20-field raw base, all 13 full-history rotation fields with the 24-hour long-turn mask, and the three compact W60 backlog fields. Tune with 2019 temporal folds, validate once on 2023, keep 2024 untouched, and report the final-tail assignment limitation explicitly. Pair the existing rotation and backlog datasets in memory rather than creating a combined feature file. |
| I | 1A | CatBoost | 03 | `catboost_1a_03.ipynb` | Full eight-field W60 backlog ablation | Hold CatBoost Experiment 02's classifier, rows, full-history rotation fields, 24-hour mask, and temporal folds fixed. Compare its three compact W60 fields with all eight existing W60 fields, select the manifest and thresholds from 2019 only, and compare both variants once on 2023. Do not create new backlog data or a combined feature dataset. |
| I | 1A | CatBoost | 04 | `catboost_1a_04.ipynb` | Airport-wide and same-airline W60 backlog | Hold CatBoost Experiment 02's classifier, rows, rotation mask, and compact airport-wide W60 fields fixed. Compare the exact control with a 41-field variant adding same-airline W60 pending count, completed count, mean signed delay, delay rate, and pending share. Select the manifest and thresholds from 2019 only, then compare both once on 2023 without creating a combined dataset. |
| I | 1A | Multilayer perceptron | 01 | `mlp_1a_01.ipynb` | Feed-forward neural network | Fill missing values, encode categories, and scale inputs. Tune a small regularized network with early stopping before considering a deeper model. |
| I | 1A | Multilayer perceptron | 02 | `mlp_1a_02.ipynb` | Full-history rotation and 60-minute backlog | Give a regularized Keras network the exact 36 source fields used by Logistic Regression Experiment 08 and CatBoost Experiment 02. Compare two compact one-hot MLP architectures and two learning rates with fold-local chronological early stopping, select the threshold from fixed-epoch 2019 held-out predictions, validate once on 2023, and retain the final-tail assignment limitation. |
| I | 1A | CatBoost / MLP blend | 01 | `ensemble_1a_01.ipynb` | Training-only probability blend | Regenerate aligned 2019 out-of-fold probabilities from the selected CatBoost Experiment 04 and MLP Experiment 02 configurations. Select among five predeclared CatBoost/MLP weights by mean temporal average precision, select the operating threshold from aggregated 2019 predictions, and evaluate the fixed blend once on 2023. Do not create a combined feature or prediction dataset, and keep 2024 untouched. |
| I | 1A | LR / DT / RF imbalance study | 01 | `imbalance_1a_01.ipynb` | Handling the smaller delayed class | Using the same folds, compare no adjustment, class weights, random over-sampling, SMOTE/SMOTENC, valid uses of ADASYN, and SMOTE-ENN. Select by average precision rather than accuracy. |
| I | 1A | CatBoost calibration | 01 | `calibration_1a_01.ipynb` | CatBoost 04 probability calibration | Compare unchanged CatBoost 04 probabilities with sigmoid and isotonic corrections using forward-chained held-out 2019 predictions. Select primarily by mean temporal Brier score, fit the selected calibrator using held-out 2019 probabilities, and evaluate the fixed result once on 2023. Retain the uncalibrated model when neither correction improves training-only reliability. |
| I | 1A | CatBoost audit | 01 | `catboost_audit_1a_01.ipynb` | Frozen selected-model subgroup and SHAP audit | Refit CatBoost 04's fixed 41-field configuration on all 2019 rows, reproduce its unchanged 2023 result, and audit ranking, calibration, and fixed-threshold errors by month, time of day, airline, route, weather, and 2019-defined traffic level. Calculate CatBoost SHAP values on a fixed 5,000-row 2023 sample and explain representative true-negative, false-positive, false-negative, and true-positive flights. Do not retune from audit results or load 2024. |
| II | 2A | Logistic regression | 01 | `logistic_regression_2a_01.ipynb` | Arrival before pushback | Fit the compact pre-pushback feature set using schedule, route, planned traffic at the origin, and origin weather available by prediction time. |
| II | 2B | Logistic regression | 01 | `logistic_regression_2b_01.ipynb` | Arrival after pushback | First add signed `DepDelay` to the 2A features. Test calculated departure-time features only as a clearly documented follow-up. |
| II | 2C | Logistic regression | 01 | `logistic_regression_2c_01.ipynb` | Arrival after takeoff | Add `TaxiOut` and takeoff-time features to the same 2B base and measure the improvement over 2B. |
| II | 2A | Decision tree | 01 | `decision_tree_2a_01.ipynb` | Arrival before pushback | Apply the selected single-tree search to the compact pre-pushback feature set. |
| II | 2B | Decision tree | 01 | `decision_tree_2b_01.ipynb` | Arrival after pushback | Add signed `DepDelay` to the 2A tree and compare changes in feature importance and recall. |
| II | 2C | Decision tree | 01 | `decision_tree_2c_01.ipynb` | Arrival after takeoff | Add taxi-out and takeoff fields without including duplicate versions of the same information. |
| II | 2A | Random forest | 01 | `random_forest_2a_01.ipynb` | Arrival forest before pushback | Apply the selected Random Forest setup to the pre-pushback feature set. Retune only settings that appear sensitive to prediction time. |
| II | 2B | Random forest | 01 | `random_forest_2b_01.ipynb` | Arrival forest after pushback | Add signed `DepDelay` and measure its added value without using late-aircraft or aircraft-rotation outcomes. |
| II | 2C | Random forest | 01 | `random_forest_2c_01.ipynb` | Arrival forest after takeoff | Add taxi-out and takeoff information and compare 2A, 2B, and 2C on exactly the same flight rows. |
| II | 2A | CatBoost | 01 | `catboost_2a_01.ipynb` | Boosted arrival model before pushback | Let CatBoost handle categorical fields directly using the pre-pushback feature set, and stop training based on time-ordered validation. |
| II | 2B | CatBoost | 01 | `catboost_2b_01.ipynb` | Boosted arrival model after pushback | Add signed `DepDelay` to the unchanged 2A base and measure the value of pushback information. |
| II | 2C | CatBoost | 01 | `catboost_2c_01.ipynb` | Boosted arrival model after takeoff | Add taxi-out and takeoff information to the unchanged 2B base and measure the value of waiting until airborne. |

After Stage I, carry another classifier into Stage II if its 2019 time-based validation average precision is clearly
better than the best main classifier, or if it offers a useful balance of recall, probability quality, and computing
cost. This means creating matching 2A, 2B, and 2C notebooks. It does not mean choosing a final winner from 2019 alone.

## Reference models and project scope

| Reference | Models or methods tested in the paper | How this project uses the paper |
|---|---|---|
| [Snell et al.](resources/docs/02_Snell_MLFlightDelayPrediction.pdf) | Logistic regression, KNN, bagging, decision tree, RepTree, random forest, neural network, SVM, and SMOTE. | These classifier families are included in Stage I. A cost-complexity-pruned tree is used as the scikit-learn equivalent of RepTree. Although Snell reports scenarios using delay fields, `DepDel15` and `DepDelay` are not allowed as inputs to Model 1A or 2A. |
| [Zoutendijk and Mitici](resources/docs/03_Zoutendijk_ProbabilisticFlightDelay.pdf) | Airline, airport, weather, schedule, and traffic inputs; evaluation of flight-specific delay uncertainty. | The paper informs the feature design and the project's checks of predicted-probability quality. The core experiments still predict the binary 15-minute target. |
| [Li](resources/docs/04_Li_DelayPropagationPrediction.pdf) | Random forest, random-forest recursive feature elimination, SMOTE, and chained delay-propagation variants. | Random Forest and training-only RFECV are included. A separate Model 1A extension now tests one causally masked preceding inbound leg. Longer aircraft sequences and network-wide delay spread remain outside scope. |
| [AlBassam](resources/docs/05_AlBassam_MLDelayEval.pdf) | Decision tree, random forest, SVC, logistic regression, KNN, and Naive Bayes with random over-sampling, SMOTE, and ADASYN. | All six classifiers and all three class-balancing methods are included in Stage I. Balancing is performed only within training folds. Actual arrival, delay-cause, and previous-flight fields are not used because they are unavailable at the project's early prediction times or directly describe an outcome. |
| [Chen and Li](resources/docs/06_Chen_ChainedDelayPrediction.pdf) | Random forest, recursive feature elimination, SMOTE, and chained delay-propagation variants. | Random Forest, RFECV, and SMOTE are included. The rotation extension follows only the immediately preceding known inbound; recursive or network-wide propagation remains outside scope. |
| [Pineda-Jaramillo et al.](resources/docs/15_Pineda_ExplainableDelayML.pdf) | Logistic regression, decision tree, Naive Bayes, KNN, RBF SVM, LDA, AdaBoost, Extra Trees, random forest, and gradient boosting; SMOTE-ENN; SHAP and Sobol explanations. | All ten classifiers are included in Stage I. SMOTE-ENN is part of the class-balancing study, and SHAP is required for final models that support it. Weather observed at destination landing or origin takeoff is not used for 1A or 2A because it is not available at prediction time. |
| [Beltman et al.](resources/docs/16_Beltman_DepartureDelayForecast.pdf) | CatBoost and neural-network methods at several pre-departure prediction times. | CatBoost is included as a core classifier. The paper's changing 90-to-15-minute prediction times are not reproduced because the current annual sources do not provide equivalent rolling operational snapshots. |

The excluded longer aircraft-chain and changing-time-horizon designs may still be useful, but they need information
this project does not collect or does not allow at the relevant prediction time. They should be reconsidered only if
the project formally adds those data sources and updates its prediction-time rules.
