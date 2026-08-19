# Capstone Project - Berkeley ML and AI

## Table of Contents

- [Overview](#overview)
  - [Predict departure delay](#predict-departure-delay)
  - [Predict arrival delay](#predict-arrival-delay)
  - [How this report is organized](#how-this-report-is-organized)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
  - [Data Sources](#data-sources)
  - [Data Coverage](#data-coverage)
  - [Why arrival-delay prediction requires more data](#why-arrival-delay-prediction-requires-more-data)
    - [Departure-delay scenario: Model 1A](#departure-delay-scenario-model-1a)
    - [Arrival-delay scenario: Models 2A, 2B, and 2C](#arrival-delay-scenario-models-2a-2b-and-2c)
  - [Data Flow](#data-flow)
    - [Departure-delay data flow: Model 1A](#departure-delay-data-flow-model-1a)
    - [Arrival-delay data flow: Models 2A, 2B, and 2C](#arrival-delay-data-flow-models-2a-2b-and-2c)
  - [Matching Records by Time](#matching-records-by-time)
  - [Model Outcomes and Available Information](#model-outcomes-and-available-information)
  - [EDA](#eda)
    - [List of Figures](#list-of-figures)
- [Data Preparation](#data-preparation)
  - [Data Directory](#data-directory)
  - [BTS](#bts)
  - [ASPM](#aspm)
  - [NOAA](#noaa)
  - [Merged](#merged)
  - [Features](#features)
  - [Operational Backlog Data](#operational-backlog-data)
  - [Aircraft Rotation Data](#aircraft-rotation-data)
- [Modeling](#modeling)
  - [Model 1A: operational features drove the gains](#model-1a-operational-features-drove-the-gains)
  - [Arrival models: observed operations drove the gains](#arrival-models-observed-operations-drove-the-gains)
  - [Experiment controls](#experiment-controls)
- [Evaluation](#evaluation)
  - [Final 2024 evaluation](#final-2024-evaluation)
  - [Interpretation and limitations](#interpretation-and-limitations)
  - [Post-test sensitivity](#post-test-sensitivity)
- [Deployment](#deployment)
- [Next Steps](#next-steps)
- [References](#references)
  - [Data Sources](#data-sources-1)
  - [Papers](#papers)
- [Appendix A](Appendix-A.md)
  - [BTS Column Selection and Dictionary](Appendix-A.md#bts-column-selection-and-dictionary)
    - [Prediction-time eligibility of key BTS operational fields](Appendix-A.md#prediction-time-eligibility-of-key-bts-operational-fields)
    - [BTS columns retained for modeling](Appendix-A.md#bts-columns-retained-for-modeling)
    - [BTS columns removed during processing](Appendix-A.md#bts-columns-removed-during-processing)
  - [ASPM Column Selection and Dictionary](Appendix-A.md#aspm-column-selection-and-dictionary)
    - [ASPM columns retained for modeling](Appendix-A.md#aspm-columns-retained-for-modeling)
    - [ASPM columns removed during processing](Appendix-A.md#aspm-columns-removed-during-processing)
  - [NOAA Column Selection and Dictionary](Appendix-A.md#noaa-column-selection-and-dictionary)
    - [NOAA columns retained for modeling](Appendix-A.md#noaa-columns-retained-for-modeling)
    - [NOAA columns removed or transformed during processing](Appendix-A.md#noaa-columns-removed-or-transformed-during-processing)
- [Appendix B](Appendix-B.md)
  - [Joined Data Column Dictionary](Appendix-B.md#joined-data-column-dictionary)
    - [Joined BTS flight columns](Appendix-B.md#joined-bts-flight-columns)
    - [Joined ASPM planned-demand columns](Appendix-B.md#joined-aspm-planned-demand-columns)
    - [Joined NOAA weather columns](Appendix-B.md#joined-noaa-weather-columns)
- [Appendix C](Appendix-C.md)
  - [Data Understanding](Appendix-C.md#data-understanding)
    - [Datasets](Appendix-C.md#datasets)
    - [Data Flow](Appendix-C.md#data-flow)
      - [Departure-delay data flow: Model 1A](Appendix-C.md#departure-delay-data-flow-model-1a)
      - [Arrival-delay data flow: Models 2A, 2B, and 2C](Appendix-C.md#arrival-delay-data-flow-models-2a-2b-and-2c)
  - [Baseline Feature Engineering](Appendix-C.md#baseline-feature-engineering)
    - [Base Engineered feature dictionary](Appendix-C.md#base-engineered-feature-dictionary)
  - [Operational Backlog Feature Engineering](Appendix-C.md#operational-backlog-feature-engineering)
    - [Backlog feature dictionary](Appendix-C.md#backlog-feature-dictionary)
    - [Same-airline backlog feature dictionary](Appendix-C.md#same-airline-backlog-feature-dictionary)
    - [Backlog implementation, validation, and scope](Appendix-C.md#backlog-implementation-validation-and-scope)
  - [Aircraft Rotation Feature Engineering](Appendix-C.md#aircraft-rotation-feature-engineering)
    - [Rotation feature dictionary](Appendix-C.md#rotation-feature-dictionary)
    - [Rotation implementation and validation](Appendix-C.md#rotation-implementation-and-validation)
    - [Full-history rotation audit](Appendix-C.md#full-history-rotation-audit)
    - [Rotation assignment limitation](Appendix-C.md#rotation-assignment-limitation)
- [Appendix D](Appendix-D.md)
  - [Overview](Appendix-D.md#overview)
  - [Reusable feature allowlists](Appendix-D.md#reusable-feature-allowlists)
    - [BL-D-20 — raw departure baseline](Appendix-D.md#bl-d-20--raw-departure-baseline)
    - [BL-D-27 — compact engineered departure allowlist](Appendix-D.md#bl-d-27--compact-engineered-departure-allowlist)
    - [BL-D-54 — broad engineered departure allowlist](Appendix-D.md#bl-d-54--broad-engineered-departure-allowlist)
    - [BL-D-34 — compact engineered departure allowlist](Appendix-D.md#bl-d-34--compact-engineered-departure-allowlist)
    - [RT-n and RTF-n — Rotation additions](Appendix-D.md#rt-n-and-rtf-n--rotation-additions)
    - [BK-t-n and BK-t-n — Airport-wide backlog additions](Appendix-D.md#bk-t-n-and-bk-t-n--airport-wide-backlog-additions)
    - [BKA-60-5 Same-airline backlog addition](Appendix-D.md#bka-60-5-same-airline-backlog-addition)
    - [BL-A-27 — arrival baseline](Appendix-D.md#bl-a-27--arrival-baseline)
  - [Model 1A experiments](Appendix-D.md#model-1a-experiments)
    - [Logistic-regression sequence](Appendix-D.md#logistic-regression-sequence)
    - [Decision-tree and Random-Forest sequence](Appendix-D.md#decision-tree-and-random-forest-sequence)
    - [CatBoost sequence](Appendix-D.md#catboost-sequence)
    - [Neural network, ensemble, calibration, and audit](Appendix-D.md#neural-network-ensemble-calibration-and-audit)
  - [Arrival-model experiments](Appendix-D.md#arrival-model-experiments)
- [Appendix E](Appendix-E.md)
  - [Experiment protocol](Appendix-E.md#experiment-protocol)
  - [Primary binary-classification experiments](Appendix-E.md#primary-binary-classification-experiments)
  - [Final 2024 evaluation](Appendix-E.md#final-2024-evaluation)
  - [Post-test combined-training sensitivity](Appendix-E.md#post-test-combined-training-sensitivity)
  - [Reference models and project scope](Appendix-E.md#reference-models-and-project-scope)
- [Appendix F](Appendix-F.md)
  - [Results recording rules](Appendix-F.md#results-recording-rules)
  - [Experiment configurations](Appendix-F.md#experiment-configurations)
  - [Ranking and calibration results](Appendix-F.md#ranking-and-calibration-results)
  - [Operating-threshold results](Appendix-F.md#operating-threshold-results)
    - [Model 1A logistic-regression comparison](Appendix-F.md#model-1a-logistic-regression-comparison)
    - [Model 1A decision-tree comparison](Appendix-F.md#model-1a-decision-tree-comparison)
    - [Model 1A random-forest comparison](Appendix-F.md#model-1a-random-forest-comparison)
    - [Model 1A exact-manifest Random Forest comparison](Appendix-F.md#model-1a-exact-manifest-random-forest-comparison)
    - [Model 1A CatBoost comparison](Appendix-F.md#model-1a-catboost-comparison)
    - [Model 1A neural-network comparison](Appendix-F.md#model-1a-neural-network-comparison)
    - [Model 1A same-airline CatBoost comparison](Appendix-F.md#model-1a-same-airline-catboost-comparison)
    - [Model 1A final CatBoost search](Appendix-F.md#model-1a-final-catboost-search)
    - [Model 1A CatBoost/MLP blend comparison](Appendix-F.md#model-1a-catboostmlp-blend-comparison)
    - [Model 1A CatBoost calibration comparison](Appendix-F.md#model-1a-catboost-calibration-comparison)
    - [Model 1A CatBoost subgroup and SHAP audit](Appendix-F.md#model-1a-catboost-subgroup-and-shap-audit)
    - [Model 2A/2B/2C logistic-regression timing comparison](Appendix-F.md#model-2a2b2c-logistic-regression-timing-comparison)
    - [Model 2A CatBoost comparisons](Appendix-F.md#model-2a-catboost-comparisons)
    - [Model 2B pushback-margin and CatBoost comparisons](Appendix-F.md#model-2b-pushback-margin-and-catboost-comparisons)
    - [Model 2C CatBoost comparison](Appendix-F.md#model-2c-catboost-comparison)
    - [Model 2C schedule-margin comparison](Appendix-F.md#model-2c-schedule-margin-comparison)
    - [Model 2C ridge-logistic comparison](Appendix-F.md#model-2c-ridge-logistic-comparison)
    - [Model 2C Gaussian Naive Bayes comparison](Appendix-F.md#model-2c-gaussian-naive-bayes-comparison)
    - [Model 2C support-vector-machine comparison](Appendix-F.md#model-2c-support-vector-machine-comparison)
    - [Linear discriminant analysis comparisons](Appendix-F.md#linear-discriminant-analysis-comparisons)
    - [Model 2C CatBoost schedule-margin comparison](Appendix-F.md#model-2c-catboost-schedule-margin-comparison)
    - [Model 2C neural-network comparison](Appendix-F.md#model-2c-neural-network-comparison)
    - [Model 2C CatBoost/MLP blend comparison](Appendix-F.md#model-2c-catboostmlp-blend-comparison)
  - [Final 2024 evaluation](Appendix-F.md#final-2024-evaluation)
    - [Final ranking and probability results](Appendix-F.md#final-ranking-and-probability-results)
    - [Final operating-threshold results](Appendix-F.md#final-operating-threshold-results)
  - [Post-test combined-training sensitivity](Appendix-F.md#post-test-combined-training-sensitivity)
    - [Combined-training ranking and probability results](Appendix-F.md#combined-training-ranking-and-probability-results)
    - [Combined-training operating-threshold results](Appendix-F.md#combined-training-operating-threshold-results)
- [Appendix G](Appendix-G.md)
  - [Confusion matrix](Appendix-G.md#confusion-matrix)
  - [Threshold-dependent classification metrics](Appendix-G.md#threshold-dependent-classification-metrics)
  - [Probability ranking and calibration metrics](Appendix-G.md#probability-ranking-and-calibration-metrics)
  - [Related evaluation terms](Appendix-G.md#related-evaluation-terms)

## Overview

This capstone project investigates whether machine learning can predict significant delays for individual domestic flights
departing from or arriving at John F. Kennedy International Airport (JFK).

A significant delay is defined as a departure or arrival delay of 15 minutes or more. The project also explores 
which flight, airport, and weather conditions are most closely associated with delays.

The project combines three main data sources:

- Bureau of Transportation Statistics (BTS) [flight schedules and performance data](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr)
- Aviation System Performance Metrics (ASPM) [airport traffic and congestion data](https://www.aspm.faa.gov/apm/sys/main.asp)
- National Oceanic and Atmospheric Administration (NOAA) [weather observations](https://www.ncei.noaa.gov/data/local-climatological-data/access/)

The analysis is organized into two model categories.

### Predict departure delay

1. **Model 1A:** For a flight departing from JFK, predict before pushback whether the flight departs at least 15 minutes late.

### Predict arrival delay

1. **Model 2A:** For a flight arriving at JFK, predict before pushback at the flight origin whether it arrives at least 15 minutes late.
2. **Model 2B:** For a flight arriving at JFK, predict immediately after pushback at the flight origin whether it arrives at least 15 minutes late, using the actual departure delay.
3. **Model 2C:** For a flight arriving at JFK, predict immediately after takeoff at the flight origin whether it arrives at least 15 minutes late, using the actual departure, taxi-out, and takeoff information.

The arrival-delay category applies the same baseline flight-level classification approach as the departure-delay category:
combine schedule, route, planned airport demand, and weather information to predict whether an individual
flight crosses the 15-minute delay threshold. 

Models 2A, 2B, and 2C update the arrival-delay prediction as the flight progresses through its operational timeline. 
Model 2A makes an initial prediction using information available before pushback. Model 2B revises that prediction after the flight departs, 
using the actual departure time and delay. Model 2C updates it once more using observed taxi-out and takeoff information. 
Together, the three models show how arrival-delay predictions can become more informed and potentially more accurate as 
actual operating information replaces earlier assumptions.

Model 1A uses BTS flights departing from JFK and uses JFK ASPM and NOAA data. Models 2A, 2B, and 2C use BTS flights arriving at JFK from many different origin
airports. A complete implementation therefore requires appropriate ASPM and NOAA coverage for those origins, NOAA
station mapping, source-specific cleaning and validation, and joins for every inbound flight.

Each model follows a clear prediction timeline, and a field is included only if it would be known at that time. Information
recorded later is excluded even when it appears in the historical datasets. This prevents the model from learning from
the future, often called data leakage. BTS provides the historical event values used for this analysis. A working
operational model would need equivalent gate-out and takeoff information from a suitable live source. Some 
BTS columns represent events—such as gate departure and takeoff—that an airline or  airport operational system can observe 
when they occur.

### How this report is organized

The main report summarizes the approach and findings. The appendices provide the detailed record of the work,
including dataset preparation and cleaning, joins, feature engineering, model inputs and configurations, model
execution, validation, and final evaluation.

| Appendix | What it documents |
|---|---|
| [Appendix A](Appendix-A.md) | BTS, ASPM, and NOAA cleaning decisions, retained columns, and source-column dictionaries. |
| [Appendix B](Appendix-B.md) | How the source datasets are joined and the columns in the joined departure and arrival datasets. |
| [Appendix C](Appendix-C.md) | Dataset production flows and the baseline, backlog, same-airline backlog, and aircraft-rotation feature engineering. |
| [Appendix D](Appendix-D.md) | Reusable feature allowlists and the exact inputs used by each model experiment. |
| [Appendix E](Appendix-E.md) | Experiment rules, completed and unattempted experiments, final model designs, and project scope. |
| [Appendix F](Appendix-F.md) | Completed model configurations, execution results, validation results, final 2024 evaluation, and combined-training sensitivity. |
| [Appendix G](Appendix-G.md) | Definitions of the classification, ranking, and probability-quality measures used in evaluation. |

## Business Understanding

Flight delays create costs and disruption for passengers, airlines, and airports. Earlier warning of a likely delay 
can help airlines communicate with passengers, adjust staffing and gate plans, and prepare for possible missed connections. 
Airports can also use this information to better understand when congestion or weather is likely to affect operations.

The departure-delay question asks:

- Can a departure delay of 15 minutes or more be identified before pushback?

The arrival-delay questions ask:

- Can an arrival delay of 15 minutes or more be identified before pushback?
- How much does the arrival prediction improve once the actual departure delay is known?
- How much more does the arrival prediction improve once taxi-out and takeoff information is known?

The analysis examines JFK. A useful result does more than produce a
yes-or-no answer. It provides a reliable estimate of delay risk and clearly shows which schedule, airport, and weather
conditions influence the prediction.

The models are intended as decision-support tools, not as proof that a particular factor caused a delay. Success is 
judged by how well the models identify delayed flights, how often their warnings are correct, and whether their results 
can be explained in a useful way.

## Data Understanding

The project brings together flight, airport, and weather data. Each record represents one flight. JFK is the origin for
Model 1A and the destination for Models 2A, 2B, and 2C. Airport and weather records
are added without changing this one-row-per-flight structure.

### Data Sources

| Source | Information used in this project | Level of detail |
|---|---|---|
| Bureau of Transportation Statistics (BTS) | Flight dates and schedules, airline, origin, destination, distance, and departure and arrival outcomes | One row per flight |
| Aviation System Performance Metrics (ASPM) | Scheduled arrivals and departures used as measures of planned airport demand | One row per airport and hour |
| National Oceanic and Atmospheric Administration (NOAA) | Temperature, humidity, visibility, precipitation, wind, and reported weather conditions | One row per weather observation |

BTS provides the individual flight records and the two outcomes the models predict. `DepDel15` identifies flights 
that departed at least 15 minutes late, while `ArrDel15` identifies flights that arrived at least 15 minutes late. ASPM 
describes scheduled airport demand, and NOAA describes the weather observed before departure.

### Data Coverage

The modeling population consists of domestic flights departing from and arriving at JFK in 2019, 2023, and 2024. These years were
selected to represent periods before and after the COVID-19 pandemic when the airport was operating at or near normal
capacity. The 2019 data provides a pre-pandemic baseline, while 2023 and 2024 show flight operations after the major
pandemic-related disruptions had passed.

The 2019 flight data will be used for training, the 2023 data for model development and validation, and the 2024 data for final testing. 
The split preserves time order so that the models are trained on earlier flights and evaluated
on later flights they have not seen.

### Why arrival-delay prediction requires more data

Each flight is represented by one BTS row containing both its `Origin` and `Dest`. The tables below identify the
airport-specific ASPM and NOAA context attached to that row in the two model categories.

#### Departure-delay scenario: Model 1A

| Airport role | BTS information used                                       | ASPM context                                 | NOAA context                                        | Data footprint                 |
|---|------------------------------------------------------------|----------------------------------------------|-----------------------------------------------------|--------------------------------|
| JFK origin | Schedule, airline, route, distance, and `DepDel15` target  | JFK planned traffic near scheduled departure | Latest JFK weather available by the prediction time | BTS, ASPM, NOAA for JFK Origin |
| Flight destination | Destination and schedule information from the same BTS row | -                                            | -                                                   | -                              |

#### Arrival-delay scenario: Models 2A, 2B, and 2C

| Airport role | BTS information used | ASPM context                                   | NOAA context                                                   | Data footprint                         |
|---|---|------------------------------------------------|----------------------------------------------------------------|----------------------------------------|
| Each flight origin | Schedule, airline, route, and Model 2B/2C operating fields | Planned traffic near departure for that origin | Latest origin weather available by the model's prediction time | BTS, ASPM, NOAA for 54 origin airports |
| JFK destination | Destination identity, scheduled arrival information, and `ArrDel15` target from the same BTS row | -                                              | -                                                              | BTS for JFK Destination                |

Both scenarios use the same baseline idea: combine flight-level BTS information with planned airport demand and
weather to classify whether a flight crosses the 15-minute delay threshold. Model 1A always departs from JFK, so one ASPM dataset and one NOAA station can serve every row. In the
arrival scenario, the departure airport changes from flight to flight. Each included origin needs ASPM coverage, a NOAA
station mapping, source cleaning and validation, and its own prediction-time-safe joins.


Model 1A expanded beyond the baseline with an engineered operation feature set. It added recent JFK and
same-airline departure backlogs, plus aircraft-rotation information describing whether the assigned aircraft had
arrived and how much turnaround time was available before pushback. These additions were developed to improve Model
1A, which must predict departure delay before it occurs.

The arrival models did not require this expanded operational
feature set: Model 2A provides the early baseline, Model 2B can use actual departure delay, and Model 2C also adds
taxi-out time. Those observed values already summarize much of the delay accumulated before the arrival prediction.
Models 2A,B,C must collect and validate consistent ASPM and NOAA context across the inbound flights.
Destination ASPM and weather are not part of the baseline arrival design. With the arrival scenario, weather observed at landing is
prohibited because it was not available when the prediction was made.

The work includes:

- Checking data quality, missing values, unusual values, and the balance between delayed and on-time flights
- Exploring which flight, airport, and weather conditions are most closely associated with delays
- Creating useful features from dates, times, flights, airport, and weather conditions
- Comparing logistic regression, random forest, and gradient-boosting models such as CatBoost
- Measuring model performance on later flights that were not used for training
- Explaining which factors have the strongest effect on each model's predictions

The project focuses on individual flights. A separate Model 1A extension now tests the immediately preceding aircraft
leg at JFK, but longer previous-flight chains and delay spread through an airline network remain out of scope. The
approach is informed by the flight-level delay research of Snell, Zoutendijk, and Pineda. The final analysis focuses on
model performance and the factors associated with delays at JFK.

### Data Flow

The two scenarios use the same basic processing and cleaning steps, but they prepare different groups of flights and
airport conditions. The departure path needs data for JFK only. The arrival path first gathers flights bound for JFK
and then gathers conditions at each flight's origin. The departure scenario also added the operational feature set.
In the diagrams below, `YEAR` represents 2019, 2023, or 2024.

#### Departure-delay data flow: Model 1A

The diagram below intentionally shows the baseline Model 1A flow. This keeps the comparison with the arrival flow
simple; the [complete Model 1A data flow in Appendix C](Appendix-C.md#departure-delay-data-flow-model-1a) also shows the
engineered backlog and aircraft-rotation datasets.

![Departure-delay data flow for Model 1A](resources/diagrams/departure-data-flow.svg)

At a high level, Model 1A combines cleaned JFK departure records from BTS with planned airport demand from ASPM and
recent weather from NOAA. The result has one row per eligible departure, with baseline features that would be known
before pushback.

[Appendix A](Appendix-A.md) documents source cleaning and retained columns. [Appendix B](Appendix-B.md) explains the
join, [Appendix C](Appendix-C.md) describes the complete feature engineering flow, and [Appendix D](Appendix-D.md)
records the exact model allowlists. Experiment and training rules are documented in [Appendix E](Appendix-E.md).

#### Arrival-delay data flow: Models 2A, 2B, and 2C

![Arrival-delay data flow for Models 2A, 2B, and 2C](resources/diagrams/arrival-data-flow.svg)

At a high level, the arrival flow begins with JFK-bound flights from 54 origin airports. Each flight is joined with
planned airport demand and recent weather from its origin, producing one row per eligible arrival. This requires more
source coverage than the departure flow, which uses JFK conditions only.

One feature dataset supports all three arrival prediction times. Model 2A uses information available before pushback,
Model 2B adds actual departure delay, and Model 2C adds taxi-out and takeoff information.

[Appendix A](Appendix-A.md) documents source cleaning and retained columns. [Appendix B](Appendix-B.md) explains the
arrival join, [Appendix C](Appendix-C.md#baseline-feature-engineering) describes the feature engineering, and
[Appendix D](Appendix-D.md#arrival-model-experiments) records the exact model allowlists. Experiment and training rules
are documented in [Appendix E](Appendix-E.md).

### Matching Records by Time

The scheduled departure date and time, stored in `DATE` ([Appendix B: Joined BTS flight columns](Appendix-B.md#joined-bts-flight-columns)), provides the main time for each BTS flight. Each flight is matched
with [ASPM schedule records for the previous, current, and next clock hours](Appendix-B.md#joined-aspm-planned-demand-columns). The next-hour values are planned counts known
ahead of time, not future operating results. Each flight is also matched with the most recent NOAA observation available
before its scheduled departure ([Appendix B: Joined NOAA weather columns](Appendix-B.md#joined-noaa-weather-columns)). The merge notebook keeps source timestamps and timing differences so the matches can be checked.

This time-based matching prevents a model from using airport conditions or weather observations that occurred after the prediction was made. It also allows the age of the matched information to be checked before modeling ([Appendix E: Experiment protocol](Appendix-E.md#experiment-protocol)).

### Model Outcomes and Available Information

| Category  | Model | Outcome | Information available when the prediction is made |
|-----------|---|---|---|
| Departure | Model 1A | `DepDel15` | Flight schedule, earlier airport conditions, and earlier weather observations |
| Arrival   | Model 2A | `ArrDel15` | The same general pre-pushback information as Model 1A, collected for the flight origin |
| Arrival   | Model 2B | `ArrDel15` | Model 2A information plus the actual departure time and departure delay |
| Arrival   | Model 2C | `ArrDel15` | Model 2B information plus taxi-out and takeoff information |

The shared feature files include measures of time of day, season, route, distance, planned traffic, and weather. Each
model notebook uses only the fields allowed at its prediction time, so information recorded later in the flight is left
out of that model.

### EDA

The exploratory data analysis notebooks examine data quality, target balance, temporal and route-related delay patterns,
weather, planned airport traffic, and differences across the available datasets. The figures are organized below by
their source notebook.

#### List of Figures

| Figure | Visualization | Description |
|---:|---|---|
| 1 | [Missing values](eda/01_data_quality_and_targets.ipynb#missing-values) | Counts and percentages of missing values by column. |
| 2 | [Delay target balance](eda/01_data_quality_and_targets.ipynb#target-balance) | Class balance for the departure- and arrival-delay targets. |
| 3 | [Daily flight volume](eda/01_data_quality_and_targets.ipynb#flight-volume-over-the-year) | Number of scheduled JFK departures on each date. |
| 4 | [Timing of matched source records](eda/01_data_quality_and_targets.ipynb#timing-of-matched-source-records) | ASPM time offsets and the age of matched NOAA observations. |
| 5 | [Monthly delay rate](eda/02_delay_patterns.ipynb#delay-rate-by-month) | Departure- and arrival-delay rates by calendar month. |
| 6 | [Delay rate by day of week](eda/02_delay_patterns.ipynb#delay-rate-by-day-of-week) | Delay rates from Monday through Sunday. |
| 7 | [Delay rate by scheduled departure hour](eda/02_delay_patterns.ipynb#delay-rate-by-scheduled-departure-hour) | Hourly pattern of delay risk across the operating day. |
| 8 | [Delay rate by time of day](eda/02_delay_patterns.ipynb#broad-time-of-day-comparison) | Delay rates for broad overnight, morning, afternoon, and evening periods. |
| 9 | [Delay rate for the busiest airlines](eda/02_delay_patterns.ipynb#airlines) | Delay rates for the airlines operating the most JFK departures. |
| 10 | [Delay rate for the busiest destinations](eda/02_delay_patterns.ipynb#destinations) | Delay rates for the most frequently served destinations. |
| 11 | [Delay rate by BTS distance group](eda/02_delay_patterns.ipynb#distance-group) | Delay rates across BTS route-distance bands. |
| 12 | [Departure-delay rate by airline and weekday](eda/02_delay_patterns.ipynb#airline-and-weekday-interaction) | Airline-by-weekday heatmap of departure-delay rates. |
| 13 | [Departure-delay rate by month and scheduled hour](eda/02_delay_patterns.ipynb#month-and-scheduled-hour-interaction) | Month-by-hour heatmap showing seasonal changes in daily delay patterns. |
| 14 | [Departure and arrival delay outcomes](eda/02_delay_patterns.ipynb#relationship-between-departure-and-arrival-outcomes) | Joint distribution of departure and arrival delay classifications. |
| 15 | [Departure-delay distributions](eda/02_delay_patterns.ipynb#departure-delay-duration) | Signed departure delays and nonnegative delay durations. |
| 16 | [Weather distributions](eda/03_weather_and_congestion.ipynb#weather-distributions) | Distributions of temperature, humidity, visibility, precipitation, and wind. |
| 17 | [Delay rate by reported weather](eda/03_weather_and_congestion.ipynb#adverse-weather-comparison) | Delay rates with and without reported adverse weather. |
| 18 | [Delay rate during reported weather conditions](eda/03_weather_and_congestion.ipynb#individual-weather-conditions) | Delay rates associated with individual reported weather conditions. |
| 19 | [Delay rate by visibility](eda/03_weather_and_congestion.ipynb#visibility) | Delay rates across operationally interpretable visibility categories. |
| 20 | [Scheduled airport traffic around departure](eda/03_weather_and_congestion.ipynb#scheduled-airport-traffic) | Planned traffic distributions for the previous, current, and next hours. |
| 21 | [Delay rate by three-hour scheduled traffic](eda/03_weather_and_congestion.ipynb#delay-rate-by-traffic-level) | Delay rates across quintiles of three-hour planned traffic. |
| 22 | [Departure-delay rate across current-hour scheduled demand](eda/03_weather_and_congestion.ipynb#current-hour-arrivals-and-departures) | Delay risk across combinations of scheduled departures and arrivals. |
| 23 | [Departure delay by weather and traffic](eda/03_weather_and_congestion.ipynb#weather-and-congestion-together) | Interaction between adverse weather and planned traffic level. |
| 24 | [Selected numeric correlations](eda/03_weather_and_congestion.ipynb#correlation-overview) | Correlations among targets, weather, traffic, and source-age fields. |
| 25 | [JFK flights by year](eda/04_airport_year_comparison.ipynb#dataset-size-and-basic-coverage) | Flight counts for the 2019, 2023, and 2024 JFK datasets. |
| 26 | [JFK delay rates by year](eda/04_airport_year_comparison.ipynb#departure-and-arrival-delay-rates) | Departure- and arrival-delay class balance across years. |
| 27 | [JFK delay-rate heatmap](eda/04_airport_year_comparison.ipynb#delay-rate-heatmaps) | Compact comparison of departure and arrival delay rates by year. |
| 28 | [Monthly JFK delay rates by year](eda/04_airport_year_comparison.ipynb#monthly-patterns) | Seasonal delay patterns compared across the three study years. |
| 29 | [Hourly JFK delay rates by year](eda/04_airport_year_comparison.ipynb#scheduled-hour-patterns) | Daily delay build-up compared across the three study years. |
| 30 | [JFK scheduled traffic around departure by year](eda/04_airport_year_comparison.ipynb#airport-traffic-distributions) | Planned previous-, current-, and next-hour traffic across years. |
| 31 | [JFK weather and source timing by year](eda/04_airport_year_comparison.ipynb#weather-and-source-timing-differences) | Visibility, ASPM offsets, and NOAA observation age across years. |

## Data Preparation

The data moves through a consistent path. Raw BTS, ASPM, and NOAA datasets are processed and cleaned, joined into one
row per eligible flight, and extended with features for modeling. The departure path uses JFK data; the arrival path
uses JFK-bound flights and conditions from each flight's origin. The directory tree below follows these steps. `YEAR`
represents 2019, 2023, or 2024; `AIRPORT` and `STATION` identify an airport and its matched NOAA weather station.

[Appendix A](Appendix-A.md) documents source cleaning and retained columns, and [Appendix B](Appendix-B.md) describes
the joined datasets. [Appendix C](Appendix-C.md#datasets) provides the detailed dataset inventory, production flow, and
feature-engineering documentation.

### Data Directory

```text
data/
├── bts/
│   ├── L_UNIQUE_CARRIERS.csv
│   ├── raw/
│   │   └── YEAR/
│   │       ├── On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_YEAR_1.csv
│   │       ├── ... monthly CSVs through YEAR_12.csv
│   │       ├── YEAR_csv_files.zip
│   │       ├── JFK.csv
│   │       └── ...
│   ├── processed/
│   │   ├── _csv_files.zip
│   │   ├── JFK_YEAR.csv
│   │   └── ...
│   ├── cleaned/
│   │   ├── _csv_files.zip
│   │   ├── JFK_YEAR.csv
│   │   └── ...
│   └── cleaned_JFK_YEAR.csv
├── aspm/
│   ├── raw/
│   │   └── aspm_output/
│   │       └── run_YEAR_AIRPORT/
│   │           ├── aspm_YEAR_AIRPORT.csv
│   │           └── raw_html.zip
│   ├── processed/
│   │   ├── _csv_files.zip
│   │   ├── JFK_YEAR.csv
│   │   └── ...
│   ├── cleaned/
│   │   ├── _csv_files.zip
│   │   ├── JFK_YEAR.csv
│   │   └── ...
│   └── cleaned_JFK_YEAR.csv
├── noaa/
│   ├── raw/
│   │   ├── airport_station_map.csv
│   │   └── YEAR/
│   │       ├── YEAR_csv_files.zip
│   │       └── STATION.csv
│   ├── processed/
│   │   ├── _csv_files.zip
│   │   ├── JFK_YEAR.csv
│   │   └── ...
│   ├── cleaned/
│   │   ├── _csv_files.zip
│   │   ├── JFK_YEAR.csv
│   │   └── ...
│   └── cleaned_JFK_YEAR.csv
├── merged/
│   ├── JFK_YEAR_departures.csv
│   └── JFK_YEAR_arrivals.csv
└── features/
    ├── JFK_YEAR_departures.csv
    ├── JFK_YEAR_departures_backlog_airline_w60.csv
    ├── JFK_YEAR_departures_backlog_w30.csv
    ├── JFK_YEAR_departures_backlog_w60.csv
    ├── JFK_YEAR_departures_rotation.csv
    ├── JFK_YEAR_departures_rotation_full_history.csv
    └── JFK_YEAR_arrivals.csv
```

### BTS

The BTS On-Time Performance dataset provides flight schedules, airline and route information, operating times, and the
departure and arrival delay targets. Cleaning produces one row per completed domestic flight, standardizes the flight
identifiers and time fields, and removes cancelled, diverted, or duplicate records. [Appendix A](Appendix-A.md#bts-column-selection-and-dictionary)
documents the BTS column selection and dictionary.

Model 1A uses flights departing from JFK and predicts `DepDel15`. Models 2A, 2B, and 2C use flights arriving at JFK and
predict `ArrDel15`. Later operating values, such as actual departure and takeoff information, are available only to the
models that make predictions after those events occur. [Appendix B](Appendix-B.md#joined-bts-flight-columns) documents
the BTS columns retained after joining, [Appendix C](Appendix-C.md#datasets) shows the dataset flow, and
[Appendix D](Appendix-D.md) records the exact fields allowed for each experiment.

### ASPM

ASPM provides the planned number of arrivals and departures for each airport and hour. These schedule counts represent
expected airport demand known before departure. Actual ASPM delay and on-time measures are excluded because they are
reported too late for the project's prediction times. Cleaning standardizes the airport-hour records and checks their
coverage. [Appendix A](Appendix-A.md#aspm-column-selection-and-dictionary) documents the ASPM column selection and
dictionary.

Model 1A uses planned demand at JFK. The arrival models use planned demand at each flight's origin. [Appendix B](Appendix-B.md#joined-aspm-planned-demand-columns)
documents the hourly values added during the join, [Appendix C](Appendix-C.md#base-engineered-feature-dictionary)
describes the traffic features derived from them, and [Appendix D](Appendix-D.md) records which fields each experiment
may use.

### NOAA

NOAA provides weather observations from the airport where each flight departs. Each project airport is mapped to a
weather station, and cleaning selects usable reports and standardizes precipitation, reported conditions, and wind.
Only observations available by the prediction time are used; older values may fill short gaps, but longer gaps remain
missing for the model to handle. [Appendix A](Appendix-A.md#noaa-column-selection-and-dictionary) documents the NOAA
cleaning decisions and column dictionary.

Model 1A uses weather observed at JFK. The arrival models use weather from each flight's origin; destination weather
near landing is not part of the baseline design. [Appendix B](Appendix-B.md#joined-noaa-weather-columns) documents the
weather match and retained fields, [Appendix C](Appendix-C.md#base-engineered-feature-dictionary) describes the derived
weather features, and [Appendix D](Appendix-D.md) records which fields each experiment may use.

### Merged

The merged datasets combine each eligible BTS flight with planned airport demand and recent weather while preserving one
row per flight. Source airport codes and timestamps remain available so the matches and their timing can be checked.
Unmatched source records are reported rather than silently removing the flight.

The departure dataset uses traffic and weather at JFK. The arrival dataset uses conditions at each flight's origin.
Some retained columns support auditing or later prediction times and are not valid inputs for every model.
[Appendix B](Appendix-B.md#joined-data-column-dictionary) documents the joined columns and matching rules, while
[Appendix C](Appendix-C.md#datasets) shows how the merged datasets are produced and used.

### Features

The feature datasets preserve the merged flight rows and add fields calculated with fixed rules. BL-D provides the
baseline departure features, and BL-A provides the shared arrival features used across Models 2A, 2B, and 2C. Model 1A
also uses separate backlog and aircraft-rotation datasets that add operational context without replacing the baseline.

Shared feature creation does not learn preprocessing settings or select model inputs. Those choices are made within the
experiments using training data and explicit allowlists. [Appendix C](Appendix-C.md) documents the feature engineering
and complete data flows, [Appendix D](Appendix-D.md#reusable-feature-allowlists) defines the allowlists, and
[Appendix E](Appendix-E.md#experiment-protocol) records the experiment rules.

### Operational Backlog Data

Airport backlog is a snapshot of recent JFK operations at a flight's scheduled departure time. It counts
earlier-scheduled flights that have pushed back and those still waiting. Completed-flight delays show how well the
airport is clearing that demand. Unlike ASPM's planned traffic counts, backlog describes what has actually happened.

Model 1A tested airport-wide 30- and 60-minute windows and a same-airline 60-minute view. The preferred model uses three
airport-wide fields and five same-airline fields from the 60-minute datasets. The same-airline addition produced a small,
consistent improvement over airport-wide backlog alone.

Backlog features were implemented only for JFK departures. Historical BTS data reconstructs the snapshots; live use
would require timely schedule and gate-out events. [Appendix C](Appendix-C.md#operational-backlog-feature-engineering)
documents their construction, validation, and scope. [Appendix D](Appendix-D.md#bk-t-n-and-bk-t-n--airport-wide-backlog-additions)
lists the exact model allowlists, and [Appendix F](Appendix-F.md#model-1a-same-airline-catboost-comparison) reports the
experiment results.

### Aircraft Rotation Data

Aircraft rotation links a JFK departure to the preceding inbound flight flown by the same aircraft. At the scheduled
departure time, it describes whether the aircraft has arrived, how much turnaround time is available, and any inbound
delay already observed. It produced the largest improvement among the operational feature families added to Model 1A.

The first rotation dataset used limited inbound history. A later audit showed that full JFK movement history produced
more complete matches, so a separate full-history dataset was added without changing the original. The selected design
uses the full-history dataset and does not use rotation values for scheduled turns longer than 24 hours.

One limitation is important: BTS records the aircraft that ultimately operated the flight, not necessarily the aircraft
assigned at prediction time. Rotation results are therefore retrospective upper bounds. Live use would require
timestamped aircraft assignments and current arrival events. [Appendix C](Appendix-C.md#aircraft-rotation-feature-engineering)
documents construction, validation, and limitations. [Appendix D](Appendix-D.md#rt-n-and-rtf-n--rotation-additions)
lists the rotation allowlists, and [Appendix F](Appendix-F.md#model-1a-logistic-regression-comparison) reports their
measured effect.

## Modeling

The project uses one model to predict departure delay and three models to update an arrival-delay prediction as a
flight progresses. All four classify whether a flight will cross the 15-minute delay threshold.

| Model | Prediction time | Flights and target | Information available |
|---|---|---|---|
| 1A | Before pushback | JFK departures; predict `DepDel15` | Schedule, route, planned JFK traffic, JFK weather, recent backlog, and aircraft rotation state |
| 2A | Before pushback at the origin | JFK arrivals; predict `ArrDel15` | Schedule, route, planned origin traffic, and origin weather |
| 2B | Immediately after pushback | Same JFK arrivals and target as 2A | Model 2A information plus actual gate-out time and departure delay |
| 2C | Immediately after takeoff | Same JFK arrivals and target as 2A | Model 2B information plus taxi-out time and actual takeoff time |

Experiments were append-only. Each new notebook answered a specific feature or classifier question without changing an
earlier result. [Appendix D](Appendix-D.md) gives the exact feature allowlists, [Appendix E](Appendix-E.md) identifies
the completed and unattempted experiments, and [Appendix F](Appendix-F.md) records configurations and results.

### Model 1A: operational features drove the gains

Model 1A must predict before the target flight pushes back, so its own operating outcome is not available. General
schedule, calendar, route, traffic, and weather transformations did not improve the raw logistic-regression baseline.
The useful gains came from features that describe current operations at JFK. Each decision was made with 2019 data;
the table shows how the progression performed on 2023 validation data.

| Modeling decision | What changed | 2023 average precision |
|---|---|---:|
| Raw logistic baseline | Schedule, route, planned traffic, and weather | 0.3959 |
| Add airport backlog | Recent JFK departures still waiting or already completed | 0.4184 |
| Use aircraft rotation | Replace airport-wide context with the assigned inbound aircraft's state | 0.6515 |
| Improve rotation history | Use full JFK movement history and mask turns over 24 hours | 0.6960 |
| Combine rotation and backlog | Add the selected 60-minute airport-wide backlog | 0.7053 |
| Change to CatBoost | Learn nonlinear interactions from the same operational inputs | 0.7473 |
| Add same-airline backlog | Add recent operating pressure within the target airline | **0.7526** |

Aircraft rotation produced the largest feature gain. Random Forest reached 0.7409 using CatBoost 04's exact 41 fields,
which confirms that the operational inputs—not CatBoost alone—drive most of the improvement. CatBoost used those inputs
best. Broader feature sets, deeper CatBoost variants, probability calibration, and a CatBoost/MLP blend did not provide
enough additional value to replace CatBoost 04.

CatBoost 04 is therefore the preferred Model 1A design. Its rotation fields rely on the final aircraft recorded by BTS,
so its performance remains a retrospective upper bound until prediction-time aircraft assignments are available.

### Arrival models: observed operations drove the gains

Models 2A, 2B, and 2C use the same JFK-bound flights. Their main difference is when the prediction is made and what has
been observed by then.

| Model | New operating information | Preferred design | 2023 average precision |
|---|---|---|---:|
| 2A: before pushback | None; schedule, route, planned origin traffic, and weather only | Logistic Regression 01 | 0.4019 |
| 2B: after pushback | Actual departure delay and time remaining until scheduled arrival | Logistic Regression 02 | 0.8704 |
| 2C: after takeoff | Taxi-out time, actual takeoff time, and updated time remaining | Logistic Regression 02 | **0.9145** |

Pushback information was transformative: average precision increased by 0.4685 because actual departure delay directly
shows how late the flight began operating. Taxi-out information added another 0.0441 after takeoff because time spent
on the ground consumes more of the flight's schedule buffer. The one-time 2024 test confirmed the same pattern, with
average precision of 0.3482, 0.8757, and 0.9268 for Models 2A, 2B, and 2C.

More complex classifiers did not produce a clear overall improvement for the arrival models. CatBoost was close for
Model 2B, and an MLP was close for Model 2C, but their small ranking gains came with weaker probability or classification
results. Logistic regression remained the best balanced choice. The tradeoff is practical: later predictions are much
stronger, but they leave less time to act.

### Experiment controls

- Model and feature choices used chronological folds within 2019. Selected designs were then trained on all 2019 data
  and validated on 2023.
- The 2024 datasets were opened only after the models, features, and thresholds were frozen. They were not used to revise
  a model.
- Models 2A, 2B, and 2C used identical flight rows. Only information available at each prediction time was admitted.
- Imputation, encoding, scaling, feature selection, class handling, calibration, and threshold selection were fitted
  using training data only.
- Average precision was the primary ranking measure because delayed flights were the smaller class. ROC AUC, Brier
  score, precision, recall, F1, and MCC provided supporting checks; [Appendix G](Appendix-G.md) defines these measures.

## Evaluation

### Final 2024 evaluation

The models and decision thresholds were selected with 2019 data, validated on 2023, frozen, and then evaluated once on 2024.
Nothing was changed in response to the final-test results. Higher average precision (AP), ROC AUC, and F1 are
better; lower Brier score is better.

| Model and prediction time | Frozen design | 2023 AP | 2024 AP | 2024 ROC AUC | 2024 Brier | 2024 F1 at frozen threshold |
|---|---|---:|---:|---:|---:|---:|
| 1A — before pushback | CatBoost 04 | 0.7526 | **0.7250** | 0.8517 | 0.0983 | 0.6321 at 0.31 |
| 2A — before pushback | Logistic Regression 01 | 0.4019 | **0.3482** | 0.6569 | 0.1603 | 0.3993 at 0.22 |
| 2B — after pushback | Logistic Regression 02 | 0.8704 | **0.8757** | 0.9254 | 0.0626 | 0.7992 at 0.39 |
| 2C — after takeoff | Logistic Regression 02 | 0.9145 | **0.9268** | 0.9611 | 0.0465 | 0.8528 at 0.45 |

[Appendix F](Appendix-F.md#final-2024-evaluation) provides the complete final-test tables, including precision, recall,
balanced accuracy, and MCC.

### Interpretation and limitations

Model 2C is clearly more predictive than Model 1A. On 2024, Model 2C reached 0.9268 AP, 0.9611 ROC AUC, and 0.8528 F1,
compared with 0.7250, 0.8517, and 0.6321 for Model 1A. This difference is expected. Model 2C observes actual departure
delay, taxi-out time, and takeoff time, which directly show how much delay has accumulated. Model 1A must predict before
the target flight moves, using aircraft availability, backlog, schedule, and weather.

The arrival sequence reinforces this point. Model 2A reached only 0.3482 AP before pushback. Actual departure delay
raised Model 2B AP to 0.8757, and taxi-out and takeoff information raised Model 2C AP to 0.9268. Models 2B and 2C also
performed slightly better in 2024 than in 2023. Later predictions are stronger, but they leave less time to act.

Model 1A is still meaningfully predictive: its 0.7250 AP is well above the 2024 departure-delay rate of 0.2038, and it
provides warning before pushback. Its rotation result remains a retrospective upper bound because BTS does not provide
the aircraft assignment known at prediction time.

The Model 1A audit found weaker performance overnight and at low traffic levels. Aircraft turn time, inbound arrival
delay, rotation state, and backlog were the leading fitted contributions. These are predictive associations, not proof
of cause, and the audit did not change the selected model. Full subgroup and SHAP results are in
[Appendix F](Appendix-F.md#model-1a-catboost-subgroup-and-shap-audit).

### Post-test sensitivity

After the final evaluation, the same frozen designs were refitted on combined 2019 and 2023 data. The check produced
small gains for Models 1A and 2A, little change for Model 2B, and no meaningful gain for Model 2C. It does not replace
the final evaluation; confirming a retraining benefit would require another untouched year. Details are in
[Appendix F](Appendix-F.md#post-test-combined-training-sensitivity).

## Deployment

## Next Steps

Further improvement to Model 1A will likely require better operational data rather than more transformations of the
existing BTS, NOAA, and baseline ASPM fields. The strongest current gains came from aircraft rotation and backlog, so
the most promising additions would describe what the airline, airport, and FAA knew at the scheduled-departure cutoff.

| Additional data | Potential value |
|---|---|
| Timestamped aircraft assignment and inbound ETA | Confirms which aircraft was assigned and when it was expected to reach JFK. This would improve rotation while removing the final-`Tail_Number` limitation. |
| FAA traffic-management information | EDCTs, ground stops, Ground Delay Programs, and flight-plan updates can identify flights expected to remain on the ground. |
| Detailed airport capacity | Current runway configuration, departure demand, available departure rate, and recent throughput describe whether JFK can process its planned departures. |
| Surface and gate operations | Taxi queues, gate occupancy, runway queues, and deicing activity measure congestion more directly than the reconstructed backlog. |
| Destination and enroute conditions | Destination weather, airport restrictions, and convective airspace constraints can cause a JFK departure to be held. |
| Multi-leg aircraft propagation | Earlier legs, accumulated delay, aircraft swaps, and scheduled recovery time can show how disruption reaches the target flight. |

Elevated [ASPM access](https://www.aspm.faa.gov/getInfo.asp) could provide individual-flight records and more detailed
airport information. Useful fields include quarter-hour `AAR` (arrival capacity), `ADR` (departure capacity), departure
and arrival demand, facility-reported traffic, and runway configuration, as documented in the
[ASPM quarter-hour dataset](https://www.aspm.faa.gov/aspmhelp/index/ASPM_Data_Download__Detail_By_Quarter_Hour.html).
Individual-flight EDCT and flight-plan fields would also be valuable because an
[EDCT](https://www.aspm.faa.gov/aspmhelp/index/Expect_Departure_Clearance_Times_%28EDCT%29.html) represents an FAA-imposed
runway-release time.

Additional FAA sources include requested-access [SWIM data services](https://aa.data.faa.gov/) for traffic-flow and
terminal events and [STDDS](https://www.faa.gov/air_traffic/technology/swim/stdds) for surface movement, tower events,
and runway visual range. Forecast weather could add TAFs, SIGMETs, thunderstorms, snow, and destination conditions;
the [Aviation Weather Center API](https://www.connect.aviationweather.gov/data/api/) shows the available live products,
although a historical experiment would need archived forecasts.

More extensive propagation should be attempted only with timestamped aircraft assignments and inbound estimates.
Extending the final BTS tail across several legs would amplify the current retrospective limitation. Every added field
must preserve the same rule used throughout the project: use the value known at the prediction cutoff, not a final
corrected record or later outcome.

## References

### Data Sources

1. [BTS - Aviation Data Library](https://www.transtats.bts.gov/databases.asp?Z1qr_VQ=E&Z1qr_Qr5p=N8vn6v10&f7owrp6_VQF=D)
2. [BTS - Airline On-Time Performance Data](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr)
3. [NOAA - Access](https://www.ncei.noaa.gov/access)
4. [NOAA - Local Climatological Data](https://www.ncei.noaa.gov/data/local-climatological-data/access/)
5. [ASPM - Aviation System Performance Metrics](https://www.aspm.faa.gov/apm/sys/main.asp)

### Papers
1. Kenney Snell, Jozef Zurada, Jan Kozak, and Zahra Hatami, *Predicting Flight Delays Using Machine Learning*. **Primary reference.**
2. Micha Zoutendijk and Mihaela Mitici, *Probabilistic Flight Delay Predictions Using Machine Learning and Applications to the Flight-to-Gate Assignment Problem*. **Primary reference.**
3. Juan Pineda-Jaramillo, Claudia Munoz, Rodrigo Mesa-Arango, Carlos Gonzalez-Calderon, and Anne Lange, *Integrating Multiple Data Sources for Improved Flight Delay Prediction Using Explainable Machine Learning*. **Primary reference.**
4. Meng Li, *Air Traffic Delay Prediction Based on Machine Learning and Delay Propagation*.
5. Jun Chen and Meng Li, *Chained Predictions of Flight Delay Using Machine Learning*.
6. Maarten Beltman, Marta Ribeiro, Jasper de Wilde, and Junzi Sun, *Dynamically Forecasting Airline Departure Delay Probability Distributions for Individual Flights Using Supervised Learning*.
7. Sarah Ahmed A. AlBassam, *Flight Delay Prediction: Evaluating Machine Learning Algorithms for Enhanced Accuracy*.
