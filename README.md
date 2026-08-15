# Capstone Project - Berkeley ML and AI (in progress)
## Initial Report and Exploratory Data Analysis

## Table of Contents

- [Overview](#overview)
  - [Predict departure delay](#predict-departure-delay)
  - [Predict arrival delay](#predict-arrival-delay)
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
    - [Rotation Matching Audit](#rotation-matching-audit)
  - [Current Model 1A Feature Assembly](#current-model-1a-feature-assembly)
- [Modeling](#modeling)
  - [Experiment roadmap](#experiment-roadmap)
  - [How experiments are compared](#how-experiments-are-compared)
  - [Arrival prediction comparison](#arrival-prediction-comparison)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [References](#references)
  - [Data Sources](#data-sources-1)
  - [Papers](#papers)
- [Appendix A](#appendix-a)
  - [BTS Column Selection and Dictionary](#bts-column-selection-and-dictionary)
    - [Prediction-time eligibility of key BTS operational fields](#prediction-time-eligibility-of-key-bts-operational-fields)
    - [BTS columns retained for modeling](#bts-columns-retained-for-modeling)
    - [Dropped airline, airport, time-block, and operational columns](#dropped-airline-airport-time-block-and-operational-columns)
    - [Dropped diversion summary columns](#dropped-diversion-summary-columns)
    - [Dropped diversion-stop columns](#dropped-diversion-stop-columns)
    - [Dropped export artifact](#dropped-export-artifact)
  - [ASPM Column Selection and Dictionary](#aspm-column-selection-and-dictionary)
    - [ASPM columns retained for modeling](#aspm-columns-retained-for-modeling)
    - [ASPM columns dropped during processing](#aspm-columns-dropped-during-processing)
    - [ASPM names after merge](#aspm-names-after-merge)
  - [NOAA Column Selection and Dictionary](#noaa-column-selection-and-dictionary)
    - [NOAA columns retained for modeling](#noaa-columns-retained-for-modeling)
    - [NOAA fields dropped during processing](#noaa-fields-dropped-during-processing)
    - [NOAA names after merge](#noaa-names-after-merge)
- [Appendix B](#appendix-b)
  - [Joined Data Column Dictionary](#joined-data-column-dictionary)
    - [Joined BTS flight columns](#joined-bts-flight-columns)
    - [Joined ASPM planned-demand columns](#joined-aspm-planned-demand-columns)
    - [Joined NOAA weather columns](#joined-noaa-weather-columns)
  - [Feature Engineering](#feature-engineering)
    - [Engineered feature dictionary](#engineered-feature-dictionary)
    - [Feature selection and timing rules](#feature-selection-and-timing-rules)
  - [Operational Backlog Feature Engineering](#operational-backlog-feature-engineering)
    - [Backlog feature dictionary](#backlog-feature-dictionary)
    - [Same-airline backlog feature dictionary](#same-airline-backlog-feature-dictionary)
    - [Backlog timing and availability rules](#backlog-timing-and-availability-rules)
  - [Aircraft Rotation Feature Engineering](#aircraft-rotation-feature-engineering)
    - [Rotation feature dictionary](#rotation-feature-dictionary)
    - [Rotation timing, leakage, and availability rules](#rotation-timing-leakage-and-availability-rules)
  - [Current Model 1A Operational Feature Manifest](#current-model-1a-operational-feature-manifest)
- [Appendix C](#appendix-c)
  - [Experiment protocol](#experiment-protocol)
  - [Primary binary-classification experiments](#primary-binary-classification-experiments)
  - [Reference models and project scope](#reference-models-and-project-scope)
- [Appendix D](#appendix-d)
  - [Results recording rules](#results-recording-rules)
  - [Experiment configurations](#experiment-configurations)
  - [Ranking and calibration results](#ranking-and-calibration-results)
  - [Operating-threshold results](#operating-threshold-results)
    - [Current Model 1A logistic-regression comparison](#current-model-1a-logistic-regression-comparison)
    - [Current Model 1A exact-manifest Random Forest comparison](#current-model-1a-exact-manifest-random-forest-comparison)
    - [Current Model 1A CatBoost/MLP blend comparison](#current-model-1a-catboostmlp-blend-comparison)
    - [Current Model 1A CatBoost calibration comparison](#current-model-1a-catboost-calibration-comparison)
    - [Current Model 1A CatBoost subgroup and SHAP audit](#current-model-1a-catboost-subgroup-and-shap-audit)
- [Appendix E](#appendix-e)
  - [Confusion matrix](#confusion-matrix)
  - [Threshold-dependent classification metrics](#threshold-dependent-classification-metrics)
  - [Probability ranking and calibration metrics](#probability-ranking-and-calibration-metrics)
  - [Related evaluation terms](#related-evaluation-terms)
- [Appendix F](#appendix-f)

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

The arrival-delay category applies the same basic flight-level classification approach as the departure-delay category:
combine schedule, route, planned airport demand, and weather information to predict whether an individual
flight crosses the 15-minute delay threshold. 

Models 2A, 2B, and 2C update the arrival-delay prediction as the flight progresses through its operational timeline. 
Model 2A makes an initial prediction using information available before pushback. Model 2B revises that prediction after the flight departs, 
using the actual departure time and delay. Model 2C updates it once more using observed taxi-out and takeoff information. 
Together, the three models show how arrival-delay predictions can become more informed and potentially more accurate as 
actual operating information replaces earlier assumptions.

Arrival-delay modeling has substantially larger and more complex data requirements. Model 1A uses flights departing from JFK and can use JFK ASPM and NOAA data. Models 2A, 2B, and 2C use flights arriving at JFK from many different origin
airports. A complete implementation therefore requires appropriate ASPM and NOAA coverage for those origins, NOAA
station mapping, source-specific cleaning and validation, and joins for every inbound flight.

Each model follows a clear prediction timeline, and a field is included only if it would be known at that time. Information
recorded later is excluded even when it appears in the historical datasets. This prevents the model from learning from
the future, often called data leakage. BTS provides the historical event values used for this analysis. A working
operational model would need equivalent gate-out and takeoff information from a suitable live source. Some 
BTS columns represent events—such as gate departure and takeoff—that an airline or  airport operational system can observe 
when they occur.

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

Both scenarios use the same basic idea: combine flight-level BTS information with planned airport demand and
weather to classify whether a flight crosses the 15-minute delay threshold. The difference is the scale of the external
data work. Model 1A always departs from JFK, so one ASPM dataset and one NOAA station can serve every row. In the
arrival scenario, the departure airport changes from flight to flight. Each included origin needs ASPM coverage, a NOAA
station mapping, source cleaning and validation, and its own prediction-time-safe joins.

Models 2B and 2C also introduce later BTS operating events, but those fields do not create the main data expansion. The
larger burden comes from collecting and validating consistent ASPM and NOAA context across the inbound flights.
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
and then gathers conditions at each flight's origin. In the diagrams below, `YEAR` represents 2019, 2023, or 2024.

#### Departure-delay data flow: Model 1A

![Departure-delay data flow for Model 1A](resources/diagrams/departure-data-flow.svg)

For each year, the cleaned BTS, ASPM, and NOAA files provide flight, planned airport traffic, and weather data for JFK.
The departure merge keeps flights whose `Origin` is `JFK`, with one row for each eligible flight. It adds JFK's scheduled
arrivals and departures for the hour before, the hour containing, and the hour after the scheduled departure. It also
adds the latest JFK weather report available by scheduled departure, provided that the report is no more than 90 minutes
old.

`feature_departures.ipynb` then adds fixed, calculated features that would be available before pushback. The saved table
also keeps the `DepDel15` target and columns needed for checking the data. When Model 1A is trained, the approved input
list in `notebooks/feature_engineering.py` selects only the permitted predictors. Steps that learn from data—including
filling missing values, encoding categories, scaling, selecting features, balancing classes, and choosing a decision
threshold—are fitted on training data only.

#### Arrival-delay data flow: Models 2A, 2B, and 2C

![Arrival-delay data flow for Models 2A, 2B, and 2C](resources/diagrams/arrival-data-flow.svg)

The arrival path covers flights to JFK from the working group of 54 origin airports. `data/bts/cat_bts.py` combines the
cleaned airport files, keeps flights whose `Dest` is `JFK`, standardizes flight numbers, and removes duplicate copies of
the same flight. The resulting flight table identifies which origins are needed. The ASPM and NOAA scripts then combine
the cleaned traffic and weather files for those origins. This is the main reason the arrival preparation is larger than
the departure preparation.

For each JFK-bound flight, `merge_arrivals.ipynb` looks up planned traffic and weather at that flight's `Origin`. It adds
the origin's planned traffic for the previous, current, and next hours, together with the latest origin weather report
available by scheduled departure and no more than 90 minutes old. The merged file keeps the airport codes, source times,
and time differences needed to check each match. The current design does not add JFK destination traffic or weather
data.

`feature_arrivals.ipynb` creates one feature table that supports all three arrival prediction times. Model 2A uses only
information available before pushback. Model 2B adds the actual gate-departure time and departure delay. Model 2C also
adds taxi-out and takeoff information. Separate approved input lists in `notebooks/feature_engineering.py` make sure that
information from a later stage of the flight cannot be used for an earlier prediction.

See [Appendix B: Feature Engineering](#feature-engineering) for the candidate features, their construction, and their rationale.

### Matching Records by Time

The scheduled departure date and time, stored in `DATE`, provides the main time for each BTS flight. Each flight is matched
with ASPM schedule records for the previous, current, and next clock hours. The next-hour values are planned counts known
ahead of time, not future operating results. Each flight is also matched with the most recent NOAA observation available
before its scheduled departure. The merge notebook keeps source timestamps and timing differences so the matches can be checked.

This time-based matching prevents a model from using airport conditions or weather observations that occurred after the prediction was made. It also allows the age of the matched information to be checked before modeling.

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
│   │       └── JFK.csv
│   ├── processed/
│   │   ├── _csv_files.zip
│   │   └── JFK_YEAR.csv
│   ├── cleaned/
│   │   ├── _csv_files.zip
│   │   └── JFK_YEAR.csv
│   └── cleaned_JFK_YEAR.csv
├── aspm/
│   ├── raw/
│   │   └── aspm_output/
│   │       └── run_YEAR_AIRPORT/
│   │           ├── aspm_YEAR_AIRPORT.csv
│   │           └── raw_html.zip
│   ├── processed/
│   │   ├── _csv_files.zip
│   │   └── JFK_YEAR.csv
│   ├── cleaned/
│   │   ├── _csv_files.zip
│   │   └── JFK_YEAR.csv
│   └── cleaned_JFK_YEAR.csv
├── noaa/
│   ├── raw/
│   │   ├── airport_station_map.csv
│   │   └── YEAR/
│   │       ├── YEAR_csv_files.zip
│   │       └── STATION.csv
│   ├── processed/
│   │   ├── _csv_files.zip
│   │   └── JFK_YEAR.csv
│   ├── cleaned/
│   │   ├── _csv_files.zip
│   │   └── JFK_YEAR.csv
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

`YEAR` stands for 2019, 2023, or 2024. `AIRPORT` is an airport included in the project, and `STATION` is the NOAA weather
station matched to that airport. Within each source directory, `cleaned/` holds one cleaned CSV for each airport and
year. The `cleaned_JFK_YEAR.csv` files combine the airport data needed for flights arriving at JFK, giving the arrival
merge one consolidated input from each source.

This layout makes it possible to follow the data from the original sources to the final model-ready features. Along the
way, the preparation steps check that every merged or feature row still represents one eligible flight and that a model
uses only information that would have been available when its prediction is made. Model 1A predicts departure delay for
flights leaving JFK. Models 2A, 2B, and 2C predict arrival delay for flights headed to JFK, using conditions at the
flight's origin and adding operating information as it becomes available.

### BTS

The `data/bts/` directory provides the flight records and delay targets for both scenarios. Each cleaned row represents
one completed domestic flight. BTS On-Time Performance data is downloaded as monthly files covering flights reported by
the included United States airlines. For each airport and year, these raw files are filtered into one annual
`AIRPORT.csv` containing flights that either depart from or arrive at that airport.

`process_bts.ipynb` keeps the schedule, airline, route, distance, operating-event, and outcome fields needed for the
project. It removes cancelled and diverted flights because their delay outcomes cannot be compared fairly with those of
completed flights. It then writes `data/bts/processed/AIRPORT_YEAR.csv`. `clean_bts.ipynb` standardizes and checks dates,
clock fields, airport and carrier codes, duplicate-flight keys, numeric ranges, and delay labels. It also creates `DATE`,
the scheduled-departure timestamp used by both merge paths, and writes the cleaned airport file to
`data/bts/cleaned/AIRPORT_YEAR.csv`.

The two scenarios use these cleaned records differently:

- **Departure scenario:** `data/bts/cleaned/JFK_YEAR.csv` provides flights with `JFK` as their `Origin`. `DepDel15` is the
  target for Model 1A.
- **Arrival scenario:** `data/bts/cat_bts.py` combines the cleaned airport-year files, keeps flights with `JFK` as their
  `Dest`, standardizes flight numbers, removes duplicate rows created by overlapping files, and writes
  `data/bts/cleaned_JFK_YEAR.csv`. `ArrDel15` is the target shared by Models 2A, 2B, and 2C. The BTS row also keeps the
  pushback and takeoff fields that Models 2B and 2C may use at their later prediction times.

### ASPM

The `data/aspm/` directory provides the planned number of arrivals and departures at each airport, rather than results
for individual flights. Each cleaned row represents one airport and one clock hour. The raw data contains one annual
hourly file for each airport. `process_aspm.ipynb` removes supporting calculation fields and actual performance measures
that are published too late for the project's prediction times. It keeps the airport and time identifiers together with
the scheduled hourly departure and arrival counts, then writes `data/aspm/processed/AIRPORT_YEAR.csv`.

`clean_aspm.ipynb` checks airport-hour keys, numeric values, duplicate records, row order, and hourly coverage. It fills
in missing clock hours where needed and creates an hourly `DATE` timestamp before writing
`data/aspm/cleaned/AIRPORT_YEAR.csv`. The scheduled counts are plans known before departure. Actual ASPM delay and
on-time measures are not used as predictors.

- **Departure scenario:** the merge uses the cleaned JFK file because every Model 1A flight leaves from JFK.
- **Arrival scenario:** `data/aspm/cat_aspm.py` finds the required origin airports in the table of JFK-bound flights and
  combines their cleaned airport-year files into `data/aspm/cleaned_JFK_YEAR.csv`. Each flight is matched to planned
  demand at its origin, not at its JFK destination.

### NOAA

The `data/noaa/` directory provides weather observations from the airport where each flight departs. Each cleaned row is
one selected weather report for an airport and time. Raw Local Climatological Data is stored in one annual CSV for each
weather station. `airport_station_map.csv` links every project airport to its NOAA station so the same airport code can
be used across the data sources.

`process_noaa.ipynb` keeps the useful weather measurements, removes reports with no usable observations, and writes
`data/noaa/processed/AIRPORT_YEAR.csv`. `clean_noaa.ipynb` selects one preferred report for each timestamp, converts trace
precipitation to a small numeric value, turns reported-weather text into condition indicators, and converts wind
direction and speed into `WindX` and `WindY`. A missing continuous measurement may be filled from an earlier report only
when that report is no more than 90 minutes old. Missing values at the beginning of a file and across longer gaps are
left for the model's training pipeline to handle. The cleaned file is written to
`data/noaa/cleaned/AIRPORT_YEAR.csv`.

- **Departure scenario:** the cleaned JFK station-year file provides weather observed at JFK before scheduled departure.
- **Arrival scenario:** `data/noaa/cat_noaa.py` combines the cleaned station data for the origin airports found in the
  JFK-bound BTS table and writes `data/noaa/cleaned_JFK_YEAR.csv`. Each flight receives weather from its origin.
  Destination weather and weather observed near landing are not part of the baseline design.

### Merged

The `data/merged/` directory contains one row for each eligible flight after the flight, planned-demand, and weather data
have been brought together. The merge notebooks keep source airport codes, timestamps, and time differences so that the
airport matches and prediction timing can be checked. If a source record cannot be matched, the notebooks report it
instead of silently removing the flight.

- **Departure scenario:** `merge_departures.ipynb` keeps flights with `Origin == JFK`. It joins JFK ASPM records for the
  previous, current, and next clock hours around scheduled departure and adds the latest JFK NOAA report available at or
  before scheduled departure, as long as it is no more than 90 minutes old. It writes
  `data/merged/JFK_YEAR_departures.csv`.
- **Arrival scenario:** `merge_arrivals.ipynb` keeps flights with `Dest == JFK` and matches the other sources using each
  flight's `Origin`. It adds the origin's three ASPM periods and latest NOAA report available by scheduled departure, as
  long as the report is no more than 90 minutes old. It then writes `data/merged/JFK_YEAR_arrivals.csv`. ASPM and NOAA
  data for the JFK destination are not added.

The next-hour ASPM values in both files are planned schedule counts known in advance, not future operating results. The
merged files keep source and outcome columns for checking the data, but this does not mean that every retained column may
be used as a model predictor.

### Features

The `data/features/` directory contains the feature files used by the modeling notebooks. Each row still represents the
same flight as the matching row in the merged file. This step creates features using fixed formulas only. It does not
fit or choose imputers, encoders, scalers, feature selectors, resampling methods, or decision thresholds. Those steps
belong to modeling rather than shared feature creation.

- **Departure scenario:** `feature_departures.ipynb` adds the shared features available before pushback and writes
  `data/features/JFK_YEAR_departures.csv` for Model 1A.
- **Arrival scenario:** `feature_arrivals.ipynb` writes `data/features/JFK_YEAR_arrivals.csv`. This file contains all
  features needed across the three arrival prediction times. Model 2A uses only schedule and origin information
  available before pushback. Model 2B adds the actual gate-departure time and departure-delay information. Model 2C also
  allows taxi-out and wheels-off information.

These shared files intentionally contain more columns than any one model may use. Each model notebook uses the allowlist
for its prediction time—the named set of columns permitted at that point—from `notebooks/feature_engineering.py`. Any
preprocessing that must learn from the data is fitted on the training data only.

### Operational Backlog Data

The backlog datasets form a separate, additive feature path. They do not replace or modify the shared departure and
arrival feature files described above. This project-specific strategy was introduced to represent realized airport
operating pressure that is not present in the reference-based engineered features. ASPM describes planned hourly
demand; backlog instead reconstructs which earlier-scheduled flights have already pushed back and which remain pending
at a sample flight's prediction cutoff.

The current implementation covers Model 1A JFK departures. The shared helper
[`feature_engineering_backlog.py`](notebooks/feature_engineering_backlog.py) contains the deterministic calculations,
and [`feature_departures_backlog.ipynb`](notebooks/feature_departures_backlog.ipynb) provides the parameterized annual
workflow. For each year, the notebook reads the unchanged `data/features/JFK_YEAR_departures.csv` file. It has generated
separately named `data/features/JFK_YEAR_departures_backlog_w30.csv` and
`data/features/JFK_YEAR_departures_backlog_w60.csv` files for 2019, 2023, and 2024. Both paths retain the original 112
columns and append eight window-specific backlog columns, producing 120 columns without changing the flight rows or
their order.

For a sample flight with scheduled-departure timestamp `T = DATE`, the cohort contains flights scheduled in
`[T - 30 minutes, T)`. The strict upper boundary excludes the sample flight and every other flight scheduled at the
same timestamp. An earlier-scheduled cohort flight is completed only when its reconstructed gate-out timestamp,
`DATE + DepDelay`, is strictly earlier than `T`; otherwise it is pending at the snapshot. Final delay values contribute
to aggregates only for completed cohort flights. Counts are zero when a cohort is empty. Rates and means remain missing
when no completed history exists so the model pipeline can learn their treatment from training data.

The W60 sensitivity applies the identical causal construction to `[T - 60 minutes, T)`. It represents sustained rather
than immediate airport pressure and is maintained as a separate dataset so its results can be compared directly with
W30 without changing the established feature path. Logistic Regression Experiment 08 selected the W60 version over
the corresponding W30 combination. CatBoost Experiment 03 then compared all eight airport-wide W60 fields with the
compact pending-count, completed-count, and mean-signed-delay set while holding the classifier and rotation inputs
fixed. The compact three-field set remained selected, so the comprehensive file is retained for audit and ablation
while the smaller manifest is used by the preferred models.

The same-airline W60 extension is another separate path. The deterministic helper
[`feature_engineering_backlog_airline.py`](notebooks/feature_engineering_backlog_airline.py) applies the established
timing rules independently within normalized `Reporting_Airline` groups, and the parameterized
[`feature_departures_backlog_airline.ipynb`](notebooks/feature_departures_backlog_airline.ipynb) writes
`data/features/JFK_YEAR_departures_backlog_airline_w60.csv`. The 2019 and 2023 files retain the original 112 columns and
append nine `AIRLINE_BACKLOG_W60_...` fields, producing 121 columns without changing row identity or order. The 2024
file is deliberately deferred until the model design is frozen so the final-test year remains outside development.

The generated training file contains ten reporting airlines; 36,464 of 107,430 rows have at least one earlier-scheduled
same-airline departure still pending in the prior hour. The 2023 development file contains eight airlines and 43,531
of 109,983 rows with pending same-airline work. Mean pending count is 0.5685 in 2019 and 0.7171 in 2023. Every
same-airline/cutoff group passes the simultaneous-flight consistency check. Different airlines at one cutoff may have
different values by design. CatBoost Experiment 04 subsequently selected five of the nine same-airline fields using
2019 temporal folds: pending count, completed count, mean signed delay, delay rate, and pending share. Adding them to
the compact airport-wide W60 and rotation base raised 2023 average precision from 0.7473 to 0.7526. The full nine-field
file remains the reusable preparation output; the five-field choice belongs to the model allowlist rather than the
shared feature generator.

The same concept applies to Models 2A, 2B, and 2C, but their origin-specific backlog datasets have not yet been
implemented. An arrival version must calculate each sample's cohort from **all departures at that flight's `Origin`**,
not only flights traveling to JFK. Each origin must be grouped and evaluated in its own local scheduled time; airport
timestamps must not be mixed into one cross-time-zone rolling window. A planned arrival output such as
`data/features/JFK_YEAR_arrivals_backlog_w30.csv` would retain the same rows across 2A, 2B, and 2C so paired backlog and
non-backlog experiments remain directly comparable.

Historical BTS fields reconstruct these snapshots for model development. A deployed version would require timely
schedule and gate-out events at JFK for Model 1A and at every included origin for the arrival models. The construction,
column definitions, leakage safeguards, and feature-selection considerations are documented separately in
[Appendix B: Operational Backlog Feature Engineering](#operational-backlog-feature-engineering).

### Aircraft Rotation Data

Aircraft rotation is implemented as another separate, additive Model 1A path. It does not change the standard or
backlog feature definitions. [`feature_departures_rotation.ipynb`](notebooks/feature_departures_rotation.ipynb) reads
the comprehensive `data/features/AIRPORT_YEAR_departures.csv` target table and the cleaned
`data/bts/cleaned_AIRPORT_YEAR.csv` table of flights arriving at that airport. The deterministic calculations are in
[`feature_engineering_rotation.py`](notebooks/feature_engineering_rotation.py). The parameterized workflow generated
`data/features/JFK_YEAR_departures_rotation.csv` for 2019, 2023, and 2024. Each file preserves the 112 standard columns
and appends 13 rotation features plus eight audit fields, for 133 columns without changing row count or order.

These first rotation files use the project's cleaned, working-airport cohort and remain intact as the Experiment 05
source. The current preferred path instead uses the separately named full-history files described below. Both paths
share the same 13-feature and eight-audit-field schema, so the history source can be compared without changing the
feature definitions.

For a target flight with scheduled-departure cutoff `T`, local airport schedule times are first converted to UTC. The
target aircraft's immediately preceding known JFK schedule event is then identified. A preceding scheduled arrival is
a rotation match. A preceding scheduled departure blocks any older arrival so a stale inbound leg is not silently
reused after the aircraft has already been scheduled to leave JFK. The inbound scheduled-arrival date is reconstructed
across time zones by selecting the destination-local date whose UTC duration most closely matches BTS
`CRSElapsedTime`; the residual remains in an audit column.

The key causal distinction is whether the assigned inbound aircraft had actually arrived by `T`. If it had arrived,
its signed arrival delay, 15-minute delay indicator, and actual ground time were observable and may be used. If it had
not arrived, those eventual outcomes remain missing; only the observable not-arrived state and the number of minutes
since its scheduled arrival are retained. The annual outputs passed all row-preservation, feature-identity, and
causal-masking checks:

| Year | Departure rows | Rotation matches | Match rate | Not arrived by cutoff | Not-arrived departure delay rate |
|---:|---:|---:|---:|---:|---:|
| 2019 | 107,430 | 102,741 | 95.64% | 2,638 | 99.77% |
| 2023 | 109,983 | 102,716 | 93.39% | 2,995 | 99.63% |
| 2024 | 104,715 | 98,350 | 93.92% | 2,660 | 99.74% |

These rates show why rotation can be substantially more informative than aggregate backlog: an aircraft that has not
yet reached JFK cannot normally operate its assigned departure on time. These descriptive rates alone do not establish
outside-year model performance; the controlled 2023 results are recorded in Appendix D. The historical reconstruction
also relies on the final BTS `Tail_Number`. A deployable version therefore
requires a timestamped aircraft-assignment feed and live arrival events; unless the recorded tail can be shown to match
the assignment known at `T`, rotation experiment results must be presented as a retrospective upper bound. The full
feature definitions and boundary limitations are documented in
[Appendix B: Aircraft Rotation Feature Engineering](#aircraft-rotation-feature-engineering).

#### Rotation Matching Audit

[`audit_rotation.ipynb`](notebooks/audit_rotation.ipynb) is a parameterized, read-only audit of the rotation match
history for one airport and year. It compares the existing cohort-limited history with every BTS movement present in
the raw airport file. The fuller history includes completed, non-diverted inbound arrivals and treats every
non-cancelled outbound movement—including a flight later diverted—as a blocking departure event. The audit does not
write or replace any feature dataset.

The audit was run for each project year. Raw history adds flights from 21–23 origins outside the current modeling
airport cohort and has a material effect on the exact preceding-leg reconstruction:

| Year | Added inbound history | Match rate, current → full | Different prior inbound | Rotation status changed | Turns >24 h, current → full |
|---:|---:|---:|---:|---:|---:|
| 2019 | 17,315 | 95.64% → 96.65% | 8,443 (7.86%) | 2,065 | 10,668 → 8,281 |
| 2023 | 19,595 | 93.39% → 95.54% | 8,940 (8.13%) | 3,698 | 10,822 → 8,409 |
| 2024 | 16,031 | 93.92% → 95.93% | 7,447 (7.11%) | 3,083 | 10,297 → 8,343 |

Full history reduces apparent turns longer than 24 hours by 19–22%, although 7.6–8.0% of target rows still have such
long matched turns. Those remaining cases can include genuine overnight or maintenance turns, movements outside BTS,
year-boundary truncation, and final-tail assignment limitations. Most importantly, the principal signal is robust:
the departure-delay rate for aircraft not arrived by the prediction cutoff remains approximately 99.7–99.8% under
full history.

The audit supports an append-only improvement before a rotation ablation experiment: generate a separately named
full-history rotation dataset and retain the original files and Logistic Regression 1A Experiment 05 unchanged. This
will isolate the value of more complete movement history from changes to the feature family or classifier.

That recommendation is implemented by
[`feature_departures_rotation_full_history.ipynb`](notebooks/feature_departures_rotation_full_history.ipynb), which
generated `data/features/JFK_YEAR_departures_rotation_full_history.csv` for 2019, 2023, and 2024. Each file preserves
the target rows, the 112-column base table, and the established 21-column rotation schema. The only intentional change
is that rotation order uses every eligible raw BTS movement at JFK: completed, non-diverted inbound arrivals and all
non-cancelled outbound blocking events. The original cohort-limited rotation files remain intact.

The full-history files are now the selected rotation source for Model 1A experiments after Logistic Regression
Experiment 06. The 24-hour rule is applied in the model notebook rather than written into these reusable datasets:
matches with scheduled turns over 1,440 minutes have their rotation values masked and receive the distinct
`LONG_TURN_EXCLUDED` status, while every target row is retained.

| Year | Departure rows | Full-history matches | Match rate | Not arrived by cutoff | Scheduled turns over 24 h |
|---:|---:|---:|---:|---:|---:|
| 2019 | 107,430 | 103,834 | 96.65% | 3,133 | 8,281 |
| 2023 | 109,983 | 105,082 | 95.54% | 3,565 | 8,409 |
| 2024 | 104,715 | 100,454 | 95.93% | 3,113 | 8,343 |

The 2019 and 2023 values are used during model development. The 2024 file has been generated and validated but remains
outside experiment selection. The selected 41-field model additionally needs the deferred 2024 same-airline backlog
file before its single final-test evaluation.

### Current Model 1A Feature Assembly

The selected operational feature families remain separate preparation outputs. CatBoost Experiment 04 and Random
Forest Experiment 02 pair them in memory only after confirming identical flight keys, row order, target values, and
backlog cutoffs. No `departures_rotation_backlog` CSV is created.

| Preparation source | Fields admitted by the current Model 1A allowlist | Role |
|---|---:|---|
| `JFK_YEAR_departures_rotation_full_history.csv` | 20 raw pre-pushback fields + 13 rotation fields | Schedule, planned demand, weather, and aircraft state; rotation values use the 24-hour mask. |
| `JFK_YEAR_departures_backlog_w60.csv` | 3 | Airport-wide pending count, completed count, and mean signed departure delay. |
| `JFK_YEAR_departures_backlog_airline_w60.csv` | 5 | Same-airline pending count, completed count, mean signed delay, delay rate, and pending share. |
| **Total** | **41** | Six categorical and 35 numeric source fields before model-specific preprocessing. |

The source CSVs remain comprehensive and retain audit and target outcomes. Presence is not permission: each experiment
uses the explicit 41-field allowlist and prohibits target-flight operating outcomes, raw `Tail_Number`, and the eight
rotation audit fields. Learned imputation, encoding, scaling, feature selection, calibration, and class handling remain
inside training-only model pipelines rather than data preparation.

[Appendix A](#appendix-a) documents the [BTS](#bts-column-selection-and-dictionary),
[ASPM](#aspm-column-selection-and-dictionary), and [NOAA](#noaa-column-selection-and-dictionary) source-column
decisions. [Appendix B](#appendix-b) documents the
[joined data columns](#joined-data-column-dictionary) and
[feature-engineering analysis](#feature-engineering).

## Modeling

The project uses four models to answer two related questions: whether a flight will leave JFK at least 15 minutes late,
and whether a flight headed to JFK will arrive at least 15 minutes late. The arrival prediction is updated as more
information becomes available during the flight.

| Model | Prediction time | Flights and target | Information available |
|---|---|---|---|
| 1A | Before pushback | JFK departures; predict `DepDel15` | Schedule, route, planned JFK traffic, and JFK weather |
| 2A | Before pushback at the origin | JFK arrivals; predict `ArrDel15` | Schedule, route, planned origin traffic, and origin weather |
| 2B | Immediately after pushback | Same JFK arrivals and target as 2A | Model 2A information plus actual gate-out time and departure delay |
| 2C | Immediately after takeoff | Same JFK arrivals and target as 2A | Model 2B information plus taxi-out time and actual takeoff time |

[Appendix C](#appendix-c) contains the full experiment plan and notebook registry. [Appendix D](#appendix-d) records the
settings and results for completed experiments.

### Experiment roadmap

The core experiment plan has two stages. Both focus on the project's main goal: predicting whether a flight will be at
least 15 minutes late.

| Stage | Main question | Work included |
|---|---|---|
| I | Which approach works best for predicting a JFK departure delay before pushback? | Compare Model 1A feature sets and a broad group of methods: logistic regression, decision trees, K-nearest neighbors (KNN), Naive Bayes, support vector machine (SVM), linear discriminant analysis (LDA), bagging, Random Forest, Extra Trees, boosting, CatBoost, and a neural network. Also compare methods for handling the smaller delayed class and for improving predicted probabilities. |
| II | How much does arrival prediction improve when pushback and takeoff information become available? | Compare logistic regression, decision tree, Random Forest, and CatBoost across Models 2A, 2B, and 2C using the same JFK-arrival flights. A method from Stage I may also be included if it provides a clear benefit. |

Stage I provides a broad but manageable comparison on Model 1A. Stage II then applies the main model families to the
three arrival prediction times. This avoids running every possible method and feature set at every prediction time while
still allowing a strong Stage I method to be carried into the arrival comparison.

The [experiment registry](#primary-binary-classification-experiments) lists the Stage I and II notebooks.

### How experiments are compared

Every experiment follows the same rules so the results can be compared fairly:

1. **Keep the flight populations consistent.** Model 1A uses JFK departures and `DepDel15`. Models 2A, 2B, and 2C use
   the same JFK-arrival flights and `ArrDel15`. Only the information available at each arrival prediction time changes.
2. **Respect time order.** Features and model settings are chosen by moving through complete days of 2019 in time order:
   earlier periods are used for training and later periods for checking the model. The selected modeling process is then
   trained on 2019 and checked on the complete 2023 dataset. The 2024 data is reserved for the final test after the model
   and decision threshold have been chosen.
3. **Prevent future information from entering the model.** Each notebook has a clear list of fields allowed at its
   prediction time. Any step that learns from the data—including filling missing values, converting categories into
   numeric inputs, scaling values, choosing features, balancing the two outcome classes, adjusting probabilities, and
   selecting a decision threshold—uses training data only.
4. **Use measures suited to an uncommon outcome.** Delayed flights are the smaller class, so average precision is the
   main measure used to rank models. ROC AUC provides another view of ranking, and the Brier score checks the quality of
   predicted probabilities. At a chosen cutoff, precision shows how often delay alerts are correct, recall shows how
   many delayed flights are found, and F1 balances those two measures. Ordinary accuracy is supporting information
   because it can look good even when a model misses many delayed flights. Appendix C lists the additional measures.
5. **Choose the decision threshold separately.** Results are reported at the standard 0.50 cutoff and at a cutoff chosen
   from 2019 training-period predictions. The 2023 or 2024 outcomes are not used to choose that cutoff.
6. **Compare like with like.** Modeling methods are compared within the same model and flight population. Models 2A,
   2B, and 2C are compared on identical arrival rows so any difference reflects the newly available flight information
   rather than a change in the sample.

The full rules for repeatability, class balancing, model explanations, and checks across months, airlines, routes,
weather, and traffic levels are in [Appendix C: Experiment protocol](#experiment-protocol).

### Arrival prediction comparison

Models 2A, 2B, and 2C are designed as a controlled comparison. They use the same flights, date ranges, target, and basic
schedule, route, traffic, and weather fields. Model 2B adds information known at pushback, and Model 2C adds information
known at takeoff. This setup measures the value of waiting for actual operating information.

The field rules follow the same timing. `DepTime` and `DepDelay` are not available to Model 2A because pushback has not
occurred. `TaxiOut` and `WheelsOff` are not available to Models 2A or 2B because they are not fully known until takeoff.
This keeps information from later in the flight out of the earlier predictions.

Earlier research suggests that actual departure information can improve arrival-delay prediction, especially once the
departure delay is known. This project tests that result across the same JFK-bound flights. The comparison also captures
a practical tradeoff: a prediction made after takeoff may be more accurate, but it gives airlines and passengers less
time to act.

Arrival prediction may also be harder than departure prediction. Model 1A always describes operations at JFK, while the
arrival models include flights from many origins with different weather, traffic, and airport conditions. Using the same
arrival rows for Models 2A, 2B, and 2C makes that added variation part of every arrival experiment rather than a source
of unfair differences between them.

Completed experiment configurations, model-comparison measures, and decision-threshold results are recorded in
[Appendix D](#appendix-d).

## Evaluation

CatBoost Experiment 04 is the preferred completed Model 1A departure candidate. On the complete 2023 development year,
its average precision is 0.7526, ROC AUC is 0.8546, and Brier score is 0.1086. At the 0.31 threshold selected only from
held-out 2019 predictions, precision is 0.7327, recall is 0.5858, F1 is 0.6511, and MCC is 0.5640. The simpler Random
Forest and MLP comparisons confirm the nonlinear operational-feature gain but do not improve enough to replace it; a
CatBoost/MLP blend and two probability corrections also fail to provide a material outside-year benefit.

The selected-model audit shows that the aggregate score hides meaningful operating differences. Ranking and recall are
lowest for the small 00:00–05:59 departure group and rise as the airport becomes busier: the 2019-defined low-traffic
quartile has 2023 AP 0.5601 and recall 0.3785, compared with AP 0.8145 and recall 0.6739 in the high-traffic quartile.
The model underpredicts the 2023 delay rate in every month and in all four weather groups. The largest month-level gap
is July at -0.0453; the low-visibility gap is -0.0364. These are monitoring findings, not reasons to retune on 2023.
The 0.31 threshold and unchanged CatBoost probabilities remain selected.

Global SHAP values on a fixed 5,000-row 2023 sample identify actual and log actual turn time as the leading contribution
magnitudes, followed by inbound arrival delay, rotation status, scheduled departure time, and airport-wide pending
count. Airport-wide and same-airline backlog fields both appear in the leading set. These explanations show fitted
associations rather than causal effects, and the rotation-based result remains a retrospective upper bound until the
aircraft assignment known at the prediction cutoff can be verified.

For arrivals, the controlled logistic-regression comparison confirms the planned information ladder. Model 2A reaches
2023 AP 0.4019 before pushback. Adding signed departure delay raises Model 2B AP to 0.8682, and adding realized taxi-out
and takeoff information raises Model 2C AP to 0.9090. The 2024 departure and arrival outcomes remain reserved for the
single final evaluation after the complete model design is frozen.

## Deployment

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

# Appendix A

This appendix explains which BTS, ASPM, and NOAA columns are kept, which are removed, and when each retained field is
available to the four models. It describes the cleaned source files that feed the departure and arrival merges.

## BTS Column Selection and Dictionary

The downloaded BTS data contains 110 columns. Processing removes cancelled and diverted flights and drops 83 columns
that are not needed. Cleaning checks the remaining fields and adds `DATE`, producing 28 columns before the ASPM and NOAA
data are joined.

A field must be both useful and available at the time a prediction is made. Some BTS fields describe events that happen
later in the flight. Using one of those fields too early would give the model information from the future, often called
time or target leakage.

BTS is a historical reporting source, but some of its columns describe events—such as pushback and takeoff—that an
airline or airport system could observe when they happen. `DepTime`, `TaxiOut`, and `WheelsOff` are therefore kept, but
their use depends on the model's prediction time. `DepTime` becomes available to Model 2B after pushback. `TaxiOut` is
not complete, and `WheelsOff` is not known, until takeoff, so they become available only to Model 2C.

`clean_bts.ipynb` writes one 28-column file for each airport and year under `data/bts/cleaned/`. For the arrival
scenario, `data/bts/cat_bts.py` combines the required origin-airport files into `data/bts/cleaned_JFK_YEAR.csv` without
changing the column layout.

### Prediction-time eligibility of key BTS operational fields

| Field | Model 1A: before pushback | Model 2A: before pushback | Model 2B: after pushback | Model 2C: after takeoff |
|---|---|---|---|---|
| Scheduled fields such as `CRSDepTime`, `CRSArrTime`, and `CRSElapsedTime` | Eligible | Eligible | Eligible | Eligible |
| `DepTime` and departure-delay fields derived from gate-out | Excluded: not yet known | Excluded: not yet known | Eligible: gate-out has occurred | Eligible |
| `DepDel15` | Target only | Excluded: not yet known | Eligible: gate-out has occurred | Eligible |
| `TaxiOut` and `WheelsOff` | Excluded: takeoff has not occurred | Excluded: takeoff has not occurred | Excluded: takeoff has not occurred | Eligible |
| `ArrDel15` | Not used | Target only | Target only | Target only |
| Other arrival outcomes and post-arrival fields | Excluded | Excluded | Excluded | Excluded |

“Dropped” means that a field is not needed or is not appropriate for this project. It may still be useful in other
aviation studies.

### BTS columns retained for modeling

| Column | Meaning | Why it is retained / how it is used |
|---|---|---|
| `Year` | Calendar year of the flight. | Retained for coverage checks and comparison across 2019, 2023, and 2024. |
| `Quarter` | Calendar quarter, numbered 1 through 4. | Retained as a calendar field; it may later be redundant with month-based features. |
| `Month` | Calendar month, numbered 1 through 12. | Retained for seasonal analysis and feature engineering. |
| `DayofMonth` | Day of the month. | Retained for calendar validation and possible calendar features. |
| `DayOfWeek` | Day of week, where BTS uses 1 for Monday through 7 for Sunday. | Retained for weekly-pattern analysis and feature engineering. |
| `FlightDate` | Scheduled flight date without a time component. | Converted to a pandas datetime and retained as the base calendar date. |
| `Reporting_Airline` | BTS reporting carrier code. | Retained as the main airline identifier. |
| `Tail_Number` | Aircraft registration or tail number. | Retained for audit and validation. Standard models do not use the registration as a predictor; the separate rotation path uses it only to match a target departure to a preceding inbound leg. |
| `Flight_Number_Reporting_Airline` | Flight number assigned by the reporting carrier. | Retained as a flight identifier and possible categorical feature; it is not treated as a continuous quantity. |
| `Origin` | Three-letter origin airport code. | Retained to identify flights originating at JFK for Model 1A and the departure airport for Models 2A, 2B, and 2C. |
| `OriginState` | Two-letter state code for the origin airport. | Retained as a compact geographic field. |
| `Dest` | Three-letter destination airport code. | Retained to identify flights arriving at JFK for Models 2A, 2B, and 2C. |
| `DestState` | Two-letter state code for the destination airport. | Retained as a compact geographic field. |
| `CRSDepTime` | Computer Reservation System scheduled departure time in local HHMM form. | Retained because it is known before departure and is used to construct the exact scheduled departure timestamp. |
| `DepTime` | Actual gate departure time in local HHMM form. | Retained as the event-time field for Model 2B and for descriptive analysis. It is excluded from Models 1A and 2A because gate-out has not occurred when those predictions are made. BTS supplies the historical value; a deployed Model 2B would require an operational gate-out feed. |
| `DepDelay` | Actual gate departure time minus scheduled departure time, in minutes; negative values indicate an early departure. | Retained as a Model 2B predictor because it is known once gate-out occurs. It is excluded from Models 1A and 2A. Because it overlaps the other departure-delay fields, the Model 2B feature list should avoid unnecessary duplicate versions of the same information. |
| `DepDelayMinutes` | Nonnegative departure delay in minutes; early departures are recorded as zero. | Retained as a possible Model 2B representation and to validate `DepDelay` and `DepDel15`. It is excluded from pre-pushback predictors and need not be included together with every related departure-delay field. |
| `DepDel15` | Indicator equal to 1 when departure delay is at least 15 minutes. | Target for Model 1A. It is excluded from Model 2A but may be used as a compact post-pushback input for Model 2B because the departure outcome is known at gate-out. It is never used to predict itself in Model 1A. |
| `DepartureDelayGroups` | Departure delay grouped into ordered 15-minute ranges. | Retained for validation, EDA, and possible Model 2B use. It is excluded before pushback and is not automatically included alongside the continuous and binary departure-delay fields. |
| `TaxiOut` | Minutes from gate departure to wheels-off. | Retained as a Model 2C predictor and for validation and EDA. It is excluded from Models 1A, 2A, and 2B because the full taxi-out duration is unknown immediately after pushback. |
| `WheelsOff` | Actual takeoff time in local HHMM form. | Retained as a Model 2C predictor and for validation and EDA. It is excluded from Models 1A, 2A, and 2B because takeoff occurs after their prediction times. As with `DepTime`, BTS is the historical source rather than the proposed live event feed. |
| `CRSArrTime` | Scheduled arrival time in the destination's local HHMM form. | Retained because it is known from the schedule before departure. |
| `WheelsOn` | Actual landing time in destination-local HHMM form. | Retained for audit and rotation-source validation. It is not a direct predictor for the target flight. |
| `TaxiIn` | Minutes from wheels-on to gate arrival. | Retained for audit and rotation-source validation. It remains unavailable for the target flight at the project's prediction times. |
| `ArrTime` | Actual gate-arrival time in destination-local HHMM form. | Retained for audit and reconstruction of a preceding aircraft leg. It is prohibited as a target-flight predictor. |
| `ArrDelay` | Signed actual gate-arrival delay in minutes. | Retained so an already-arrived preceding leg can contribute its observed delay in the separate rotation path. The target flight's value is prohibited. |
| `ArrDelayMinutes` | Nonnegative gate-arrival delay in minutes. | Retained for audit and validation; it is not a direct Model 1A input. |
| `ArrDel15` | Indicator equal to 1 when arrival delay is at least 15 minutes. | Target for Models 2A, 2B, and 2C. It is never used as a predictor because it is known only after arrival. |
| `ArrivalDelayGroups` | Arrival delay grouped into ordered 15-minute ranges. | Retained for target validation and EDA. It is excluded from every model input because it describes the completed arrival outcome. |
| `CRSElapsedTime` | Scheduled gate-to-gate elapsed time, in minutes. | Retained as a schedule and route characteristic known before departure. |
| `ActualElapsedTime` | Actual gate-to-gate elapsed time in minutes. | Retained for audit and rotation-source validation; it remains prohibited for the target flight before completion. |
| `Distance` | Published distance between origin and destination, in miles. | Retained as a route characteristic. |
| `DistanceGroup` | BTS distance band based on 250-mile intervals. | Retained for grouped analysis and possible categorical modeling. |
| `DATE` | Constructed scheduled departure timestamp created from FlightDate and CRSDepTime. | Added during cleaning. It supports sorting, time-based ASPM and NOAA matching, chronological splitting, and time feature engineering. |

### Dropped airline, airport, time-block, and operational columns

| Column | Meaning | Why it is dropped |
|---|---|---|
| `DOT_ID_Reporting_Airline` | Permanent numeric identifier assigned by the U.S. Department of Transportation to the reporting carrier. | Dropped because Reporting_Airline already supplies the carrier identity in a more readable form. |
| `IATA_CODE_Reporting_Airline` | IATA code for the reporting carrier. | Dropped because it duplicates the retained Reporting_Airline code. |
| `OriginAirportID` | BTS numeric identifier for the origin airport. | Dropped because the retained Origin code uniquely and more readably identifies the airport for this project. |
| `OriginAirportSeqID` | Sequence identifier for a specific historical version of the origin airport record. | Dropped because historical airport-record versioning is not needed and Origin is retained. |
| `OriginCityMarketID` | BTS identifier for the origin city market, which may include multiple airports. | Dropped because the project analyzes named airports and retains Origin. |
| `OriginCityName` | Origin city and state name. | Dropped because Origin and OriginState retain the required location information with less redundancy. |
| `OriginStateFips` | Numeric FIPS code for the origin state. | Dropped because OriginState provides the needed state identifier. |
| `OriginStateName` | Full name of the origin state. | Dropped because it duplicates OriginState. |
| `OriginWac` | World Area Code for the origin. | Dropped because this domestic-airport project does not require the broader geographic code. |
| `DestAirportID` | BTS numeric identifier for the destination airport. | Dropped because the retained Dest code uniquely and more readably identifies the airport for this project. |
| `DestAirportSeqID` | Sequence identifier for a specific historical version of the destination airport record. | Dropped because historical airport-record versioning is not needed and Dest is retained. |
| `DestCityMarketID` | BTS identifier for the destination city market, which may include multiple airports. | Dropped because the project analyzes named airports and retains Dest. |
| `DestCityName` | Destination city and state name. | Dropped because Dest and DestState retain the required location information with less redundancy. |
| `DestStateFips` | Numeric FIPS code for the destination state. | Dropped because DestState provides the needed state identifier. |
| `DestStateName` | Full name of the destination state. | Dropped because it duplicates DestState. |
| `DestWac` | World Area Code for the destination. | Dropped because this domestic-airport project does not require the broader geographic code. |
| `DepTimeBlk` | BTS scheduled departure time block. | Dropped because more precise time-of-day features can be derived from CRSDepTime and DATE. |
| `ArrTimeBlk` | BTS scheduled arrival time block. | Dropped because the more precise CRSArrTime is retained and can be used to derive time categories. |
| `Cancelled` | Indicator that the flight was cancelled. | Cancelled flights are removed because they have no completed departure or arrival outcome; the now-constant indicator is then dropped. |
| `CancellationCode` | BTS code for the reason a flight was cancelled. | Dropped after cancelled flights are removed; it is not applicable to the remaining completed flights. |
| `Diverted` | Indicator that the flight was diverted. | Diverted flights are removed because their scheduled-destination arrival outcome is not comparable with an ordinary completed flight; the now-constant indicator is then dropped. |
| `AirTime` | Minutes between wheels-off and wheels-on. | Dropped because it is known only after landing and is outside all four prediction times. |
| `Flights` | BTS row-count field, normally equal to 1 for each flight record. | Dropped because it is an administrative constant rather than a useful flight characteristic. |
| `CarrierDelay` | Minutes of arrival delay attributed to the air carrier. | Dropped because delay-cause minutes are assigned after the outcome and directly reveal that a delay occurred. |
| `WeatherDelay` | Minutes of arrival delay attributed to weather. | Dropped because delay-cause minutes are assigned after the outcome and would leak future information. |
| `NASDelay` | Minutes of arrival delay attributed to the National Airspace System. | Dropped because delay-cause minutes are assigned after the outcome and would leak future information. |
| `SecurityDelay` | Minutes of arrival delay attributed to security. | Dropped because delay-cause minutes are assigned after the outcome and would leak future information. |
| `LateAircraftDelay` | Minutes of arrival delay attributed to a late-arriving aircraft. | Dropped because it is a post-event cause field and aircraft-delay propagation is outside the project scope. |
| `FirstDepTime` | First gate departure time at the origin airport. | Dropped because it describes an irregular event observed after the scheduled prediction time and is sparsely populated. |
| `TotalAddGTime` | Total time away from the gate for a flight that returns to the gate or is cancelled. | Dropped because it is observed after pushback and is not available at the model prediction times. |
| `LongestAddGTime` | Longest period away from the gate for a flight that returns to the gate or is cancelled. | Dropped because it is observed after pushback and is not available at the model prediction times. |

### Dropped diversion summary columns

| Column | Meaning | Why it is dropped |
|---|---|---|
| `DivAirportLandings` | Number of diversion-airport landings. | Dropped because diverted flights are removed from the modeling population. |
| `DivReachedDest` | Indicator that a diverted flight eventually reached its scheduled destination. | Dropped because diverted flights are removed from the modeling population. |
| `DivActualElapsedTime` | Elapsed time for a diverted flight that eventually reaches its scheduled destination. | Dropped because diverted flights are removed and the value is only known after the flight. |
| `DivArrDelay` | Difference between scheduled and actual arrival time for a diverted flight that reaches its scheduled destination. | Dropped because diverted flights are removed and the value directly describes a post-flight outcome. |
| `DivDistance` | Distance between the scheduled destination and the final diversion airport; zero when the diverted flight reaches its scheduled destination. | Dropped because diverted flights are removed from the modeling population. |

### Dropped diversion-stop columns

BTS can record details for as many as five diversion stops. Because diverted flights are removed before modeling, every stop-level field is also removed.

| Column | Meaning | Why it is dropped |
|---|---|---|
| `Div1Airport` | Airport code for diversion stop 1. | Dropped because diverted flights are removed; diversion-stop airport code is outside the completed-flight prediction scope. |
| `Div1AirportID` | BTS numeric airport identifier for diversion stop 1. | Dropped because diverted flights are removed; diversion-stop numeric airport identifier is outside the completed-flight prediction scope. |
| `Div1AirportSeqID` | BTS airport-record sequence identifier for diversion stop 1. | Dropped because diverted flights are removed; diversion-stop historical airport sequence identifier is outside the completed-flight prediction scope. |
| `Div1WheelsOn` | Actual wheels-on time at diversion stop 1. | Dropped because diverted flights are removed; diversion-stop wheels-on time is outside the completed-flight prediction scope. |
| `Div1TotalGTime` | Total ground time at diversion stop 1. | Dropped because diverted flights are removed; diversion-stop total ground time is outside the completed-flight prediction scope. |
| `Div1LongestGTime` | Longest ground-time period at diversion stop 1. | Dropped because diverted flights are removed; diversion-stop longest ground-time period is outside the completed-flight prediction scope. |
| `Div1WheelsOff` | Actual wheels-off time from diversion stop 1. | Dropped because diverted flights are removed; diversion-stop wheels-off time is outside the completed-flight prediction scope. |
| `Div1TailNum` | Aircraft tail number recorded for diversion segment 1. | Dropped because diverted flights are removed; diversion-stop aircraft tail number is outside the completed-flight prediction scope. |
| `Div2Airport` | Airport code for diversion stop 2. | Dropped because diverted flights are removed; diversion-stop airport code is outside the completed-flight prediction scope. |
| `Div2AirportID` | BTS numeric airport identifier for diversion stop 2. | Dropped because diverted flights are removed; diversion-stop numeric airport identifier is outside the completed-flight prediction scope. |
| `Div2AirportSeqID` | BTS airport-record sequence identifier for diversion stop 2. | Dropped because diverted flights are removed; diversion-stop historical airport sequence identifier is outside the completed-flight prediction scope. |
| `Div2WheelsOn` | Actual wheels-on time at diversion stop 2. | Dropped because diverted flights are removed; diversion-stop wheels-on time is outside the completed-flight prediction scope. |
| `Div2TotalGTime` | Total ground time at diversion stop 2. | Dropped because diverted flights are removed; diversion-stop total ground time is outside the completed-flight prediction scope. |
| `Div2LongestGTime` | Longest ground-time period at diversion stop 2. | Dropped because diverted flights are removed; diversion-stop longest ground-time period is outside the completed-flight prediction scope. |
| `Div2WheelsOff` | Actual wheels-off time from diversion stop 2. | Dropped because diverted flights are removed; diversion-stop wheels-off time is outside the completed-flight prediction scope. |
| `Div2TailNum` | Aircraft tail number recorded for diversion segment 2. | Dropped because diverted flights are removed; diversion-stop aircraft tail number is outside the completed-flight prediction scope. |
| `Div3Airport` | Airport code for diversion stop 3. | Dropped because diverted flights are removed; diversion-stop airport code is outside the completed-flight prediction scope. |
| `Div3AirportID` | BTS numeric airport identifier for diversion stop 3. | Dropped because diverted flights are removed; diversion-stop numeric airport identifier is outside the completed-flight prediction scope. |
| `Div3AirportSeqID` | BTS airport-record sequence identifier for diversion stop 3. | Dropped because diverted flights are removed; diversion-stop historical airport sequence identifier is outside the completed-flight prediction scope. |
| `Div3WheelsOn` | Actual wheels-on time at diversion stop 3. | Dropped because diverted flights are removed; diversion-stop wheels-on time is outside the completed-flight prediction scope. |
| `Div3TotalGTime` | Total ground time at diversion stop 3. | Dropped because diverted flights are removed; diversion-stop total ground time is outside the completed-flight prediction scope. |
| `Div3LongestGTime` | Longest ground-time period at diversion stop 3. | Dropped because diverted flights are removed; diversion-stop longest ground-time period is outside the completed-flight prediction scope. |
| `Div3WheelsOff` | Actual wheels-off time from diversion stop 3. | Dropped because diverted flights are removed; diversion-stop wheels-off time is outside the completed-flight prediction scope. |
| `Div3TailNum` | Aircraft tail number recorded for diversion segment 3. | Dropped because diverted flights are removed; diversion-stop aircraft tail number is outside the completed-flight prediction scope. |
| `Div4Airport` | Airport code for diversion stop 4. | Dropped because diverted flights are removed; diversion-stop airport code is outside the completed-flight prediction scope. |
| `Div4AirportID` | BTS numeric airport identifier for diversion stop 4. | Dropped because diverted flights are removed; diversion-stop numeric airport identifier is outside the completed-flight prediction scope. |
| `Div4AirportSeqID` | BTS airport-record sequence identifier for diversion stop 4. | Dropped because diverted flights are removed; diversion-stop historical airport sequence identifier is outside the completed-flight prediction scope. |
| `Div4WheelsOn` | Actual wheels-on time at diversion stop 4. | Dropped because diverted flights are removed; diversion-stop wheels-on time is outside the completed-flight prediction scope. |
| `Div4TotalGTime` | Total ground time at diversion stop 4. | Dropped because diverted flights are removed; diversion-stop total ground time is outside the completed-flight prediction scope. |
| `Div4LongestGTime` | Longest ground-time period at diversion stop 4. | Dropped because diverted flights are removed; diversion-stop longest ground-time period is outside the completed-flight prediction scope. |
| `Div4WheelsOff` | Actual wheels-off time from diversion stop 4. | Dropped because diverted flights are removed; diversion-stop wheels-off time is outside the completed-flight prediction scope. |
| `Div4TailNum` | Aircraft tail number recorded for diversion segment 4. | Dropped because diverted flights are removed; diversion-stop aircraft tail number is outside the completed-flight prediction scope. |
| `Div5Airport` | Airport code for diversion stop 5. | Dropped because diverted flights are removed; diversion-stop airport code is outside the completed-flight prediction scope. |
| `Div5AirportID` | BTS numeric airport identifier for diversion stop 5. | Dropped because diverted flights are removed; diversion-stop numeric airport identifier is outside the completed-flight prediction scope. |
| `Div5AirportSeqID` | BTS airport-record sequence identifier for diversion stop 5. | Dropped because diverted flights are removed; diversion-stop historical airport sequence identifier is outside the completed-flight prediction scope. |
| `Div5WheelsOn` | Actual wheels-on time at diversion stop 5. | Dropped because diverted flights are removed; diversion-stop wheels-on time is outside the completed-flight prediction scope. |
| `Div5TotalGTime` | Total ground time at diversion stop 5. | Dropped because diverted flights are removed; diversion-stop total ground time is outside the completed-flight prediction scope. |
| `Div5LongestGTime` | Longest ground-time period at diversion stop 5. | Dropped because diverted flights are removed; diversion-stop longest ground-time period is outside the completed-flight prediction scope. |
| `Div5WheelsOff` | Actual wheels-off time from diversion stop 5. | Dropped because diverted flights are removed; diversion-stop wheels-off time is outside the completed-flight prediction scope. |
| `Div5TailNum` | Aircraft tail number recorded for diversion segment 5. | Dropped because diverted flights are removed; diversion-stop aircraft tail number is outside the completed-flight prediction scope. |

### Dropped export artifact

| Column | Meaning | Why it is dropped |
|---|---|---|
| `Unnamed: 109` | Empty column created by a trailing delimiter in the downloaded CSV export. | Dropped because it contains no information and is a file-format artifact. |

## ASPM Column Selection and Dictionary

The downloaded ASPM hourly file contains 18 columns. The project keeps the airport, date, hour, scheduled departures,
and scheduled arrivals. Cleaning also creates `DATE`, giving each cleaned ASPM file six columns. The scheduled counts
describe planned airport demand. Because historical schedules can include later revisions, the project treats these
counts as the best available summary of the schedule rather than as a perfect record of what was known at every moment.

The other 13 source columns are removed. They are calculation fields or summaries of actual airport performance, such
as on-time percentages and average delays. ASPM generally publishes these results in the following day's update, which
is too late for the project's prediction times. Using the previous operating hour would not solve that publication
delay.

`clean_aspm.ipynb` writes one six-column file for each airport and year under `data/aspm/cleaned/`. For the arrival
scenario, `data/aspm/cat_aspm.py` combines the required origin-airport files into `data/aspm/cleaned_JFK_YEAR.csv`. The
combined file uses the same six columns.

### ASPM columns retained for modeling

| Column | Meaning | Why it is retained / how it is used |
|---|---|---|
| `airport` | Three-letter code for the airport represented by the hourly record. | Retained as the airport identifier and merge key. |
| `report_date` | Calendar date associated with the hourly record. | Converted to a pandas datetime and combined with `Hour` to create the hourly timestamp. |
| `Hour` | Local airport hour, represented by an integer from 0 through 23. | Retained to identify the hourly reporting period and construct `DATE`. |
| `Scheduled Departures` | Number of flights scheduled to depart during the hour. | Retained as a planned measure of departure demand and airport workload. ASPM has already counted the scheduled flights by hour, so the project does not need to rebuild that count from BTS. |
| `Scheduled Arrivals` | Number of flights scheduled to arrive during the hour. | Retained as a planned measure of arrival demand and airport workload under the same hourly-count assumption. |
| `DATE` | Constructed hourly timestamp created by combining `report_date` and `Hour`. | Added during cleaning and used for sorting, checking hourly coverage, and matching ASPM records to flights. |

### ASPM columns dropped during processing

| Column | Meaning | Why it is dropped |
|---|---|---|
| `Departures For Metric Computation` | Number of qualifying departures used by ASPM to calculate the reported departure performance metrics. | Dropped because it is a supporting count for later performance calculations, not a measure of planned demand. `Scheduled Departures` provides the schedule count needed here. |
| `Arrivals For Metric Computation` | Number of qualifying arrivals used by ASPM to calculate the reported arrival performance metrics. | Dropped because it is a supporting count for later performance calculations, not a measure of planned demand. `Scheduled Arrivals` provides the schedule count needed here. |
| `% On-Time Gate Departures` | Percentage of qualifying flights that departed the gate on time during the hour. | Dropped primarily because this realized hourly result is not published near enough to prediction time. It is also an aggregate departure-delay outcome that closely parallels `DepDel15`. |
| `% On-Time Airport Departures` | Percentage of qualifying flights whose airport departure was on time during the hour. | Dropped primarily because it is not available near real time. It also duplicates related departure-performance information and summarizes an outcome that has already occurred. |
| `% On-Time Gate Arrivals` | Percentage of qualifying flights that arrived at the gate on time during the hour. | Dropped primarily because it is not available near real time. It is also an aggregate arrival-delay outcome that closely parallels `ArrDel15`. |
| `Average Gate Departure Delay` | Average difference between scheduled and actual gate departure time for qualifying flights in the hour. | Dropped because ASPM publishes it too late for the intended prediction. It also summarizes a departure-delay outcome that has already occurred. |
| `Average Taxi Out Time` | Average number of minutes from gate departure to wheels-off for the flights represented in the hour. | Dropped because it is an actual operating result that ASPM publishes too late for the intended prediction times. Analysis after the flight is outside the project scope. |
| `Average Taxi Out Delay` | Average taxi-out delay beyond the expected or unimpeded taxi-out time. | Dropped because it is a realized congestion measure that ASPM publishes too late for the intended prediction times. |
| `Average Airport Departure Delay` | Average airport departure delay for qualifying flights in the hour, including delay accumulated before takeoff. | Dropped because ASPM publishes it too late for the intended prediction. It also combines gate and taxi-out results that have already occurred. |
| `Average Airborne Delay` | Average reported airborne delay for the flights represented in the hour. | Dropped because it is a realized operating measure that ASPM publishes too late for the intended prediction times. |
| `Average Taxi In Delay` | Average taxi-in delay for arriving flights represented in the hour. | Dropped because it is a realized congestion measure that ASPM publishes too late for the intended prediction times. |
| `Average Block Delay` | Average difference between scheduled and actual gate-to-gate elapsed time for qualifying flights. | Dropped primarily because this completed-flight result is not available near prediction time. It also combines several operating phases and is closely related to the arrival outcome. |
| `Average Gate Arrival Delay` | Average difference between scheduled and actual gate arrival time for qualifying flights in the hour. | Dropped primarily because ASPM publishes it too late for the intended prediction. It is also a direct aggregate of arrival-delay outcomes and closely parallels `ArrDel15`. |

### ASPM names after merge

Each ASPM field receives a prefix identifying the previous, current, or next clock hour. Spaces are replaced with
underscores and names are capitalized consistently.

| Before merge | Previous hour | Current hour | Next hour |
|---|---|---|---|
| `airport` | `ASPM_PREVIOUS_AIRPORT` | `ASPM_CURRENT_AIRPORT` | `ASPM_NEXT_AIRPORT` |
| `report_date` | `ASPM_PREVIOUS_REPORT_DATE` | `ASPM_CURRENT_REPORT_DATE` | `ASPM_NEXT_REPORT_DATE` |
| `Hour` | `ASPM_PREVIOUS_HOUR` | `ASPM_CURRENT_HOUR` | `ASPM_NEXT_HOUR` |
| `Scheduled Departures` | `ASPM_PREVIOUS_SCHEDULED_DEPARTURES` | `ASPM_CURRENT_SCHEDULED_DEPARTURES` | `ASPM_NEXT_SCHEDULED_DEPARTURES` |
| `Scheduled Arrivals` | `ASPM_PREVIOUS_SCHEDULED_ARRIVALS` | `ASPM_CURRENT_SCHEDULED_ARRIVALS` | `ASPM_NEXT_SCHEDULED_ARRIVALS` |
| `DATE` | `ASPM_PREVIOUS_DATE` | `ASPM_CURRENT_DATE` | `ASPM_NEXT_DATE` |

The merge also keeps a requested lookup timestamp and an offset from scheduled departure for each period. The next-hour
offset is positive because that record describes planned demand after the flight's departure hour begins.

## NOAA Column Selection and Dictionary

The NOAA data describes weather at the airport where a flight departs: JFK for Model 1A and the flight's origin airport
for Models 2A, 2B, and 2C. Only hourly fields that are useful for delay prediction are kept. Daily and monthly summaries
are removed because they do not describe conditions at a specific time and may include weather observed later.

The merge uses the latest NOAA observation available at or before scheduled departure, as long as it is no more than 90
minutes old. Later reports are not added to Models 2B and 2C. When a weather value is filled during cleaning, it may use
only an earlier observation within the same 90-minute limit; future weather is never used.

`clean_noaa.ipynb` writes an 18-column file for each airport and year under `data/noaa/cleaned/`. For the arrival
scenario, `data/noaa/cat_noaa.py` combines the origin-airport files and adds `AIRPORT`, producing the 19-column
`data/noaa/cleaned_JFK_YEAR.csv` file.

### NOAA columns retained for modeling

| Column | Plain-English description | Why it is retained |
|---|---|---|
| `DATE` | Date and time of the weather observation. | Used to match each flight with weather already observed by the prediction time. |
| `AIRPORT` | Airport code linked to the weather station. | Added only to the consolidated arrival input so each flight can be matched to weather at its origin. |
| `HourlyDewPointTemperature` | Temperature at which moisture begins to condense. | Helps describe how much moisture is in the air. |
| `HourlyDryBulbTemperature` | Air temperature reported by the station. | Provides the main temperature measurement. |
| `HourlyPrecipitation` | Amount of precipitation reported for the observation period. | Measures the amount of rain or melted precipitation. Trace amounts are kept as a small positive value. |
| `HourlyRelativeHumidity` | Relative humidity percentage. | Describes how moist the air is. |
| `HourlyVisibility` | Distance that can be seen clearly from the station. | Poor visibility can affect airport operations. |
| `HourlyWindSpeed` | Reported wind speed. | Strong winds can affect runway and flight operations. |
| `Rain` | Indicates that rain was reported. | Provides a simple rain feature. |
| `Drizzle` | Indicates that drizzle was reported. | Separates drizzle from other precipitation. |
| `Snow` | Indicates that snow was reported. | Identifies snow conditions that may affect operations. |
| `Fog` | Indicates that fog was reported. | Identifies an important cause of poor visibility. |
| `Mist` | Indicates that mist was reported. | Identifies a lighter visibility restriction. |
| `Thunderstorm` | Indicates that a thunderstorm was reported. | Identifies severe weather that may disrupt flights. |
| `FreezingPrecip` | Indicates that the weather report contains a freezing-condition code. | Identifies freezing weather that may affect aircraft and runways. |
| `Showers` | Indicates that showers were reported. | Distinguishes showers from other weather reports. |
| `PrecipOccurred` | Indicates that precipitation is supported by either a measured amount or a rain, snow, or drizzle report. | Combines several sources into one general precipitation feature. |
| `WindX` | East-west part of the wind, calculated from wind speed and direction. | Represents wind direction without the artificial break between 0 and 360 degrees. |
| `WindY` | North-south part of the wind, calculated from wind speed and direction. | Works with `WindX` to represent both wind direction and strength. |

`HourlyPresentWeatherType` is used to create the weather indicators and is then removed. `HourlyWindDirection` is used with wind speed to create `WindX` and `WindY`, then removed. These derived fields are easier for a model to use than the original coded weather text and compass degrees.

### NOAA fields dropped during processing

| Field group | Quick description | Why it is dropped |
|---|---|---|
| Station and report metadata | Station ID, name, location, elevation, report type, and source. | Each file already represents a selected airport weather station, so these fields add little useful flight-level information. |
| Other hourly measurements | Pressure, sky condition, wet-bulb temperature, and wind-gust fields. | Left out to keep the project focused on a smaller core weather set and avoid additional missing-data and parsing work. |
| Sunrise and sunset | Daily sunrise and sunset times. | Calendar information can be derived later if it becomes useful. |
| Daily summaries | Daily averages, totals, minimums, maximums, snow, wind, and weather summaries. | They mix daily and hourly detail and may include weather observed after the prediction time. |
| Monthly and climate summaries | Monthly averages, totals, extremes, degree days, normals, and counts of weather days. | They describe a month or climate period rather than conditions near an individual flight. |
| Short-duration precipitation summaries | Maximum precipitation amounts and ending times for several durations. | These are sparse summary extremes rather than regular observations near the flight. |
| Backup-station, equipment, and remarks fields | Backup location details, equipment history, and coded remarks. | They are metadata or require extra processing outside the capstone scope. |

### NOAA names after merge

During the merge, `DATE` becomes `NOAA_DATE` so it is not confused with the flight timestamp. For the arrival scenario,
`AIRPORT` becomes `NOAA_AIRPORT` and remains in the merged file as a field for checking the join. The departure file does
not need that field because every weather match is for JFK. The other weather names remain unchanged, and
`NOAA_AGE_MINUTES` records how old the observation is at scheduled departure.

# Appendix B

This appendix explains the columns saved after the BTS, ASPM, and NOAA joins, the fixed reference-informed features,
and the separate backlog and aircraft-rotation paths developed during Model 1A experimentation. It covers both the
departure and arrival scenarios, distinguishes reusable feature outputs from model-specific allowlists, and shows when
fields are available at each prediction cutoff.

## Joined Data Column Dictionary

`data/merged/JFK_YEAR_departures.csv` contains one row for each eligible completed flight leaving JFK. It has 77 columns:
34 BTS flight fields, 24 ASPM planned-demand and join-checking fields, and 19 NOAA weather and join-checking fields.
The currently published `data/merged/JFK_YEAR_arrivals.csv` files contain one row for each eligible completed flight
headed to JFK and retain their established 72-column schema: 28 BTS fields, 24 ASPM fields, and 20 NOAA fields including
`NOAA_AIRPORT`. The six restored full-flight outcome fields are needed by the new departure-rotation source path, not by
the completed arrival experiments.

The merged files intentionally keep targets, descriptive outcomes, source timestamps, and operating events from
different points in the flight. They are broad audit and feature-building files, so a column's presence does not mean
that every model may use it. `feature_departures.ipynb` and `feature_arrivals.ipynb` add fixed calculated fields and write
the 112-column departure and 116-column arrival files under `data/features/`. Each model notebook then applies the
feature list allowed at its prediction time. The departure files include six restored BTS arrival fields for audit and
rotation-source validation. The experiment notebooks read these shared feature files directly rather than creating
another layer of model-specific CSV files.

### Joined BTS flight columns

| Column | Description | Use and availability |
|---|---|---|
| `Year` | Calendar year of the flight. | Used for coverage checks and time-based data splits. |
| `Quarter` | Calendar quarter, from 1 through 4. | Calendar field available before departure. |
| `Month` | Calendar month, from 1 through 12. | Used for seasonal analysis and feature engineering. |
| `DayofMonth` | Day number within the month. | Calendar field available before departure. |
| `DayOfWeek` | BTS weekday number, with Monday as 1 and Sunday as 7. | Used to examine and model weekly patterns. |
| `FlightDate` | Scheduled flight date without the departure time. | Base date available from the schedule. |
| `Reporting_Airline` | BTS reporting carrier code. | Airline identifier available before departure. |
| `Tail_Number` | Aircraft registration number. | Audit-only in standard models; used as a matching key, not a direct predictor, in the separate rotation path. |
| `Flight_Number_Reporting_Airline` | Flight number assigned by the reporting airline. | Flight identifier and possible categorical feature, not a numeric measurement. |
| `Origin` | Three-letter departure-airport code. | Identifies the origin and is part of the ASPM join key. |
| `OriginState` | Two-letter state code for the origin. | Compact origin-location field. |
| `Dest` | Three-letter destination-airport code. | Identifies the route destination. |
| `DestState` | Two-letter state code for the destination. | Compact destination-location field. |
| `CRSDepTime` | Scheduled departure time in local HHMM form. | Known before departure and used to construct the scheduled timestamp. |
| `DepTime` | Actual gate departure or pushback time in local HHMM form. | Available only after pushback; eligible for Model 2B, not Models 1A or 2A. |
| `DepDelay` | Actual gate departure minus scheduled departure, in minutes. | Completed departure outcome used for validation and a possible Model 2B predictor; unavailable before pushback. |
| `DepDelayMinutes` | Nonnegative departure delay in minutes. | Outcome field used for validation and possible Model 2B input. |
| `DepDel15` | Indicates a departure delay of at least 15 minutes. | Target for Model 1A and possible post-pushback input for Model 2B. |
| `DepartureDelayGroups` | Departure delay grouped into 15-minute ranges. | Outcome field for EDA and validation; unavailable before pushback. |
| `TaxiOut` | Minutes from gate departure to takeoff. | Available only after takeoff; eligible for Model 2C and excluded from Models 1A, 2A, and 2B. |
| `WheelsOff` | Actual takeoff time in local HHMM form. | Available only at takeoff; eligible for Model 2C and excluded from Models 1A, 2A, and 2B. |
| `CRSArrTime` | Scheduled arrival time in the destination's local HHMM form. | Schedule field known before departure. |
| `ArrTime` | Actual gate-arrival time in destination-local HHMM form. | Restored for rotation-source audit; completed-flight outcome and never a direct Model 1A predictor. |
| `ArrDelay` | Actual gate arrival minus scheduled arrival, in signed minutes. | Used only for an already-arrived preceding aircraft in the separate rotation path; the target flight's value is prohibited. |
| `ArrDelayMinutes` | Nonnegative arrival-delay minutes. | Retained for audit; unavailable for the target flight at Model 1A prediction time. |
| `ArrDel15` | Indicates an arrival delay of at least 15 minutes. | Target for Models 2A, 2B, and 2C; never a predictor. |
| `ArrivalDelayGroups` | Arrival delay grouped into 15-minute ranges. | Completed-flight outcome used only for EDA and target validation. |
| `WheelsOn` | Actual landing time in destination-local HHMM form. | Restored for rotation-source audit; unavailable for the target flight before pushback. |
| `TaxiIn` | Minutes from landing to gate arrival. | Restored for rotation-source audit; unavailable for the target flight before pushback. |
| `CRSElapsedTime` | Scheduled gate-to-gate travel time in minutes. | Route and schedule feature known before departure. |
| `ActualElapsedTime` | Actual gate-to-gate elapsed time in minutes. | Restored for audit; completed-flight outcome and not a direct Model 1A predictor. |
| `Distance` | Published route distance in miles. | Route feature known before departure. |
| `DistanceGroup` | BTS distance band based on 250-mile intervals. | Grouped route-length field available before departure. |
| `DATE` | Scheduled departure timestamp constructed from `FlightDate` and `CRSDepTime`. | Main flight timestamp used for sorting and time-based joins. |

### Joined ASPM planned-demand columns

ASPM supplies planned schedule counts for the previous, current, and next clock hours around scheduled departure. For a
departure flight these records describe JFK; for an arrival flight they describe that flight's origin. They are schedule
values known ahead of time, not future operating results. Each offset is the ASPM timestamp minus the scheduled departure
timestamp.

| Column | Description |
|---|---|
| `ASPM_PREVIOUS_LOOKUP_DATE` | Previous full clock hour requested for the ASPM join. |
| `ASPM_CURRENT_LOOKUP_DATE` | Beginning of the flight's scheduled departure hour requested for the ASPM join. |
| `ASPM_NEXT_LOOKUP_DATE` | Beginning of the next clock hour requested for the ASPM join. |
| `ASPM_PREVIOUS_AIRPORT` | Airport code returned by the previous-hour match. |
| `ASPM_PREVIOUS_REPORT_DATE` | ASPM calendar date associated with the previous-hour record. |
| `ASPM_PREVIOUS_HOUR` | Hour number of the previous-hour record. |
| `ASPM_PREVIOUS_SCHEDULED_DEPARTURES` | Flights scheduled to depart during the previous hour. |
| `ASPM_PREVIOUS_SCHEDULED_ARRIVALS` | Flights scheduled to arrive during the previous hour. |
| `ASPM_PREVIOUS_DATE` | Timestamp of the matched previous-hour ASPM record. |
| `ASPM_PREVIOUS_OFFSET_MINUTES` | Previous-hour timestamp relative to scheduled departure; normally −119 through −60 minutes. |
| `ASPM_CURRENT_AIRPORT` | Airport code returned by the current-hour match. |
| `ASPM_CURRENT_REPORT_DATE` | ASPM calendar date associated with the current-hour record. |
| `ASPM_CURRENT_HOUR` | Hour number of the current-hour record. |
| `ASPM_CURRENT_SCHEDULED_DEPARTURES` | Flights scheduled to depart during the current hour. |
| `ASPM_CURRENT_SCHEDULED_ARRIVALS` | Flights scheduled to arrive during the current hour. |
| `ASPM_CURRENT_DATE` | Timestamp of the matched current-hour ASPM record. |
| `ASPM_CURRENT_OFFSET_MINUTES` | Current-hour timestamp relative to scheduled departure; normally −59 through 0 minutes. |
| `ASPM_NEXT_AIRPORT` | Airport code returned by the next-hour match. |
| `ASPM_NEXT_REPORT_DATE` | ASPM calendar date associated with the next-hour record. |
| `ASPM_NEXT_HOUR` | Hour number of the next-hour record. |
| `ASPM_NEXT_SCHEDULED_DEPARTURES` | Flights scheduled to depart during the next hour. |
| `ASPM_NEXT_SCHEDULED_ARRIVALS` | Flights scheduled to arrive during the next hour. |
| `ASPM_NEXT_DATE` | Timestamp of the matched next-hour ASPM record. |
| `ASPM_NEXT_OFFSET_MINUTES` | Next-hour timestamp relative to scheduled departure; normally 1 through 60 minutes. |

A few next-hour fields are null for flights in the final hour of December 31 because the requested ASPM record belongs
to the next annual file. These are documented year-boundary gaps, not zero-traffic hours.

### Joined NOAA weather columns

NOAA supplies the latest origin-airport weather observation at or before scheduled departure, within the 90-minute
matching limit. The observation timestamp and age remain in the merged files so future or overly old matches can be
detected.

| Column | Description | Use and availability |
|---|---|---|
| `NOAA_AIRPORT` | Airport code linked to the matched weather station. | Present only in the arrival file; confirms that weather came from the flight's origin. |
| `NOAA_DATE` | Timestamp of the matched NOAA observation. | Must be at or before scheduled departure. |
| `HourlyDewPointTemperature` | Dew-point temperature reported by the station. | Describes moisture in the air. |
| `HourlyDryBulbTemperature` | Air temperature reported by the station. | Main temperature measurement. |
| `HourlyPrecipitation` | Precipitation amount for the observation period. | Continuous precipitation measurement; trace amounts use a small positive value. |
| `HourlyRelativeHumidity` | Relative humidity percentage. | Describes atmospheric moisture. |
| `HourlyVisibility` | Horizontal visibility reported by the station. | Measures visibility conditions that may affect operations. |
| `HourlyWindSpeed` | Reported wind speed. | Measures wind strength. |
| `Rain` | Indicates that rain was reported. | 0/1 indicator for rain. |
| `Drizzle` | Indicates that drizzle was reported. | 0/1 indicator for drizzle. |
| `Snow` | Indicates that snow was reported. | 0/1 indicator for snow. |
| `Fog` | Indicates that fog was reported. | 0/1 indicator for fog and poor visibility. |
| `Mist` | Indicates that mist was reported. | 0/1 indicator for mist and reduced visibility. |
| `Thunderstorm` | Indicates that a thunderstorm was reported. | 0/1 indicator for thunderstorms. |
| `FreezingPrecip` | Indicates that the report contains a freezing-condition code. | 0/1 indicator for freezing precipitation. |
| `Showers` | Indicates that showers were reported. | 0/1 indicator for showers. |
| `PrecipOccurred` | Indicates precipitation based on a measured amount or a rain, snow, or drizzle report. | One 0/1 indicator combining several signs of precipitation. |
| `WindX` | East-west wind component calculated from speed and direction. | Numeric representation of the east-west wind direction and strength. |
| `WindY` | North-south wind component calculated from speed and direction. | Used with `WindX` to represent the wind vector. |
| `NOAA_AGE_MINUTES` | Scheduled departure time minus the NOAA observation time, in minutes. | Must be nonnegative and within the allowed weather-match tolerance. |

## Feature Engineering

The feature notebooks add useful schedule, calendar, route, planned-traffic, and weather measures while keeping one row
per flight. `feature_departures.ipynb` adds 35 common pre-pushback fields to the departure merge. The same 35 fields are
added to the arrival merge, followed by nine fields based on actual pushback and takeoff information. This produces
`data/features/JFK_YEAR_departures.csv` with 112 columns and `data/features/JFK_YEAR_arrivals.csv` with 116 columns.

The feature choices are supported by the three primary references. Snell combines flight records with hourly weather
and emphasizes schedule, airline, route, traffic, and weather. Zoutendijk and Mitici use airline, airport, distance,
scheduled traffic, weather, and time-cycle features. Pineda-Jaramillo et al. combine flight, airport, geographic, and
weather data and examine which fields contribute to predictions.

The common pre-pushback fields are available to Models 1A and 2A and remain available to Models 2B and 2C. Model 2B adds
information known after pushback, and Model 2C adds taxi-out and takeoff information. In the table below, **Pre** means
all four models, **2B and 2C** means first available after pushback, and **2C** means first available after takeoff. Raw
fields such as `Reporting_Airline`, `Origin`, `Dest`, `CRSElapsedTime`, `Distance`, the selected NOAA measurements,
`WindX`, `WindY`, and `NOAA_AGE_MINUTES` remain possible inputs even though they are not repeated in the engineered
feature table.

### Engineered feature dictionary

| Feature | Construction and description | Availability | Justification and evidence |
|---|---|---|---|
| `SCHED_DEP_MINUTE_OF_DAY` | Convert `CRSDepTime` from HHMM to minutes after local midnight. | Pre | Provides a valid numeric representation of scheduled departure time. Scheduled time is used by all three primary references, and Pineda identifies time-of-day effects as important. |
| `SCHED_DEP_HOUR` | Integer hour from `SCHED_DEP_MINUTE_OF_DAY`. | Pre | Gives an interpretable grouping for EDA and simple models and supports comparison of peak-hour delay rates. Snell discusses scheduled time blocks and peak-hour patterns. |
| `SCHED_DEP_TIME_SIN` | `sin(2π * SCHED_DEP_MINUTE_OF_DAY / 1440)`. | Pre | Represents the daily cycle without placing 23:59 far from 00:00. Zoutendijk and Mitici explicitly encode time features with sine and cosine. |
| `SCHED_DEP_TIME_COS` | `cos(2π * SCHED_DEP_MINUTE_OF_DAY / 1440)`. | Pre | Completes the cyclical representation of scheduled departure time. |
| `SCHED_ARR_MINUTE_OF_DAY` | Convert `CRSArrTime` from destination-local HHMM to minutes after local midnight. | Pre | Captures the scheduled arrival period without implying that origin and destination clocks share a time zone. It must not be subtracted from scheduled departure time; `CRSElapsedTime` is the valid duration field. |
| `SCHED_ARR_TIME_SIN` | `sin(2π * SCHED_ARR_MINUTE_OF_DAY / 1440)`. | Pre | Preserves the daily periodicity of the destination-local scheduled arrival time. |
| `SCHED_ARR_TIME_COS` | `cos(2π * SCHED_ARR_MINUTE_OF_DAY / 1440)`. | Pre | Completes the cyclical representation of scheduled arrival time. |
| `TIME_OF_DAY` | Interpretable category derived from scheduled departure time, with fixed morning, afternoon, evening, and overnight bands documented before modeling. | Pre | Snell discusses time-of-day slots, Pineda uses a departure-period category, and Zoutendijk and Mitici select time of day. This field is especially useful for EDA; a model may use it instead of the sine and cosine pair to avoid repeating the same information. |
| `IS_WEEKEND` | 1 when `DayOfWeek` is 6 or 7; otherwise 0. | Pre | Provides a simple weekly schedule distinction. Snell discusses weekend flags, while all three references include or discuss weekday effects. |
| `DAY_OF_WEEK_SIN` | `sin(2π * (DayOfWeek - 1) / 7)`. | Pre | Preserves adjacency between Sunday and Monday. Zoutendijk and Mitici explicitly apply trigonometric encoding to day of week. |
| `DAY_OF_WEEK_COS` | `cos(2π * (DayOfWeek - 1) / 7)`. | Pre | Completes the weekly cyclical representation. |
| `DAY_OF_YEAR` | Ordinal day from `FlightDate`, from 1 through 365 or 366. | Pre | Represents position within the year and is among the schedule features used by Zoutendijk and Mitici. |
| `DAY_OF_YEAR_SIN` | `sin(2π * (DAY_OF_YEAR - 1) / days_in_year)`. | Pre | Represents annual seasonality continuously and handles leap years through `days_in_year`. |
| `DAY_OF_YEAR_COS` | `cos(2π * (DAY_OF_YEAR - 1) / days_in_year)`. | Pre | Completes the annual cyclical representation. |
| `MONTH_SIN` | `sin(2π * (Month - 1) / 12)`. | Pre | Represents month as a cycle. Zoutendijk and Mitici explicitly use month sine and cosine, and Pineda reports month effects. |
| `MONTH_COS` | `cos(2π * (Month - 1) / 12)`. | Pre | Completes the monthly cyclical representation. |
| `YEAR_PERIOD` | Treat 2019, 2023, and 2024 as categories rather than as a continuous numeric trend. | Pre | Separates the pre-pandemic baseline from the two post-pandemic periods without assuming a steady year-to-year change. Zoutendijk and Mitici use year, but a period category better fits this project's nonconsecutive years. |
| `ROUTE` | Concatenate `Origin` and `Dest` as an origin-destination category. | Pre | Preserves the flight-leg identity highlighted by Snell and the airport/destination effects emphasized by Zoutendijk and Mitici and Pineda. |
| `AIRLINE_FLIGHT_ID` | Concatenate `Reporting_Airline` and `Flight_Number_Reporting_Airline`; treat the result as categorical. | Pre | Avoids treating a flight number as a continuous quantity and distinguishes identical numbers used by different airlines. Snell and Pineda both retain scheduled flight and airline identity. |
| `AIRLINE_DEST` | Concatenate `Reporting_Airline` and `Dest` as a categorical interaction. | Pre | Provides one limited, interpretable service-pattern interaction instead of a large arbitrary interaction set. Airline and destination are supported individually across the primary references. |
| `LOG_DISTANCE` | `log1p(Distance)`. | Pre | Retains route-length ordering while reducing right skew. Distance is selected or discussed by all three primary references. The raw distance should remain available for tree models and interpretation. |
| `SCHEDULED_SPEED_PROXY` | `60 * Distance / CRSElapsedTime`, when elapsed time is positive. | Pre | Summarizes the relationship between route length and scheduled gate-to-gate duration. It uses only schedule information available before pushback, but should be compared with its two source fields to avoid repeating the same information. |
| `ASPM_PREVIOUS_TOTAL_SCHEDULED_TRAFFIC` | Previous-hour scheduled departures plus previous-hour scheduled arrivals. | Pre | Summarizes planned airport workload immediately before the flight. Snell supports airport congestion/traffic measures, and Zoutendijk and Mitici use scheduled-flight counts near the flight time. |
| `ASPM_CURRENT_TOTAL_SCHEDULED_TRAFFIC` | Current-hour scheduled departures plus current-hour scheduled arrivals. | Pre | Measures planned workload during the scheduled departure hour. |
| `ASPM_NEXT_TOTAL_SCHEDULED_TRAFFIC` | Next-hour scheduled departures plus next-hour scheduled arrivals. | Pre | Measures planned workload just after the scheduled departure hour. These are schedule counts known ahead of time, not future realized outcomes. |
| `ASPM_THREE_HOUR_SCHEDULED_DEPARTURES` | Sum scheduled departures across the previous, current, and next hours. | Pre | Provides a three-hour view of planned departure demand, similar to the nearby scheduled-flight window used by Zoutendijk and Mitici. |
| `ASPM_THREE_HOUR_SCHEDULED_ARRIVALS` | Sum scheduled arrivals across the previous, current, and next hours. | Pre | Separates planned arrival demand from departure demand because each can load airport resources differently. |
| `ASPM_THREE_HOUR_TOTAL_SCHEDULED_TRAFFIC` | Sum `ASPM_THREE_HOUR_SCHEDULED_DEPARTURES` and `ASPM_THREE_HOUR_SCHEDULED_ARRIVALS`. | Pre | Provides the main compact congestion feature supported by Snell's traffic-volume discussion and Zoutendijk and Mitici's scheduled-flight window. |
| `ASPM_CURRENT_MINUS_PREVIOUS_TRAFFIC` | Current-hour total scheduled traffic minus previous-hour total. | Pre | Indicates whether planned airport workload is building or easing near departure without using realized performance. |
| `ASPM_NEXT_MINUS_CURRENT_TRAFFIC` | Next-hour total scheduled traffic minus current-hour total. | Pre | Shows whether planned demand is expected to rise or fall in the next hour, using schedule information already known at prediction time. |
| `ASPM_MAX_HOURLY_TRAFFIC` | Maximum of the previous-, current-, and next-hour total scheduled traffic. | Pre | Captures the local planned peak without imposing a learned high-traffic threshold. |
| `TEMP_DEWPOINT_SPREAD` | `HourlyDryBulbTemperature - HourlyDewPointTemperature`. | Pre | Provides a compact moisture-related measure while retaining the underlying observations. Zoutendijk and Mitici select temperature/dew point features, and Snell and Pineda support weather integration. |
| `LOG_PRECIPITATION` | `log1p(max(HourlyPrecipitation, 0))`. | Pre | Keeps the difference between trace and heavy precipitation while reducing the influence of a small number of very large values. Snell and Pineda include precipitation-related weather information. |
| `WEATHER_CONDITION_COUNT` | Sum `Rain`, `Drizzle`, `Snow`, `Fog`, `Mist`, `Thunderstorm`, `FreezingPrecip`, and `Showers`. | Pre | Gives an interpretable measure of how many adverse condition types are reported without inventing severity weights. |
| `ADVERSE_WEATHER` | 1 when any of the eight weather-condition indicators is 1; otherwise 0. | Pre | Supplies a compact general-weather flag for linear baselines while the component indicators remain available. The primary references consistently support weather as a predictor. |
| `ACTUAL_DEP_MINUTE_OF_DAY` | Convert `DepTime` from HHMM to minutes after local midnight. | 2B and 2C | Represents the known gate-out time once pushback occurs. Snell directly compares arrival-delay models without and with actual departure information. |
| `ACTUAL_DEP_TIME_SIN` | `sin(2π * ACTUAL_DEP_MINUTE_OF_DAY / 1440)`. | 2B and 2C | Encodes actual pushback time without a midnight discontinuity. |
| `ACTUAL_DEP_TIME_COS` | `cos(2π * ACTUAL_DEP_MINUTE_OF_DAY / 1440)`. | 2B and 2C | Completes the cyclical representation of actual pushback time. |
| `DEPARTED_EARLY` | 1 when signed `DepDelay` is less than 0; otherwise 0. | 2B and 2C | Preserves the distinction between early and non-early departures if a nonnegative delay transform is tested. |
| `LOG_DEP_DELAY_MINUTES` | `log1p(DepDelayMinutes)`. | 2B and 2C | Reduces the influence of very long departure delays while keeping their order. It should be compared with signed `DepDelay` rather than automatically included with every related departure-delay field. |
| `ACTUAL_TAKEOFF_MINUTE_OF_DAY` | Convert `WheelsOff` from HHMM to minutes after local midnight. Treat `2400` as minute zero; invalid or missing values remain missing. | 2C | Represents the known takeoff time once the aircraft is airborne without treating HHMM as an ordinary number. |
| `ACTUAL_TAKEOFF_TIME_SIN` | `sin(2π * ACTUAL_TAKEOFF_MINUTE_OF_DAY / 1440)`. | 2C | Encodes actual takeoff time without a midnight discontinuity. |
| `ACTUAL_TAKEOFF_TIME_COS` | `cos(2π * ACTUAL_TAKEOFF_MINUTE_OF_DAY / 1440)`. | 2C | Completes the cyclical representation of actual takeoff time. |
| `LOG_TAXI_OUT_MINUTES` | `log1p(TaxiOut)` when `TaxiOut` is nonnegative; invalid or missing values remain missing. | 2C | Reduces the influence of unusually long taxi-out times for linear models. The original `TaxiOut` value remains available for tree models and interpretation. |

### Feature selection and timing rules

Use the following rules when selecting features and preparing data:

- Start with the fixed baseline fields. Add calculated features only when they have a clear purpose and improve the
  model. Avoid keeping several fields that express the same information.
- Choose features separately for each model type. Logistic regression may benefit from sine and cosine time features
  that preserve daily or weekly cycles, along with summarized weather measures. Tree models may work well with the
  original values and may not need every calculated version.
- Learn every data-preparation choice from the training data only. This includes filling missing values, scaling numeric
  values, converting categories, selecting features, balancing the two outcome classes, and choosing numeric cutoffs.
  Development and test data are used only to evaluate the finished choice.
- Do not create flags such as `LOW_VISIBILITY`, `HIGH_WIND`, `ASPM_HIGH_TRAFFIC`, or flags based on source-record age
  until their cutoffs have a clear operational meaning or have been selected using training data.
- Add large sets of interaction features or historical delay-rate features only through a documented follow-up
  experiment. If historical rates are added, calculate them from earlier completed flights only. Exclude the flight
  being predicted, never use development or test outcomes, and reduce unstable rates for groups with few observations.
- For Model 2B, begin with a small update such as `DepDelay`, where negative values mean an early departure and positive
  values mean a late departure. Then test whether a larger departure-information set improves the result. `DepTime`,
  `DepDelay`, `DepDelayMinutes`, `DepDel15`, and `DepartureDelayGroups` describe much of the same event, so they should
  not all be included automatically. `TaxiOut` and `WheelsOff` are available only to Model 2C and must remain excluded
  from Models 1A, 2A, and 2B.
- For Model 2C, retain raw `TaxiOut` and `WheelsOff` and compare them with the derived takeoff-time encodings and
  `LOG_TAXI_OUT_MINUTES`. Do not automatically include every raw and transformed representation when they express the
  same information. A `WheelsOff` value of `2400` maps to minute zero for the cyclical time representation; the feature
  does not imply a same-day takeoff date.
- Model 1A uses flights with `Origin` equal to JFK. Models 2A, 2B, and 2C must instead use inbound flights with `Dest`
  equal to JFK. Their ASPM and NOAA data must describe each flight's origin and must be available by that model's
  prediction time.
- Do not substitute JFK outbound data for the origin data required by the arrival models. Do not use destination weather
  observed at landing in a pre-pushback model. Destination weather is allowed only if it came from a forecast or
  observation that was available by the prediction cutoff.
- Keep source timestamps such as `FlightDate`, `DATE`, ASPM lookup and report dates, and `NOAA_DATE` so the joins can be
  checked. Do not use these timestamps directly as model predictors.
- Keep missing next-hour ASPM matches at annual file boundaries as missing unless they can be recovered from the
  following year's planned schedule file. Do not replace them with zero or copy values from the current hour.
- Keep `Tail_Number` audit-only in standard models. The separate rotation path may use it only as a matching key under
  the timing and assignment rules below; the raw registration is not a predictor. Longer aircraft chains,
  tail-identity effects, and network-propagation features remain excluded.

## Operational Backlog Feature Engineering

The backlog features are documented separately from the preceding feature dictionary because they were developed as a
project-specific follow-up rather than selected primarily from the reviewed reference papers. They are not presented
as a novel research method. Their purpose is narrower: measure recent realized airport operating state while preserving
the prediction cutoff and keeping all existing feature datasets and experiments intact.

The implementation is deterministic and row preserving. It uses the schedule and gate-out outcomes of other flights to
reconstruct what an operational event feed would have shown at the sample cutoff. It performs no imputation, scaling,
feature selection, or class balancing. Those learned steps remain inside the model pipeline. The window length is a
notebook parameter and is included in the output name and column prefix. The implemented paths use 30 minutes with
prefix `BACKLOG_W30` and 60 minutes with prefix `BACKLOG_W60`.

### Backlog feature dictionary

Let `T` be the sample flight's scheduled departure timestamp. Let the trailing scheduled cohort contain only flights
from the same airport with scheduled timestamps in `[T - 30 minutes, T)`. A cohort flight is *completed* when its actual
gate-out timestamp is earlier than `T` and *pending* otherwise. Delay summaries use completed members only.

| Feature | Construction and description | Missing-value behavior | Availability and purpose |
|---|---|---|---|
| `BACKLOG_W30_SCHEDULED_COUNT` | Number of same-airport flights scheduled during the trailing 30-minute cohort. | Complete integer; zero for an empty cohort. | Planned local workload immediately before the sample cutoff. |
| `BACKLOG_W30_COMPLETED_COUNT` | Cohort flights whose reconstructed gate-out timestamp is strictly before `T`. | Complete integer; zero when none completed. | Recent realized throughput available from gate-out events. |
| `BACKLOG_W30_PENDING_COUNT` | `SCHEDULED_COUNT - COMPLETED_COUNT`; earlier-scheduled cohort flights that have not pushed back by `T`. | Complete integer; zero when none are pending. | Main queue-pressure measure; selected as the strongest backlog field in Logistic Regression Experiment 04. |
| `BACKLOG_W30_DELAYED_DEPARTURE_COUNT` | Completed cohort flights with `DepDel15 == 1`. | Complete integer; zero when no completed cohort flight is delayed. | Count of recently observed 15-minute departure delays; no pending flight's eventual label is used. |
| `BACKLOG_W30_DELAY_RATE` | `DELAYED_DEPARTURE_COUNT / COMPLETED_COUNT`. | Missing when `COMPLETED_COUNT == 0`. | Recent observed delay frequency. The denominator prevents an empty history from being interpreted as a zero delay rate. |
| `BACKLOG_W30_MEAN_DEP_DELAY` | Mean signed `DepDelay` among completed cohort flights; early departures remain negative. | Missing when `COMPLETED_COUNT == 0`. | Recent signed gate-out performance; selected in Experiment 04's compact backlog set. |
| `BACKLOG_W30_MEAN_DEP_DELAY_MINUTES` | Mean nonnegative `DepDelayMinutes` among completed cohort flights. | Missing when `COMPLETED_COUNT == 0`. | Recent delay severity without offset from early departures. |
| `BACKLOG_W30_TOTAL_DEP_DELAY_MINUTES` | Sum of nonnegative `DepDelayMinutes` among completed cohort flights. | Complete numeric value; zero when no completed delay minutes are observed. | Accumulated recent completed-flight delay burden. |

The three count fields are intentionally related:

```text
BACKLOG_W30_SCHEDULED_COUNT
    = BACKLOG_W30_COMPLETED_COUNT + BACKLOG_W30_PENDING_COUNT
```

The W60 dataset repeats the same eight definitions with prefix `BACKLOG_W60` and cohort
`[T - 60 minutes, T)`. Its causal boundaries, missing-value behavior, arithmetic identities, and leakage safeguards are
identical; only the trailing duration changes.

Linear models should not automatically include every related count and delay representation. Logistic Regression
Experiment 04 selected the compact W30 set consisting of `PENDING_COUNT`, `COMPLETED_COUNT`, and signed
`MEAN_DEP_DELAY`; Experiment 08 retained the same three definitions when W60 replaced W30. CatBoost Experiment 03 then
tested all eight W60 fields with nonlinear trees while holding the classifier, target rows, and rotation inputs fixed.
The full set did not improve 2019 temporal AP or 2023 validation, so the compact three-field W60 allowlist remains the
current airport-wide choice. The output file still retains all eight fields for traceability and future ablations.

### Same-airline backlog feature dictionary

The same-airline W60 dataset applies the identical cutoff, completion, and delay-observability rules separately within
each normalized `Reporting_Airline`. For a sample flight operated by airline `A`, only earlier-scheduled airline `A`
departures in `[T - 60 minutes, T)` enter its cohort. Other airlines never contribute to its same-airline fields. The
airline is schedule information known before `T`; it is not inferred from a later event.

Eight fields repeat the airport-wide definitions under the `AIRLINE_BACKLOG_W60` prefix. A ninth field explicitly
represents the fraction of scheduled same-airline work that remains pending:

| Feature | Construction and description | Missing-value behavior |
|---|---|---|
| `AIRLINE_BACKLOG_W60_SCHEDULED_COUNT` | Same-airline flights scheduled in `[T - 60 minutes, T)`. | Complete integer; zero for an empty cohort. |
| `AIRLINE_BACKLOG_W60_COMPLETED_COUNT` | Same-airline cohort flights whose reconstructed gate-out is before `T`. | Complete integer; zero when none completed. |
| `AIRLINE_BACKLOG_W60_PENDING_COUNT` | Same-airline `SCHEDULED_COUNT - COMPLETED_COUNT`. | Complete integer; zero when none are pending. |
| `AIRLINE_BACKLOG_W60_DELAYED_DEPARTURE_COUNT` | Completed same-airline cohort flights with `DepDel15 == 1`. | Complete integer; zero when none qualify. |
| `AIRLINE_BACKLOG_W60_DELAY_RATE` | Delayed count divided by completed count. | Missing when no same-airline cohort flight has completed. |
| `AIRLINE_BACKLOG_W60_MEAN_DEP_DELAY` | Mean signed `DepDelay` among completed same-airline cohort flights. | Missing when none completed. |
| `AIRLINE_BACKLOG_W60_MEAN_DEP_DELAY_MINUTES` | Mean nonnegative `DepDelayMinutes` among completed same-airline cohort flights. | Missing when none completed. |
| `AIRLINE_BACKLOG_W60_TOTAL_DEP_DELAY_MINUTES` | Sum of nonnegative delay minutes among completed same-airline cohort flights. | Complete numeric value; zero when no completed delay minutes are observed. |
| `AIRLINE_BACKLOG_W60_PENDING_SHARE` | `PENDING_COUNT / SCHEDULED_COUNT`. | Missing when the same-airline scheduled cohort is empty. |

The output is intentionally comprehensive so model notebooks can compare compact and broader manifests without
rebuilding the feature data. CatBoost Experiment 04 completed the first controlled comparison and selected pending
count, completed count, mean signed delay, delay rate, and pending share using 2019 temporal folds, while retaining the
airport-wide compact W60 fields as a control. This five-field extension is now part of the preferred 41-field Model 1A
allowlist. The other four same-airline fields remain in the shared file but are not admitted automatically.

### Backlog timing and availability rules

- The sample cutoff is the scheduled departure timestamp `DATE`. The trailing interval includes its lower boundary and
  excludes `T`, so the sample never supplies its own outcome and flights scheduled simultaneously receive the same
  backlog state.
- Same-airline fields add `Reporting_Airline` to the grouping key. Flights from the same airline at the same cutoff
  receive identical state; flights from different airlines at that cutoff may receive different state. Airline codes
  are stripped, uppercased, required to be complete, and never learned from an outcome.
- For the completed-flight test, actual gate-out is reconstructed as `DATE + DepDelay`. This handles midnight rollover
  without interpreting the HHMM-formatted `DepTime` as an ordinary number.
- A completed cohort flight contributes delay values only when its gate-out event occurred strictly before `T`. A
  pending flight contributes to pending count but not its eventual `DepDelay`, `DepDelayMinutes`, or `DepDel15`.
- These fields are time-local operational snapshots, not group statistics learned from the full training or validation
  year. A validation row may use earlier events from that validation year because those events would already have been
  observed by its cutoff; it never uses a later event or its own outcome.
- The current Model 1A files group one-airport JFK departure populations. An arrival implementation must use a separate
  all-departure reference population grouped by each sample's `Origin`. Computing origin backlog from only JFK-bound
  flights would undercount airport activity and is not allowed.
- Origin groups use local scheduled timestamps. Backlog windows must never compare local clock values across airports
  as though they shared a time zone.
- The annual implementation uses only events present in that annual input. A sample near the beginning of January may
  lack events from the final 30 minutes of the preceding annual file; this boundary condition must be reported rather
  than filled with future outcomes.
- Backlog is available at the scheduled-departure snapshot for Model 1A and, once the origin implementation exists, for
  Models 2A, 2B, and 2C. Model 2B and 2C must use the exact same pre-pushback backlog base as 2A before adding the sample
  flight's pushback or takeoff information.
- Historical development uses BTS to reconstruct event state. Deployment requires a live or sufficiently timely
  schedule and gate-out feed; ASPM planned counts alone cannot reproduce completed and pending flight state.
- New window lengths, overdue-minute severity measures, or backlog trends must be implemented as new, separately named
  datasets and experiments. Existing `backlog_w30` files and published results remain unchanged under the project's
  append-only guideline.

## Aircraft Rotation Feature Engineering

The rotation features are documented separately because they extend the reference-informed schedule, traffic, and
weather design with the operating state of the assigned aircraft. The original cohort-limited path writes
`JFK_YEAR_departures_rotation.csv`; the append-only current path writes
`JFK_YEAR_departures_rotation_full_history.csv`. Both contain the same 13 predictor fields and eight audit fields, and
both preserve the standard 112 columns and target-row order. The full-history file is the selected source for current
Model 1A experiments, while the original files and their Experiment 05 results remain intact.

Let `T` be the target flight's scheduled departure. The target's BTS `Tail_Number` is matched against flights scheduled
to arrive at the same airport. All origin-local scheduled departures, destination-local scheduled arrivals, and target
cutoffs are converted to UTC before ordering. The immediately preceding known airport event for that tail determines
the match state. A previous arrival supplies a candidate inbound leg; a previous departure prevents reuse of an older
arrival. The target row itself supplies no departure or arrival outcome to these calculations.

### Rotation feature dictionary

| Feature | Construction and description | Missing-value behavior | Availability and purpose |
|---|---|---|---|
| `ROTATION_STATUS` | Categorical state: `ARRIVED`, `NOT_ARRIVED`, `NO_PRIOR_EVENT`, `PREVIOUS_EVENT_DEPARTURE`, or `MISSING_TAIL`. | Complete category. | Compact description of rotation coverage and the assigned aircraft's observable state at `T`. |
| `ROTATION_MATCH_FOUND` | 1 when the immediately preceding known tail event is an inbound arrival; otherwise 0. | Complete binary indicator. | Distinguishes a true zero or observed value from the absence of a usable rotation match. |
| `ROTATION_INBOUND_ORIGIN` | Origin airport of the matched preceding inbound leg. | Missing without a match. | Categorical context for the preceding flight; models may compare it with or omit it to control cardinality. |
| `ROTATION_SCHEDULED_TURN_MINUTES` | `T` minus the matched inbound's scheduled gate-arrival timestamp. | Missing without a match. | Planned ground time available from the schedule. |
| `ROTATION_LOG_SCHEDULED_TURN_MINUTES` | `log1p(ROTATION_SCHEDULED_TURN_MINUTES)`. | Missing without a match. | Skew-reduced scheduled-turn representation for linear models. |
| `ROTATION_INBOUND_ARRIVED_BY_CUTOFF` | 1 when the matched inbound's actual gate arrival is at or before `T`. | Complete binary indicator; 0 without a match. | Shows whether the assigned aircraft is physically available by scheduled departure. |
| `ROTATION_INBOUND_NOT_ARRIVED_BY_CUTOFF` | 1 for a matched inbound whose actual gate arrival is after `T`. | Complete binary indicator; 0 without a match. | Main late-aircraft state available from a live arrival feed at `T`. |
| `ROTATION_INBOUND_OVERDUE_MINUTES` | For `NOT_ARRIVED`, minutes from scheduled inbound arrival through `T`; zero for `ARRIVED`. | Missing without a match. | Severity of the observable failure to arrive by schedule; does not reveal the future actual arrival time. |
| `ROTATION_LOG_INBOUND_OVERDUE_MINUTES` | `log1p(ROTATION_INBOUND_OVERDUE_MINUTES)`. | Missing without a match. | Skew-reduced overdue duration for linear models. |
| `ROTATION_ACTUAL_TURN_MINUTES` | `T` minus actual inbound gate arrival, calculated only when the aircraft had arrived by `T`. | Missing for `NOT_ARRIVED` and unmatched rows. | Amount of ground time already available at the prediction cutoff. |
| `ROTATION_LOG_ACTUAL_TURN_MINUTES` | `log1p(ROTATION_ACTUAL_TURN_MINUTES)`. | Missing unless actual turn time is observable. | Skew-reduced actual-turn representation. |
| `ROTATION_INBOUND_ARR_DELAY` | Signed `ArrDelay` of the matched inbound only when that arrival occurred by `T`. | Missing for `NOT_ARRIVED` and unmatched rows. | Carries observed arrival performance into the next leg without exposing a future outcome. |
| `ROTATION_INBOUND_DELAYED_15` | `ArrDel15` of the matched inbound only when that arrival occurred by `T`. | Missing for `NOT_ARRIVED` and unmatched rows. | Compact observed preceding-leg delay indicator. |

Eight additional audit fields are saved but are not predictors: `ROTATION_TARGET_CUTOFF_UTC`,
`ROTATION_PRIOR_REPORTING_AIRLINE`, `ROTATION_PRIOR_FLIGHT_NUMBER`, `ROTATION_PRIOR_FLIGHT_DATE`,
`ROTATION_PRIOR_SCHEDULED_DEPARTURE_UTC`, `ROTATION_PRIOR_SCHEDULED_ARRIVAL_UTC`,
`ROTATION_PRIOR_ACTUAL_ARRIVAL_UTC`, and `ROTATION_SCHEDULE_RECONSTRUCTION_ERROR_MINUTES`. The actual-arrival audit
timestamp follows the same masking rule as the actual rotation features and is blank unless the arrival had occurred by
`T`.

### Rotation timing, leakage, and availability rules

- `T = DATE` remains the Model 1A prediction cutoff. Target-flight `DepDelay`, `DepDel15`, `ArrDelay`, `ArrDel15`, and
  all later target outcomes are prohibited as predictors.
- BTS uses the origin-local `FlightDate` for a leg and a destination-local `CRSArrTime`. Candidate destination dates are
  converted to UTC, and the date whose duration best agrees with `CRSElapsedTime` is selected. The nonnegative residual
  is retained for audit. Across the generated files, only 11 matched 2019 rows, 10 matched 2023 rows, and two matched
  2024 rows have a residual greater than one minute; none exceeds 60 minutes.
- A matched inbound's actual arrival and delay are exposed only when its reconstructed actual arrival timestamp is at or
  before `T`. For `NOT_ARRIVED`, the future actual arrival timestamp, final delay, and actual turn remain missing. The
  historical final arrival is used only to reconstruct whether the arrival event had occurred; deployment obtains the
  same state from a live event feed.
- `Tail_Number` is a join key, never a direct categorical predictor. BTS records the final aircraft that operated the
  target flight, not a timestamped history of assignments. A last-minute aircraft swap could make that value unknown at
  `T` and create retrospective assignment leakage. Until a timestamped assignment source verifies this assumption,
  rotation model results must be labeled a retrospective upper bound rather than an immediately deployable score.
- The original cohort-limited rotation reference contains only flights retained by the project's working-airport
  filter. In that append-only baseline, `immediately preceding` means the preceding **known in-cohort** JFK event, not
  necessarily the aircraft's complete network history. A previous in-cohort departure blocks a stale inbound, but
  omitted out-of-cohort legs can still create unusually long apparent turns. The separate full-history path is used by
  current experiments so this limitation remains documented without silently changing Experiment 05.
- The append-only full-history path addresses that source limitation without changing the original files. It reads all
  raw BTS movements at the airport, admits only completed non-diverted inbound arrivals, and uses every non-cancelled
  outbound movement as a blocker. Experiment 06 additionally masks matches with scheduled turns over 24 hours using a
  distinct category selected from the 2019 temporal comparison; no target rows are removed.
- Only same-year rotation history is used. Early-January targets may lack a preceding December arrival. Such rows remain
  `NO_PRIOR_EVENT`; they are not filled from future flights or assigned a zero-valued turn.
- Missing-match indicators and categories are explicit. Numeric missing values remain missing for the model pipeline to
  impute or handle natively. Missing is not silently interpreted as zero.
- The standard 112 columns and all eight rotation audit fields remain available for traceability, but each model
  experiment must use an explicit allowlist. It should compare a compact nonduplicative rotation set with the broader
  set rather than automatically using raw and log versions of the same duration.
- Extension to another target airport is supported when its code is present in the helper's documented time-zone map
  and a corresponding cleaned inbound table exists. Adding an airport requires adding and validating its IANA time
  zone before generating data.

## Current Model 1A Operational Feature Manifest

The preferred retrospective Model 1A feature hypothesis combines the most useful members of the separate preparation
paths without materializing another dataset. Model notebooks load the full-history rotation, airport-wide W60, and
same-airline W60 files and require exact equality of these flight identity fields before pairing rows:
`FlightDate`, `Reporting_Airline`, `Flight_Number_Reporting_Airline`, `Origin`, `Dest`, `CRSDepTime`, `Tail_Number`, and
`DepDel15`. The two backlog cutoffs must also match exactly. These checks prevent a plausible-looking but misaligned
horizontal concatenation.

| Feature family | Included fields | Count |
|---|---|---:|
| Raw pre-pushback categorical | `Month`, `DayOfWeek`, `Reporting_Airline`, `Dest` | 4 |
| Raw pre-pushback numeric | `CRSDepTime`, `CRSArrTime`, `CRSElapsedTime`, `Distance`; previous, current, and next ASPM scheduled departure and arrival counts; dew point, dry-bulb temperature, precipitation, relative humidity, visibility, and wind speed | 16 |
| Full-history aircraft rotation | All 13 `ROTATION_...` predictor fields in the dictionary above, with turns over 24 hours masked and labeled `LONG_TURN_EXCLUDED` | 13 |
| Compact airport-wide W60 | `BACKLOG_W60_PENDING_COUNT`, `BACKLOG_W60_COMPLETED_COUNT`, `BACKLOG_W60_MEAN_DEP_DELAY` | 3 |
| Selected same-airline W60 | `AIRLINE_BACKLOG_W60_PENDING_COUNT`, `AIRLINE_BACKLOG_W60_COMPLETED_COUNT`, `AIRLINE_BACKLOG_W60_MEAN_DEP_DELAY`, `AIRLINE_BACKLOG_W60_DELAY_RATE`, `AIRLINE_BACKLOG_W60_PENDING_SHARE` | 5 |
| **Total** | Six categorical and 35 numeric source fields | **41** |

The manifest is an allowlist, not a new CSV schema. CatBoost handles its six categorical fields natively. The controlled
Random Forest comparison one-hot encodes them and applies fold-local numeric imputation, producing 210 prepared fields;
that expansion is model preprocessing rather than feature engineering. CatBoost Experiment 04 selected the
same-airline extension, and Random Forest Experiment 02 independently confirmed that the rotation and backlog families
carry the performance gain, while CatBoost remained the better classifier.

The manifest deliberately excludes the target flight's `DepTime`, departure-delay outcomes, taxi and takeoff events,
arrival outcomes, raw `Tail_Number`, timestamps, and rotation audit columns. Historical backlog outcomes enter only
after their gate-out events, and inbound rotation outcomes enter only after the aircraft has arrived by the cutoff. The
final-tail assignment limitation still makes this a retrospective upper bound. The 2024 full-history and airport-wide
W60 files are prepared, but the same-airline W60 file is intentionally deferred until the design is frozen; therefore
the complete 41-field final-test input has not yet been assembled or evaluated.

# Appendix C

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
   uses a smaller, nonduplicative set of calculated features from Appendix B. Models 2B and 2C must begin with the same
   pre-pushback fields as Model 2A before adding information that becomes available later.
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

# Appendix D

This appendix records the experiments that have been completed. Appendix C describes the plan; Appendix D shows what
was actually run and the results. Separate tables record the model setup, the quality of its rankings and probabilities,
and its results at specific decision thresholds. This makes it clear when a result changes because the threshold changed
rather than because the fitted model changed.

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

# Appendix E

This appendix is a reference for the evaluation terms used in this project. For all four models, the **positive class**
is a flight delayed by at least 15 minutes (`DepDel15 = 1` or `ArrDel15 = 1`), and the **negative class** is a flight
that is not delayed by at least 15 minutes. The definitions below therefore describe finding delayed flights, although
the same definitions apply to any binary-classification problem.

## Confusion matrix

A predicted probability becomes a class prediction after a decision threshold is applied. The confusion matrix counts
the four possible combinations of predicted and actual class.

![Confusion matrix for the flight-delay classifier](resources/diagrams/confusion-matrix.svg)

Let $N = TP + TN + FP + FN$. Metrics based on these four counts change when the decision threshold changes.

## Threshold-dependent classification metrics

| Metric | Definition | Interpretation for this project |
|---|---|---|
| **Accuracy** | $\frac{TP + TN}{N}$ | Fraction of all flights classified correctly. Higher is better. Because non-delayed flights are more common, accuracy can be high even when many delayed flights are missed. |
| **Balanced accuracy** | $\frac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)$ | Average of recall for delayed flights and recall for non-delayed flights. Higher is better. It gives the two classes equal weight even when their sizes differ; 0.5 is the no-skill reference for a binary problem. |
| **Precision** | $\frac{TP}{TP+FP}$ | Among flights for which the model issues a delay alert, the fraction that are actually delayed. Higher precision means fewer false alarms. |
| **Recall** | $\frac{TP}{TP+FN}$ | Among flights that are actually delayed, the fraction found by the model. Higher recall means fewer missed delays. Recall is also called **sensitivity** or the **true-positive rate (TPR)**. |
| **F1 score** | $2\frac{\mathrm{Precision}\times\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}$ | Harmonic mean of precision and recall. Higher is better. F1 is high only when both alert correctness and delayed-flight coverage are high, but it does not use true negatives. |
| **Matthews correlation coefficient (MCC)** | $\frac{TP\times TN-FP\times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | Correlation between the predicted and actual binary classes using all four confusion-matrix cells. It is useful with unequal class sizes. MCC ranges from -1 (complete disagreement) through 0 (no correlation) to 1 (perfect agreement). |

**Specificity**, or the **true-negative rate (TNR)**, is $\frac{TN}{TN+FP}$: the fraction of non-delayed flights correctly
identified. Balanced accuracy is the average of sensitivity and specificity.

## Probability ranking and calibration metrics

These metrics use predicted probabilities rather than one set of class predictions, so they do not change when only the
classification threshold changes.

| Metric | Definition | Interpretation for this project |
|---|---|---|
| **ROC AUC** | Area under the receiver operating characteristic curve, which plots recall (TPR) against the false-positive rate, $\frac{FP}{FP+TN}$, over all thresholds. | Probability that a randomly selected delayed flight receives a higher model score than a randomly selected non-delayed flight, with half credit for ties. Higher is better: 0.5 represents random ranking and 1.0 perfect ranking. Because it weights performance on both classes, it can look strong even when precision for the smaller delayed class is modest. |
| **Average precision (AP)** | Weighted mean of precision as recall increases: $AP=\sum_n(R_n-R_{n-1})P_n$, where $P_n$ and $R_n$ are precision and recall at successive thresholds. | Summarizes the precision-recall curve and emphasizes ranking quality for the delayed class. Higher is better. The no-skill reference is the delayed-flight prevalence, so AP should be read relative to the delay rate of the evaluated data. AP is the project's main model-ranking metric. |
| **Brier score** | Mean squared probability error: $\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2$, where $p_i$ is the predicted delay probability and $y_i$ is 1 for a delay and 0 otherwise. | Measures the accuracy of the probabilities, not just their order. Lower is better: 0 is perfect and 1 is the worst possible binary score. It reflects both calibration and the model's ability to separate the classes and should be compared on the same flight population. |

## Related evaluation terms

| Term | Definition |
|---|---|
| **Predicted probability** | The model's estimated probability $p_i$ that a flight belongs to the delayed class. |
| **Decision threshold** | The cutoff used to convert a probability into an alert. A flight is predicted delayed when $p_i$ is at or above the threshold. Lowering the threshold usually increases recall and false alarms; raising it usually increases precision and missed delays. |
| **Prevalence** | Fraction of evaluated flights that are actually delayed: $\frac{TP+FN}{N}$. It is the no-skill reference for average precision and can change across years or flight populations. |
| **Precision-recall curve** | Precision plotted against recall as the decision threshold varies. It shows the tradeoff between finding more delayed flights and keeping alerts correct. AP summarizes this curve. |
| **ROC curve** | TPR plotted against the false-positive rate as the decision threshold varies. ROC AUC summarizes this curve. |
| **Calibration** | Agreement between predicted probabilities and observed frequencies. For example, among flights assigned a delay probability near 0.30, about 30% should be delayed if the model is well calibrated. A calibration (reliability) curve displays this relationship; the Brier score provides a numeric probability-error summary. |

No single metric answers every evaluation question. In this project, AP and ROC AUC describe ranking, Brier score and
the calibration curve describe probability quality, and the confusion-matrix metrics describe behavior at a stated
operating threshold.

# Appendix F

This appendix is a temporary placeholder for extensions that may be explored if time remains after the four core
15-minute classification models—1A, 2A, 2B, and 2C—are complete. These extensions are not required for the capstone's
main goals.

| Optional stage | Possible extension | Minimum plan |
|---|---|---|
| III | Predict several delay categories | Use the existing BTS departure and arrival delay-group fields to test whether a multi-class model can distinguish levels of delay. Keep these results separate from the binary 15-minute classifier results. |
| IV | Predict delay minutes or a range of likely outcomes | Begin with signed departure-delay minutes. An arrival version would first require adding a signed arrival-delay target and regenerating the arrival feature files. Possible methods include Random Forest, neural-network, mixture-density, and CatBoost ensemble approaches. |

If this work is completed, its methods and results should be folded into the relevant Modeling, Evaluation, Appendix C,
and Appendix D sections. If time does not allow it, Appendix F and its single table-of-contents entry can be removed
without changing the core experiment plan or the documented results for Models 1A, 2A, 2B, and 2C.
