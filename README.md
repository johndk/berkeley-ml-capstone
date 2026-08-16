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
  - [Feature Engineering](Appendix-C.md#feature-engineering)
    - [Engineered feature dictionary](Appendix-C.md#engineered-feature-dictionary)
    - [Feature selection and timing rules](Appendix-C.md#feature-selection-and-timing-rules)
  - [Operational Backlog Feature Engineering](Appendix-C.md#operational-backlog-feature-engineering)
    - [Backlog feature dictionary](Appendix-C.md#backlog-feature-dictionary)
    - [Same-airline backlog feature dictionary](Appendix-C.md#same-airline-backlog-feature-dictionary)
    - [Backlog timing and availability rules](Appendix-C.md#backlog-timing-and-availability-rules)
  - [Aircraft Rotation Feature Engineering](Appendix-C.md#aircraft-rotation-feature-engineering)
    - [Rotation feature dictionary](Appendix-C.md#rotation-feature-dictionary)
    - [Rotation timing, leakage, and availability rules](Appendix-C.md#rotation-timing-leakage-and-availability-rules)
  - [Current Model 1A Operational Feature Manifest](Appendix-C.md#current-model-1a-operational-feature-manifest)
- [Appendix D](Appendix-D.md)
  - [Experiment protocol](Appendix-D.md#experiment-protocol)
  - [Primary binary-classification experiments](Appendix-D.md#primary-binary-classification-experiments)
  - [Reference models and project scope](Appendix-D.md#reference-models-and-project-scope)
- [Appendix E](Appendix-E.md)
  - [Results recording rules](Appendix-E.md#results-recording-rules)
  - [Experiment configurations](Appendix-E.md#experiment-configurations)
  - [Ranking and calibration results](Appendix-E.md#ranking-and-calibration-results)
  - [Operating-threshold results](Appendix-E.md#operating-threshold-results)
    - [Current Model 1A logistic-regression comparison](Appendix-E.md#current-model-1a-logistic-regression-comparison)
    - [Current Model 1A exact-manifest Random Forest comparison](Appendix-E.md#current-model-1a-exact-manifest-random-forest-comparison)
    - [Current Model 1A CatBoost/MLP blend comparison](Appendix-E.md#current-model-1a-catboostmlp-blend-comparison)
    - [Current Model 1A CatBoost calibration comparison](Appendix-E.md#current-model-1a-catboost-calibration-comparison)
    - [Current Model 1A CatBoost subgroup and SHAP audit](Appendix-E.md#current-model-1a-catboost-subgroup-and-shap-audit)
- [Appendix F](Appendix-F.md)
  - [Confusion matrix](Appendix-F.md#confusion-matrix)
  - [Threshold-dependent classification metrics](Appendix-F.md#threshold-dependent-classification-metrics)
  - [Probability ranking and calibration metrics](Appendix-F.md#probability-ranking-and-calibration-metrics)
  - [Related evaluation terms](Appendix-F.md#related-evaluation-terms)

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

See [Appendix C: Feature Engineering](Appendix-C.md#feature-engineering) for the candidate features, their construction, and their rationale.

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
[Appendix C: Operational Backlog Feature Engineering](Appendix-C.md#operational-backlog-feature-engineering).

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
outside-year model performance; the controlled 2023 results are recorded in [Appendix E](Appendix-E.md). The historical reconstruction
also relies on the final BTS `Tail_Number`. A deployable version therefore
requires a timestamped aircraft-assignment feed and live arrival events; unless the recorded tail can be shown to match
the assignment known at `T`, rotation experiment results must be presented as a retrospective upper bound. The full
feature definitions and boundary limitations are documented in
[Appendix C: Aircraft Rotation Feature Engineering](Appendix-C.md#aircraft-rotation-feature-engineering).

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

[Appendix A](Appendix-A.md) documents the [BTS](Appendix-A.md#bts-column-selection-and-dictionary),
[ASPM](Appendix-A.md#aspm-column-selection-and-dictionary), and [NOAA](Appendix-A.md#noaa-column-selection-and-dictionary) source-column
decisions. [Appendix B](Appendix-B.md) documents the
[joined data columns](Appendix-B.md#joined-data-column-dictionary), and [Appendix C](Appendix-C.md) documents the
[feature-engineering analysis](Appendix-C.md#feature-engineering).

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

[Appendix D](Appendix-D.md) contains the full experiment plan and notebook registry. [Appendix E](Appendix-E.md) records the
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

The [experiment registry](Appendix-D.md#primary-binary-classification-experiments) lists the Stage I and II notebooks.

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
   because it can look good even when a model misses many delayed flights. [Appendix D](Appendix-D.md) lists the additional measures.
5. **Choose the decision threshold separately.** Results are reported at the standard 0.50 cutoff and at a cutoff chosen
   from 2019 training-period predictions. The 2023 or 2024 outcomes are not used to choose that cutoff.
6. **Compare like with like.** Modeling methods are compared within the same model and flight population. Models 2A,
   2B, and 2C are compared on identical arrival rows so any difference reflects the newly available flight information
   rather than a change in the sample.

The full rules for repeatability, class balancing, model explanations, and checks across months, airlines, routes,
weather, and traffic levels are in [Appendix D: Experiment protocol](Appendix-D.md#experiment-protocol).

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
[Appendix E](Appendix-E.md).

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
