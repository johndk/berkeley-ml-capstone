# Capstone Project - Berkeley ML and AI 
## Initial Report and Exploratory Data Analysis

## Table of Contents

- [Overview](#overview)
  - [Predict departure delay](#predict-departure-delay)
  - [Predict arrival delay](#predict-arrival-delay)
  - [Why arrival-delay prediction requires more data](#why-arrival-delay-prediction-requires-more-data)
    - [Departure-delay scenario: Model 1A](#departure-delay-scenario-model-1a)
    - [Arrival-delay scenario: Models 2A, 2B, and 2C](#arrival-delay-scenario-models-2a-2b-and-2c)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
  - [Data Sources](#data-sources)
  - [Data Coverage](#data-coverage)
  - [Data Flow](#data-flow)
  - [Matching Records by Time](#matching-records-by-time)
  - [Model Outcomes and Available Information](#model-outcomes-and-available-information)
  - [EDA](#eda)
    - [List of Figures](#list-of-figures)
- [Data Preparation](#data-preparation)
  - [Data Directory](#data-directory)
  - [Raw Data](#raw-data)
  - [Processed and Cleaned Data](#processed-and-cleaned-data)
  - [Model Data](#model-data)
- [Modeling](#modeling)
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
  - [Joined Model-Assembly Column Dictionary](#joined-model-assembly-column-dictionary)
    - [Joined BTS flight columns](#joined-bts-flight-columns)
    - [Joined ASPM planned-demand columns](#joined-aspm-planned-demand-columns)
    - [Joined NOAA weather columns](#joined-noaa-weather-columns)
  - [Feature Engineering](#feature-engineering)
    - [Initial core engineered features](#initial-core-engineered-features)
    - [Feature selection and leakage rules](#feature-selection-and-leakage-rules)

## Overview

This capstone project investigates whether machine learning can predict significant delays for individual domestic flights
departing from or arriving at John F. Kennedy International Airport (JFK).

A significant delay is defined as a departure or arrival delay of 15 minutes or more. The project also explores 
which flight, airport, and weather conditions are most closely associated with delays.

The project combines three main data sources:

- Bureau of Transportation Statistics (BTS) [flight schedules and performance data](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr)
- Aviation System Performance Metrics (ASPM) [airport traffic and congestion data](https://www.aspm.faa.gov/apm/sys/main.asp)
- National Oceanic and Atmospheric Administration (NOAA) [weather observations](https://www.ncei.noaa.gov/data/local-climatological-data/access/)

The JFK analysis is organized into two model categories.

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

Each model follows a clear prediction time, and a field is included only if it would be known at that time. Information
recorded later is excluded even when it appears in the historical datasets. This prevents the model from learning from
the future, often called data leakage. BTS provides the historical event values used for this analysis. A working
operational model would need equivalent gate-out and takeoff information from a suitable live source. Some 
BTS columns represent events—such as gate departure and takeoff—that an airline or  airport operational system can observe 
when they occur.

## Business Understanding

Flight delays create costs and disruption for passengers, airlines, and airports. Earlier warning of a likely delay 
can help airlines communicate with passengers, adjust staffing and gate plans, and prepare for possible missed connections. 
Airports can also use this information to better understand when congestion or weather is likely to affect operations.

The primary research question is:

- Can a departure delay of 15 minutes or more be identified before pushback?

~~If time allows,~~ The arrival-delay extension asks:

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

The exact division of the years and flights into training, development, and final test sets remains to be decided. 
The split preserves time order so that the models are trained on earlier flights and evaluated
on later flights they have not seen. Files use a consistent `JFK_YEAR.csv` naming pattern.

### Why arrival-delay prediction requires more data

Each flight is represented by one BTS row containing both its `Origin` and `Dest`. The tables below identify the
airport-specific ASPM and NOAA context attached to that row in the baseline design.

#### Departure-delay scenario: Model 1A

| Airport role | BTS information used                                       | ASPM context                                 | NOAA context                                        | Data footprint                 |
|---|------------------------------------------------------------|----------------------------------------------|-----------------------------------------------------|--------------------------------|
| JFK origin | Schedule, airline, route, distance, and `DepDel15` target  | JFK planned traffic near scheduled departure | Latest JFK weather available by the prediction time | BTS, ASPM, NOAA for JFK Origin |
| Flight destination | Destination and schedule information from the same BTS row | -                                            | -                                                   | -                              |

#### Arrival-delay scenario: Models 2A, 2B, and 2C

| Airport role | BTS information used | ASPM context                                   | NOAA context                                                   | Data footprint                                    |
|---|---|------------------------------------------------|----------------------------------------------------------------|---------------------------------------------------|
| Each flight origin | Schedule, airline, route, and Model 2B/2C operating fields | Planned traffic near departure for that origin | Latest origin weather available by the model's prediction time | BTS, ASPM, NOAA for as many as 79 origin airports |
| JFK destination | Destination identity, scheduled arrival information, and `ArrDel15` target from the same BTS row | -                                              | -                                                              | BTS for JFK Destination                           |

Both scenarios use the same basic idea: combine flight-level BTS information with planned airport demand and
weather to classify whether a flight crosses the 15-minute delay threshold. The difference is the scale of the external
data work. Model 1A always departs from JFK, so one ASPM dataset and one NOAA station can serve every row. In the
arrival scenario, the departure airport changes from flight to flight. Each included origin needs ASPM coverage, a NOAA
station mapping, source cleaning and validation, and its own prediction-time-safe joins.

Models 2B and 2C also introduce later BTS operating events, but those fields do not create the main data expansion. The
larger burden comes from collecting and validating consistent ASPM and NOAA context across the inbound origin set.
Destination ASPM and weather are not part of the baseline arrival design. Weather later observed at landing is
prohibited because it was not available when the prediction was made.

The work includes:

- Checking data quality, missing values, unusual values, and the balance between delayed and on-time flights
- Exploring which flight, airport, and weather conditions are most closely associated with delays
- Creating useful features from dates, times, flights, airport, and weather conditions
- Comparing logistic regression, random forest, and gradient-boosting models such as CatBoost
- Measuring model performance on later flights that were not used for training
- Explaining which factors have the strongest effect on each model's predictions

The project focuses on individual flights. Aircraft rotations, previous-flight chains, and delay spread through an airline 
network are out of scope. The approach is informed by the flight-level delay research of Snell, Zoutendijk, and Pineda. 
The final analysis focuses on model performance and the factors associated with delays at JFK.

### Data Flow

```text
Raw BTS  ──→ Process BTS  ──→ Clean BTS  ─────┐
                                              │
Raw ASPM ──→ Process ASPM ──→ Clean ASPM ─────┼──→ Cleaned source data
                                              │               │
Raw NOAA ──→ Process NOAA ──→ Clean NOAA ─────┘               │
                                                              │
                     ┌───────────────────┬────────────────────┼───────────────────┐
                     ▼                   ▼                    ▼                   ▼
          notebooks/model_1a.ipynb  model_2a.ipynb      model_2b.ipynb      model_2c.ipynb
          Join and engineer         Join and engineer   Join and engineer   Join and engineer
          JFK departures            JFK arrivals        JFK arrivals        JFK arrivals
                     │                   │                    │                   │
                     ▼                   ▼                    ▼                   ▼
             data/models/         data/models/        data/models/        data/models/
             JFK_YEAR_m1a.csv     JFK_YEAR_m2a.csv    JFK_YEAR_m2b.csv    JFK_YEAR_m2c.csv
```

Processing first makes each source easier to use. Cleaning then checks the quality and consistency of the data. The
four model notebooks load the cleaned sources directly, perform their permitted time-safe joins, and create the
corresponding model datasets. `model_1a.ipynb` produces the departure dataset. Within each notebook, the baseline feature set
is evaluated first and engineered features are then applied on top of that baseline to determine whether they improve
performance. See [Appendix B: Feature Engineering](#feature-engineering) for the candidate features, their construction and rationale.

### Matching Records by Time

The scheduled departure date and time, stored in `DATE`, provides the main time for each BTS flight. Each flight is matched
with ASPM schedule records for the previous, current, and next clock hours. The next-hour values are planned counts known
ahead of time, not future operating results. Each flight is also matched with the most recent NOAA observation available
before its scheduled departure. The model notebook keeps source timestamps and timing differences in its working
dataframe so the matches can be checked.

This time-based matching prevents a model from using airport conditions or weather observations that occurred after the prediction was made. It also allows the age of the matched information to be checked before modeling.

### Model Outcomes and Available Information

| Category  | Model | Outcome | Information available when the prediction is made |
|-----------|---|---|---|
| Departure | Model 1A | `DepDel15` | Flight schedule, earlier airport conditions, and earlier weather observations |
| Arrival   | Model 2A | `ArrDel15` | The same general pre-pushback information as Model 1A, collected for the flight origin |
| Arrival   | Model 2B | `ArrDel15` | Model 2A information plus the actual departure time and departure delay |
| Arrival   | Model 2C | `ArrDel15` | Model 2B information plus taxi-out and takeoff information |

Each model notebook engineers features that summarize time of day, season, route, distance, congestion, and weather. Each
saved model dataset excludes information recorded after its stated prediction time.

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
│   ├── raw/
│   │   ├── filter.py
│   │   ├── 2019/
│   │   │   ├── On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2019_1/
│   │   │   │   ├── On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2019_1.csv
│   │   │   │   └── readme.html
│   │   │   ├── On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2019_2/
│   │   │   │   ├── On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2019_2.csv
│   │   │   │   └── readme.html
│   │   │   ├── ... same monthly structure through 2019_12
│   │   │   └── JFK.csv
│   │   ├── 2023/
│   │   │   └── ... same monthly structure and annual JFK file
│   │   └── 2024/
│   │       └── ... same monthly structure and annual JFK file
│   ├── processed/
│   │   ├── JFK_2019.csv
│   │   └── ... through JFK_2024.csv
│   └── cleaned/
│       ├── JFK_2019.csv
│       └── ... through JFK_2024.csv
├── aspm/
│   ├── raw/
│   │   ├── download_aspm_hourly_v3.py
│   │   ├── download_aspm_2019.sh
│   │   ├── download_aspm_2023.sh
│   │   ├── download_aspm_2024.sh
│   │   └── aspm_output/
│   │       ├── run_2019_JFK/
│   │       │   ├── aspm_2019_JFK.csv
│   │       │   └── raw_html/
│   │       │       ├── JFK_2019-01-01.html
│   │       │       ├── JFK_2019-01-02.html
│   │       │       └── ... one HTML response per day through 2019-12-31
│   │       └── ... corresponding JFK runs for 2023 and 2024
│   ├── processed/
│   │   ├── JFK_2019.csv
│   │   └── ... through JFK_2024.csv
│   └── cleaned/
│       ├── JFK_2019.csv
│       └── ... through JFK_2024.csv
├── noaa/
│   ├── raw/
│   │   ├── 2019/
│   │   │   └── 74486094789.csv
│   │   ├── 2023/
│   │   │   └── 74486094789.csv
│   │   └── 2024/
│   │       └── 74486094789.csv
│   ├── processed/
│   │   ├── JFK_2019.csv
│   │   └── ... through JFK_2024.csv
│   └── cleaned/
│       ├── JFK_2019.csv
│       └── ... through JFK_2024.csv
└── models/
    ├── JFK_2019_m1a.csv
    ├── JFK_2019_m2a.csv
    ├── JFK_2019_m2b.csv
    ├── JFK_2019_m2c.csv
    └── ... same four model dataset files for 2023 and 2024
        ... and model variants
```

The folders represent the main stages of the data:

### Raw Data

The `raw` folders contain data as downloaded or first collected, along with the scripts used to retrieve or separate the source files.

BTS On-Time Performance data is provided as a separate compressed download for each month. Each monthly file contains 
domestic flights reported by all included United States carriers, rather than data for a single airport. After the monthly files 
are downloaded and extracted, `filter.py` reads all 12 files for a year, keeps flights where JFK is either the origin or
destination, and combines the matching records into one annual `JFK.csv` file. For example, the 12 monthly files under
`data/bts/raw/2019/` are used to produce `JFK.csv` in that same directory. This annual airport file becomes the input to
the BTS processing notebook.

ASPM data is collected with `download_aspm_hourly_v3.py`, which requests one FAA ASPM hourly report for JFK for each
calendar day. The year-specific shell scripts supply the date range and airport code. Each run saves the original daily
HTML responses under `raw_html/` and combines the extracted hourly records into one annual airport CSV, such as
`aspm_2019_JFK.csv`. Keeping the HTML responses preserves the original reports while the annual CSV becomes the input to
the ASPM processing notebook.

NOAA Local Climatological Data is downloaded as one annual CSV for the weather station selected to represent JFK.
Station `74486094789` represents JFK. The files are grouped by year under `data/noaa/raw/` and provide the raw weather
observations used to produce the JFK inputs for the NOAA processing notebook.

### Processed and Cleaned Data

The `processed` folders contain smaller working files with the records and source columns needed by the project. Processing
removes empty, redundant, or out-of-scope fields before the more detailed cleaning checks begin.

The work differs slightly by source. BTS processing removes cancelled and diverted flights and keeps the schedule,
route, and outcome fields needed for analysis. ASPM processing keeps the airport and time identifiers together with
scheduled hourly arrivals and departures. NOAA processing keeps the selected hourly weather measurements and removes
rows that contain no usable weather observations.



The `cleaned` folders contain the standardized source files used directly by the model notebooks. Cleaning converts or constructs
timestamps, sorts the records, and checks the expected columns, missing values, duplicate keys, numeric ranges, category
codes, time coverage, and outcome consistency. Validation checks report possible problems without changing otherwise
valid records.

NOAA requires some additional preparation. Trace precipitation is converted to a small numeric value, weather codes are
turned into plain condition indicators, and wind direction and speed are used to create `WindX` and `WindY`. Missing
continuous measurements are also filled during the current cleaning process. Before modeling, this fill method must be
limited to weather already observed at the relevant prediction time so a later observation cannot influence an earlier
flight.

The cleaned BTS, ASPM, and NOAA JFK-year files remain separate until a model notebook loads them. Each model notebook
keeps one row per eligible flight and adds the appropriate scheduled airport demand and time-matched weather observation.

Cleaning a field does not automatically make it a valid predictor. Some retained BTS fields are outcomes, validation
fields, or operating events available only to a later model. Fields published too late to support the prediction, such
as ASPM's realized performance measures, are removed. The model-ready files later enforce the exact information
available at each prediction time and prevent time leakage.

[Appendix A](#appendix-a) documents the source-column decisions in more detail: the
[BTS column selection and timing rules](#bts-column-selection-and-dictionary),
[ASPM planned-demand fields](#aspm-column-selection-and-dictionary),
and [NOAA weather fields](#noaa-column-selection-and-dictionary).
[Appendix B](#appendix-b) documents the
[joined model-assembly columns](#joined-model-assembly-column-dictionary) and
[feature-engineering analysis](#feature-engineering).

### Model Data

The `models` folder contains the initial baseline dataset for each prediction scenario and year:
`JFK_YEAR_m1a.csv`, `JFK_YEAR_m2a.csv`, `JFK_YEAR_m2b.csv`, and `JFK_YEAR_m2c.csv`. The `_m1a`, `_m2a`, `_m2b`, and
`_m2c` suffixes identify the model scenario and its prediction-time information boundary. Each baseline file contains
only the baseline predictors and target permitted for that scenario.

Engineered features are created on top of the corresponding fixed baseline dataset. The baseline files remain unchanged
so every engineered-feature experiment starts from the same reproducible data. Baseline and engineered variants are
defined as feature projections and compared within the modeling workflow rather than replacing the `_m1a`, `_m2a`,
`_m2b`, or `_m2c` files. The candidate engineered features are documented in
[Appendix B: Feature Engineering](#feature-engineering).

The processing and cleaning work is recorded in separate notebooks for BTS, ASPM, and NOAA. The four model-assembly
notebooks perform the scenario-specific joins, validation, and fixed baseline column projection directly from those
cleaned inputs. Feature engineering and model experiments then begin with the resulting baseline files.

## Modeling

Baseline models for the capstone check-in requirement - "Develop an appropriate classification or 
regression ML model to utilize as baseline for your analysis" - can be found here:

- [decision_tree_01.ipynb](models/decision_tree_01.ipynb)
- [decision_tree_02.ipynb](models/decision_tree_02.ipynb)
- [logistic_regression_01.ipynb](models/logistic_regression_01.ipynb)
- [logistic_regression_02.ipynb](models/logistic_regression_02.ipynb)

### What the baseline models show

The main point is simple: the current models are a starting point, not the finished analysis. They use a small set of
basic inputs and do somewhat better than guessing, but they do not yet identify delayed flights well enough. The next
stage will describe each flight more carefully, try stronger models, and test whether they can predict flights that
occur later in time.

The four notebooks above are four experiments for Model 1A. They do not yet represent all four project models. The
`_01` notebooks contain the selected logistic-regression and decision-tree models. The `_02` notebooks show how their
settings were chosen with grid searches. All four notebooks use the same 2019 departure data and the same 20 basic
schedule, airline, destination, airport-traffic, and weather columns. They do not yet use the engineered features in
Appendix B, and they have not yet been tested with the 2023 or 2024 data.

The models have some ability to place delayed flights above on-time flights when flights are ordered from highest to
lowest predicted risk, but that ability is still limited. The score used to measure the quality of that ordered list,
called average precision, is 0.3595 for logistic regression and 0.3968 for the decision tree. A model with no useful
ordering ability would score close to the delayed-flight rate of 0.1887. A second ranking measure, ROC AUC, gives the
models scores of 0.6996 and 0.7060. Both measures show the same basic result: the models have learned something, but
they still confuse many delayed and on-time flights.

The default logistic-regression threshold of 0.50 identifies only 3.04% of delayed flights. Lowering the threshold to
0.225 raises that figure to 55.66%. However, only 32.60% of its delay warnings are correct, so it also creates many false
alarms. The decision tree identifies 64.07% of delayed flights, but only 30.39% of its delay warnings are correct. These
results show why overall accuracy is not enough. Because most flights are on time, a model can appear accurate simply
by predicting that almost every flight will be on time. The project must report both how many real delays are found and
how many warnings are correct.

Changing the logistic-regression threshold does not make the underlying model better at ranking flights by risk. It
only changes how many flights are labeled as delayed. Improving the model itself will require better inputs, additional
model types, and stronger testing.

### Planned improvements

The next work will proceed in a clear sequence:

1. **Describe each flight more clearly with the data already collected.** The baseline currently treats scheduled times
   such as `2300` as ordinary numbers. That does not tell the model that 23:00 is close to 00:00. The improved version
   will represent the clock in a way that keeps nearby times close together. Other planned inputs include time of day,
   season, route, airline and destination combinations, total scheduled traffic around departure, changes in scheduled
   traffic, and simple adverse-weather indicators.
   [Pineda et al.](resources/docs/15_Pineda_ExplainableDelayML.pdf) found that time of day, destination, weather, and
   combinations of inputs helped explain delays.
   [Li](resources/docs/04_Li_DelayPropagationPrediction.pdf) and
   [Chen and Li](resources/docs/06_Chen_ChainedDelayPrediction.pdf) also found airport operating conditions useful.
   Groups of new inputs will be added one at a time so that any improvement can be traced to a specific change.

2. **Try models that can find patterns the current models miss.** A single decision tree makes all of its choices through
   one tree. A random forest combines the results of many different trees. A gradient-boosted model builds a series of
   trees, with each new tree trying to correct mistakes made by the earlier trees. These are reasonable next models for
   this project. [Snell et al.](resources/docs/02_Snell_MLFlightDelayPrediction.pdf) and
   [AlBassam](resources/docs/05_AlBassam_MLDelayEval.pdf) found that combined-tree models and methods for handling the
   smaller delayed class improved results in their datasets. Their reported scores cannot be compared directly with
   this project because the airports, features, and prediction times differ. Class weighting will be tested, and SMOTE
   may also be tested. SMOTE creates additional training examples that resemble delayed flights, giving the model more
   delayed examples from which to learn. These created examples must remain in training data and must never enter
   validation or test data.

3. **Test models in time order.** The current random split places flights from throughout 2019 into both training and
   test sets. Flights from the same day can therefore appear on both sides of the split and share similar weather,
   schedules, congestion, or disruptions. That makes the test data more similar to the training data than future
   flights would be. The improved test will keep complete days together and always train on earlier flights before
   evaluating later flights. Earlier parts of 2019 can be used to predict later parts of 2019 while features and model
   settings are being developed. The selected model can then be trained on 2019 and checked on 2023. After all choices
   are settled, 2024 can remain untouched for the final test. This will show whether the model continues to work after
   schedules, airlines, traffic levels, and operating conditions change. Beltman et al. also kept complete days
   together to reduce the chance that closely related same-day information appeared in both training and test data.

4. **Check whether the predicted chances of delay are believable.** If the model gives a group of flights a 30% chance
   of delay, roughly 30% of those flights should actually be delayed. A simple chart can compare predicted chances with
   what actually happened. A summary measure called the Brier score can report the size of the difference. Believable
   probabilities are important because the best warning threshold depends on how costly it is to miss a delay compared
   with issuing a false warning. [Zoutendijk and Mitici](resources/docs/03_Zoutendijk_ProbabilisticFlightDelay.pdf)
   and [Beltman et al.](resources/docs/16_Beltman_DepartureDelayForecast.pdf) show why the uncertainty around a delay
   prediction can be useful for airport and airline decisions.

5. **Explain what the models learned.** Logistic-regression coefficients and decision-tree feature importance provide
   a starting point. More complicated models are harder to understand because their answer comes from many decisions
   spread across many trees. SHAP is a method for turning those decisions into a readable explanation. For one flight,
   it shows which inputs raised the predicted chance of delay and which inputs lowered it. Across many flights, it shows
   which inputs most often influenced the predictions.

   SHAP was not covered in the course, and it is not needed to fit a model. It is valuable for this capstone because the
   result should explain more than which model earned the highest score. It should also show whether the model is using
   reasonable information, such as time, weather, route, and planned airport traffic. It may also reveal a problem, such
   as a model depending too heavily on one airline or one route. SHAP explains what the model learned from the data; it
   does not prove that an input caused a delay. Pineda et al. use this kind of explanation to connect model results with
   practical flight-delay factors.

The final comparison will also examine where each model fails, including results by month, time of day, airline, route,
weather, and airport traffic. This may reveal that a model works well under normal conditions but poorly during severe
weather or unusually congested periods.

### Why the arrival models may improve

Models 2A, 2B, and 2C predict the same arrival-delay outcome at three different points in the flight:

| Model | Prediction time | Information added |
|---|---|---|
| 2A | Before pushback at the origin | Schedule, route, origin airport traffic, and origin weather |
| 2B | Immediately after pushback | Model 2A information plus actual gate-out time and departure delay |
| 2C | Immediately after takeoff | Model 2B information plus taxi-out time and actual takeoff time |

There is good reason to expect Models 2B and 2C to improve on Model 2A. Li and Chen found that departure delay was the
most important input for predicting arrival delay, and their predictions improved as actual departure information was
added. Snell et al. also reported better arrival predictions after adding actual departure-delay information.

The improvement is still something this project must demonstrate. Zoutendijk and Mitici found that arrival delays were
harder to predict than departure delays in their data. This project's arrival data also includes many origin airports,
while Model 1A always begins at JFK. The arrival problem may therefore contain more variation in weather, traffic, and
airport operations.

Models 2A, 2B, and 2C will be compared using the same flights and the same date ranges. Only the newly available
operating information will change from one model to the next. This will show whether actual departure delay and takeoff
information really improve the predictions. It will also show an important tradeoff: a prediction made after takeoff
may be more accurate, but it gives the airline and passengers less advance warning.

The timing rules must remain strict. `DepTime` and `DepDelay` cannot be used in Model 2A because they are not known
before pushback. `TaxiOut` and `WheelsOff` cannot be used in Models 2A or 2B because they are not fully known until
takeoff. Keeping these fields out of the earlier models ensures that any improvement is based on information that
would truly have been available at that prediction time.

## Evaluation

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

This appendix documents source-column selection, processing decisions, and timing eligibility for BTS, ASPM, and NOAA.

## BTS Column Selection and Dictionary

The downloaded BTS data contains 110 columns. Processing removes cancelled and diverted flights and drops 83
unneeded columns. Cleaning validates the remaining fields and adds `DATE`, producing 28 columns before 
the ASPM and NOAA merge join.

Column selection is based on both usefulness and timing. A value may be useful for describing a completed flight but 
still be an invalid predictor if it becomes known only after the model's prediction time. Using such a value would give 
the model information from the future, commonly called target or time leakage.

BTS publishes the On-Time Performance data as a historical reporting dataset. However, some BTS columns represent events—such as gate departure and takeoff—that an airline or 
airport operational system can observe when they occur.

`DepTime`, `TaxiOut`, and `WheelsOff` are retained because they mark useful and clearly different information points.
`DepTime` is the actual gate-out or pushback time and is eligible for Model 2B. `TaxiOut` is not complete, and `WheelsOff` 
is not known, until takeoff; both are excluded from Models 1A, 2A, and 2B. They become eligible for Model 2C, which updates 
the arrival-delay prediction immediately after takeoff.

### Prediction-time eligibility of key BTS operational fields

| Field | Model 1A: before pushback | Model 2A: before pushback | Model 2B: after pushback | Model 2C: after takeoff |
|---|---|---|---|---|
| Scheduled fields such as `CRSDepTime`, `CRSArrTime`, and `CRSElapsedTime` | Eligible | Eligible | Eligible | Eligible |
| `DepTime` and departure-delay fields derived from gate-out | Excluded: not yet known | Excluded: not yet known | Eligible: gate-out has occurred | Eligible |
| `DepDel15` | Target only | Excluded: not yet known | Eligible: gate-out has occurred | Eligible |
| `TaxiOut` and `WheelsOff` | Excluded: takeoff has not occurred | Excluded: takeoff has not occurred | Excluded: takeoff has not occurred | Eligible |
| `ArrDel15` | Not used | Target only | Target only | Target only |
| Other arrival outcomes and post-arrival fields | Excluded | Excluded | Excluded | Excluded |

“Dropped” means a field is unnecessary or inappropriate for this capstone design; it does not mean the field has no value in other aviation studies.

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
| `Tail_Number` | Aircraft registration or tail number. | Retained for audit and validation. It is not intended as a model feature because aircraft-chain and propagation modeling are out of scope. |
| `Flight_Number_Reporting_Airline` | Flight number assigned by the reporting carrier. | Retained as a flight identifier and possible categorical feature; it is not treated as a continuous quantity. |
| `Origin` | Three-letter origin airport code. | Retained to identify flights originating at JFK for Model 1A and the departure airport for Models 2A, 2B, and 2C. |
| `OriginState` | Two-letter state code for the origin airport. | Retained as a compact geographic field. |
| `Dest` | Three-letter destination airport code. | Retained to identify flights arriving at JFK for Models 2A, 2B, and 2C. |
| `DestState` | Two-letter state code for the destination airport. | Retained as a compact geographic field. |
| `CRSDepTime` | Computer Reservation System scheduled departure time in local HHMM form. | Retained because it is known before departure and is used to construct the exact scheduled departure timestamp. |
| `DepTime` | Actual gate departure time in local HHMM form. | Retained as the event-time field for Model 2B and for descriptive analysis. It is excluded from Models 1A and 2A because gate-out has not occurred when those predictions are made. BTS supplies the historical value; a deployed Model 2B would require an operational gate-out feed. |
| `DepDelay` | Actual gate departure time minus scheduled departure time, in minutes; negative values indicate an early departure. | Retained as a Model 2B predictor because it is known once gate-out occurs. It is excluded from Models 1A and 2A. Because it overlaps the other departure-delay fields, the final Model 2B projection should avoid unnecessary duplicate representations. |
| `DepDelayMinutes` | Nonnegative departure delay in minutes; early departures are recorded as zero. | Retained as a possible Model 2B representation and to validate `DepDelay` and `DepDel15`. It is excluded from pre-pushback predictors and need not be included together with every related departure-delay field. |
| `DepDel15` | Indicator equal to 1 when departure delay is at least 15 minutes. | Target for Model 1A. It is excluded from Model 2A but may be used as a compact post-pushback input for Model 2B because the departure outcome is known at gate-out. It is never used to predict itself in Model 1A. |
| `DepartureDelayGroups` | Departure delay grouped into ordered 15-minute ranges. | Retained for validation, EDA, and possible Model 2B use. It is excluded before pushback and is not automatically included alongside the continuous and binary departure-delay fields. |
| `TaxiOut` | Minutes from gate departure to wheels-off. | Retained as a Model 2C predictor and for validation and EDA. It is excluded from Models 1A, 2A, and 2B because the full taxi-out duration is unknown immediately after pushback. |
| `WheelsOff` | Actual takeoff time in local HHMM form. | Retained as a Model 2C predictor and for validation and EDA. It is excluded from Models 1A, 2A, and 2B because takeoff occurs after their prediction times. As with `DepTime`, BTS is the historical source rather than the proposed live event feed. |
| `CRSArrTime` | Scheduled arrival time in the destination's local HHMM form. | Retained because it is known from the schedule before departure. |
| `ArrDel15` | Indicator equal to 1 when arrival delay is at least 15 minutes. | Target for Models 2A, 2B, and 2C. It is never used as a predictor because it is known only after arrival. |
| `ArrivalDelayGroups` | Arrival delay grouped into ordered 15-minute ranges. | Retained for target validation and EDA. It is excluded from every model input because it describes the completed arrival outcome. |
| `CRSElapsedTime` | Scheduled gate-to-gate elapsed time, in minutes. | Retained as a schedule and route characteristic known before departure. |
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
| `WheelsOn` | Actual landing time in local HHMM form. | Dropped because it occurs after takeoff and would leak future information into arrival-delay prediction. |
| `TaxiIn` | Minutes from wheels-on to gate arrival. | Dropped because it is known only after landing and would leak the arrival outcome. |
| `ArrTime` | Actual gate arrival time in local HHMM form. | Dropped because it directly reveals the arrival outcome. |
| `ArrDelay` | Actual arrival time minus scheduled arrival time, in minutes. | Dropped because the project uses the binary ArrDel15 target and the numeric value directly reveals that target. |
| `ArrDelayMinutes` | Nonnegative arrival delay in minutes. | Dropped because it directly determines the binary ArrDel15 target. |
| `ArrTimeBlk` | BTS scheduled arrival time block. | Dropped because the more precise CRSArrTime is retained and can be used to derive time categories. |
| `Cancelled` | Indicator that the flight was cancelled. | Cancelled flights are removed because they have no completed departure or arrival outcome; the now-constant indicator is then dropped. |
| `CancellationCode` | BTS code for the reason a flight was cancelled. | Dropped after cancelled flights are removed; it is not applicable to the remaining completed flights. |
| `Diverted` | Indicator that the flight was diverted. | Diverted flights are removed because their scheduled-destination arrival outcome is not comparable with an ordinary completed flight; the now-constant indicator is then dropped. |
| `ActualElapsedTime` | Actual gate-to-gate elapsed time, in minutes. | Dropped because it is known only after arrival and would leak future operational information. |
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

The downloaded ASPM hourly file contains 18 columns. The project keeps the airport, date, hour, scheduled departures, and scheduled arrivals. Cleaning also creates `DATE`, giving the cleaned ASPM data six columns. The scheduled counts represent planned airport demand and are assumed to be accurate hourly summaries of the schedule. Historical schedule counts may include later revisions, so this is a practical project assumption.

The other 13 source columns are dropped. They contain supporting calculation counts or summaries of actual airport performance, such as on-time percentages and average delays. ASPM generally publishes these operational results as part of the following day's update, so they are not available close enough to departure or arrival time for the project's predictions. Looking at the previous operating hour would not solve that publication delay.

### ASPM columns retained for modeling

| Column | Meaning | Why it is retained / how it is used |
|---|---|---|
| `airport` | Three-letter code for the airport represented by the hourly record. | Retained as the airport identifier and merge key. |
| `report_date` | Calendar date associated with the hourly record. | Converted to a pandas datetime and combined with `Hour` to create the hourly timestamp. |
| `Hour` | Local airport hour, represented by an integer from 0 through 23. | Retained to identify the hourly reporting period and construct `DATE`. |
| `Scheduled Departures` | Number of flights scheduled to depart during the hour. | Retained as a planned measure of departure demand and airport workload. ASPM is assumed to have performed the hourly schedule roll-up that would otherwise be calculated from BTS. |
| `Scheduled Arrivals` | Number of flights scheduled to arrive during the hour. | Retained as a planned measure of arrival demand and airport workload under the same ASPM schedule-roll-up assumption. |
| `DATE` | Constructed hourly timestamp created by combining `report_date` and `Hour`. | Added during cleaning and used as the time key for sorting, coverage validation, and the lagged BTS merge. |

### ASPM columns dropped during processing

| Column | Meaning | Why it is dropped |
|---|---|---|
| `Departures For Metric Computation` | Number of qualifying departures used by ASPM to calculate the reported departure performance metrics. | Dropped because it is a later-published auxiliary denominator, not a planned-demand measure. `Scheduled Departures` provides the relevant schedule-based count. |
| `Arrivals For Metric Computation` | Number of qualifying arrivals used by ASPM to calculate the reported arrival performance metrics. | Dropped because it is a later-published auxiliary denominator, not a planned-demand measure. `Scheduled Arrivals` provides the relevant schedule-based count. |
| `% On-Time Gate Departures` | Percentage of qualifying flights that departed the gate on time during the hour. | Dropped primarily because this realized hourly result is not published near enough to prediction time. It is also an aggregate departure-delay outcome that closely parallels `DepDel15`. |
| `% On-Time Airport Departures` | Percentage of qualifying flights whose airport departure was on time during the hour. | Dropped primarily because it is not available near real time. It also duplicates related departure-performance information and summarizes an outcome that has already occurred. |
| `% On-Time Gate Arrivals` | Percentage of qualifying flights that arrived at the gate on time during the hour. | Dropped primarily because it is not available near real time. It is also an aggregate arrival-delay outcome that closely parallels `ArrDel15`. |
| `Average Gate Departure Delay` | Average difference between scheduled and actual gate departure time for qualifying flights in the hour. | Dropped primarily because ASPM publishes it too late for the intended prediction. It is also a target-proximate departure outcome that overlaps other operating measures. |
| `Average Taxi Out Time` | Average number of minutes from gate departure to wheels-off for the flights represented in the hour. | Dropped because it is a realized operating measure that ASPM publishes too late for the intended prediction times. Retrospective analysis of this field is outside the project scope. |
| `Average Taxi Out Delay` | Average taxi-out delay beyond the expected or unimpeded taxi-out time. | Dropped because it is a realized congestion measure that ASPM publishes too late for the intended prediction times. |
| `Average Airport Departure Delay` | Average airport departure delay for qualifying flights in the hour, including delay accumulated before takeoff. | Dropped primarily because ASPM publishes it too late for the intended prediction. It is also a composite departure outcome that overlaps gate and taxi-out performance. |
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

The NOAA data describes weather observed near JFK. Only hourly fields that are useful for the delay models are kept.
Daily and monthly summaries are dropped because they do not describe conditions at a specific prediction time and may
include weather that occurred later.

Each model uses the most recent NOAA observation available by its prediction time. Later observations are excluded to prevent data leakage. Missing weather values must also be handled with a past-only method so that an earlier record is not filled using future weather.

### NOAA columns retained for modeling

| Column | Plain-English description | Why it is retained |
|---|---|---|
| `DATE` | Date and time of the weather observation. | Used to match each flight with weather already observed by the prediction time. |
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

During the merge, `DATE` becomes `NOAA_DATE` so it is not confused with the flight timestamp. The weather feature names remain unchanged. `NOAA_AGE_MINUTES` records how old the matched weather observation is at the relevant flight time.

# Appendix B

This appendix documents the joined model-assembly columns and the candidate engineered features derived from them.

## Joined Model-Assembly Column Dictionary

Inside `model_1a.ipynb`, the joined working dataframe contains one row per completed, non-diverted flight departing JFK.
Before feature engineering and final projection, it has 71 columns: 28 BTS flight fields, 24 ASPM planned-demand and
join-audit fields, and 19 NOAA weather and join-audit fields.

This in-memory dataframe is intentionally broader than the saved model dataset. It contains targets, descriptive
outcomes, source timestamps, and operational events from different points in time. A column appearing here does not mean
it is eligible for every model. The final projection keeps only information available at the stated prediction time.

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
| `Tail_Number` | Aircraft registration number. | Retained for audit; not planned as a predictor because aircraft-chain modeling is out of scope. |
| `Flight_Number_Reporting_Airline` | Flight number assigned by the reporting airline. | Flight identifier and possible categorical feature, not a numeric measurement. |
| `Origin` | Three-letter departure-airport code. | Identifies the origin and is part of the ASPM join key. |
| `OriginState` | Two-letter state code for the origin. | Compact origin-location field. |
| `Dest` | Three-letter destination-airport code. | Identifies the route destination. |
| `DestState` | Two-letter state code for the destination. | Compact destination-location field. |
| `CRSDepTime` | Scheduled departure time in local HHMM form. | Known before departure and used to construct the scheduled timestamp. |
| `DepTime` | Actual gate departure or pushback time in local HHMM form. | Available only after pushback; eligible for Model 2B, not Models 1A or 2A. |
| `DepDelay` | Actual gate departure minus scheduled departure, in minutes. | Model 1A target information and a possible Model 2B predictor; unavailable before pushback. |
| `DepDelayMinutes` | Nonnegative departure delay in minutes. | Outcome field used for validation and possible Model 2B input. |
| `DepDel15` | Indicates a departure delay of at least 15 minutes. | Target for Model 1A and possible post-pushback input for Model 2B. |
| `DepartureDelayGroups` | Departure delay grouped into 15-minute ranges. | Outcome field for EDA and validation; unavailable before pushback. |
| `TaxiOut` | Minutes from gate departure to takeoff. | Available only after takeoff; eligible for Model 2C and excluded from Models 1A, 2A, and 2B. |
| `WheelsOff` | Actual takeoff time in local HHMM form. | Available only at takeoff; eligible for Model 2C and excluded from Models 1A, 2A, and 2B. |
| `CRSArrTime` | Scheduled arrival time in the destination's local HHMM form. | Schedule field known before departure. |
| `ArrDel15` | Indicates an arrival delay of at least 15 minutes. | Target for Models 2A, 2B, and 2C; never a predictor. |
| `ArrivalDelayGroups` | Arrival delay grouped into 15-minute ranges. | Completed-flight outcome used only for EDA and target validation. |
| `CRSElapsedTime` | Scheduled gate-to-gate travel time in minutes. | Route and schedule feature known before departure. |
| `Distance` | Published route distance in miles. | Route feature known before departure. |
| `DistanceGroup` | BTS distance band based on 250-mile intervals. | Grouped route-length field available before departure. |
| `DATE` | Scheduled departure timestamp constructed from `FlightDate` and `CRSDepTime`. | Main flight timestamp used for sorting and time-based joins. |

### Joined ASPM planned-demand columns

ASPM supplies planned schedule counts for the previous, current, and next clock hours around scheduled departure. These
are schedule values known ahead of time, not future operating results. The offset is calculated as the ASPM timestamp
minus the flight's scheduled departure timestamp.

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

NOAA supplies the latest weather observation at or before scheduled departure. The observation timestamp and age remain
in the working dataframe so future or stale matches can be detected.

| Column | Description | Use and availability |
|---|---|---|
| `NOAA_DATE` | Timestamp of the matched NOAA observation. | Must be at or before the relevant prediction time. |
| `HourlyDewPointTemperature` | Dew-point temperature reported by the station. | Describes moisture in the air. |
| `HourlyDryBulbTemperature` | Air temperature reported by the station. | Main temperature measurement. |
| `HourlyPrecipitation` | Precipitation amount for the observation period. | Continuous precipitation measurement; trace amounts use a small positive value. |
| `HourlyRelativeHumidity` | Relative humidity percentage. | Describes atmospheric moisture. |
| `HourlyVisibility` | Horizontal visibility reported by the station. | Measures visibility conditions that may affect operations. |
| `HourlyWindSpeed` | Reported wind speed. | Measures wind strength. |
| `Rain` | Indicates that rain was reported. | Binary weather-condition feature. |
| `Drizzle` | Indicates that drizzle was reported. | Binary weather-condition feature. |
| `Snow` | Indicates that snow was reported. | Binary weather-condition feature. |
| `Fog` | Indicates that fog was reported. | Binary low-visibility feature. |
| `Mist` | Indicates that mist was reported. | Binary visibility-condition feature. |
| `Thunderstorm` | Indicates that a thunderstorm was reported. | Binary severe-weather feature. |
| `FreezingPrecip` | Indicates that the report contains a freezing-condition code. | Binary freezing-weather feature. |
| `Showers` | Indicates that showers were reported. | Binary weather-condition feature. |
| `PrecipOccurred` | Indicates precipitation based on a measured amount or a rain, snow, or drizzle report. | Combined precipitation feature. |
| `WindX` | East-west wind component calculated from speed and direction. | Model-friendly representation of wind direction and strength. |
| `WindY` | North-south wind component calculated from speed and direction. | Used with `WindX` to represent the wind vector. |
| `NOAA_AGE_MINUTES` | Scheduled departure time minus the NOAA observation time, in minutes. | Must be nonnegative and within the allowed weather-match tolerance. |

## Feature Engineering

The initial feature set is intentionally compact and interpretable. It combines schedule, calendar, route, planned airport
demand, and weather information while preserving one row per flight. This design follows the three primary references:
Snell combines BTS flight records with hourly NOAA weather and emphasizes schedule, airline, route, traffic, and weather
variables; Zoutendijk and Mitici use airline, airport, distance, scheduled traffic, weather, and cyclical time encodings;
and Pineda-Jaramillo et al. combine flight, airport, geographic, and weather data and analyze the features that influence
the resulting predictions.

The same pre-pushback feature block is intended for Models 1A and 2A and is carried forward into Models 2B and 2C. Model
2B adds information available once pushback has occurred, and Model 2C adds taxi-out and takeoff information. In the
table below, **Pre** means all four models, while **2B and 2C** marks information first available at pushback. Raw fields
such as `Reporting_Airline`, `Origin`, `Dest`, `CRSElapsedTime`, `Distance`, the selected NOAA measurements, `WindX`,
`WindY`, and `NOAA_AGE_MINUTES` remain candidate inputs even though they are not repeated as engineered features.

### Initial core engineered features

| Feature | Construction and description | Availability | Justification and evidence |
|---|---|---|---|
| `SCHED_DEP_MINUTE_OF_DAY` | Convert `CRSDepTime` from HHMM to minutes after local midnight. | Pre | Provides a valid numeric representation of scheduled departure time. Scheduled time is used by all three primary references, and Pineda identifies time-of-day effects as important. |
| `SCHED_DEP_HOUR` | Integer hour from `SCHED_DEP_MINUTE_OF_DAY`. | Pre | Gives an interpretable grouping for EDA and simple models and supports comparison of peak-hour delay rates. Snell discusses scheduled time blocks and peak-hour patterns. |
| `SCHED_DEP_TIME_SIN` | `sin(2π * SCHED_DEP_MINUTE_OF_DAY / 1440)`. | Pre | Represents the daily cycle without placing 23:59 far from 00:00. Zoutendijk and Mitici explicitly encode time features with sine and cosine. |
| `SCHED_DEP_TIME_COS` | `cos(2π * SCHED_DEP_MINUTE_OF_DAY / 1440)`. | Pre | Completes the cyclical representation of scheduled departure time. |
| `SCHED_ARR_MINUTE_OF_DAY` | Convert `CRSArrTime` from destination-local HHMM to minutes after local midnight. | Pre | Captures the scheduled arrival period without implying that origin and destination clocks share a time zone. It must not be subtracted from scheduled departure time; `CRSElapsedTime` is the valid duration field. |
| `SCHED_ARR_TIME_SIN` | `sin(2π * SCHED_ARR_MINUTE_OF_DAY / 1440)`. | Pre | Preserves the daily periodicity of the destination-local scheduled arrival time. |
| `SCHED_ARR_TIME_COS` | `cos(2π * SCHED_ARR_MINUTE_OF_DAY / 1440)`. | Pre | Completes the cyclical representation of scheduled arrival time. |
| `TIME_OF_DAY` | Interpretable category derived from scheduled departure time, with fixed morning, afternoon, evening, and overnight bands documented before modeling. | Pre | Snell discusses time-of-day slots, Pineda uses a departure-period category, and Zoutendijk and Mitici select time of day. This field is especially useful for EDA; models may use it instead of, or compare it with, the cyclical pair to limit redundancy. |
| `IS_WEEKEND` | 1 when `DayOfWeek` is 6 or 7; otherwise 0. | Pre | Provides a simple weekly schedule distinction. Snell discusses weekend flags, while all three references include or discuss weekday effects. |
| `DAY_OF_WEEK_SIN` | `sin(2π * (DayOfWeek - 1) / 7)`. | Pre | Preserves adjacency between Sunday and Monday. Zoutendijk and Mitici explicitly apply trigonometric encoding to day of week. |
| `DAY_OF_WEEK_COS` | `cos(2π * (DayOfWeek - 1) / 7)`. | Pre | Completes the weekly cyclical representation. |
| `DAY_OF_YEAR` | Ordinal day from `FlightDate`, from 1 through 365 or 366. | Pre | Represents position within the year and is among the schedule features used by Zoutendijk and Mitici. |
| `DAY_OF_YEAR_SIN` | `sin(2π * (DAY_OF_YEAR - 1) / days_in_year)`. | Pre | Represents annual seasonality continuously and handles leap years through `days_in_year`. |
| `DAY_OF_YEAR_COS` | `cos(2π * (DAY_OF_YEAR - 1) / days_in_year)`. | Pre | Completes the annual cyclical representation. |
| `MONTH_SIN` | `sin(2π * (Month - 1) / 12)`. | Pre | Represents month as a cycle. Zoutendijk and Mitici explicitly use month sine and cosine, and Pineda reports month effects. |
| `MONTH_COS` | `cos(2π * (Month - 1) / 12)`. | Pre | Completes the monthly cyclical representation. |
| `YEAR_PERIOD` | Treat 2019, 2023, and 2024 as categories rather than as a continuous numeric trend. | Pre | Separates the pre-pandemic baseline from the two post-pandemic periods without assuming a linear yearly effect. Zoutendijk and Mitici use year, but this project's discontinuous coverage makes a period indicator more defensible. |
| `ROUTE` | Concatenate `Origin` and `Dest` as an origin-destination category. | Pre | Preserves the flight-leg identity highlighted by Snell and the airport/destination effects emphasized by Zoutendijk and Mitici and Pineda. |
| `AIRLINE_FLIGHT_ID` | Concatenate `Reporting_Airline` and `Flight_Number_Reporting_Airline`; treat the result as categorical. | Pre | Avoids treating a flight number as a continuous quantity and distinguishes identical numbers used by different airlines. Snell and Pineda both retain scheduled flight and airline identity. |
| `AIRLINE_DEST` | Concatenate `Reporting_Airline` and `Dest` as a categorical interaction. | Pre | Provides one limited, interpretable service-pattern interaction instead of a large arbitrary interaction set. Airline and destination are supported individually across the primary references. |
| `LOG_DISTANCE` | `log1p(Distance)`. | Pre | Retains route-length ordering while reducing right skew. Distance is selected or discussed by all three primary references. The raw distance should remain available for tree models and interpretation. |
| `SCHEDULED_SPEED_PROXY` | `60 * Distance / CRSElapsedTime`, when elapsed time is positive. | Pre | Summarizes the relationship between route length and scheduled gate-to-gate duration. It is schedule-derived and time-safe, but should be checked for redundancy with its two source fields before final selection. |
| `ASPM_PREVIOUS_TOTAL_SCHEDULED_TRAFFIC` | Previous-hour scheduled departures plus previous-hour scheduled arrivals. | Pre | Summarizes planned airport workload immediately before the flight. Snell supports airport congestion/traffic measures, and Zoutendijk and Mitici use scheduled-flight counts near the flight time. |
| `ASPM_CURRENT_TOTAL_SCHEDULED_TRAFFIC` | Current-hour scheduled departures plus current-hour scheduled arrivals. | Pre | Measures planned workload during the scheduled departure hour. |
| `ASPM_NEXT_TOTAL_SCHEDULED_TRAFFIC` | Next-hour scheduled departures plus next-hour scheduled arrivals. | Pre | Measures planned workload just after the scheduled departure hour. These are schedule counts known ahead of time, not future realized outcomes. |
| `ASPM_THREE_HOUR_SCHEDULED_DEPARTURES` | Sum scheduled departures across the previous, current, and next hours. | Pre | Approximates the local two-hour-neighborhood scheduled-flight feature used by Zoutendijk and Mitici while matching the available ASPM clock-hour records. |
| `ASPM_THREE_HOUR_SCHEDULED_ARRIVALS` | Sum scheduled arrivals across the previous, current, and next hours. | Pre | Separates planned arrival demand from departure demand because each can load airport resources differently. |
| `ASPM_THREE_HOUR_TOTAL_SCHEDULED_TRAFFIC` | Sum `ASPM_THREE_HOUR_SCHEDULED_DEPARTURES` and `ASPM_THREE_HOUR_SCHEDULED_ARRIVALS`. | Pre | Provides the main compact congestion feature supported by Snell's traffic-volume discussion and Zoutendijk and Mitici's scheduled-flight window. |
| `ASPM_CURRENT_MINUS_PREVIOUS_TRAFFIC` | Current-hour total scheduled traffic minus previous-hour total. | Pre | Indicates whether planned airport workload is building or easing near departure without using realized performance. |
| `ASPM_NEXT_MINUS_CURRENT_TRAFFIC` | Next-hour total scheduled traffic minus current-hour total. | Pre | Adds the forward planned-demand slope using only schedule information already known at prediction time. |
| `ASPM_MAX_HOURLY_TRAFFIC` | Maximum of the previous-, current-, and next-hour total scheduled traffic. | Pre | Captures the local planned peak without imposing a learned high-traffic threshold. |
| `TEMP_DEWPOINT_SPREAD` | `HourlyDryBulbTemperature - HourlyDewPointTemperature`. | Pre | Provides a compact moisture-related measure while retaining the underlying observations. Zoutendijk and Mitici select temperature/dew point features, and Snell and Pineda support weather integration. |
| `LOG_PRECIPITATION` | `log1p(max(HourlyPrecipitation, 0))`. | Pre | Reduces precipitation skew while retaining trace and heavy precipitation distinctions. Snell and Pineda include precipitation-related weather information. |
| `WEATHER_CONDITION_COUNT` | Sum `Rain`, `Drizzle`, `Snow`, `Fog`, `Mist`, `Thunderstorm`, `FreezingPrecip`, and `Showers`. | Pre | Gives an interpretable measure of how many adverse condition types are reported without inventing severity weights. |
| `ADVERSE_WEATHER` | 1 when any of the eight weather-condition indicators is 1; otherwise 0. | Pre | Supplies a compact general-weather flag for linear baselines while the component indicators remain available. The primary references consistently support weather as a predictor. |
| `ACTUAL_DEP_MINUTE_OF_DAY` | Convert `DepTime` from HHMM to minutes after local midnight. | 2B and 2C | Represents the known gate-out time once pushback occurs. Snell directly compares arrival-delay models without and with actual departure information. |
| `ACTUAL_DEP_TIME_SIN` | `sin(2π * ACTUAL_DEP_MINUTE_OF_DAY / 1440)`. | 2B and 2C | Encodes actual pushback time without a midnight discontinuity. |
| `ACTUAL_DEP_TIME_COS` | `cos(2π * ACTUAL_DEP_MINUTE_OF_DAY / 1440)`. | 2B and 2C | Completes the cyclical representation of actual pushback time. |
| `DEPARTED_EARLY` | 1 when signed `DepDelay` is less than 0; otherwise 0. | 2B and 2C | Preserves the distinction between early and non-early departures if a nonnegative delay transform is tested. |
| `LOG_DEP_DELAY_MINUTES` | `log1p(DepDelayMinutes)`. | 2B and 2C | Reduces the influence of very long departure delays while retaining their ordering. It should be compared with signed `DepDelay` rather than automatically included with every redundant departure-delay field. |

### Feature selection and leakage rules

The initial baseline should begin with the core table and the retained raw predictors, then remove redundant
representations based only on the training data and model family. For example, a linear model may benefit from cyclical
time features and compact weather summaries, while a tree model may not need both every raw input and every derived
summary. Any one-hot, frequency, or target encoding, imputation, scaling, threshold selection, feature selection, or
class-balancing step must be fitted on training data only.

Threshold features such as `LOW_VISIBILITY`, `HIGH_WIND`, `ASPM_HIGH_TRAFFIC`, and source-staleness flags are deferred
until their cutoffs are justified by an operational definition or selected from training data. Broad interaction sets
and historical delay-rate features are also deferred. If historical rates are added later, the current row must be
shifted out, only earlier completed flights may contribute, small groups must be smoothed, and development/test outcomes
must never be used.

For Model 2B, the first comparison should use a minimal actual-departure update, such as signed `DepDelay` alone, followed
by a deliberately chosen fuller representation. `DepTime`, `DepDelay`, `DepDelayMinutes`, `DepDel15`, and
`DepartureDelayGroups` encode overlapping information and should not all be included automatically. `TaxiOut` and
`WheelsOff` remain excluded from Models 1A, 2A, and 2B.

The current Model 1A assembly logic contains flights **departing** JFK. Before building Models 2A, 2B, and 2C as defined
in this README, their notebooks must create the inbound population with `Dest` equal to JFK and attach
schedule, ASPM, and time-safe NOAA information for the flight origin at the relevant prediction time. The
outbound rows or destination weather observed at landing must not be reused as a substitute. Pineda uses destination
weather observed around landing, but that information is unavailable before pushback; it is eligible here only if a
forecast or observation was demonstrably available by the model's prediction cutoff.

Source timestamps (`FlightDate`, `DATE`, ASPM lookup/report dates, and `NOAA_DATE`) remain audit columns rather than
predictors. The nine documented missing next-hour ASPM matches at annual file boundaries must remain missing or be
recovered from the following year's planned schedule file; they must not be filled with zero or copied from the current
hour. `Tail_Number` remains audit-only, and no aircraft rotation, previous-flight chain, turnaround, tail-sequence, or
network-propagation feature is part of this feature set.
