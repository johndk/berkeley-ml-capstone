# Capstone Project

## Overview

This capstone investigates whether machine learning can predict significant delays for individual flights departing from three major United States airports:

- John F. Kennedy International Airport (JFK)
- Chicago O'Hare International Airport (ORD)
- Hartsfield-Jackson Atlanta International Airport (ATL)

A significant delay is defined as a departure or arrival delay of 15 minutes or more. The project will also explore which flight, airport, and weather conditions are most closely associated with delays.

The project has three related prediction goals:

1. Before pushback, predict whether a flight will depart at least 15 minutes late.
2. Before pushback, predict whether a flight will arrive at least 15 minutes late.
3. Immediately after pushback, predict whether a flight will arrive at least 15 minutes late using its actual departure delay.

The first two models will use only information available before pushback. The third will add the actual departure time and departure delay. Comparing the second and third models will show how much arrival predictions improve once the flight has pushed back. Information recorded after takeoff or arrival will not be used to make predictions.

The project combines three main data sources:

- Bureau of Transportation Statistics (BTS) flight schedules and performance data
- Aviation System Performance Metrics (ASPM) airport traffic and congestion data
- National Oceanic and Atmospheric Administration (NOAA) weather observations

Each flight will be matched with the most recent airport and weather information available before the prediction is made. This prevents the models from using information from the future.

The work will include:

- Checking data quality, missing values, unusual values, and the balance between delayed and on-time flights
- Exploring which flight, airport, and weather conditions are most closely associated with delays
- Creating useful features from dates, times, flights, airport, and weather conditions
- Comparing logistic regression, random forest, and gradient-boosting models such as CatBoost
- Measuring model performance on later flights that were not used for training
- Explaining which factors have the strongest effect on each model's predictions

The project focuses on individual flights. Aircraft rotations, previous-flight chains, and delay spread through an airline network are outside its scope. The approach is informed by the flight-level delay research of Snell, Zoutendijk, and Pineda. The final analysis will compare both model performance and the factors associated with delays across JFK, ORD, and ATL.

## Business Understanding

Flight delays create costs and disruption for passengers, airlines, and airports. Earlier warning of a likely delay can help airlines communicate with passengers, adjust staffing and gate plans, and prepare for possible missed connections. Airports can also use this information to better understand when congestion or weather is likely to affect operations.

This project asks three practical questions:

- Can a departure delay of 15 minutes or more be identified before pushback?
- Can an arrival delay of 15 minutes or more be identified before pushback?
- How much does the arrival prediction improve once the actual departure delay is known?

The analysis will also compare JFK, ORD, and ATL because the conditions associated with delay may differ by airport. As emphasized by Snell, Zoutendijk, and Pineda, a useful result should do more than produce a yes-or-no answer. It should provide a reliable estimate of delay risk and clearly show which schedule, airport, and weather conditions influenced the prediction.

The models are intended as decision-support tools, not as proof that a particular factor caused a delay. Success will be judged by how well the models identify delayed flights, how often their warnings are correct, and whether their results can be explained in a useful way.

## Data Understanding

The project brings together flight, airport, and weather data. The record of analysis is one flight departing from JFK, ORD, or ATL. Airport and weather records are added to each flight without changing this one-row-per-flight structure.

### Data Sources

| Source | Information used in this project | Level of detail |
|---|---|---|
| Bureau of Transportation Statistics (BTS) | Flight dates and schedules, airline, origin, destination, distance, and departure and arrival outcomes | One row per flight |
| Aviation System Performance Metrics (ASPM) | Scheduled arrivals and departures and recent measures of airport congestion and delay | One row per airport and hour |
| National Oceanic and Atmospheric Administration (NOAA) | Temperature, humidity, visibility, precipitation, wind, and reported weather conditions | One row per weather observation |

BTS provides the individual flight records and the two outcomes the models will predict. `DepDel15` identifies flights that departed at least 15 minutes late, while `ArrDel15` identifies flights that arrived at least 15 minutes late. ASPM describes recent conditions at the departure airport, and NOAA describes the weather observed there before departure.

### Data Coverage

The project is designed to compare three airports—JFK, ORD, and ATL—using data from 2019, 2023, and 2024. These years were selected to represent periods before and after the COVID-19 pandemic when the airports were operating at or near normal capacity. The 2019 data provides a pre-pandemic baseline, while 2023 and 2024 show flight operations after the major pandemic-related disruptions had passed.

Before modeling begins, a decision will be made about how to divide the years and flights into training, development, and final test sets. The split will preserve time order so that the models are trained on earlier flights and evaluated on later flights they have not seen. Files use a consistent `AIRPORT_YEAR.csv` naming pattern so that the same processing steps can be applied to each airport and year.

### Data Directory

```text
data/
├── bts/
│   ├── L_UNIQUE_CARRIERS.csv
│   ├── raw/
│   │   ├── filter.py
│   │   ├── 2019/
│   │   │   ├── On_Time_Reporting_..._2019_1/
│   │   │   │   ├── On_Time_Reporting_..._2019_1.csv
│   │   │   │   └── readme.html
│   │   │   ├── On_Time_Reporting_..._2019_2/
│   │   │   │   ├── On_Time_Reporting_..._2019_2.csv
│   │   │   │   └── readme.html
│   │   │   ├── ... same monthly structure through 2019_12
│   │   │   ├── ATL.csv
│   │   │   ├── JFK.csv
│   │   │   └── ORD.csv
│   │   ├── 2023/
│   │   │   └── ... same monthly and airport structure
│   │   └── 2024/
│   │       └── ... same monthly and airport structure
│   ├── processed/
│   │   ├── ATL_2019.csv
│   │   ├── JFK_2019.csv
│   │   ├── ORD_2019.csv
│   │   └── ... through ATL, JFK, and ORD for 2024
│   └── cleaned/
│       ├── ATL_2019.csv
│       ├── JFK_2019.csv
│       ├── ORD_2019.csv
│       └── ... through ATL, JFK, and ORD for 2024
├── aspm/
│   ├── raw/
│   │   ├── download_aspm_hourly_v3.py
│   │   ├── download_aspm_2019.sh
│   │   ├── download_aspm_2023.sh
│   │   ├── download_aspm_2024.sh
│   │   └── aspm_output/
│   │       ├── run_2019_ATL/
│   │       │   ├── aspm_2019_ATL.csv
│   │       │   └── raw_html/
│   │       │       ├── ATL_2019-01-01.html
│   │       │       ├── ATL_2019-01-02.html
│   │       │       └── ... one HTML response per day through 2019-12-31
│   │       ├── run_2019_JFK/
│   │       ├── run_2019_ORD/
│   │       └── ... through ATL, JFK, and ORD for 2024
│   ├── processed/
│   │   ├── ATL_2019.csv
│   │   ├── JFK_2019.csv
│   │   ├── ORD_2019.csv
│   │   └── ... through ATL, JFK, and ORD for 2024
│   └── cleaned/
│       ├── ATL_2019.csv
│       ├── JFK_2019.csv
│       ├── ORD_2019.csv
│       └── ... through ATL, JFK, and ORD for 2024
├── noaa/
│   ├── raw/
│   │   ├── isd-history.csv
│   │   ├── 2019/
│   │   │   ├── 72219013874.csv
│   │   │   ├── 72530094846.csv
│   │   │   └── 74486094789.csv
│   │   ├── 2023/
│   │   │   └── ... same three weather stations
│   │   └── 2024/
│   │       └── ... same three weather stations
│   ├── processed/
│   │   ├── ATL_2019.csv
│   │   ├── JFK_2019.csv
│   │   ├── ORD_2019.csv
│   │   └── ... through ATL, JFK, and ORD for 2024
│   └── cleaned/
│       ├── ATL_2019.csv
│       ├── JFK_2019.csv
│       ├── ORD_2019.csv
│       └── ... through ATL, JFK, and ORD for 2024
├── merged/
│   ├── ATL_2019.csv
│   ├── JFK_2019.csv
│   ├── ORD_2019.csv
│   └── ... through ATL, JFK, and ORD for 2024
├── features/
│   └── all_airports_features.csv
└── models/
    ├── model_1_depdel15_pre_pushback.csv
    ├── model_2a_arrdel15_pre_pushback.csv
    └── model_2b_arrdel15_post_pushback.csv
```

The folders represent the main stages of the data:

#### Raw

The `raw` folders contain data as downloaded or first collected, along with the scripts used to retrieve or separate the source files.

BTS On-Time Performance data is provided as a separate compressed download for each month. Each monthly file contains flights reported by all included United States carriers, rather than data for a single airport. After the monthly files are downloaded and extracted, `filter.py` reads all 12 files for a year, keeps flights where JFK, ORD, or ATL is either the origin or destination, and combines the matching records into one annual file for each airport. For example, the 12 monthly files under `data/bts/raw/2019/` are used to produce `ATL.csv`, `JFK.csv`, and `ORD.csv` in that same directory. These annual airport files become the inputs to the BTS processing notebook.

#### Processed

The `processed` folders contain the useful source columns in a more consistent format.

#### Cleaned

The `cleaned` folders contain data that has been checked for missing values, duplicate records, data types, and time ordering.

#### Merged

The `merged` folder contains one airport-year flight file with the appropriate ASPM and NOAA observations attached.

#### Features

The `features` folder will contain the combined, feature-engineered dataset used as the common source for all three models.

#### Models

The `models` folder will contain three model-ready CSV files projected from the feature dataset. Each file will include only the predictors and outcome allowed at that prediction time.

The feature and model CSV files shown above are planned names and do not exist yet. The processing and cleaning work is recorded in separate notebooks for BTS, ASPM, and NOAA. A separate merge notebook combines the three cleaned sources.

### Data Flow

```text
Raw BTS  ──→ Process BTS  ──→ Clean BTS  ─────┐
                                              │
Raw ASPM ──→ Process ASPM ──→ Clean ASPM ─────┼──→ Merge
                                              │
Raw NOAA ──→ Process NOAA ──→ Clean NOAA ─────┘
                                                      │
                                                      ▼
                                             Merged flight data
                                                      │
                                                      ▼
                                             Feature engineering
                                                      │
                                                      ▼
                                     all_airports_features.csv
                                                      │
                            ┌─────────────────────────┼─────────────────────────┐
                            ▼                         ▼                         ▼
              model_1_depdel15_         model_2a_arrdel15_        model_2b_arrdel15_
              pre_pushback.csv          pre_pushback.csv          post_pushback.csv
              Departure delay           Arrival delay             Arrival delay
              Before pushback           Before pushback           After pushback
```

Processing first makes each source easier to use. Cleaning then checks the quality and consistency of the data. The cleaned sources are merged into a single flight-level dataset. Feature engineering will create additional values from the existing dates, times, routes, congestion measures, and weather conditions. The resulting data will then be separated into the three model datasets according to what information is allowed at each prediction time.

### Matching Records by Time

The scheduled departure date and time, stored in `DATE`, provides the main time for each BTS flight. Each flight is matched with an earlier ASPM hourly record and the most recent NOAA observation available before its scheduled departure. The merged data keeps the source timestamps and calculates the age of each matched observation in minutes.

This time-based matching is important because a model should not use airport conditions or weather observations that occurred after the prediction was made. It also allows the age of the matched information to be checked before modeling.

### Model Outcomes and Available Information

| Model | Outcome | Information available when the prediction is made |
|---|---|---|
| Model 1 | `DepDel15` | Flight schedule, earlier airport conditions, and earlier weather observations |
| Model 2A | `ArrDel15` | The same information available to Model 1 |
| Model 2B | `ArrDel15` | Model 2A information plus the actual departure time and departure delay |

Before the model datasets are created, the merged data will be explored and expanded with features that summarize time of day, season, route, distance, congestion, and weather. Information recorded after takeoff or arrival will not be included as a model input.

## Data Preparation

## Modeling

## Evaluation

## Deployment

## References

1. Kenney Snell, Jozef Zurada, Jan Kozak, and Zahra Hatami, *Predicting Flight Delays Using Machine Learning*. **Primary reference.**
2. Micha Zoutendijk and Mihaela Mitici, *Probabilistic Flight Delay Predictions Using Machine Learning and Applications to the Flight-to-Gate Assignment Problem*. **Primary reference.**
3. Juan Pineda-Jaramillo, Claudia Munoz, Rodrigo Mesa-Arango, Carlos Gonzalez-Calderon, and Anne Lange, *Integrating Multiple Data Sources for Improved Flight Delay Prediction Using Explainable Machine Learning*. **Primary reference.**
4. Meng Li, *Air Traffic Delay Prediction Based on Machine Learning and Delay Propagation*.
5. Jun Chen and Meng Li, *Chained Predictions of Flight Delay Using Machine Learning*.
6. Maarten Beltman, Marta Ribeiro, Jasper de Wilde, and Junzi Sun, *Dynamically Forecasting Airline Departure Delay Probability Distributions for Individual Flights Using Supervised Learning*.
