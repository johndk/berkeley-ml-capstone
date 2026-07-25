**ATTENTION: This submission is in progress. Expected completion Saturday, July 25, 2026.** 

# Capstone Project - Berkeley ML and AI 
## Initial Report and Exploratory Data Analysis

## Overview

This capstone investigates whether machine learning can predict significant delays for individual flights departing from 
or arriving at three major United States airports:

- John F. Kennedy International Airport (JFK)
- Chicago O'Hare International Airport (ORD)
- Hartsfield-Jackson Atlanta International Airport (ATL)

A significant delay is defined as a departure or arrival delay of 15 minutes or more. The project also explores 
which flight, airport, and weather conditions are most closely associated with delays.

The project has three base prediction goals and one stretch goal:

1. **Model 1:** For a flight departing from JFK, ORD, or ATL, predict before pushback whether the flight departs at least 15 minutes late.
2. **Model 2A:** For a flight arriving at JFK, ORD, or ATL, predict before pushback at the flight origin whether it arrives at least 15 minutes late.
3. **Model 2B:** For a flight arriving at JFK, ORD, or ATL, predict immediately after pushback at the flight origin whether it arrives at least 15 minutes late, using the actual departure delay.
4. **Stretch Model 2C:** For a flight arriving at JFK, ORD, or ATL, predict immediately after takeoff at the flight origin whether it arrives at least 15 minutes late, using the actual departure, taxi-out, and takeoff information.

The first two models use only information available before pushback. The third adds the actual departure time
and departure delay. The stretch model adds information that becomes known at takeoff, including taxi-out time.
These stages show how arrival predictions improve as new operating information becomes available.

Each model follows a clear prediction time. A field is included only if it would be known at that time; information
recorded later is excluded even when it appears in the historical dataset. This prevents the model from learning from
the future, often called data leakage. BTS provides the historical event values used for this analysis, but a working
operational model would need equivalent gate-out and takeoff information from a suitable live source. Some 
BTS columns represent events—such as gate departure and takeoff—that an airline or  airport operational system can observe 
when they occur

The project combines three main data sources:

- Bureau of Transportation Statistics (BTS) [flight schedules and performance data](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr)
- Aviation System Performance Metrics (ASPM) [airport traffic and congestion data](https://www.aspm.faa.gov/apm/sys/main.asp)
- National Oceanic and Atmospheric Administration (NOAA) [weather observations](https://www.ncei.noaa.gov/data/local-climatological-data/access/)

Each flight is matched with ASPM scheduled airport demand and the most recent NOAA weather information available before the prediction is 
made. This prevents the models from using information from the future.

The work includes:

- Checking data quality, missing values, unusual values, and the balance between delayed and on-time flights
- Exploring which flight, airport, and weather conditions are most closely associated with delays
- Creating useful features from dates, times, flights, airport, and weather conditions
- Comparing logistic regression, random forest, and gradient-boosting models such as CatBoost
- Measuring model performance on later flights that were not used for training
- Explaining which factors have the strongest effect on each model's predictions

The project focuses on individual flights. Aircraft rotations, previous-flight chains, and delay spread through an airline 
network are out of scope. The approach is informed by the flight-level delay research of Snell, Zoutendijk, and Pineda. 
The final analysis compares both model performance and the factors associated with delays across JFK, ORD, and ATL.

## Business Understanding

Flight delays create costs and disruption for passengers, airlines, and airports. Earlier warning of a likely delay 
can help airlines communicate with passengers, adjust staffing and gate plans, and prepare for possible missed connections. 
Airports can also use this information to better understand when congestion or weather is likely to affect operations.

This project asks three core questions and one stretch question:

- Can a departure delay of 15 minutes or more be identified before pushback?
- Can an arrival delay of 15 minutes or more be identified before pushback?
- How much does the arrival prediction improve once the actual departure delay is known?
- As a stretch goal, how much more does the arrival prediction improve once taxi-out and takeoff information is known?

The analysis also compares JFK, ORD, and ATL because the conditions associated with delay may differ by airport. 
As emphasized by Snell, Zoutendijk, and Pineda, a useful result does more than produce a yes-or-no answer. It 
provides a reliable estimate of delay risk and clearly shows which schedule, airport, and weather conditions influence the 
prediction.

The models are intended as decision-support tools, not as proof that a particular factor caused a delay. Success is 
judged by how well the models identify delayed flights, how often their warnings are correct, and whether their results 
can be explained in a useful way.

## Data Understanding

The project brings together flight, airport, and weather data. Each record represents one flight. The airport of interest
is the origin for Model 1 and the destination for Models 2A, 2B, and the possible Model 2C. Airport and weather records
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

The project is designed to compare three airports—JFK, ORD, and ATL—using data from 2019, 2023, and 2024. These years 
were selected to represent periods before and after the COVID-19 pandemic when the airports were operating at or near 
normal capacity. The 2019 data provides a pre-pandemic baseline, while 2023 and 2024 show flight operations after the 
major pandemic-related disruptions had passed.

The exact division of the years and flights into training, development, and final test sets remains to be decided. 
The split preserves time order so that the models are trained on earlier flights and evaluated 
on later flights they have not seen. Files use a consistent `AIRPORT_YEAR.csv` naming pattern so that the same processing 
steps can be applied to each airport and year.

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
│   ├── ATL_2019.csv
│   ├── JFK_2019.csv
│   ├── ORD_2019.csv
│   └── ... through ATL, JFK, and ORD for 2024
└── models/
    ├── ATL_2019_m1.csv
    ├── ATL_2019_m2a.csv
    ├── ATL_2019_m2b.csv
    ├── JFK_2019_m1.csv
    ├── JFK_2019_m2a.csv
    ├── JFK_2019_m2b.csv
    └── ... same three model files for each airport and year
```

The folders represent the main stages of the data:

#### Raw

The `raw` folders contain data as downloaded or first collected, along with the scripts used to retrieve or separate the source files.

BTS On-Time Performance data is provided as a separate compressed download for each month. Each monthly file contains 
domestic flights reported by all included United States carriers, rather than data for a single airport. After the monthly files 
are downloaded and extracted, `filter.py` reads all 12 files for a year, keeps flights where JFK, ORD, or ATL is either
the origin or destination, and combines the matching records into one annual file for each airport. For example, 
the 12 monthly files under `data/bts/raw/2019/` are used to produce `ATL.csv`, `JFK.csv`, and `ORD.csv` in that same 
directory. These annual airport files become the inputs to the BTS processing notebook.

ASPM data is collected with `download_aspm_hourly_v3.py`, which requests one FAA ASPM hourly report for each airport and
calendar day. The year-specific shell scripts supply the date range and airport code. Each run saves the original daily
HTML responses under `raw_html/` and combines the extracted hourly records into one annual airport CSV, such as
`aspm_2019_ATL.csv`. Keeping the HTML responses preserves the original reports while the annual CSV becomes the input to
the ASPM processing notebook.

NOAA Local Climatological Data is downloaded as one annual CSV for the weather station selected to represent each airport.
Station `72219013874` represents ATL, `72530094846` represents ORD, and `74486094789` represents JFK. The files are grouped
by year under `data/noaa/raw/`; for example, the three station files in `2019/` provide the raw weather observations used
to produce the ATL, ORD, and JFK inputs for the NOAA processing notebook.

#### Processed

The `processed` folders contain smaller working files with the records and source columns needed by the project. Processing
removes empty, redundant, or out-of-scope fields before the more detailed cleaning checks begin.

The work differs slightly by source. BTS processing removes cancelled and diverted flights and keeps the schedule,
route, and outcome fields needed for analysis. ASPM processing keeps the airport and time identifiers together with
scheduled hourly arrivals and departures. NOAA processing keeps the selected hourly weather measurements and removes
rows that contain no usable weather observations.

#### Cleaned

The `cleaned` folders contain the standardized source files used by the merge step. Cleaning converts or constructs
timestamps, sorts the records, and checks the expected columns, missing values, duplicate keys, numeric ranges, category
codes, time coverage, and outcome consistency. Validation checks report possible problems without changing otherwise
valid records.

NOAA requires some additional preparation. Trace precipitation is converted to a small numeric value, weather codes are
turned into plain condition indicators, and wind direction and speed are used to create `WindX` and `WindY`. Missing
continuous measurements are also filled during the current cleaning process. Before modeling, this fill method must be
limited to weather already observed at the relevant prediction time so a later observation cannot influence an earlier
flight.

The cleaned BTS, ASPM, and NOAA airport-year files remain separate until the merge step. The merge keeps one row per
flight and adds the appropriate scheduled airport demand and time-matched weather observation.

Cleaning a field does not automatically make it a valid predictor. Some retained BTS fields are outcomes, validation
fields, or operating events available only to a later model. Fields published too late to support the prediction, such
as ASPM's realized performance measures, are removed. The model-ready files later enforce the exact information
available at each prediction time and prevent time leakage.

Appendix A describes the retained and removed columns, source-specific transformations, and prediction-time restrictions
in more detail.

#### Merged

The `merged` folder contains one airport-year flight file with the appropriate ASPM and NOAA observations attached.

#### Features

The `features` folder contains one feature-engineered file for each airport and year. These files remain partitioned by airport and year unless a modeling requirement provides a clear reason to combine them.

#### Models

The `models` folder contains three model-ready projections for each airport-year feature file. The `_m1`, `_m2a`, and `_m2b` suffixes identify the three prediction scenarios. Each file includes only the predictors and outcome allowed at that prediction time.

The processing and cleaning work is recorded in separate notebooks for BTS, ASPM, and NOAA. A separate merge notebook combines the three cleaned sources.

### Data Flow

```text
Raw BTS  ──→ Process BTS  ──→ Clean BTS  ─────┐
                                              │
Raw ASPM ──→ Process ASPM ──→ Clean ASPM ─────┼──→ Merge
                                              │       │
Raw NOAA ──→ Process NOAA ──→ Clean NOAA ─────┘       │ 
                                                      │
                                                      ▼
                                             Merged flight data
                                           AIRPORT_YEAR.csv
                                                      │
                                                      ▼
                                             Feature engineering
                                                      │
                                                      ▼
                                         features/AIRPORT_YEAR.csv
                                                      │
                            ┌─────────────────────────┼─────────────────────────┐
                            ▼                         ▼                         ▼
           models/AIRPORT_YEAR_m1.csv  models/AIRPORT_YEAR_m2a.csv  models/AIRPORT_YEAR_m2b.csv
           Departure delay             Arrival delay                Arrival delay
           Before pushback             Before pushback              After pushback
```

Processing first makes each source easier to use. Cleaning then checks the quality and consistency of the data. The cleaned sources are merged into a single flight-level dataset. Feature engineering creates additional values from the existing dates, times, routes, congestion measures, and weather conditions. The resulting data is then separated into the three base model datasets according to what information is allowed at each prediction time. A fourth projection is added if stretch Model 2C is implemented.

### Matching Records by Time

The scheduled departure date and time, stored in `DATE`, provides the main time for each BTS flight. Each flight is matched with an ASPM hourly schedule record and the most recent NOAA observation available before its scheduled departure. The merged data keeps the source timestamps and calculates the age of each matched observation in minutes.

This time-based matching prevents a model from using airport conditions or weather observations that occurred after the prediction was made. It also allows the age of the matched information to be checked before modeling.

### Model Outcomes and Available Information

| Model | Outcome | Information available when the prediction is made |
|---|---|---|
| Model 1 | `DepDel15` | Flight schedule, earlier airport conditions, and earlier weather observations |
| Model 2A | `ArrDel15` | The same information available to Model 1 |
| Model 2B | `ArrDel15` | Model 2A information plus the actual departure time and departure delay |
| Stretch Model 2C | `ArrDel15` | Model 2B information plus taxi-out and takeoff information |

The merged data is explored and expanded with features that summarize time of day, season, route, distance, congestion, and weather. Each model excludes information recorded after its stated prediction time.

## Modeling

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

# Appendix A

## BTS Column Selection and Dictionary

The downloaded BTS airport-year file contains 110 columns. Processing removes cancelled and diverted flights and drops 83 
source columns that are redundant, describe events outside the project scope, or would reveal information that is not 
available when a prediction is made. Cleaning validates the remaining fields and adds `DATE`, producing 28 columns before 
the ASPM and NOAA merge.

Column selection is based on both usefulness and timing. A value may be useful for describing a completed flight but 
still be an invalid predictor if it becomes known only after the model's prediction time. Using such a value would give 
the model information from the future, commonly called target or time leakage. For this reason, retaining a column in the 
cleaned data does not automatically make it an input to every model. Separate model-ready datasets enforce the information 
boundary for each prediction.

BTS publishes the On-Time Performance data as a historical reporting dataset; it is not the proposed live source for an
operational model. However, some BTS columns represent events—such as gate departure and takeoff—that an airline or 
airport operational system can observe when they occur. This capstone uses the historical BTS values to test how 
predictions change at those event times. It therefore makes an explicit deployment assumption: an operational version 
of Model 2B would receive gate-departure information from a suitable live feed, not wait for the later BTS publication.

`DepTime`, `TaxiOut`, and `WheelsOff` are retained because they mark useful and clearly different information points. `DepTime` is the actual gate-out or pushback time and is eligible for Model 2B. `TaxiOut` is not complete, and `WheelsOff` is not known, until takeoff; both are excluded from Models 1, 2A, and 2B. They remain available for validation, EDA, and a possible stretch Model 2C that would update the arrival-delay prediction immediately after takeoff. Arrival results remain targets or retrospective analysis fields and are never predictors.

### Prediction-time eligibility of key BTS operational fields

| Field | Model 1: before pushback | Model 2A: before pushback | Model 2B: after pushback | Possible Model 2C: after takeoff |
|---|---|---|---|---|
| Scheduled fields such as `CRSDepTime`, `CRSArrTime`, and `CRSElapsedTime` | Eligible | Eligible | Eligible | Eligible |
| `DepTime` and departure-delay fields derived from gate-out | Excluded: not yet known | Excluded: not yet known | Eligible: gate-out has occurred | Eligible |
| `DepDel15` | Target only | Excluded: not yet known | Eligible: gate-out has occurred | Eligible |
| `TaxiOut` and `WheelsOff` | Excluded: takeoff has not occurred | Excluded: takeoff has not occurred | Excluded: takeoff has not occurred | Eligible |
| `ArrDel15` | Not used | Target only | Target only | Target only |
| Other arrival outcomes and post-arrival fields | Excluded | Excluded | Excluded | Excluded |

“Dropped” means a field is unnecessary or inappropriate for this capstone design; it does not mean the field has no value in other aviation studies.

### Columns retained before merge

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
| `Origin` | Three-letter origin airport code. | Retained to identify flights originating at the airport of interest for Model 1 and the departure airport for Models 2A and 2B. |
| `OriginState` | Two-letter state code for the origin airport. | Retained as a compact geographic field. |
| `Dest` | Three-letter destination airport code. | Retained to identify flights arriving at the airport of interest for Models 2A and 2B. |
| `DestState` | Two-letter state code for the destination airport. | Retained as a compact geographic field. |
| `CRSDepTime` | Computer Reservation System scheduled departure time in local HHMM form. | Retained because it is known before departure and is used to construct the exact scheduled departure timestamp. |
| `DepTime` | Actual gate departure time in local HHMM form. | Retained as the event-time field for Model 2B and for descriptive analysis. It is excluded from Models 1 and 2A because gate-out has not occurred when those predictions are made. BTS supplies the historical value; a deployed Model 2B would require an operational gate-out feed. |
| `DepDelay` | Actual gate departure time minus scheduled departure time, in minutes; negative values indicate an early departure. | Retained as a Model 2B predictor because it is known once gate-out occurs. It is excluded from Models 1 and 2A. Because it overlaps the other departure-delay fields, the final Model 2B projection should avoid unnecessary duplicate representations. |
| `DepDelayMinutes` | Nonnegative departure delay in minutes; early departures are recorded as zero. | Retained as a possible Model 2B representation and to validate `DepDelay` and `DepDel15`. It is excluded from pre-pushback predictors and need not be included together with every related departure-delay field. |
| `DepDel15` | Indicator equal to 1 when departure delay is at least 15 minutes. | Target for Model 1. It is excluded from Model 2A but may be used as a compact post-pushback input for Model 2B because the departure outcome is known at gate-out. It is never used to predict itself in Model 1. |
| `DepartureDelayGroups` | Departure delay grouped into ordered 15-minute ranges. | Retained for validation, EDA, and possible Model 2B use. It is excluded before pushback and is not automatically included alongside the continuous and binary departure-delay fields. |
| `TaxiOut` | Minutes from gate departure to wheels-off. | Retained for validation, EDA, and a possible after-takeoff Model 2C. It is excluded from Models 1, 2A, and 2B because the full taxi-out duration is unknown immediately after pushback. |
| `WheelsOff` | Actual takeoff time in local HHMM form. | Retained for validation, EDA, and a possible after-takeoff Model 2C. It is excluded from Models 1, 2A, and 2B because takeoff occurs after their prediction times. As with `DepTime`, BTS is the historical source rather than the proposed live event feed. |
| `CRSArrTime` | Scheduled arrival time in the destination's local HHMM form. | Retained because it is known from the schedule before departure. |
| `ArrDel15` | Indicator equal to 1 when arrival delay is at least 15 minutes. | Target for Models 2A and 2B, and for a possible Model 2C. It is never used as a predictor because it is known only after arrival. |
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
| `AirTime` | Minutes between wheels-off and wheels-on. | Dropped because it is known only after landing and is outside all three prediction times. |
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

These operational fields could be used in a separate historical analysis, but that work is outside the capstone scope. Excluding them keeps the feature set focused on information that is useful and reasonably available at prediction time.

### ASPM columns retained before merge

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

ASPM columns receive an `ASPM_` prefix during the merge so their origin remains clear in the combined dataset. Spaces are replaced with underscores.

| Before merge | After merge |
|---|---|
| `airport` | `ASPM_Airport` |
| `report_date` | `ASPM_ReportDate` |
| `Hour` | `ASPM_Hour` |
| `Scheduled Departures` | `ASPM_Scheduled_Departures` |
| `Scheduled Arrivals` | `ASPM_Scheduled_Arrivals` |
| `DATE` | `ASPM_DATE` |

The merge also adds `ASPM_LOOKUP_DATE`, the preceding full hour requested for each flight, and `ASPM_AGE_MINUTES`, the difference between the scheduled flight time and the matched ASPM observation.

## NOAA Column Selection and Dictionary

The NOAA data describes weather observed near each airport. Only hourly fields that are useful for the delay models are kept. Daily and monthly summaries are dropped because they do not describe conditions at a specific prediction time and may include weather that occurred later.

Each model uses the most recent NOAA observation available by its prediction time. Later observations are excluded to prevent data leakage. Missing weather values must also be handled with a past-only method so that an earlier record is not filled using future weather.

### NOAA fields retained before merge

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
