# Appendix A

[Back to the main report](README.md)

This appendix explains which BTS, ASPM, and NOAA columns are kept, which are removed, and when each retained field is
available to the four models. It describes the cleaned source datasets that feed the departure and arrival merges.
Joined column names and merge-audit fields are documented in
[Appendix B](Appendix-B.md#joined-data-column-dictionary).

## BTS Column Selection and Dictionary

The downloaded BTS dataset contains 110 columns. Processing removes cancelled and diverted flights, drops 77 unused
source fields, and retains 33 flight, route, schedule, outcome, and audit columns. Cleaning validates those fields and
adds `DATE`, producing the current 34-column BTS dataset before the ASPM and NOAA data are joined. The 2019, 2023, and
2024 cleaned datasets have the same column layout.

A field must be both useful and available at the time a prediction is made. Some BTS fields describe events that happen
later in the flight. Using one of those fields too early would give the model information from the future, often called
time or target leakage.

BTS is a historical reporting source, but some of its columns describe events—such as pushback and takeoff—that an
airline or airport system could observe when they happen. `DepTime`, `TaxiOut`, and `WheelsOff` are therefore kept, but
their use depends on the model's prediction time. `DepTime` becomes available after pushback. `TaxiOut` is not complete,
and `WheelsOff` is not known, until takeoff. The implemented Model 2B uses signed `DepDelay` rather than raw `DepTime`;
the implemented Model 2C derives log taxi-out duration and cyclical takeoff-time features from `TaxiOut` and
`WheelsOff` rather than admitting the two raw fields as duplicate predictors.

`clean_bts.ipynb` writes one 34-column dataset for each airport and year under `data/bts/cleaned/`. For the arrival
scenario, `data/bts/cat_bts.py` combines the required origin-airport datasets into `data/bts/cleaned_JFK_YEAR.csv` without
changing the column layout.

### Prediction-time eligibility of key BTS operational fields

| Field | Model 1A: before pushback | Model 2A: before pushback | Model 2B: after pushback | Model 2C: after takeoff |
|---|---|---|---|---|
| Scheduled fields such as `CRSDepTime`, `CRSArrTime`, and `CRSElapsedTime` | Eligible | Eligible | Eligible | Eligible |
| `DepTime` and departure-delay fields derived from gate-out | Excluded: not yet known | Excluded: not yet known | Eligible after gate-out; Experiment 01 uses signed `DepDelay` only | Eligible |
| Target flight's `DepDel15` | Target only | Excluded: not yet known | Eligible after gate-out, but excluded from the current Model 2B experiment | Eligible, but excluded from the current Model 2C experiment |
| `TaxiOut` and `WheelsOff` | Excluded: takeoff has not occurred | Excluded: takeoff has not occurred | Excluded: takeoff has not occurred | Eligible sources for engineered taxi-out and takeoff-time features |
| Target flight's `ArrDel15` | Not used | Target only | Target only | Target only |
| Other target-flight arrival outcomes and post-arrival fields | Excluded | Excluded | Excluded | Excluded |

This eligibility table describes fields belonging to the flight being predicted. The operational feature extensions
also use completed events from *earlier* flights, but only after those events were observable by the target flight's
prediction cutoff. Earlier departures contribute `DepDelay`, `DepDelayMinutes`, and `DepDel15` to airport-wide and
same-airline backlog summaries. An already-arrived preceding aircraft leg can contribute `ArrDelay` and `ArrDel15` to
the rotation features. The target flight's future outcome fields remain prohibited.

“Dropped” means that a field is not needed or is not appropriate for this project. It may still be useful in other
aviation studies.

### BTS columns retained for modeling

| Column | Meaning | Why it is retained / how it is used |
|---|---|---|
| `Year` | Calendar year of the flight. | Retained for coverage checks and comparison across 2019, 2023, and 2024. |
| `Quarter` | Calendar quarter, numbered 1 through 4. | Retained for calendar validation and candidate analysis; current completed models do not use it directly. |
| `Month` | Calendar month, numbered 1 through 12. | Retained for seasonal analysis and feature engineering. |
| `DayofMonth` | Day of the month. | Retained for calendar validation. Current calendar features are derived primarily from `FlightDate` and `DayOfWeek`. |
| `DayOfWeek` | Day of week, where BTS uses 1 for Monday through 7 for Sunday. | Retained for weekly-pattern analysis and feature engineering. |
| `FlightDate` | Scheduled flight date without a time component. | Converted to a pandas datetime and retained as the base calendar date. |
| `Reporting_Airline` | BTS reporting carrier code. | Retained as the main airline identifier. |
| `Tail_Number` | Aircraft registration or tail number. | Retained for audit and validation. Standard models do not use the registration as a predictor; the separate rotation path uses it only to match a target departure to a preceding inbound leg. |
| `Flight_Number_Reporting_Airline` | Flight number assigned by the reporting carrier. | Retained for row identity, audit, and the engineered `AIRLINE_FLIGHT_ID` tested in the broad Model 1A experiment; it is not treated as a continuous quantity. |
| `Origin` | Three-letter origin airport code. | Retained to identify flights originating at JFK for Model 1A and the departure airport for Models 2A, 2B, and 2C. |
| `OriginState` | Two-letter state code for the origin airport. | Retained as a compact geographic candidate and for route validation; current completed models use the airport code rather than the state. |
| `Dest` | Three-letter destination airport code. | Retained to identify flights arriving at JFK for Models 2A, 2B, and 2C. |
| `DestState` | Two-letter state code for the destination airport. | Retained as a compact geographic candidate and for route validation; current completed models use the airport code rather than the state. |
| `CRSDepTime` | Computer Reservation System scheduled departure time in local HHMM form. | Retained because it is known before departure and is used to construct the exact scheduled departure timestamp. |
| `DepTime` | Actual gate departure time in local HHMM form. | Retained as an observable gate-out event time, for descriptive analysis, and as a candidate source for post-pushback clock features. The current Model 2B experiment excludes raw `DepTime` and adds signed `DepDelay` only. It is unavailable to Models 1A and 2A. A deployed post-pushback model would require an operational gate-out feed. |
| `DepDelay` | Actual gate departure time minus scheduled departure time, in minutes; negative values indicate an early departure. | The only post-pushback field added by the current Model 2B experiment. It is also used with `DATE` to reconstruct actual gate-out times for earlier flights in the backlog features. It is excluded from Models 1A and 2A as a target-flight predictor. |
| `DepDelayMinutes` | Nonnegative departure delay in minutes; early departures are recorded as zero. | Retained to validate `DepDelay` and `DepDel15` and to calculate nonnegative delay summaries for earlier completed flights in the backlog feature source. The current Model 2B experiment excludes it to avoid overlapping delay representations. |
| `DepDel15` | Indicator equal to 1 when departure delay is at least 15 minutes. | Target for Model 1A. For earlier completed departures, it contributes delayed-flight counts and rates to the backlog feature source. It is excluded from Model 2A and from the current Model 2B/2C predictor manifests, which use signed `DepDelay` instead. It is never used to predict itself for the target Model 1A flight. |
| `DepartureDelayGroups` | Departure delay grouped into ordered 15-minute ranges. | Retained for target validation and EDA. It is excluded before pushback and from the current Model 2B/2C experiments to avoid duplicating signed `DepDelay`; a later experiment could compare it as an alternative representation. |
| `TaxiOut` | Minutes from gate departure to wheels-off. | Retained for validation and EDA and as the source for Model 2C's `LOG_TAXI_OUT_MINUTES`. The raw field is excluded from the current predictor manifest and is unavailable to Models 1A, 2A, and 2B. |
| `WheelsOff` | Actual takeoff time in local HHMM form. | Retained for validation and EDA and as the source for Model 2C's cyclical actual-takeoff fields. The raw field is excluded from the current predictor manifest and is unavailable to Models 1A, 2A, and 2B. BTS is the historical source rather than the proposed live event feed. |
| `CRSArrTime` | Scheduled arrival time in the destination's local HHMM form. | Retained because it is known from the schedule before departure. |
| `WheelsOn` | Actual landing time in destination-local HHMM form. | Retained for cleaning and event-sequence audit. It is not used by the current rotation reconstruction or as a target-flight predictor. |
| `TaxiIn` | Minutes from wheels-on to gate arrival. | Retained for cleaning and duration audit. It is not used by the current rotation reconstruction and remains unavailable for the target flight at the project's prediction times. |
| `ArrTime` | Actual gate-arrival time in destination-local HHMM form. | Retained for cleaning and arrival-event audit. The current rotation reconstruction instead derives actual arrival from scheduled arrival plus signed `ArrDelay`; raw `ArrTime` is prohibited as a target-flight predictor. |
| `ArrDelay` | Signed actual gate-arrival delay in minutes. | For an already-arrived preceding aircraft leg, it is used with the reconstructed scheduled arrival to calculate actual arrival time and becomes `ROTATION_INBOUND_ARR_DELAY`. The target flight's value is prohibited. |
| `ArrDelayMinutes` | Nonnegative gate-arrival delay in minutes. | Retained for audit and validation; it is not a direct Model 1A input. |
| `ArrDel15` | Indicator equal to 1 when arrival delay is at least 15 minutes. | Target for Models 2A, 2B, and 2C. The target flight's value is never a predictor. For an already-arrived preceding aircraft leg, it becomes the causally available `ROTATION_INBOUND_DELAYED_15` feature used by Model 1A rotation experiments. |
| `ArrivalDelayGroups` | Arrival delay grouped into ordered 15-minute ranges. | Retained for target validation and EDA. It is excluded from every model input because it describes the completed arrival outcome. |
| `CRSElapsedTime` | Scheduled gate-to-gate elapsed time, in minutes. | Retained as a schedule and route characteristic known before departure. |
| `ActualElapsedTime` | Actual gate-to-gate elapsed time in minutes. | Retained for cleaning and duration audit. It is not used by the current rotation reconstruction and remains prohibited for the target flight before completion. |
| `Distance` | Published distance between origin and destination, in miles. | Retained as a route characteristic. |
| `DistanceGroup` | BTS distance band based on 250-mile intervals. | Retained for grouped analysis and candidate categorical modeling; current completed models use continuous or log distance instead. |
| `DATE` | Constructed scheduled departure timestamp created from FlightDate and CRSDepTime. | Added during cleaning. It supports sorting, time-based ASPM and NOAA matching, chronological splitting, time feature engineering, backlog cutoffs, and rotation-event reconstruction. |

### BTS columns removed during processing

The 77 removed columns are mainly duplicate airline and airport identifiers, redundant location labels, coarse time
blocks, administrative fields, diversion details, and an empty export column. Cancelled and diverted flights are
removed before their now-unused status and detail fields are dropped.

Post-event measurements that would reveal the result—such as delay-cause minutes, airtime, and irregular gate-return
times—are also removed. They are not known at the relevant prediction time and would give the model an unrealistic
advantage. The exact removal list is implemented in `clean_bts.ipynb`.

## ASPM Column Selection and Dictionary

The downloaded ASPM hourly dataset contains 18 columns. Processing removes 13 and retains the airport, date, hour,
scheduled departures, and scheduled arrivals. Cleaning also creates `DATE`, giving each cleaned ASPM dataset six columns.
The scheduled counts describe planned airport demand. Because historical schedules can include later revisions, the
project treats these counts as the best available summary of the schedule rather than as a perfect record of what was
known at every moment.

`clean_aspm.ipynb` writes one six-column dataset for each airport and year under `data/aspm/cleaned/`. For the arrival
scenario, `data/aspm/cat_aspm.py` combines the required origin-airport datasets into `data/aspm/cleaned_JFK_YEAR.csv`. The
combined dataset uses the same six columns.

In the current model implementations, only the scheduled-departure and scheduled-arrival counts become predictors.
The airport, date, hour, and constructed timestamp are retained to identify records, perform the merge, and audit the
result. The departure models can use the previous, current, and next clock-hour counts directly or derive three-hour
totals and changes from them. The next-hour counts are planned schedule values, not future realized operating results;
their use is therefore intentional, subject to the historical-schedule-snapshot limitation noted above.

### ASPM columns retained for modeling

| Column | Meaning | Why it is retained / how it is used |
|---|---|---|
| `airport` | Three-letter code for the airport represented by the hourly record. | Retained as the airport identifier and merge key; it is not used as a model predictor. |
| `report_date` | Calendar date associated with the hourly record. | Converted to a pandas datetime and combined with `Hour` to create the hourly timestamp; it is a join/audit field rather than a predictor. |
| `Hour` | Local airport hour, represented by an integer from 0 through 23. | Retained to identify the hourly reporting period and construct `DATE`; it is not passed to the models as an ASPM predictor. |
| `Scheduled Departures` | Number of flights scheduled to depart during the hour. | Retained as a planned measure of departure demand and airport workload. ASPM has already counted the scheduled flights by hour, so the project does not need to rebuild that count from BTS. |
| `Scheduled Arrivals` | Number of flights scheduled to arrive during the hour. | Retained as a planned measure of arrival demand and airport workload under the same hourly-count assumption. |
| `DATE` | Constructed hourly timestamp created by combining `report_date` and `Hour`. | Added during cleaning and used for sorting, checking hourly coverage, and matching ASPM records to flights; it is not a predictor. |

### ASPM columns removed during processing

The 13 removed columns are counts used to calculate ASPM performance metrics plus reported on-time percentages and
average gate, taxi, airborne, block, and arrival delays. These fields describe actual performance rather than planned
airport demand and are generally published in the following day's update—too late for the project's prediction times.
Using the previous operating hour would not solve that publication delay. The exact removal list is implemented in
`clean_aspm.ipynb`.

## NOAA Column Selection and Dictionary

The NOAA data describes weather at the airport where a flight departs: JFK for Model 1A and the flight's origin airport
for Models 2A, 2B, and 2C. Only hourly fields that are useful for delay prediction are kept. Daily and monthly summaries
are removed because they do not describe conditions at a specific time and may include weather observed later.

The merge uses the latest NOAA observation available at or before scheduled departure, as long as it is no more than 90
minutes old. Later reports are not added to Models 2B and 2C. When a weather value is filled during cleaning, it may use
only an earlier observation within the same 90-minute limit; future weather is never used.

When NOAA supplies multiple report types at the same timestamp, cleaning keeps one row. It prefers `FM-16`, then
`FM-15`, then `FM-12`; ties are resolved by retained-field completeness and original source order. `REPORT_TYPE` is
removed after that selection.

`clean_noaa.ipynb` writes an 18-column dataset for each airport and year under `data/noaa/cleaned/`. For the arrival
scenario, `data/noaa/cat_noaa.py` combines the origin-airport datasets and adds `AIRPORT`, producing the 19-column
`data/noaa/cleaned_JFK_YEAR.csv` dataset.

These cleaned columns form the available NOAA feature set; individual models use subsets.

### NOAA columns retained for modeling

| Column | Plain-English description | Why it is retained / how it is used |
|---|---|---|
| `DATE` | Date and time of the weather observation. | Used to match each flight with weather already observed by the prediction time; it is a join/audit field rather than a predictor. |
| `AIRPORT` | Airport code linked to the weather station. | Added only to the consolidated arrival input so each flight can be matched to weather at its origin; it is not a predictor. |
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

### NOAA columns removed or transformed during processing

The raw NOAA Local Climatological Data export uses a 125-column schema. Initial processing removes 115 columns and
carries 10 into cleaning. The removed fields are mainly station metadata, unused hourly measurements, daily and monthly
summaries, short-duration precipitation summaries, and sparse backup, equipment, or remarks fields. Summary fields are
especially unsuitable because they may include weather observed after the flight's prediction time.

Cleaning then uses three intermediate columns without retaining them directly: `REPORT_TYPE` selects among reports at
the same timestamp, `HourlyPresentWeatherType` produces the weather indicators, and `HourlyWindDirection` produces
`WindX` and `WindY`. The result contains 18 columns: the observation timestamp, six direct weather measurements, and
11 engineered weather fields. The exact processing choices are implemented in `process_noaa.ipynb` and
`clean_noaa.ipynb`.
