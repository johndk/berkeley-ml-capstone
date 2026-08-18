# Appendix D

[Back to the main report](README.md)

This appendix documents the source fields "allowed" in each implemented model experiment. It explains why each
allowlist was chosen and what changed with each experiment. Feature definitions are in
[Appendix C](Appendix-C.md); the experiment plan is in [Appendix E](Appendix-E.md); completed results are in
[Appendix F](Appendix-F.md).

## Overview

An allowlist is the set of source fields a model may use. This appendix refers to source fields before
imputation, scaling, missing-value indicators, or categorical encoding. Those training-only steps can increase the
number of prepared columns without changing the source allowlist.

Each reusable allowlist identifies its originating feature dataset. Experiments that combine allowlists from multiple categories
combine the datasets only in memory; no permanent combined feature dataset is created. Before pairing rows, they require
exact equality of `FlightDate`, `Reporting_Airline`, `Flight_Number_Reporting_Airline`, `Origin`, `Dest`, `CRSDepTime`,
`Tail_Number`, and `DepDel15`. The airport-wide and same-airline backlog cutoff timestamps must also match.

## Reusable feature allowlists

The model inputs come from baseline, airport-backlog, same-airline backlog, and aircraft-rotation feature datasets.
Departure experiments combine field subsets from these dataset categories. Arrival experiments use a field subset
from the baseline arrival dataset. [Appendix C](Appendix-C.md#datasets) defines the base dataset codes, saved
dataset names, source datasets, and production notebooks.

This section defines the reusable field subsets admitted to the models. Each allowlist code combines an Appendix C
base dataset code with the number of fields in the subset. For example, `BL-D-20` contains 20 fields from `BL-D`, and
`RTF-13` contains 13 fields from `RTF`. For backlog allowlists, the middle number is the lookback window in minutes and
the final number is the field count. For example, `BK-60-3` contains three fields from the 60-minute airport-backlog
dataset. These codes identify model inputs; they do not identify additional saved datasets.

| Code | Short explanation | Fields | Dataset category |
|---|---|---:|---|
| BL-D-20 | Raw departure baseline | 20 | Base JFK departures |
| BL-D-27 | Compact engineered departure allowlist | 27 | Base JFK departures |
| BL-D-54 | Broad engineered departure allowlist | 54 | Base JFK departures |
| BL-D-34 | Compact engineered departure allowlist | 34 | Base JFK departures |
| RT-5 | Compact aircraft-rotation context | 5 | Original limited-history departure rotation |
| RT-13 | Full aircraft-rotation context | 13 | Original limited-history departure rotation |
| RTF-3 | Schedule-only aircraft-rotation context | 3 | Full-history departure rotation |
| RTF-4 | Observable aircraft-rotation state | 4 | Full-history departure rotation |
| RTF-13 | Full aircraft-rotation context | 13 | Full-history departure rotation |
| BK-30-3 | Compact airport-wide backlog over 30 minutes | 3 | Airport-wide W30 departure backlog |
| BK-30-8 | Full airport-wide backlog over 30 minutes | 8 | Airport-wide W30 departure backlog |
| BK-60-3 | Compact airport-wide backlog over 60 minutes | 3 | Airport-wide W60 departure backlog |
| BK-60-8 | Full airport-wide backlog over 60 minutes | 8 | Airport-wide W60 departure backlog |
| BKA-60-5 | Compact same-airline backlog over 60 minutes | 5 | Same-airline W60 departure backlog |
| BL-A-27 | Pre-pushback arrival baseline | 27 | Base JFK arrivals |

The experiments use 2019 for training and selection and 2023 for validation. The 2024 datasets remain reserved for
final testing.

### BL-D-20 — raw departure baseline

**Dataset category:** Base JFK departures.

BL-D-20 contains 20 fields known before a JFK departure. It is intentionally simple and preserves raw clock, traffic, and
weather measurements.

| Fields | Type | Why they are retained / how they are used |
|---|---|---|
| `Month`, `DayOfWeek` | Categorical (2) | Represent seasonal and weekly schedule patterns in their original form. |
| `Reporting_Airline`, `Dest` | Categorical (2) | Identify the operating airline and destination, two basic sources of systematic delay differences. |
| `CRSDepTime`, `CRSArrTime` | Numeric (2) | Preserve the scheduled departure and arrival clock times known before pushback. |
| `CRSElapsedTime`, `Distance` | Numeric (2) | Describe the planned duration and length of the route. |
| `ASPM_PREVIOUS_SCHEDULED_DEPARTURES`, `ASPM_PREVIOUS_SCHEDULED_ARRIVALS` | Numeric (2) | Measure planned airport demand in the preceding clock hour. |
| `ASPM_CURRENT_SCHEDULED_DEPARTURES`, `ASPM_CURRENT_SCHEDULED_ARRIVALS` | Numeric (2) | Measure planned airport demand in the scheduled departure hour. |
| `ASPM_NEXT_SCHEDULED_DEPARTURES`, `ASPM_NEXT_SCHEDULED_ARRIVALS` | Numeric (2) | Measure planned demand in the following hour; these are schedule counts, not future operating outcomes. |
| `HourlyDewPointTemperature`, `HourlyDryBulbTemperature`, `HourlyPrecipitation`, `HourlyRelativeHumidity`, `HourlyVisibility`, `HourlyWindSpeed` | Numeric (6) | Preserve six direct weather measurements available before the scheduled departure. |

**Reasoning:** BL-D-20 provides an understandable reference point using schedule, airline, route, planned airport demand,
and weather. Later experiments can then measure whether engineered or operational features add value.

### BL-D-27 — compact engineered departure allowlist

**Dataset category:** Base JFK departures.

BL-D-27 contains 27 fields. It replaces redundant raw representations with cyclical time, compact traffic, route, and
weather features.

| Fields | Type | Why they are retained / how they are used |
|---|---|---|
| `Reporting_Airline`, `Dest`, `AIRLINE_DEST` | Categorical (3) | Represent the airline, destination, and their interaction without treating identifiers as continuous numbers. |
| `SCHED_DEP_TIME_SIN`, `SCHED_DEP_TIME_COS` | Numeric (2) | Represent scheduled departure time as a daily cycle. |
| `SCHED_ARR_TIME_SIN`, `SCHED_ARR_TIME_COS` | Numeric (2) | Represent destination-local scheduled arrival time as a daily cycle. |
| `DAY_OF_WEEK_SIN`, `DAY_OF_WEEK_COS` | Numeric (2) | Preserve the weekly cycle, including the Sunday-to-Monday boundary. |
| `DAY_OF_YEAR_SIN`, `DAY_OF_YEAR_COS` | Numeric (2) | Represent annual seasonality without an artificial year-end break. |
| `IS_WEEKEND` | Numeric | Separates weekend from weekday schedules. |
| `CRSElapsedTime`, `LOG_DISTANCE`, `SCHEDULED_SPEED_PROXY` | Numeric (3) | Describe planned duration, route length, and their relationship. |
| `ASPM_THREE_HOUR_SCHEDULED_DEPARTURES`, `ASPM_THREE_HOUR_SCHEDULED_ARRIVALS` | Numeric (2) | Summarize planned airport demand across the previous, current, and next hours. |
| `ASPM_CURRENT_MINUS_PREVIOUS_TRAFFIC`, `ASPM_NEXT_MINUS_CURRENT_TRAFFIC` | Numeric (2) | Show whether planned airport demand is rising or falling. |
| `HourlyDryBulbTemperature`, `TEMP_DEWPOINT_SPREAD` | Numeric (2) | Describe temperature and the separation between temperature and dew point. |
| `LOG_PRECIPITATION`, `HourlyVisibility` | Numeric (2) | Represent precipitation intensity and visibility with limited redundancy. |
| `WindX`, `WindY` | Numeric (2) | Represent wind direction and speed without the discontinuity at 360 degrees. |
| `ADVERSE_WEATHER` | Numeric | Flags whether any selected adverse-weather condition is present. |
| `NOAA_AGE_MINUTES` | Numeric | Records how old the matched weather observation is at prediction time. |

**Reasoning:** Linear models cannot discover cycles or nonlinear transformations on their own. BL-D-27 supplies those
representations while avoiding several versions of the same information.

### BL-D-54 — broad engineered departure allowlist

**Dataset category:** Base JFK departures.

BL-D-54 contains every pre-pushback engineered field considered safe for modeling, plus eligible raw weather and schedule
measurements.

| Fields | Type | Why they are retained / how they are used |
|---|---|---|
| `Reporting_Airline`, `Dest` | Categorical (2) | Identify the airline and destination. |
| `TIME_OF_DAY`, `YEAR_PERIOD` | Categorical (2) | Provide understandable daily and seasonal schedule bands. |
| `ROUTE`, `AIRLINE_FLIGHT_ID`, `AIRLINE_DEST` | Categorical (3) | Test route, recurring flight, and airline-destination interactions in the broad candidate pool. |
| `SCHED_DEP_MINUTE_OF_DAY`, `SCHED_DEP_HOUR` | Numeric (2) | Preserve direct scheduled-departure clock representations. |
| `SCHED_DEP_TIME_SIN`, `SCHED_DEP_TIME_COS` | Numeric (2) | Represent scheduled departure time as a daily cycle. |
| `SCHED_ARR_MINUTE_OF_DAY`, `SCHED_ARR_TIME_SIN`, `SCHED_ARR_TIME_COS` | Numeric (3) | Represent destination-local scheduled arrival time directly and cyclically. |
| `IS_WEEKEND`, `DAY_OF_WEEK_SIN`, `DAY_OF_WEEK_COS` | Numeric (3) | Describe weekend status and the weekly cycle. |
| `DAY_OF_YEAR`, `DAY_OF_YEAR_SIN`, `DAY_OF_YEAR_COS`, `MONTH_SIN`, `MONTH_COS` | Numeric (5) | Provide direct and cyclical representations of annual seasonality. |
| `LOG_DISTANCE`, `SCHEDULED_SPEED_PROXY` | Numeric (2) | Describe route length and its relationship to planned duration. |
| `ASPM_PREVIOUS_TOTAL_SCHEDULED_TRAFFIC`, `ASPM_CURRENT_TOTAL_SCHEDULED_TRAFFIC`, `ASPM_NEXT_TOTAL_SCHEDULED_TRAFFIC` | Numeric (3) | Measure total planned traffic in each adjacent clock hour. |
| `ASPM_THREE_HOUR_SCHEDULED_DEPARTURES`, `ASPM_THREE_HOUR_SCHEDULED_ARRIVALS`, `ASPM_THREE_HOUR_TOTAL_SCHEDULED_TRAFFIC` | Numeric (3) | Summarize planned demand across the three-hour window. |
| `ASPM_CURRENT_MINUS_PREVIOUS_TRAFFIC`, `ASPM_NEXT_MINUS_CURRENT_TRAFFIC`, `ASPM_MAX_HOURLY_TRAFFIC` | Numeric (3) | Describe changes and the peak in planned traffic. |
| `TEMP_DEWPOINT_SPREAD`, `LOG_PRECIPITATION` | Numeric (2) | Add compact moisture and precipitation representations. |
| `WEATHER_CONDITION_COUNT`, `ADVERSE_WEATHER` | Numeric (2) | Summarize the number and presence of adverse conditions. |
| `CRSElapsedTime` | Numeric | Retains the planned gate-to-gate duration. |
| `HourlyDryBulbTemperature`, `HourlyRelativeHumidity`, `HourlyVisibility`, `HourlyWindSpeed` | Numeric (4) | Retain direct weather measurements for the broad candidate pool. |
| `Rain`, `Drizzle`, `Snow`, `Fog`, `Mist`, `Thunderstorm`, `FreezingPrecip`, `Showers`, `PrecipOccurred` | Numeric (9) | Preserve individual weather-condition indicators so their effects can be tested separately. |
| `WindX`, `WindY` | Numeric (2) | Represent wind direction and speed continuously. |
| `NOAA_AGE_MINUTES` | Numeric | Records how old the matched weather observation is at prediction time. |

**Reasoning:** BL-D-54 tests whether a larger safe candidate pool and training-only L1 selection outperform the
hand-curated BL-D-27 representation. Breadth is the controlled change; prediction timing is unchanged.

### BL-D-34 — compact engineered departure allowlist

**Dataset category:** Base JFK departures.

BL-D-34 contains 34 fields. It retains raw ordered values that trees can split directly and adds selected engineered
summaries.

| Fields | Type | Why they are retained / how they are used |
|---|---|---|
| `Month`, `DayOfWeek` | Categorical (2) | Retain direct calendar values that a tree can split. |
| `Reporting_Airline`, `Dest` | Categorical (2) | Identify the airline and destination. |
| `TIME_OF_DAY`, `AIRLINE_DEST` | Categorical (2) | Represent the scheduled-departure period and airline-destination interaction. |
| `CRSDepTime`, `CRSArrTime` | Numeric (2) | Preserve raw scheduled clock times for direct splits. |
| `CRSElapsedTime`, `Distance` | Numeric (2) | Describe planned duration and route length. |
| `IS_WEEKEND`, `SCHEDULED_SPEED_PROXY` | Numeric (2) | Separate weekend schedules and represent the relationship between distance and planned duration. |
| `ASPM_PREVIOUS_SCHEDULED_DEPARTURES`, `ASPM_PREVIOUS_SCHEDULED_ARRIVALS` | Numeric (2) | Measure planned airport demand in the preceding hour. |
| `ASPM_CURRENT_SCHEDULED_DEPARTURES`, `ASPM_CURRENT_SCHEDULED_ARRIVALS` | Numeric (2) | Measure planned airport demand in the scheduled departure hour. |
| `ASPM_NEXT_SCHEDULED_DEPARTURES`, `ASPM_NEXT_SCHEDULED_ARRIVALS` | Numeric (2) | Measure planned demand in the following hour. |
| `ASPM_THREE_HOUR_SCHEDULED_DEPARTURES`, `ASPM_THREE_HOUR_SCHEDULED_ARRIVALS` | Numeric (2) | Summarize planned departures and arrivals across three hours. |
| `ASPM_CURRENT_MINUS_PREVIOUS_TRAFFIC`, `ASPM_NEXT_MINUS_CURRENT_TRAFFIC`, `ASPM_MAX_HOURLY_TRAFFIC` | Numeric (3) | Describe changes and the peak in planned traffic. |
| `HourlyDryBulbTemperature` | Numeric | Retains the direct air-temperature measurement. |
| `TEMP_DEWPOINT_SPREAD`, `LOG_PRECIPITATION` | Numeric (2) | Add compact moisture and precipitation representations. |
| `HourlyRelativeHumidity`, `HourlyVisibility`, `HourlyWindSpeed` | Numeric (3) | Retain direct moisture, visibility, and wind measurements. |
| `WindX`, `WindY` | Numeric (2) | Represent wind direction and speed continuously. |
| `WEATHER_CONDITION_COUNT`, `ADVERSE_WEATHER` | Numeric (2) | Summarize the number and presence of adverse conditions. |
| `NOAA_AGE_MINUTES` | Numeric | Records how old the matched weather observation is at prediction time. |

**Reasoning:** BL-D-34 avoids high-cardinality identity fields and many duplicate raw/transformed pairs. It is broader than
BL-D-20 but smaller than BL-D-54.

### Rotation additions

**Dataset categories:** RT-5 and RT-13 come from the original limited-history rotation dataset. RTF-3, RTF-4, and RTF-13 come
from the full-history rotation dataset.

RT-13 and RTF-13 contain the same fields but use rotation datasets with different history coverage.

| Field | Allowlist membership | Why it is retained / how it is used |
|---|---|---|
| `ROTATION_STATUS` | RT-5, RT-13, RTF-4, RTF-13 | Summarizes whether a preceding aircraft leg was matched and whether it had arrived by the cutoff. |
| `ROTATION_MATCH_FOUND` | RT-13, RTF-3, RTF-13 | Indicates whether the aircraft was matched to a preceding inbound leg. |
| `ROTATION_INBOUND_ORIGIN` | RT-13, RTF-3, RTF-13 | Provides the origin of the matched preceding leg as scheduled aircraft context. |
| `ROTATION_SCHEDULED_TURN_MINUTES` | RT-13, RTF-13 | Measures the planned interval from inbound arrival to target departure. |
| `ROTATION_INBOUND_ARRIVED_BY_CUTOFF` | RT-13, RTF-4, RTF-13 | Indicates that the preceding aircraft had reached JFK by prediction time. |
| `ROTATION_INBOUND_NOT_ARRIVED_BY_CUTOFF` | RT-13, RTF-4, RTF-13 | Indicates that the preceding aircraft was still unavailable at prediction time. |
| `ROTATION_INBOUND_OVERDUE_MINUTES` | RT-13, RTF-13 | Measures how long a not-yet-arrived aircraft was past its scheduled arrival. |
| `ROTATION_LOG_INBOUND_OVERDUE_MINUTES` | RT-5, RT-13, RTF-4, RTF-13 | Compresses the skewed overdue duration while preserving its ordering. |
| `ROTATION_ACTUAL_TURN_MINUTES` | RT-13, RTF-13 | Measures observable remaining turn time after the inbound aircraft has arrived. |
| `ROTATION_LOG_ACTUAL_TURN_MINUTES` | RT-5, RT-13, RTF-13 | Compresses the skewed observable turn-time measure. |
| `ROTATION_INBOUND_ARR_DELAY` | RT-5, RT-13, RTF-13 | Carries the signed arrival delay of an inbound leg that had already arrived. |
| `ROTATION_INBOUND_DELAYED_15` | RT-13, RTF-13 | Indicates whether an already-arrived inbound leg was at least 15 minutes late. |
| `ROTATION_LOG_SCHEDULED_TURN_MINUTES` | RT-5, RT-13, RTF-3, RTF-13 | Compresses scheduled turn time and provides the main planned-turn representation. |

**Reasoning:** Rotation fields describe whether the assigned aircraft is available and how its preceding trip
performed. The narrower sets separate scheduled aircraft context from live arrival state. All rotation experiments
remain retrospective upper bounds because BTS contains the final aircraft assignment rather than a timestamped
assignment available at prediction time.

### Airport-wide backlog additions

**Dataset categories:** BK-30-3 and BK-30-8 come from the airport-wide W30 departure backlog dataset. BK-60-3 and BK-60-8 come
from the airport-wide W60 departure backlog dataset.

The W30 and W60 fields use the same definitions; only the lookback window changes.

| Fields | Allowlist membership | Why they are retained / how they are used |
|---|---|---|
| `BACKLOG_W30_SCHEDULED_COUNT`, `BACKLOG_W60_SCHEDULED_COUNT` | BK-30-8; BK-60-8 | Count all flights scheduled in the trailing window to represent planned local workload. |
| `BACKLOG_W30_COMPLETED_COUNT`, `BACKLOG_W60_COMPLETED_COUNT` | BK-30-3, BK-30-8; BK-60-3, BK-60-8 | Count cohort flights that had already pushed back to represent recent throughput. |
| `BACKLOG_W30_PENDING_COUNT`, `BACKLOG_W60_PENDING_COUNT` | BK-30-3, BK-30-8; BK-60-3, BK-60-8 | Count earlier-scheduled flights still waiting to push back; this is the main queue-pressure measure. |
| `BACKLOG_W30_DELAYED_DEPARTURE_COUNT`, `BACKLOG_W60_DELAYED_DEPARTURE_COUNT` | BK-30-8; BK-60-8 | Count completed cohort flights observed to be at least 15 minutes late. |
| `BACKLOG_W30_DELAY_RATE`, `BACKLOG_W60_DELAY_RATE` | BK-30-8; BK-60-8 | Measure the share of completed cohort flights observed to be delayed. |
| `BACKLOG_W30_MEAN_DEP_DELAY`, `BACKLOG_W60_MEAN_DEP_DELAY` | BK-30-3, BK-30-8; BK-60-3, BK-60-8 | Measure mean signed departure delay among completed cohort flights. |
| `BACKLOG_W30_MEAN_DEP_DELAY_MINUTES`, `BACKLOG_W60_MEAN_DEP_DELAY_MINUTES` | BK-30-8; BK-60-8 | Measure mean nonnegative delay severity among completed cohort flights. |
| `BACKLOG_W30_TOTAL_DEP_DELAY_MINUTES`, `BACKLOG_W60_TOTAL_DEP_DELAY_MINUTES` | BK-30-8; BK-60-8 | Measure accumulated nonnegative delay among completed cohort flights. |

**Reasoning:** The compact sets represent queue pressure, recent throughput, and signed delay with little duplication.
The full sets test whether additional counts, rates, and delay totals help a nonlinear model.

### Same-airline backlog addition

**Dataset category:** Same-airline W60 departure backlog.

| Field | Why it is retained / how it is used |
|---|---|
| `AIRLINE_BACKLOG_W60_PENDING_COUNT` | Counts earlier-scheduled flights from the same airline that had not pushed back. |
| `AIRLINE_BACKLOG_W60_COMPLETED_COUNT` | Counts recent same-airline flights that had already pushed back. |
| `AIRLINE_BACKLOG_W60_MEAN_DEP_DELAY` | Measures mean signed delay among completed same-airline flights. |
| `AIRLINE_BACKLOG_W60_DELAY_RATE` | Measures the observed delayed share among completed same-airline flights. |
| `AIRLINE_BACKLOG_W60_PENDING_SHARE` | Measures the fraction of scheduled same-airline work still pending. |

**Reasoning:** Airport-wide backlog can hide differences between airlines. BKA-60-5 tests whether recent pressure within
the target flight's airline adds useful information without admitting all nine generated same-airline fields.

### BL-A-27 — arrival baseline

**Dataset category:** Base JFK arrivals.

BL-A-27 contains 27 fields available at the origin before pushback.

| Fields | Type | Why they are retained / how they are used |
|---|---|---|
| `Reporting_Airline`, `Origin` | Categorical (2) | Identify the airline and origin airport for flights arriving at JFK. |
| `SCHED_DEP_TIME_SIN`, `SCHED_DEP_TIME_COS` | Numeric (2) | Represent origin-local scheduled departure time as a daily cycle. |
| `SCHED_ARR_TIME_SIN`, `SCHED_ARR_TIME_COS` | Numeric (2) | Represent JFK-local scheduled arrival time as a daily cycle. |
| `DAY_OF_WEEK_SIN`, `DAY_OF_WEEK_COS` | Numeric (2) | Preserve the weekly cycle. |
| `DAY_OF_YEAR_SIN`, `DAY_OF_YEAR_COS` | Numeric (2) | Represent annual seasonality without a year-end break. |
| `IS_WEEKEND` | Numeric | Separates weekend from weekday schedules. |
| `CRSElapsedTime`, `LOG_DISTANCE`, `SCHEDULED_SPEED_PROXY` | Numeric (3) | Describe planned duration, route length, and their relationship. |
| `ASPM_THREE_HOUR_SCHEDULED_DEPARTURES`, `ASPM_THREE_HOUR_SCHEDULED_ARRIVALS` | Numeric (2) | Summarize planned demand at the origin across three hours. |
| `ASPM_CURRENT_MINUS_PREVIOUS_TRAFFIC`, `ASPM_NEXT_MINUS_CURRENT_TRAFFIC`, `ASPM_MAX_HOURLY_TRAFFIC` | Numeric (3) | Describe changes and the peak in planned origin traffic. |
| `HourlyDryBulbTemperature`, `TEMP_DEWPOINT_SPREAD` | Numeric (2) | Describe temperature and moisture conditions at the origin. |
| `LOG_PRECIPITATION`, `HourlyVisibility` | Numeric (2) | Represent precipitation intensity and visibility at the origin. |
| `WindX`, `WindY` | Numeric (2) | Represent origin wind direction and speed continuously. |
| `ADVERSE_WEATHER` | Numeric | Flags whether any selected adverse-weather condition is present at the origin. |
| `NOAA_AGE_MINUTES` | Numeric | Records how old the matched origin-weather observation is at prediction time. |

**Reasoning:** BL-A-27 describes the flight schedule, route, planned demand, and weather at the origin. `Dest` is constant
JFK, while `ROUTE` and `AIRLINE_DEST` would repeat `Origin` and `Reporting_Airline`.

## Model 1A experiments

### Logistic-regression sequence

| Experiment | Source allowlist | Change and reason |
|---|---:|---|
| Logistic Regression 01 | BL-D-20: 20 fields | Establishes the raw pre-pushback baseline. |
| Logistic Regression 02 | BL-D-27: 27 fields | Replaces raw and duplicate representations with a compact engineered departure allowlist. |
| Logistic Regression 03 | BL-D-54: 54 fields | Tests a broad safe pool with fold-local zero-variance removal and L1 top-50, top-100, top-200, or passthrough selection. The source allowlist changes; the prediction cutoff does not. |
| Logistic Regression 04 | BL-D-20 + BK-30-3: 23 fields; BL-D-20 + BK-30-8: 28 fields | Adds recent JFK operating state. The compact and full backlog variants test whether extra correlated summaries justify their complexity. BK-30-3 is selected. |
| Logistic Regression 05 | BL-D-20 + RT-5: 25 fields; BL-D-20 + RT-13: 33 fields | Replaces airport-wide context with flight-specific aircraft state. The experiment tests compact versus complete limited-history rotation information. RT-13 is selected. |
| Logistic Regression 06 | Six variants described below | Separates rotation-history coverage from the type of rotation information. The classifier is held fixed so differences come from the rotation input. |
| Logistic Regression 07 | BL-D-20 + RTF-13: 33-field control; plus BK-30-3: 36 fields | Adds compact 30-minute airport-wide backlog to the selected full-history, 24-hour-masked rotation design. |
| Logistic Regression 08 | BL-D-20 + RTF-13: 33-field control; plus BK-60-3: 36 fields | Changes only the backlog window from 30 to 60 minutes to test a broader operating snapshot. |

Logistic Regression 06 uses these variants:

| Variant | Dataset category | Fields | Purpose |
|---|---|---:|---|
| Raw baseline | Full-history departure rotation; base departure columns only | BL-D-20: 20 | Measures performance without rotation. |
| Cohort-history full rotation | Original limited-history departure rotation | BL-D-20 + RT-13: 33 | Reproduces the original limited-history rotation source. |
| Full-history schedule only | Full-history departure rotation | BL-D-20 + RTF-3: 23 | Tests scheduled aircraft identity and turn context without live arrival outcomes. |
| Full-history state only | Full-history departure rotation | BL-D-20 + RTF-4: 24 | Tests whether aircraft arrival state alone carries the gain. |
| Full-history full rotation | Full-history departure rotation | BL-D-20 + RTF-13: 33 | Tests all rotation fields using complete same-airport movement history. |
| Full-history full rotation, 24-hour mask | Full-history departure rotation | BL-D-20 + RTF-13: 33 | Keeps the same allowlist but masks questionable scheduled turns over 24 hours. This treatment is selected. |

### Decision-tree and Random-Forest sequence

| Experiment | Source allowlist | Change and reason |
|---|---:|---|
| Decision Tree 01 | BL-D-20: 20 fields | Rebuilds the raw baseline with a single tree so classifier effects can be compared fairly. |
| Decision Tree 02 | BL-D-34: 34 fields | Adds selected tree-friendly engineered summaries while avoiding the broad BL-D-54 pool. |
| Random Forest 01 | BL-D-34: 34 fields | Holds the allowlist fixed and changes only the classifier from one tree to a forest. |
| Random Forest 02 | BL-D-20 + RTF-13 + BK-60-3 + BKA-60-5: 41 fields | Gives Random Forest the exact preferred CatBoost 04 source allowlist. This separates feature value from classifier-family value. |

### CatBoost sequence

| Experiment | Source allowlist | Change and reason |
|---|---:|---|
| CatBoost 01 | BL-D-34: 34 fields | Holds the compact engineered departure allowlist fixed and lets CatBoost handle six categorical fields directly. |
| CatBoost 02 | BL-D-20 + RTF-13 + BK-60-3: 36 fields | Applies CatBoost to the operational feature hypothesis selected by Logistic Regression 08. The 24-hour rotation mask is retained. |
| CatBoost 03 | 36-field control versus BL-D-20 + RTF-13 + BK-60-8: 41 fields | Changes only the airport-wide backlog breadth. The full eight-field backlog does not improve the 2019 selection result, so BK-60-3 remains selected. |
| CatBoost 04 | 36-field control versus BL-D-20 + RTF-13 + BK-60-3 + BKA-60-5: 41 fields | Adds airline-specific pressure while retaining the airport-wide backlog and rotation controls. The 41-field variant is selected and is the preferred departure model. |

The selected CatBoost 04 allowlist contains 41 source fields: 20 from BL-D-20, 13 from RTF-13, three from BK-60-3,
and five from BKA-60-5. These are six categorical and 35 numeric fields before model-specific preprocessing. The
reusable allowlist tables above provide the exact field names.

### Neural network, ensemble, calibration, and audit

| Experiment | Source inputs | Change and reason |
|---|---:|---|
| MLP 02 | BL-D-20 + RTF-13 + BK-60-3: 36 fields | Uses the same source hypothesis as Logistic Regression 08 and CatBoost 02. The controlled change is the classifier and neural-network preprocessing. |
| CatBoost/MLP Blend 01 | CatBoost 04 probabilities from BL-D-20 + RTF-13 + BK-60-3 + BKA-60-5: 41 fields; MLP 02 probabilities from BL-D-20 + RTF-13 + BK-60-3: 36 fields | Blends fixed component probabilities. The MLP component does not use the same-airline fields. The experiment does not create a combined feature allowlist or retune either component. |
| Calibration 01 | BL-D-20 + RTF-13 + BK-60-3 + BKA-60-5: 41 fields | Compares probability corrections without changing the selected model or source inputs. The uncalibrated output remains selected. |
| CatBoost Audit 01 | BL-D-20 + RTF-13 + BK-60-3 + BKA-60-5: 41 fields | Reuses the frozen allowlist, classifier, and threshold for subgroup and SHAP analysis. It makes no new feature choice. |

## Arrival-model experiments

The arrival experiments are nested and use the same flight rows. Each later model adds only information newly available
at its prediction time.

| Experiment | Prediction time | Source allowlist | Change and reason |
|---|---|---:|---|
| Model 2A Logistic Regression 01 | Before pushback | BL-A-27: 27 fields | Establishes the schedule, route, planned-origin-demand, and origin-weather baseline. All target-flight operating outcomes are prohibited. |
| Model 2B Logistic Regression 01 | Immediately after pushback | BL-A-27 + `DepDelay`: 28 fields | Adds signed gate-departure delay, the smallest nonredundant description of the completed pushback event. Raw `DepTime` and alternative delay representations remain excluded. |
| Model 2C Logistic Regression 01 | Immediately after takeoff | BL-A-27 + `DepDelay` + `LOG_TAXI_OUT_MINUTES` + `ACTUAL_TAKEOFF_TIME_SIN` + `ACTUAL_TAKEOFF_TIME_COS`: 31 fields | Adds realized surface duration and takeoff time. Raw `TaxiOut` and `WheelsOff` remain excluded to avoid duplicate representations. |

## Superseded baseline notebooks

`logistic_regression_01.ipynb`, `logistic_regression_02.ipynb`, `decision_tree_01.ipynb`, and
`decision_tree_02.ipynb` are earlier baseline notebooks. All four use the same BL-D-20 source fields, but they use an older
random train/test split and missing-row policy. The numbered `*_1a_*` notebooks replace them under the current
time-ordered protocol. They use the base JFK departure dataset, remain in the repository for history, and are not
separate current experiments.
