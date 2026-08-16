# Appendix C

[Back to the main report](README.md)

This appendix describes the engineered features used by the departure and arrival models. It covers the shared
schedule, calendar, route, planned-traffic, and weather features; the separate operational-backlog and aircraft-rotation
datasets; and the rules that determine which fields each model may use at its prediction time. The joined source
columns are documented in [Appendix B](Appendix-B.md#joined-data-column-dictionary).

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
feature table. The current arrival-model base uses `WindX`, `WindY`, and `NOAA_AGE_MINUTES`; preferred CatBoost 1A
Experiment 04 instead uses the six direct NOAA measurements listed in its manifest below.

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
- Learn every data-preparation choice from 2019 training data only. This includes filling missing values, scaling
  numeric values, converting categories, selecting features, balancing the two outcome classes, and choosing numeric
  cutoffs. Use 2023 only for external validation and keep 2024 untouched until the model and threshold are frozen.
- Do not create flags such as `LOW_VISIBILITY`, `HIGH_WIND`, `ASPM_HIGH_TRAFFIC`, or flags based on source-record age
  until their cutoffs have a clear operational meaning or have been selected using training data.
- Add large sets of interaction features or historical delay-rate features only through a documented follow-up
  experiment. If historical rates are added, calculate them from earlier completed flights only. Exclude the flight
  being predicted, never use development or test outcomes, and reduce unstable rates for groups with few observations.
- Current Model 2B Experiment 01 adds only signed `DepDelay`, where negative values mean an early departure and positive
  values mean a late departure. Raw gate-out time and the nonnegative, binary, grouped, and log delay alternatives are
  excluded to keep the comparison nonredundant. Any broader update must be a separately documented experiment.
- Current Model 2C Experiment 01 adds `LOG_TAXI_OUT_MINUTES`, `ACTUAL_TAKEOFF_TIME_SIN`, and
  `ACTUAL_TAKEOFF_TIME_COS` to the complete Model 2B base. Raw `TaxiOut` and `WheelsOff` remain in the shared dataset but
  are excluded from this manifest to avoid duplicate representations. A `WheelsOff` value of `2400` maps to minute zero
  for the cyclical representation; it does not imply a same-day takeoff date.
- Model 1A uses flights with `Origin` equal to JFK. Models 2A, 2B, and 2C must instead use inbound flights with `Dest`
  equal to JFK. Their ASPM and NOAA data must describe each flight's origin and must be available by that model's
  prediction time.
- Do not substitute JFK outbound data for the origin data required by the arrival models. Do not use destination weather
  observed at landing in a pre-pushback model. Destination weather is allowed only if it came from a forecast or
  observation that was available by the prediction cutoff.
- Keep source timestamps such as `FlightDate`, `DATE`, ASPM lookup and report dates, and `NOAA_DATE` so the joins can be
  checked. Do not use these timestamps directly as model predictors.
- Keep missing next-hour ASPM matches at annual dataset boundaries as missing unless they can be recovered from the
  following year's planned schedule dataset. Do not replace them with zero or copy values from the current hour.
- Keep `Tail_Number` audit-only in standard models. The separate rotation path may use it only as a matching key under
  the timing and assignment rules below; the raw registration is not a predictor. Longer aircraft chains,
  tail-identity effects, and network-propagation features remain excluded.

The implemented arrival-model allowlists are nested so their performance differences reflect newly available
information rather than unrelated feature changes:

| Model | Prediction time | Implemented source predictors |
|---|---|---|
| 2A | Before pushback | 27 fields: `Reporting_Airline`, `Origin`, and 25 schedule-cycle, calendar, route, planned-origin-traffic, origin-weather, and NOAA-age numeric fields from the dictionary above. |
| 2B | After pushback | All 27 Model 2A fields plus signed `DepDelay` (28 total). |
| 2C | After takeoff | All 28 Model 2B fields plus `LOG_TAXI_OUT_MINUTES`, `ACTUAL_TAKEOFF_TIME_SIN`, and `ACTUAL_TAKEOFF_TIME_COS` (31 total). |

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
current airport-wide choice. The output dataset still retains all eight fields for traceability and future ablations.

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
allowlist. The other four same-airline fields remain in the shared dataset but are not admitted automatically.

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
- The current Model 1A datasets cover only JFK departures. An arrival implementation must use a separate all-departure
  reference population grouped by each sample's `Origin`. Computing origin backlog from only JFK-bound flights would
  undercount airport activity and is not allowed.
- Origin groups use local scheduled timestamps. Backlog windows must never compare local clock values across airports
  as though they shared a time zone.
- The annual implementation uses only events present in that annual input. A sample near the beginning of January may
  lack events from the final 30 minutes of the preceding annual dataset; this boundary condition must be reported rather
  than filled with future outcomes.
- Backlog is available at the scheduled-departure snapshot for Model 1A and, once the origin implementation exists, for
  Models 2A, 2B, and 2C. Model 2B and 2C must use the exact same pre-pushback backlog base as 2A before adding the sample
  flight's pushback or takeoff information.
- Historical development uses BTS to reconstruct event state. Deployment requires a live or sufficiently timely
  schedule and gate-out feed; ASPM planned counts alone cannot reproduce completed and pending flight state.
- New window lengths, overdue-minute severity measures, or backlog trends must be implemented as new, separately named
  datasets and experiments. Existing `backlog_w30` datasets and published results remain unchanged under the project's
  append-only guideline.

## Aircraft Rotation Feature Engineering

The rotation features are documented separately because they extend the reference-informed schedule, traffic, and
weather design with the operating state of the assigned aircraft. The original cohort-limited path writes
`JFK_YEAR_departures_rotation.csv`; the append-only current path writes
`JFK_YEAR_departures_rotation_full_history.csv`. Both contain the same 13 predictor fields and eight audit fields, and
both preserve the standard 112 columns and target-row order. The full-history dataset is the selected source for current
Model 1A experiments, while the original datasets and their Experiment 05 results remain intact.

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
  is retained for audit. Across the generated datasets, only 11 matched 2019 rows, 10 matched 2023 rows, and two matched
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
- The append-only full-history path addresses that source limitation without changing the original datasets. It reads all
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
same-airline W60 datasets and require exact equality of these flight identity fields before pairing rows:
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
W60 datasets are prepared, but the same-airline W60 dataset is intentionally deferred until the design is frozen;
therefore the complete 41-field final-test input has not yet been assembled or evaluated.
