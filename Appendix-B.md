# Appendix B

[Back to the main report](README.md)

This appendix describes the columns saved after the BTS, ASPM, and NOAA datasets are joined. It covers the departure
and arrival datasets, explains the purpose of each joined field, and distinguishes model inputs from fields retained
only to build features or check the joins. The engineered features built from these columns are documented in
[Appendix C](Appendix-C.md#feature-engineering).

## Joined Data Column Dictionary

Each row in `data/merged/JFK_YEAR_departures.csv` represents a completed flight leaving JFK. The dataset has 77 columns:
34 from BTS, 24 from ASPM, and 19 from NOAA. Each row in `data/merged/JFK_YEAR_arrivals.csv` represents a completed
flight headed to JFK. That dataset has 72 columns: 28 from BTS, 24 from ASPM, and 20 from NOAA. The NOAA total includes
`NOAA_AIRPORT`. The departure dataset has six additional BTS outcome fields needed to build the aircraft-rotation
features; the completed arrival experiments do not need them.

The merged datasets contain more information than any one model is allowed to use. Targets, actual outcomes, timestamps,
and operating events are kept so the data and joins can be checked and later features can be built. Their presence does
not make them valid predictors. `feature_departures.ipynb` and `feature_arrivals.ipynb` add the standard calculated
fields and write the 112-column departure and 116-column arrival datasets under `data/features/`. Each model notebook
then selects only the fields known at its prediction time. All experiments use these shared feature datasets instead of
making separate model-specific datasets.

The column layouts are checked for 2019, 2023, and 2024. Model choices are made with 2019 data and checked with 2023
data. The 2024 data remains untouched until the final model and decision threshold are fixed.

### Joined BTS flight columns

The table lists the union of BTS fields in the two merged scenarios. `WheelsOn`, `TaxiIn`, `ArrTime`, `ArrDelay`,
`ArrDelayMinutes`, and `ActualElapsedTime` are present only in the departure datasets, where they support audit and the
separate preceding-aircraft rotation path; the arrival datasets retain the other 28 BTS fields.

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
| `DepTime` | Actual gate departure or pushback time in local HHMM form. | Available only after pushback. The current Model 2B experiment excludes the raw clock value and uses signed `DepDelay` instead. |
| `DepDelay` | Actual gate departure minus scheduled departure, in minutes. | The sole post-pushback field added by the current Model 2B experiment; unavailable to Models 1A and 2A for the target flight. |
| `DepDelayMinutes` | Nonnegative departure delay in minutes. | Retained for validation but excluded from the current Model 2B/2C manifests to avoid duplicating signed `DepDelay`. |
| `DepDel15` | Indicates a departure delay of at least 15 minutes. | Target for Model 1A. It is excluded from the current arrival-model manifests, which use signed `DepDelay` after pushback. |
| `DepartureDelayGroups` | Departure delay grouped into 15-minute ranges. | Outcome field for EDA and validation; unavailable before pushback. |
| `TaxiOut` | Minutes from gate departure to takeoff. | Available only after takeoff and used to derive Model 2C's `LOG_TAXI_OUT_MINUTES`; the raw value is excluded from the current manifest. |
| `WheelsOff` | Actual takeoff time in local HHMM form. | Available only at takeoff and used to derive Model 2C's cyclical takeoff-time fields; the raw value is excluded from the current manifest. |
| `CRSArrTime` | Scheduled arrival time in the destination's local HHMM form. | Schedule field known before departure. |
| `ArrTime` | Actual gate-arrival time in destination-local HHMM form. | Restored for rotation-source audit; completed-flight outcome and never a direct Model 1A predictor. |
| `ArrDelay` | Actual gate arrival minus scheduled arrival, in signed minutes. | Used only for an already-arrived preceding aircraft in the separate rotation path; the target flight's value is prohibited. |
| `ArrDelayMinutes` | Nonnegative arrival-delay minutes. | Retained for audit; unavailable for the target flight at Model 1A prediction time. |
| `ArrDel15` | Indicates an arrival delay of at least 15 minutes. | Target for Models 2A, 2B, and 2C and never a predictor for that target flight. For Model 1A rotation experiments, an already-arrived preceding leg may supply it as `ROTATION_INBOUND_DELAYED_15`. |
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
to the next annual dataset. These are documented year-boundary gaps, not zero-traffic hours.

### Joined NOAA weather columns

NOAA supplies the latest origin-airport weather observation at or before scheduled departure, within the 90-minute
matching limit. The observation timestamp and age remain in the merged datasets so future or overly old matches can be
detected.

| Column | Description | Use and availability |
|---|---|---|
| `NOAA_AIRPORT` | Airport code linked to the matched weather station. | Present only in the arrival dataset; confirms that weather came from the flight's origin. |
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
| `NOAA_AGE_MINUTES` | Scheduled departure time minus the NOAA observation time, in minutes. | Must be nonnegative and within the allowed weather-match tolerance. It is used by the current arrival-model base but not preferred CatBoost 1A Experiment 04. |
