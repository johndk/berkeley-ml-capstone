# Exploratory Data Analysis

The notebooks in this directory explore the merged BTS, ASPM, and NOAA
airport-year datasets in `data/merged`.

## Configuration

Set `AIRPORT` and `YEAR` in `config.py` before running the first three
notebooks. Valid airports are `ATL`, `JFK`, and `ORD`; valid years are
`2019`, `2023`, and `2024`.

The cross-airport notebook loads all available airport-year files and does
not use the single-dataset configuration.

## Notebooks

1. `01_data_quality_and_targets.ipynb` checks structure, missing values,
   duplicates, target balance, and the age of matched ASPM and NOAA records.
2. `02_delay_patterns.ipynb` explores when delays occur and how delay rates
   differ by airline, destination, distance, and scheduled time.
3. `03_weather_and_congestion.ipynb` examines weather, airport traffic,
   congestion, and their relationship with departure and arrival delays.
4. `04_airport_year_comparison.ipynb` compares all available airports and
   years using memory-conscious summaries and samples.

Each notebook contains Markdown commentary before its tables and charts.
The helper columns created for EDA are descriptive aids; they are not
automatically part of the later model feature set.

## Suggested order

Run the notebooks in numeric order. The first three provide detailed EDA for
one airport-year. The fourth checks whether the main patterns and data-quality
findings differ across the full project scope.
