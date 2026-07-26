# Exploratory Data Analysis

The notebooks in this directory explore the model datasets assembled from
cleaned BTS, ASPM, and NOAA data in `data/models`.

## Configuration

Set `AIRPORT`, `YEAR`, and `MODEL` in `config.py` before running the first
three notebooks. The current airport is `JFK`; valid years are `2019`, `2023`,
and `2024`, and valid model values are `m1a`, `m2a`, `m2b`, and `m2c`.

The comparison notebook loads all available JFK model files and does not use
the single-dataset configuration.

## Notebooks

1. `01_data_quality_and_targets.ipynb` checks structure, missing values,
   duplicates, target balance, ASPM period offsets, and NOAA observation age.
2. `02_delay_patterns.ipynb` explores when delays occur and how delay rates
   differ by airline, destination, distance, and scheduled time.
3. `03_weather_and_congestion.ipynb` examines weather, planned traffic in the
   previous, current, and next hours, and their relationship with delays.
4. `04_airport_year_comparison.ipynb` compares available JFK model datasets
   and years using memory-conscious summaries and samples.

Each notebook contains Markdown commentary before its tables and charts.
The helper columns created for EDA are descriptive aids; they are not
automatically part of the later model feature set.

## Suggested order

Run the notebooks in numeric order. The first three provide detailed EDA for
one model dataset. The fourth checks whether the main patterns and data-quality
findings differ across available years and model variants.
