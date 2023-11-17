import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(42)

# Generate a DataFrame with 4 years of data
range_of_dates = pd.date_range(start="2020-01-01", end="2023-12-31")
X = pd.DataFrame(index=range_of_dates)

# Create a sequence of day numbers and add day of the year information
X["day_nr"] = range(len(X))
X["day_of_year"] = X.index.dayofyear

# Generate components of the target time series
signal_1 = 3 + 4 * np.sin(X["day_nr"] / 365 * 2 * np.pi)
signal_2 = 3 * np.sin(X["day_nr"] / 365 * 4 * np.pi + 365 / 2)
noise = np.random.normal(0, 0.85, len(X))

# Combine the components to get the target series
y = signal_1 + signal_2 + noise

# Convert the target series to a DataFrame and assign a column name
df = y.to_frame()
df.columns = ["y"]

# Save the generated data
df.to_csv("data/time_series.csv")
