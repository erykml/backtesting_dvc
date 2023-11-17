from sklearn.metrics import mean_absolute_error, mean_squared_error

# backtesting setup
BT_START_DATE = "2023-01-01"
BT_END_DATE = "2023-12-24"
BACKTEST_FREQ = "7D"
DATA_FREQ = "1D"
FCST_HORIZON = 7

BT_METRICS = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
}

# list of features
FEATURE_LIST = [
    "month_2",
    "month_3",
    "month_4",
    "month_5",
    "month_6",
    "month_7",
    "month_8",
    "month_9",
    "month_10",
    "month_11",
    "month_12",
]
