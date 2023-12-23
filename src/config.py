from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.model_definitions import *

# backtesting setup
BT_START_DATE = "2023-01-01"
BT_END_DATE = "2023-12-24"
BACKTEST_FREQ = "7D"
DATA_FREQ = "1D"
FCST_HORIZON = 3
ROLLING_WINDOW_SIZE = None

BT_METRICS = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
}
METRICS_TO_PLOT = ["mse", "mae"]

# list of features
MODELS_W_FEATURES = ["linear", "rf"]
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

# a dictionary with the available models
SELECTED_MODEL = "linear"
MODEL_DICT = {
    "naive": naive_forecast,
    "mean": mean_forecast,
    "linear": linear_model,
    "rf": rf_model,
}
