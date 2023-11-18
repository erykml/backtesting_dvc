import pandas as pd
from src.backtester import TSBacktester
from src.config import *
from src.model_definitions import *
from dvclive import Live

# load data
df = pd.read_csv("data/time_series.csv", index_col=0)
df.index = pd.to_datetime(df.index)

# generate features
dummies = pd.get_dummies(df.index.month, prefix="month", drop_first=True)
dummies.index = df.index
df = pd.concat([df, dummies], axis=1)

# run the backtest
backtester = TSBacktester(
    naive_forecast,
    BT_START_DATE,
    BT_END_DATE,
    BACKTEST_FREQ,
    DATA_FREQ,
    FCST_HORIZON,
    rolling_window_size=100,
)

backtester.run_backtest(df, target_col="y", verbose=True)
backtest_metadata, backtest_results = backtester.evaluate_backtest(
    BT_METRICS, model_name="naive"
)

print("Backtest results ----")
print(backtest_results)

with Live(save_dvc_exp=True) as live:
    live.log_params(backtest_metadata)
    for metric_name, metric_value in backtest_results.items():
        live.log_metric(metric_name, metric_value)
