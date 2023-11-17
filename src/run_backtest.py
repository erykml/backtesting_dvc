import pandas as pd
from src.backtester import TSBacktester
from src.config import *
from src.model_definitions import *

# load data
df = pd.read_csv("data/time_series.csv", index_col=0)
df.index = pd.to_datetime(df.index)

# generate features
dummies = pd.get_dummies(df.index.month, prefix="month", drop_first=True)
dummies.index = df.index
df = pd.concat([df, dummies], axis=1)

# run the backtest
backtester = TSBacktester(
    mean_forecast,
    BT_START_DATE,
    BT_END_DATE,
    BACKTEST_FREQ,
    DATA_FREQ,
    FCST_HORIZON,
    rolling_window_size=None,
)

backtester.run_backtest(df, target_col="y", verbose=True)
backtest_results = backtester.evaluate_backtest(BT_METRICS)

print("Backtest results ----")
print(pd.DataFrame(backtest_results))
