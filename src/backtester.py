import pandas as pd


class TSBacktester:
    """
    A custom time series backtester class for evaluating forecasting models.
    """

    def __init__(
        self,
        pred_func,
        start_date,
        end_date,
        backtest_freq,
        data_freq,
        forecast_horizon,
        rolling_window_size=None,
    ):
        self.pred_func = pred_func
        self.start_date = start_date
        self.end_date = end_date
        self.backtest_freq = backtest_freq
        self.data_freq = data_freq
        self.forecast_horizon = forecast_horizon
        self.rolling_window_size = rolling_window_size
        self.backtest_df = None

    def run_backtest(self, df, target_col, features=None, verbose=False):
        """
        Run the time series backtest using the specified parameters.
        """

        ts_df = df.copy()

        fcst_dates = pd.date_range(
            self.start_date, self.end_date, freq=self.backtest_freq
        )
        backtest_list = []

        for forecast_date in fcst_dates:
            test_ind = pd.date_range(
                forecast_date, periods=self.forecast_horizon, freq=self.data_freq
            )

            X_train = ts_df.loc[ts_df.index < forecast_date].copy()
            if self.rolling_window_size is not None:
                X_train = X_train.iloc[-self.rolling_window_size :]
            y_train = X_train.pop(target_col)

            X_test = ts_df.loc[ts_df.index.isin(test_ind)].copy()
            y_test = X_test.pop(target_col)

            if verbose:
                print(f"Forecasting as of {forecast_date.date()} ----")
                print(
                    f"Training data: {X_train.index.min().date()} : {X_train.index.max().date()} (n = {len(X_train)})"
                )
                print(
                    f"Test data: {X_test.index.min().date()} : {X_test.index.max().date()} (n = {len(X_test)})"
                )

            # get predictions
            y_pred = self.pred_func(
                X_train, y_train, X_test, self.forecast_horizon, features
            )

            pred_df = pd.DataFrame(
                {
                    "forecast_date": forecast_date,
                    "report_date": y_test.index,
                    "forecast": y_pred,
                    "actual": y_test,
                }
            )

            backtest_list.append(pred_df)

        backtest_df = pd.concat(backtest_list, ignore_index=True)

        # add some columns for potential score calculation
        backtest_df["horizon"] = (
            backtest_df["report_date"] - backtest_df["forecast_date"]
        ).dt.days

        self.backtest_df = backtest_df

    def evaluate_backtest(self, metrics, model_name, agg_col="horizon"):
        """
        Evaluates the backtest using specified performance metrics.
        """

        if self.backtest_df is None:
            raise ValueError(
                "Backtest was not yet executed! Please run it before evaluating"
            )

        metadata_dict = {}
        score_dict = {}

        # adding some details about the backtest
        metadata_dict["model_name"] = model_name
        metadata_dict["start_date"] = self.start_date
        metadata_dict["end_date"] = self.end_date
        metadata_dict["backtest_freq"] = self.backtest_freq
        metadata_dict["forecast_horizon"] = self.forecast_horizon
        metadata_dict["validation_type"] = (
            "rolling" if self.rolling_window_size is not None else "expanding"
        )
        metadata_dict["rolling_window_size"] = self.rolling_window_size

        # preparing for scoring
        backtest_df = self.backtest_df.copy()
        grouped = backtest_df.groupby(agg_col)

        # calculate scores and store them in a dict
        for metric, metric_func in metrics.items():
            score_dict[f"{metric}_total"] = round(
                metric_func(backtest_df["actual"], backtest_df["forecast"]), 4
            )

            for group, group_df in grouped:
                score_dict[f"{metric}_{agg_col}_{group}"] = round(
                    metric_func(group_df["actual"], group_df["forecast"]), 4
                )

        return metadata_dict, score_dict
