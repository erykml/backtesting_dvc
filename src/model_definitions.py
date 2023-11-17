import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def naive_forecast(X_train, y_train, X_test, horizon, features):
    y_pred = [y_train.iloc[-1]] * horizon
    return y_pred


def mean_forecast(X_train, y_train, X_test, horizon, features):
    y_pred = [y_train.mean()] * horizon
    return y_pred


def linear_model(X_train, y_train, X_test, horizon, features):
    if features is not None:
        X_train = X_train[features]
        X_test = X_test[features]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred


def rf_model(X_train, y_train, X_test, horizon, features):
    if features is not None:
        X_train = X_train[features]
        X_test = X_test[features]

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred
