import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor


def run_experiment(
    stock: pd.DataFrame,
    news: pd.DataFrame | None = None,
    *,
    model,
    use_news: bool = True,
    horizons: tuple[int, ...] = (1, 2, 3),
    test_fraction: float = 0.2,
) -> dict:
    """Train and evaluate a return-prediction model using a chronological split.

    Required columns:
      stock: date, close
      news: published_at (when use_news=True)

    Each target is the cumulative log return from the row's close to the
    close `horizon` stock sessions later.

    The last `max(horizons) - 1` rows before the split are dropped from training,
    because their targets use closes from the test period.

    Metrics for each horizon cover the model and a train-mean baseline, plus
    directional accuracy compared with always predicting an up move.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("horizons must contain positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must not contain duplicates")

    required_stock_columns = {"date", "close"}
    missing_stock_columns = required_stock_columns - set(stock.columns)
    if missing_stock_columns:
        raise ValueError(f"stock is missing columns: {sorted(missing_stock_columns)}")

    data = stock[["date", "close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    data["close"] = pd.to_numeric(data["close"], errors="raise")
    data = (
        data.sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )

    if data.empty:
        raise ValueError("stock contains no rows")
    if data["close"].isna().any() or (data["close"] <= 0).any():
        raise ValueError("stock close prices must be present and positive")

    if use_news:
        if news is None:
            raise ValueError("news is required when use_news=True")
        if "published_at" not in news.columns:
            raise ValueError("news must contain a 'published_at' column")

        published_dates = (
            pd.to_datetime(news["published_at"], errors="raise")
            .dt.normalize()
            .to_numpy(dtype="datetime64[ns]")
        )
        sessions = data["date"].to_numpy(dtype="datetime64[ns]")

        # Attribute each article to the next stock session, so news from a
        # given date cannot be used to predict that same date's return.
        session_positions = np.searchsorted(sessions, published_dates, side="right")
        session_positions = session_positions[session_positions < len(data)]
        counts = pd.Series(session_positions).value_counts()

        data["news_count"] = (
            pd.Series(np.arange(len(data)))
            .map(counts)
            .fillna(0)
            .to_numpy()
        )

    data["previous_return"] = np.log(data["close"] / data["close"].shift(1))

    target_columns = []
    for horizon in horizons:
        column = f"return_{horizon}d"
        data[column] = np.log(data["close"].shift(-horizon) / data["close"])
        target_columns.append(column)

    feature_columns = ["previous_return"]
    if use_news:
        feature_columns.append("news_count")

    data = data.dropna(subset=feature_columns + target_columns).reset_index(drop=True)

    split_at = int(len(data) * (1 - test_fraction))
    train_end = split_at - (max(horizons) - 1)
    if train_end < 1 or split_at >= len(data):
        raise ValueError(
            "Not enough usable rows for the requested chronological split"
        )

    train = data.iloc[:train_end]
    test = data.iloc[split_at:]

    X_train = train[feature_columns]
    X_test = test[feature_columns]

    if len(horizons) == 1:
        fitted_model = clone(model)
        fitted_model.fit(X_train, train[target_columns[0]])
        predictions = np.asarray(fitted_model.predict(X_test)).reshape(-1, 1)
        train_predictions = np.asarray(fitted_model.predict(X_train)).reshape(-1, 1)
    else:
        fitted_model = MultiOutputRegressor(clone(model))
        fitted_model.fit(X_train, train[target_columns])
        predictions = np.asarray(fitted_model.predict(X_test))
        train_predictions = np.asarray(fitted_model.predict(X_train))

    prediction_data = {"date": test["date"].to_numpy()}
    train_prediction_data = {"date": train["date"].to_numpy()}
    metrics = {}

    for index, horizon in enumerate(horizons):
        actual = test[target_columns[index]].to_numpy()
        predicted = predictions[:, index]
        baseline = np.full(len(actual), train[target_columns[index]].mean())

        prediction_data[f"actual_{horizon}d"] = actual
        prediction_data[f"predicted_{horizon}d"] = predicted
        prediction_data[f"baseline_{horizon}d"] = baseline
        train_prediction_data[f"actual_{horizon}d"] = train[target_columns[index]].to_numpy()
        train_prediction_data[f"predicted_{horizon}d"] = train_predictions[:, index]
        metrics[f"{horizon}d"] = {
            "mae": mean_absolute_error(actual, predicted),
            "rmse": np.sqrt(mean_squared_error(actual, predicted)),
            "baseline_mae": mean_absolute_error(actual, baseline),
            "baseline_rmse": np.sqrt(mean_squared_error(actual, baseline)),
            "directional_accuracy": np.mean((predicted > 0) == (actual > 0)),
            "always_up_accuracy": np.mean(actual > 0),
        }

    return {
        "metrics": metrics,
        "predictions": pd.DataFrame(prediction_data),
        "train_predictions": pd.DataFrame(train_prediction_data),
        "model": fitted_model,
        "features": feature_columns,
        "horizons": horizons,
    }