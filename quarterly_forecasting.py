from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DATE_FORMAT = "%d-%b-%y"
DEFAULT_CSV_PATH = "Meropenem_FY22_to_FY25_with_MarketCategory.csv"
MODEL_NAMES = [
    "last_quarter_naive",
    "seasonal_naive",
    "moving_average_4q",
    "lagged_ridge_regression",
]
REGRESSION_FEATURES = [
    "lag_1",
    "lag_4",
    "rolling_mean_2",
    "rolling_mean_4",
]


@dataclass
class RidgeModel:
    feature_columns: List[str]
    feature_means: np.ndarray
    feature_scales: np.ndarray
    target_mean: float
    coefficients: np.ndarray
    alpha: float


def load_meropenem_data(csv_path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load the raw file and make the time columns safe for downstream work."""
    df = pd.read_csv(csv_path)
    df["Invoice Date"] = pd.to_datetime(df["Invoice Date"], format=DATE_FORMAT, errors="coerce")
    df = df.dropna(subset=["Invoice Date"]).copy()
    df["Financial Year"] = df["Financial Year"].astype(int)
    df["Quarter"] = df["Quarter"].astype(int)
    return df


def _quarter_start(financial_year: int, quarter: int) -> pd.Timestamp:
    """Map the financial quarter to a real calendar date for sorting and plotting."""
    quarter_to_month = {1: (financial_year, 4), 2: (financial_year, 7), 3: (financial_year, 10), 4: (financial_year + 1, 1)}
    year, month = quarter_to_month[quarter]
    return pd.Timestamp(year=year, month=month, day=1)


def _next_quarter(financial_year: int, quarter: int) -> Dict[str, int]:
    if quarter < 4:
        return {"financial_year": financial_year, "quarter": quarter + 1}
    return {"financial_year": financial_year + 1, "quarter": 1}


def build_quarterly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the business to one record per financial quarter."""
    quarterly = (
        df.groupby(["Financial Year", "Quarter"], as_index=False)["Total Qty"]
        .sum()
        .rename(
            columns={
                "Financial Year": "financial_year",
                "Quarter": "quarter",
                "Total Qty": "total_qty",
            }
        )
        .sort_values(["financial_year", "quarter"])
        .reset_index(drop=True)
    )

    quarterly["quarter_start"] = quarterly.apply(
        lambda row: _quarter_start(int(row["financial_year"]), int(row["quarter"])),
        axis=1,
    )
    quarterly["quarter_label"] = (
        "FY"
        + quarterly["financial_year"].astype(str)
        + " Q"
        + quarterly["quarter"].astype(str)
    )
    quarterly["period_index"] = np.arange(1, len(quarterly) + 1)
    return quarterly


def add_forecasting_features(quarterly: pd.DataFrame) -> pd.DataFrame:
    """Create lagged and seasonal features that can be reused across models."""
    features = quarterly.copy()
    features["lag_1"] = features["total_qty"].shift(1)
    features["lag_2"] = features["total_qty"].shift(2)
    features["lag_4"] = features["total_qty"].shift(4)

    # Rolling features only use prior quarters so they stay forecast-safe.
    lagged_qty = features["total_qty"].shift(1)
    features["rolling_mean_2"] = lagged_qty.rolling(2).mean()
    features["rolling_mean_4"] = lagged_qty.rolling(4).mean()

    for quarter_number in [2, 3, 4]:
        features[f"quarter_{quarter_number}"] = (features["quarter"] == quarter_number).astype(int)

    return features


def _fit_ridge_model(
    training_frame: pd.DataFrame,
    feature_columns: List[str],
    alpha: float = 3.0,
) -> Optional[RidgeModel]:
    """Fit a small regularized regression model without external ML libraries."""
    usable = training_frame.dropna(subset=feature_columns + ["total_qty"]).copy()
    if len(usable) < max(4, len(feature_columns)):
        return None

    x = usable[feature_columns].to_numpy(dtype=float)
    y = usable["total_qty"].to_numpy(dtype=float)

    feature_means = x.mean(axis=0)
    feature_scales = x.std(axis=0, ddof=0)
    feature_scales[feature_scales == 0] = 1.0

    x_scaled = (x - feature_means) / feature_scales
    target_mean = float(y.mean())
    y_centered = y - target_mean

    identity = np.eye(x_scaled.shape[1], dtype=float)
    coefficients = np.linalg.solve(x_scaled.T @ x_scaled + alpha * identity, x_scaled.T @ y_centered)

    return RidgeModel(
        feature_columns=feature_columns,
        feature_means=feature_means,
        feature_scales=feature_scales,
        target_mean=target_mean,
        coefficients=coefficients,
        alpha=alpha,
    )


def _predict_ridge(model: RidgeModel, row: pd.Series) -> float:
    values = row[model.feature_columns].to_numpy(dtype=float)
    scaled_values = (values - model.feature_means) / model.feature_scales
    return float(model.target_mean + scaled_values @ model.coefficients)


def model_coefficients(model: Optional[RidgeModel]) -> pd.DataFrame:
    if model is None:
        return pd.DataFrame(columns=["feature", "coefficient"])

    return (
        pd.DataFrame(
            {
                "feature": model.feature_columns,
                "coefficient": model.coefficients,
            }
        )
        .sort_values("coefficient", key=lambda series: series.abs(), ascending=False)
        .reset_index(drop=True)
    )


def _predict_all_models(training_frame: pd.DataFrame, forecast_row: pd.Series) -> Dict[str, float]:
    predictions: Dict[str, float] = {}

    if len(training_frame) == 0:
        return predictions

    predictions["last_quarter_naive"] = float(training_frame["total_qty"].iloc[-1])

    if len(training_frame) >= 4:
        predictions["seasonal_naive"] = float(training_frame["total_qty"].iloc[-4])
        predictions["moving_average_4q"] = float(training_frame["total_qty"].tail(4).mean())

    ridge_model = _fit_ridge_model(training_frame, REGRESSION_FEATURES)
    if ridge_model is not None and forecast_row[REGRESSION_FEATURES].notna().all():
        predictions["lagged_ridge_regression"] = _predict_ridge(ridge_model, forecast_row)

    return predictions


def expanding_window_backtest(quarterly_features: pd.DataFrame, minimum_history: int = 8) -> pd.DataFrame:
    """Run one-step-ahead backtesting so model quality is measured on past unseen quarters."""
    records: List[Dict[str, float]] = []

    for forecast_index in range(minimum_history, len(quarterly_features)):
        training_frame = quarterly_features.iloc[:forecast_index].copy()
        forecast_row = quarterly_features.iloc[forecast_index]
        predictions = _predict_all_models(training_frame, forecast_row)

        actual_qty = float(forecast_row["total_qty"])
        for model_name, predicted_qty in predictions.items():
            error = predicted_qty - actual_qty
            records.append(
                {
                    "financial_year": int(forecast_row["financial_year"]),
                    "quarter": int(forecast_row["quarter"]),
                    "quarter_label": forecast_row["quarter_label"],
                    "model": model_name,
                    "actual_qty": actual_qty,
                    "predicted_qty": float(predicted_qty),
                    "error": float(error),
                    "absolute_error": float(abs(error)),
                    "absolute_percentage_error": float(abs(error) / actual_qty * 100) if actual_qty else np.nan,
                }
            )

    return pd.DataFrame(records)


def summarise_backtest(backtest_predictions: pd.DataFrame) -> pd.DataFrame:
    if backtest_predictions.empty:
        return pd.DataFrame(columns=["model", "forecast_count", "mae", "rmse", "mape", "mean_error"])

    summary = (
        backtest_predictions.groupby("model")
        .agg(
            forecast_count=("model", "size"),
            mae=("absolute_error", "mean"),
            rmse=("error", lambda series: float(np.sqrt(np.mean(np.square(series))))),
            mape=("absolute_percentage_error", "mean"),
            mean_error=("error", "mean"),
        )
        .reset_index()
        .sort_values(["mae", "rmse", "mape"])
        .reset_index(drop=True)
    )
    return summary


def _build_future_row(quarterly: pd.DataFrame) -> pd.DataFrame:
    latest_row = quarterly.iloc[-1]
    next_period = _next_quarter(int(latest_row["financial_year"]), int(latest_row["quarter"]))

    future_row = pd.DataFrame(
        [
            {
                "financial_year": next_period["financial_year"],
                "quarter": next_period["quarter"],
                "total_qty": np.nan,
                "quarter_start": _quarter_start(next_period["financial_year"], next_period["quarter"]),
                "quarter_label": f"FY{next_period['financial_year']} Q{next_period['quarter']}",
                "period_index": int(latest_row["period_index"]) + 1,
            }
        ]
    )

    future_with_history = pd.concat([quarterly, future_row], ignore_index=True)
    future_with_features = add_forecasting_features(future_with_history)
    return future_with_features.tail(1)


def forecast_next_quarter(
    quarterly_features: pd.DataFrame,
    metrics: pd.DataFrame,
    backtest_predictions: pd.DataFrame,
    preferred_model: Optional[str] = None,
) -> Dict[str, object]:
    """Train on the full history and forecast the next business quarter."""
    future_row = _build_future_row(quarterly_features)

    if preferred_model is None:
        if metrics.empty:
            preferred_model = "moving_average_4q"
        else:
            preferred_model = str(metrics.iloc[0]["model"])

    all_predictions = _predict_all_models(quarterly_features, future_row.iloc[0])
    if preferred_model not in all_predictions:
        raise ValueError(f"Model '{preferred_model}' could not produce a forecast for the next quarter.")

    predicted_qty = float(all_predictions[preferred_model])
    model_backtest = backtest_predictions[backtest_predictions["model"] == preferred_model].copy()
    absolute_errors = model_backtest["absolute_error"].dropna()

    if absolute_errors.empty:
        interval_80 = 0.0
        interval_95 = 0.0
    else:
        interval_80 = float(absolute_errors.quantile(0.80))
        interval_95 = float(absolute_errors.quantile(0.95))

    full_model = None
    coefficient_table = pd.DataFrame(columns=["feature", "coefficient"])
    if preferred_model == "lagged_ridge_regression":
        full_model = _fit_ridge_model(quarterly_features, REGRESSION_FEATURES)
        coefficient_table = model_coefficients(full_model)

    forecast_summary = pd.DataFrame(
        [
            {
                "selected_model": preferred_model,
                "next_quarter_label": future_row.iloc[0]["quarter_label"],
                "predicted_qty": predicted_qty,
                "lower_80": max(predicted_qty - interval_80, 0.0),
                "upper_80": predicted_qty + interval_80,
                "lower_95": max(predicted_qty - interval_95, 0.0),
                "upper_95": predicted_qty + interval_95,
            }
        ]
    )

    return {
        "forecast_summary": forecast_summary,
        "future_row": future_row.reset_index(drop=True),
        "all_model_predictions": all_predictions,
        "coefficient_table": coefficient_table,
    }


def run_forecasting_workflow(
    csv_path: str = DEFAULT_CSV_PATH,
    minimum_history: int = 8,
    preferred_model: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Convenience wrapper used by the notebook and by command-line validation."""
    raw_df = load_meropenem_data(csv_path)
    quarterly = build_quarterly_series(raw_df)
    quarterly_features = add_forecasting_features(quarterly)
    backtest_predictions = expanding_window_backtest(quarterly_features, minimum_history=minimum_history)
    metrics = summarise_backtest(backtest_predictions)
    next_quarter = forecast_next_quarter(
        quarterly_features=quarterly_features,
        metrics=metrics,
        backtest_predictions=backtest_predictions,
        preferred_model=preferred_model,
    )

    return {
        "quarterly_features": quarterly_features,
        "backtest_predictions": backtest_predictions,
        "metrics": metrics,
        "forecast_summary": next_quarter["forecast_summary"],
        "future_row": next_quarter["future_row"],
        "coefficient_table": next_quarter["coefficient_table"],
        "next_quarter_model_comparison": pd.DataFrame(
            [
                {"model": model_name, "predicted_qty": predicted_qty}
                for model_name, predicted_qty in next_quarter["all_model_predictions"].items()
            ]
        ).sort_values("predicted_qty", ascending=False).reset_index(drop=True),
    }


def save_forecasting_outputs(
    results: Dict[str, pd.DataFrame],
    metrics_path: str = "forecast_q4_backtest_metrics.csv",
    predictions_path: str = "forecast_q4_backtest_predictions.csv",
    summary_path: str = "forecast_next_quarter_summary.csv",
) -> None:
    """Persist the most useful forecasting tables for reporting or reuse."""
    results["metrics"].round(2).to_csv(metrics_path, index=False)
    results["backtest_predictions"].round(2).to_csv(predictions_path, index=False)
    results["forecast_summary"].round(2).to_csv(summary_path, index=False)


if __name__ == "__main__":
    results = run_forecasting_workflow()
    save_forecasting_outputs(results)
    print("Backtest metrics:")
    print(results["metrics"].round(2).to_string(index=False))
    print("\nNext-quarter forecast:")
    print(results["forecast_summary"].round(2).to_string(index=False))
