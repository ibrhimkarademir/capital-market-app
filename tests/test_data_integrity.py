"""Piyasa verisi ve standart CSV dosyalarının temel bütünlük testleri."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


class DataIntegrityTests(unittest.TestCase):
    """Yerel veri ve standart sonuç CSV dosyalarını doğrular."""

    def test_aapl_history_is_readable_and_valid(self) -> None:
        """AAPL geçmiş veri dosyası okunabilir ve zorunlu alanları içerir."""
        data = pd.read_csv(DATA_DIR / "aapl_history.csv")
        required_columns = {"Date", "Open", "High", "Low", "Close", "Volume"}
        self.assertFalse(data.empty)
        self.assertTrue(required_columns.issubset(data.columns))
        self.assertFalse(pd.to_datetime(data["Date"], errors="coerce").isna().any())

        for column in required_columns - {"Date"}:
            numeric_values = pd.to_numeric(data[column], errors="coerce")
            self.assertFalse(numeric_values.isna().all(), column)

    def test_recent_close_if_present(self) -> None:
        """Son kapanış yedeği varsa Date ve Close sütunları geçerli olmalıdır."""
        recent_close_path = PREDICTIONS_DIR / "recent_close.csv"
        if not recent_close_path.exists():
            self.skipTest("recent_close.csv mevcut değil.")

        data = pd.read_csv(recent_close_path)
        self.assertTrue({"Date", "Close"}.issubset(data.columns))
        self.assertFalse(data.empty)
        self.assertFalse(pd.to_datetime(data["Date"], errors="coerce").isna().any())
        self.assertFalse(pd.to_numeric(data["Close"], errors="coerce").isna().any())

    def test_standard_prediction_csv_columns_and_numbers(self) -> None:
        """Standart prediction CSV dosyalarında zorunlu sütunlar sayısal olmalıdır."""
        specs = {
            "lstm_5day_rolling_test_predictions.csv": (
                ("Date", "Actual", "Predicted"),
                ("Actual", "Predicted"),
            ),
            "tft_5day_rolling_test_predictions.csv": (
                ("Date", "Actual", "Predicted"),
                ("Actual", "Predicted"),
            ),
            "chronos_5day_rolling_test_predictions.csv": (
                ("Date", "Actual", "Predicted", "Lower", "Upper"),
                ("Actual", "Predicted", "Lower", "Upper"),
            ),
            "lstm_5day_future_forecast.csv": (
                ("Date", "Predicted_Log_Return", "Predicted_Close"),
                ("Predicted_Log_Return", "Predicted_Close"),
            ),
            "tft_5day_future_forecast.csv": (
                ("Date", "Predicted_Log_Return", "Predicted_Close"),
                ("Predicted_Log_Return", "Predicted_Close"),
            ),
            "chronos_5day_future_forecast.csv": (
                ("Date", "Predicted_Close", "Lower", "Upper"),
                ("Predicted_Close", "Lower", "Upper"),
            ),
        }

        for file_name, (required_columns, numeric_columns) in specs.items():
            with self.subTest(file=file_name):
                data = pd.read_csv(PREDICTIONS_DIR / file_name)
                self.assertFalse(data.empty)
                self.assertTrue(set(required_columns).issubset(data.columns))
                self.assertFalse(
                    pd.to_datetime(data["Date"], errors="coerce").isna().any()
                )
                for column in numeric_columns:
                    values = pd.to_numeric(data[column], errors="coerce")
                    self.assertFalse(values.isna().any(), column)

    def test_standard_metric_csv_columns_and_numbers(self) -> None:
        """Standart metrik CSV dosyaları beklenen sayısal alanları içermelidir."""
        metric_files = (
            "lstm_5day_test_metrics.csv",
            "lstm_5day_validation_metrics.csv",
            "tft_5day_test_metrics.csv",
            "tft_5day_validation_metrics.csv",
            "chronos_5day_test_metrics.csv",
        )
        required_columns = {"MSE", "RMSE", "MAE", "MAPE", "R2", "Direction_Accuracy"}
        for file_name in metric_files:
            with self.subTest(file=file_name):
                data = pd.read_csv(METRICS_DIR / file_name)
                self.assertFalse(data.empty)
                self.assertTrue(required_columns.issubset(data.columns))
                for column in required_columns:
                    self.assertFalse(
                        pd.to_numeric(data[column], errors="coerce").isna().any(),
                        column,
                    )


if __name__ == "__main__":
    unittest.main()
