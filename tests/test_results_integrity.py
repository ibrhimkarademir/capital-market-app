"""Aktif tez sonuçlarının bütünlüğünü doğrulayan testler."""

from __future__ import annotations

import json
import math
import unittest
from numbers import Real
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_ROOT / "data" / "model_results.json"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
EXPECTED_MODELS = ("LSTM", "TFT", "Chronos")
EXPECTED_FUTURE_DATES = [
    "2025-12-31",
    "2026-01-01",
    "2026-01-02",
    "2026-01-05",
    "2026-01-06",
]


def is_number(value: object) -> bool:
    """Bool olmayan sonlu sayıları doğrular."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


class ResultsIntegrityTests(unittest.TestCase):
    """Model sonuç dosyalarının beklenen akademik kapsamı koruduğunu test eder."""

    @classmethod
    def setUpClass(cls) -> None:
        """Merkezi sonuç dosyasını test sınıfı için yükler."""
        cls.results_text = RESULTS_FILE.read_text(encoding="utf-8")
        cls.results = json.loads(cls.results_text)

    def test_model_results_json_is_valid_and_expected(self) -> None:
        """JSON geçerli olmalı ve yalnızca üç tez modelini içermelidir."""
        self.assertIn("project", self.results)
        self.assertIn("models", self.results)
        self.assertEqual(tuple(self.results["models"]), EXPECTED_MODELS)

    def test_no_naive_or_baseline_in_active_results(self) -> None:
        """Naive/Baseline yöntemleri aktif sonuçlarda yer almamalıdır."""
        lowered = self.results_text.lower()
        self.assertNotIn("naive", lowered)
        self.assertNotIn("baseline", lowered)

    def test_main_metrics_are_numeric(self) -> None:
        """Üç modelin ana karşılaştırma metrikleri sayısal olmalıdır."""
        for model_key in EXPECTED_MODELS:
            metrics = self.results["models"][model_key]["metrics"]
            for metric_name in ("mae", "rmse", "r2", "direction_accuracy"):
                with self.subTest(model=model_key, metric=metric_name):
                    self.assertTrue(is_number(metrics[metric_name]))

    def test_rolling_prediction_row_counts(self) -> None:
        """Rolling test satır sayıları tez çıktılarıyla aynı kalmalıdır."""
        expected_rows = {
            "lstm_5day_rolling_test_predictions.csv": 215,
            "tft_5day_rolling_test_predictions.csv": 215,
            "chronos_5day_rolling_test_predictions.csv": 225,
        }
        for file_name, row_count in expected_rows.items():
            with self.subTest(file=file_name):
                data = pd.read_csv(PREDICTIONS_DIR / file_name)
                self.assertEqual(len(data), row_count)

    def test_required_prediction_and_forecast_files_exist(self) -> None:
        """Aktif modellerin prediction ve forecast dosyaları mevcut olmalıdır."""
        for model_key in EXPECTED_MODELS:
            model = self.results["models"][model_key]
            for field_name in ("prediction_file", "future_forecast_file"):
                with self.subTest(model=model_key, field=field_name):
                    self.assertTrue((PREDICTIONS_DIR / model[field_name]).is_file())

    def test_chronos_interval_columns_and_bounds(self) -> None:
        """Chronos tahmin aralığı sütunları ve sınırları geçerli olmalıdır."""
        for file_name in (
            "chronos_5day_rolling_test_predictions.csv",
            "chronos_5day_future_forecast.csv",
        ):
            with self.subTest(file=file_name):
                data = pd.read_csv(PREDICTIONS_DIR / file_name)
                self.assertIn("Lower", data.columns)
                self.assertIn("Upper", data.columns)
                self.assertFalse((data["Lower"] > data["Upper"]).any())

    def test_future_forecast_files_have_five_source_dates(self) -> None:
        """Üç modelin beş günlük tahmin tarihleri kaynak tez çıktılarıyla eşleşmelidir."""
        expected_files = (
            "lstm_5day_future_forecast.csv",
            "tft_5day_future_forecast.csv",
            "chronos_5day_future_forecast.csv",
        )
        for file_name in expected_files:
            with self.subTest(file=file_name):
                data = pd.read_csv(PREDICTIONS_DIR / file_name)
                self.assertEqual(len(data), 5)
                self.assertEqual(data["Date"].tolist(), EXPECTED_FUTURE_DATES)

    def test_rolling_prediction_source_dates(self) -> None:
        """Rolling test tarih aralıkları kırpılmadan korunmalıdır."""
        expected_periods = {
            "lstm_5day_rolling_test_predictions.csv": (
                "2025-02-18",
                "2025-12-23",
            ),
            "tft_5day_rolling_test_predictions.csv": (
                "2025-02-18",
                "2025-12-23",
            ),
            "chronos_5day_rolling_test_predictions.csv": (
                "2025-02-05",
                "2025-12-26",
            ),
        }
        for file_name, (first_date, last_date) in expected_periods.items():
            with self.subTest(file=file_name):
                data = pd.read_csv(PREDICTIONS_DIR / file_name)
                self.assertEqual(data["Date"].iloc[0], first_date)
                self.assertEqual(data["Date"].iloc[-1], last_date)


if __name__ == "__main__":
    unittest.main()
