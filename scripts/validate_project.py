"""Proje bütünlüğünü tek komutla doğrulayan betik."""

from __future__ import annotations

import importlib
import json
import math
import py_compile
import sys
import tempfile
from numbers import Real
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
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


class ProjectValidator:
    """Proje dosyaları için okunabilir Türkçe doğrulama raporu üretir."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.ok_count = 0

    def ok(self, message: str) -> None:
        """Başarılı kontrol mesajı yazdırır."""
        self.ok_count += 1
        print(f"[OK] {message}")

    def warn(self, message: str) -> None:
        """Kritik olmayan uyarı mesajı kaydeder."""
        self.warnings.append(message)
        print(f"[UYARI] {message}")

    def fail(self, message: str) -> None:
        """Kritik hata mesajı kaydeder."""
        self.errors.append(message)
        print(f"[HATA] {message}")

    def check_json(self) -> dict | None:
        """Merkezi model sonuç JSON dosyasını doğrular."""
        path = DATA_DIR / "model_results.json"
        try:
            results = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            self.fail(f"model_results.json okunamadı veya geçersiz: {error}")
            return None

        models = results.get("models")
        if not isinstance(models, dict):
            self.fail("model_results.json içinde models alanı geçersiz.")
            return None

        if tuple(models) != EXPECTED_MODELS:
            self.fail(
                "Aktif modeller beklenen sırada değil: "
                f"beklenen {EXPECTED_MODELS}, bulunan {tuple(models)}"
            )
        else:
            self.ok("model_results.json üç aktif modeli içeriyor.")

        lowered = json.dumps(results, ensure_ascii=False).lower()
        if "naive" in lowered or "baseline" in lowered:
            self.fail("Aktif sonuçlarda Naive/Baseline ifadesi bulunuyor.")
        else:
            self.ok("Naive Baseline aktif sonuçlarda bulunmuyor.")

        for model_key in EXPECTED_MODELS:
            metrics = models.get(model_key, {}).get("metrics", {})
            for metric_name in ("mae", "rmse", "r2", "direction_accuracy"):
                if not is_number(metrics.get(metric_name)):
                    self.fail(f"{model_key} için {metric_name} metriği sayısal değil.")
            self.ok(f"{model_key} ana metrikleri sayısal.")

        return results

    def read_csv(
        self,
        path: Path,
        required_columns: tuple[str, ...],
        numeric_columns: tuple[str, ...],
    ) -> pd.DataFrame | None:
        """Bir CSV dosyasını okuyup zorunlu tarih ve sayısal alanları doğrular."""
        if not path.exists():
            self.fail(f"CSV dosyası bulunamadı: {path.relative_to(PROJECT_ROOT)}")
            return None

        try:
            data = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeError) as error:
            self.fail(f"CSV dosyası okunamadı: {path.name} ({error})")
            return None

        if data.empty:
            self.fail(f"CSV dosyası boş: {path.name}")
            return None

        missing = [column for column in required_columns if column not in data.columns]
        if missing:
            self.fail(f"{path.name} eksik sütun içeriyor: {', '.join(missing)}")
            return None

        if "Date" in required_columns:
            invalid_dates = pd.to_datetime(data["Date"], errors="coerce").isna().sum()
            if invalid_dates:
                self.fail(f"{path.name} içinde geçersiz tarih var: {invalid_dates}")

        for column in numeric_columns:
            invalid_values = pd.to_numeric(data[column], errors="coerce").isna().sum()
            if invalid_values:
                self.fail(
                    f"{path.name} içinde {column} alanında geçersiz sayı var: "
                    f"{invalid_values}"
                )

        return data

    def check_prediction_csvs(self) -> None:
        """Standart prediction ve forecast CSV dosyalarını doğrular."""
        specs = {
            "lstm_5day_rolling_test_predictions.csv": (
                ("Date", "Actual", "Predicted"),
                ("Actual", "Predicted"),
                215,
                ("2025-02-18", "2025-12-23"),
            ),
            "tft_5day_rolling_test_predictions.csv": (
                ("Date", "Actual", "Predicted"),
                ("Actual", "Predicted"),
                215,
                ("2025-02-18", "2025-12-23"),
            ),
            "chronos_5day_rolling_test_predictions.csv": (
                ("Date", "Actual", "Predicted", "Lower", "Upper"),
                ("Actual", "Predicted", "Lower", "Upper"),
                225,
                ("2025-02-05", "2025-12-26"),
            ),
            "lstm_5day_future_forecast.csv": (
                ("Date", "Predicted_Log_Return", "Predicted_Close"),
                ("Predicted_Log_Return", "Predicted_Close"),
                5,
                None,
            ),
            "tft_5day_future_forecast.csv": (
                ("Date", "Predicted_Log_Return", "Predicted_Close"),
                ("Predicted_Log_Return", "Predicted_Close"),
                5,
                None,
            ),
            "chronos_5day_future_forecast.csv": (
                ("Date", "Predicted_Close", "Lower", "Upper"),
                ("Predicted_Close", "Lower", "Upper"),
                5,
                None,
            ),
            "recent_close.csv": (
                ("Date", "Close"),
                ("Close",),
                None,
                None,
            ),
        }

        for file_name, (required, numeric, expected_rows, expected_period) in specs.items():
            data = self.read_csv(PREDICTIONS_DIR / file_name, required, numeric)
            if data is None:
                continue

            if expected_rows is not None and len(data) != expected_rows:
                self.fail(
                    f"{file_name} satır sayısı beklenenden farklı: "
                    f"{len(data)} / {expected_rows}"
                )
            elif expected_rows is not None:
                self.ok(f"{file_name} satır sayısı doğru: {expected_rows}")

            if expected_period is not None:
                first_date, last_date = expected_period
                if data["Date"].iloc[0] != first_date or data["Date"].iloc[-1] != last_date:
                    self.fail(f"{file_name} tarih aralığı beklenenle eşleşmiyor.")
                else:
                    self.ok(f"{file_name} tarih aralığı korundu.")

            if file_name.endswith("future_forecast.csv"):
                if data["Date"].tolist() != EXPECTED_FUTURE_DATES:
                    self.fail(f"{file_name} gelecek tahmin tarihleri değişmiş.")
                else:
                    self.ok(f"{file_name} beş kaynak tahmin tarihini koruyor.")

            if {"Lower", "Upper"}.issubset(data.columns):
                if (pd.to_numeric(data["Lower"]) > pd.to_numeric(data["Upper"])).any():
                    self.fail(f"{file_name} içinde Lower > Upper olan satır var.")
                else:
                    self.ok(f"{file_name} tahmin aralığı sınırları geçerli.")

    def check_metric_csvs(self) -> None:
        """Standart metrik ve ayar CSV dosyalarını doğrular."""
        metric_files = (
            "lstm_5day_test_metrics.csv",
            "lstm_5day_validation_metrics.csv",
            "tft_5day_test_metrics.csv",
            "tft_5day_validation_metrics.csv",
            "chronos_5day_test_metrics.csv",
        )
        metric_columns = ("MSE", "RMSE", "MAE", "MAPE", "R2", "Direction_Accuracy")
        for file_name in metric_files:
            data = self.read_csv(METRICS_DIR / file_name, metric_columns, metric_columns)
            if data is not None:
                self.ok(f"{file_name} metrik sütunları geçerli.")

        settings_path = METRICS_DIR / "chronos_5day_settings.csv"
        settings_columns = (
            "model_id",
            "inference_type",
            "context_length",
            "forecast_horizon",
            "rolling_step",
            "central_forecast",
            "lower_quantile",
            "upper_quantile",
            "training_performed",
        )
        settings = self.read_csv(settings_path, settings_columns, ())
        if settings is not None:
            self.ok("chronos_5day_settings.csv ayar sütunları geçerli.")

    def check_market_data(self) -> None:
        """Yerel AAPL piyasa verisini doğrular."""
        market_columns = ("Date", "Open", "High", "Low", "Close", "Volume")
        data = self.read_csv(
            DATA_DIR / "aapl_history.csv",
            market_columns,
            ("Open", "High", "Low", "Close", "Volume"),
        )
        if data is not None:
            self.ok("Yerel AAPL piyasa verisi okunabilir ve geçerli.")

    def check_png_files(self) -> None:
        """Gerekli tez grafik PNG dosyalarının durumunu raporlar."""
        required_plots = (
            "lstm_5day_loss.png",
            "lstm_5day_rolling_test.png",
            "lstm_5day_future_forecast.png",
            "tft_5day_loss.png",
            "tft_5day_rolling_test.png",
            "tft_5day_future_forecast.png",
            "chronos_5day_rolling_test.png",
            "chronos_5day_future_forecast.png",
        )
        for file_name in required_plots:
            path = PLOTS_DIR / file_name
            if not path.exists():
                self.fail(f"PNG dosyası bulunamadı: {file_name}")
            elif path.stat().st_size <= 0:
                self.fail(f"PNG dosyası boş görünüyor: {file_name}")
            else:
                self.ok(f"PNG dosyası mevcut: {file_name}")

    def check_python_syntax(self) -> None:
        """Tüm proje Python dosyalarında sözdizimi kontrolü yapar."""
        sys.pycache_prefix = str(Path(tempfile.gettempdir()) / "capital_market_pycache")
        python_files = [
            path
            for path in PROJECT_ROOT.rglob("*.py")
            if ".venv" not in path.parts and "__pycache__" not in path.parts
        ]
        for path in python_files:
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as error:
                self.fail(f"Sözdizimi hatası: {path.relative_to(PROJECT_ROOT)} ({error})")
        self.ok(f"{len(python_files)} Python dosyasında sözdizimi kontrolü tamamlandı.")

    def check_imports(self) -> None:
        """Temel yardımcı modülleri ve sonuç yükleyici fonksiyonları doğrular."""
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        module_names = (
            "utils.data_loader",
            "utils.chart_utils",
            "utils.results_loader",
            "utils.ui_components",
            "utils.presentation_mode",
        )
        for module_name in module_names:
            try:
                importlib.import_module(module_name)
            except Exception as error:  # noqa: BLE001 - doğrulama raporu için genel yakalama
                self.fail(f"Import başarısız: {module_name} ({error})")
            else:
                self.ok(f"Import başarılı: {module_name}")

        try:
            results_loader = importlib.import_module("utils.results_loader")
        except Exception:
            return

        for function_name in (
            "load_prediction_data",
            "load_future_forecast_data",
            "load_interval_prediction_data",
            "load_interval_future_forecast_data",
        ):
            if callable(getattr(results_loader, function_name, None)):
                self.ok(f"Fonksiyon mevcut: {function_name}")
            else:
                self.fail(f"Fonksiyon eksik: {function_name}")

    def run(self) -> int:
        """Tüm kontrolleri çalıştırır ve süreç çıkış kodunu döndürür."""
        print("Proje bütünlük kontrolü başlatıldı.\n")
        self.check_json()
        self.check_market_data()
        self.check_prediction_csvs()
        self.check_metric_csvs()
        self.check_png_files()
        self.check_python_syntax()
        self.check_imports()

        print("\nÖzet")
        print(f"Başarılı kontrol sayısı: {self.ok_count}")
        print(f"Uyarı sayısı: {len(self.warnings)}")
        print(f"Kritik hata sayısı: {len(self.errors)}")

        if self.errors:
            print("Proje bütünlük kontrolü kritik hatalarla tamamlandı.")
            return 1

        print("Proje bütünlük kontrolü başarıyla tamamlandı.")
        return 0


if __name__ == "__main__":
    raise SystemExit(ProjectValidator().run())
