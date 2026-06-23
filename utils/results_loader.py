"""Tez model sonuçlarını ve tahmin dosyalarını güvenli biçimde yükleyen araçlar."""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import Final

import pandas as pd


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
PREDICTIONS_DIRECTORY: Final = PROJECT_ROOT / "results" / "predictions"
EXPECTED_MODELS: Final = ("LSTM", "TFT", "Chronos")
EXPECTED_METRICS: Final = ("mae", "rmse", "r2", "direction_accuracy")
DETAILED_METRICS: Final = (
    "mse",
    "rmse",
    "mae",
    "mape",
    "r2",
    "direction_accuracy",
)
HYPERPARAMETER_FIELDS: Final = (
    "units",
    "hidden_size",
    "dropout",
    "batch_size",
    "epochs",
    "patience",
    "learning_rate",
    "num_heads",
)
MODEL_SETTING_FIELDS: Final = (
    "context_length",
    "forecast_horizon",
    "rolling_step",
    "lower_quantile",
    "upper_quantile",
)
PROJECT_FIELDS: Final = (
    "symbol",
    "input_window",
    "forecast_horizon",
    "target_type",
)
DISPLAY_NAMES: Final = {
    "LSTM": "LSTM",
    "TFT": "TFT Tabanlı Model",
    "Chronos": "Chronos",
}

__all__ = [
    "ResultsDataError",
    "load_interval_future_forecast_data",
    "load_interval_prediction_data",
]


class ResultsDataError(RuntimeError):
    """Sonuç dosyalarıyla ilgili kullanıcıya gösterilebilir hataları temsil eder."""


def _is_finite_number(value: object) -> bool:
    """Değerin bool olmayan sonlu bir sayı olup olmadığını denetler."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_valid_metric(metric_name: str, value: object) -> bool:
    """Bir metrik değerini kendi anlamlı değer aralığına göre doğrular."""
    if not _is_finite_number(value):
        return False

    numeric_value = float(value)
    if metric_name in {"mae", "rmse"}:
        return numeric_value >= 0
    if metric_name == "direction_accuracy":
        return 0 <= numeric_value <= 100
    return True


def _normalize_metric(metric_name: str, value: Real) -> float:
    """Karşılaştırmada kullanılacak metrik ölçeğini standartlaştırır."""
    numeric_value = float(value)
    if metric_name == "direction_accuracy" and numeric_value <= 1:
        return numeric_value * 100
    return numeric_value


def _sanitize_numeric_mapping(
    values: object,
    expected_fields: tuple[str, ...],
) -> dict:
    """Sayısal bir sözlüğü beklenen alanlarla güvenli biçimde sınırlar."""
    mapping = values if isinstance(values, dict) else {}
    return {
        field: mapping.get(field) if _is_finite_number(mapping.get(field)) else None
        for field in expected_fields
    }


def _resolve_prediction_file(file_path: str | Path) -> Path:
    """Göreli bir dosya adını güvenli tahmin klasörü içinde çözer."""
    if file_path is None or not str(file_path).strip():
        raise ResultsDataError("Tahmin dosyası yolu belirtilmedi.")

    requested_path = Path(str(file_path).strip())
    if requested_path.is_absolute():
        raise ResultsDataError("Mutlak tahmin dosyası yolları kabul edilmez.")
    if ".." in requested_path.parts:
        raise ResultsDataError(
            "Tahmin dosyası yolunda üst klasöre geçiş ('..') kullanılamaz."
        )

    predictions_root = PREDICTIONS_DIRECTORY.resolve()
    resolved_path = (predictions_root / requested_path).resolve()

    try:
        resolved_path.relative_to(predictions_root)
    except ValueError as error:
        raise ResultsDataError(
            "Tahmin dosyası yalnızca results/predictions klasöründen okunabilir."
        ) from error

    if not resolved_path.exists():
        standard_path = Path("results") / "predictions" / requested_path
        raise ResultsDataError(
            f"Standart tahmin dosyası bulunamadı: {standard_path.as_posix()}"
        )

    return resolved_path


def _read_prediction_csv(file_path: str | Path) -> pd.DataFrame:
    """Güvenli tahmin yolundaki CSV dosyasını okur."""
    resolved_path = _resolve_prediction_file(file_path)
    try:
        return pd.read_csv(resolved_path)
    except (OSError, pd.errors.ParserError, UnicodeError) as error:
        raise ResultsDataError(
            f"Tahmin dosyası okunamadı ({resolved_path.name}): {error}"
        ) from error


def load_model_results(file_path: str | Path) -> dict:
    """Model sonuçlarını UTF-8 JSON dosyasından okuyup doğrular."""
    path = Path(file_path)
    if not path.exists():
        raise ResultsDataError(f"Model sonuç dosyası bulunamadı: {path}")

    try:
        with path.open("r", encoding="utf-8") as results_file:
            results = json.load(results_file)
    except json.JSONDecodeError as error:
        raise ResultsDataError(
            f"Model sonuç dosyası geçerli JSON içermiyor ({path}): "
            f"{error.msg}, satır {error.lineno}"
        ) from error
    except (OSError, UnicodeError) as error:
        raise ResultsDataError(
            f"Model sonuç dosyası okunamadı ({path}): {error}"
        ) from error

    validate_results_schema(results)
    return results


def validate_results_schema(results: object) -> bool:
    """Sonuç sözlüğünün zorunlu proje ve model alanlarını doğrular."""
    if not isinstance(results, dict):
        raise ResultsDataError("Model sonuçlarının kök yapısı bir nesne olmalıdır.")

    project = results.get("project")
    models = results.get("models")

    if not isinstance(project, dict):
        raise ResultsDataError("Model sonuçlarında 'project' alanı eksik veya geçersiz.")
    if not isinstance(models, dict):
        raise ResultsDataError("Model sonuçlarında 'models' alanı eksik veya geçersiz.")

    missing_project_fields = [
        field for field in PROJECT_FIELDS if field not in project
    ]
    if missing_project_fields:
        raise ResultsDataError(
            "Proje bilgilerinde zorunlu alanlar eksik: "
            + ", ".join(missing_project_fields)
        )

    for model_key, model_result in models.items():
        if not isinstance(model_result, dict):
            raise ResultsDataError(
                f"{model_key} model kaydı bir nesne biçiminde olmalıdır."
            )

        metrics = model_result.get("metrics", {})
        if metrics is not None and not isinstance(metrics, dict):
            raise ResultsDataError(
                f"{model_key} modelinin 'metrics' alanı geçersiz."
            )

    return True


def get_model_result(results: dict, model_key: str) -> dict:
    """İstenen model kaydını eksik alanlar için güvenli varsayılanlarla döndürür."""
    models = results.get("models", {}) if isinstance(results, dict) else {}
    model_result = models.get(model_key)

    if not isinstance(model_result, dict):
        return {
            "display_name": DISPLAY_NAMES.get(model_key, model_key),
            "status": "Model kaydı bulunamadı",
            "metrics": {metric: None for metric in EXPECTED_METRICS},
            "test_metrics": {metric: None for metric in DETAILED_METRICS},
            "validation_metrics": {metric: None for metric in DETAILED_METRICS},
            "hyperparameters": {
                field: None for field in HYPERPARAMETER_FIELDS
            },
            "prediction_file": None,
            "future_forecast_file": None,
            "result_type": None,
            "generated_period": None,
            "inference_type": None,
            "model_id": None,
            "context_length": None,
            "model_settings": {},
            "notes": None,
        }

    metrics = model_result.get("metrics")
    safe_metrics = metrics if isinstance(metrics, dict) else {}

    return {
        "display_name": model_result.get("display_name")
        or DISPLAY_NAMES.get(model_key, model_key),
        "status": model_result.get("status") or "Durum bilgisi eklenmedi",
        "metrics": {
            metric: (
                safe_metrics.get(metric)
                if _is_valid_metric(metric, safe_metrics.get(metric))
                else None
            )
            for metric in EXPECTED_METRICS
        },
        "test_metrics": _sanitize_numeric_mapping(
            model_result.get("test_metrics"),
            DETAILED_METRICS,
        ),
        "validation_metrics": _sanitize_numeric_mapping(
            model_result.get("validation_metrics"),
            DETAILED_METRICS,
        ),
        "hyperparameters": _sanitize_numeric_mapping(
            model_result.get("hyperparameters"),
            HYPERPARAMETER_FIELDS,
        ),
        "prediction_file": model_result.get("prediction_file"),
        "future_forecast_file": model_result.get("future_forecast_file"),
        "result_type": model_result.get("result_type"),
        "generated_period": model_result.get("generated_period"),
        "inference_type": model_result.get("inference_type"),
        "model_id": model_result.get("model_id"),
        "context_length": (
            model_result.get("context_length")
            if _is_finite_number(model_result.get("context_length"))
            else None
        ),
        "model_settings": (
            model_result.get("model_settings")
            if isinstance(model_result.get("model_settings"), dict)
            else {}
        ),
        "notes": model_result.get("notes"),
    }


def get_available_models(results: dict) -> list[str]:
    """Sonuç dosyasında geçerli kayıt yapısı bulunan model anahtarlarını döndürür."""
    models = results.get("models", {}) if isinstance(results, dict) else {}
    return [
        model_key
        for model_key in EXPECTED_MODELS
        if isinstance(models.get(model_key), dict)
    ]


def load_prediction_data(file_path: str | Path) -> pd.DataFrame:
    """Tahmin CSV dosyasını güvenli klasörden okuyup sütunlarını doğrular."""
    prediction_data = _read_prediction_csv(file_path)

    normalized_columns = {
        str(column).strip().lower(): column for column in prediction_data.columns
    }
    required_columns = {"date": "Date", "actual": "Actual", "predicted": "Predicted"}
    missing_columns = [
        display_name
        for normalized_name, display_name in required_columns.items()
        if normalized_name not in normalized_columns
    ]
    if missing_columns:
        raise ResultsDataError(
            "Tahmin dosyasında zorunlu sütunlar eksik: "
            + ", ".join(missing_columns)
        )

    prepared_data = prediction_data.rename(
        columns={
            normalized_columns[normalized_name]: display_name
            for normalized_name, display_name in required_columns.items()
        }
    )[["Date", "Actual", "Predicted"]].copy()

    prepared_data["Date"] = pd.to_datetime(prepared_data["Date"], errors="coerce")
    prepared_data["Actual"] = pd.to_numeric(
        prepared_data["Actual"], errors="coerce"
    )
    prepared_data["Predicted"] = pd.to_numeric(
        prepared_data["Predicted"], errors="coerce"
    )

    if prepared_data.empty:
        raise ResultsDataError("Tahmin dosyası boş.")
    if prepared_data[["Date", "Actual", "Predicted"]].isna().any().any():
        raise ResultsDataError(
            "Tahmin dosyasında geçersiz tarih veya sayısal değer bulunuyor."
        )

    return prepared_data


def load_future_forecast_data(file_path: str | Path) -> pd.DataFrame:
    """Beş günlük sabit gelecek tahmin CSV dosyasını yükleyip doğrular."""
    forecast_data = _read_prediction_csv(file_path)
    normalized_columns = {
        str(column).strip().lower(): column for column in forecast_data.columns
    }
    required_columns = {
        "date": "Date",
        "predicted_log_return": "Predicted_Log_Return",
        "predicted_close": "Predicted_Close",
    }
    missing_columns = [
        display_name
        for normalized_name, display_name in required_columns.items()
        if normalized_name not in normalized_columns
    ]
    if missing_columns:
        raise ResultsDataError(
            "Gelecek tahmin dosyasında zorunlu sütunlar eksik: "
            + ", ".join(missing_columns)
        )

    prepared_data = forecast_data.rename(
        columns={
            normalized_columns[normalized_name]: display_name
            for normalized_name, display_name in required_columns.items()
        }
    )[["Date", "Predicted_Log_Return", "Predicted_Close"]].copy()
    prepared_data["Date"] = pd.to_datetime(prepared_data["Date"], errors="coerce")
    prepared_data["Predicted_Log_Return"] = pd.to_numeric(
        prepared_data["Predicted_Log_Return"], errors="coerce"
    )
    prepared_data["Predicted_Close"] = pd.to_numeric(
        prepared_data["Predicted_Close"], errors="coerce"
    )

    if prepared_data.empty:
        raise ResultsDataError("Gelecek tahmin dosyası boş.")
    if prepared_data.isna().any().any():
        raise ResultsDataError(
            "Gelecek tahmin dosyasında geçersiz tarih veya sayısal değer bulunuyor."
        )

    return prepared_data.sort_values("Date").reset_index(drop=True)


def load_interval_prediction_data(file_path: str | Path) -> pd.DataFrame:
    """Tahmin aralığı içeren rolling test CSV dosyasını yükleyip doğrular."""
    prediction_data = _read_prediction_csv(file_path)
    normalized_columns = {
        str(column).strip().lower(): column for column in prediction_data.columns
    }
    required_columns = {
        "date": "Date",
        "actual": "Actual",
        "predicted": "Predicted",
        "lower": "Lower",
        "upper": "Upper",
    }
    missing_columns = [
        display_name
        for normalized_name, display_name in required_columns.items()
        if normalized_name not in normalized_columns
    ]
    if missing_columns:
        raise ResultsDataError(
            "Tahmin aralıklı rolling test dosyasında zorunlu sütunlar eksik: "
            + ", ".join(missing_columns)
        )

    prepared_data = prediction_data.rename(
        columns={
            normalized_columns[normalized_name]: display_name
            for normalized_name, display_name in required_columns.items()
        }
    )[["Date", "Actual", "Predicted", "Lower", "Upper"]].copy()
    prepared_data["Date"] = pd.to_datetime(prepared_data["Date"], errors="coerce")
    for column_name in ("Actual", "Predicted", "Lower", "Upper"):
        prepared_data[column_name] = pd.to_numeric(
            prepared_data[column_name],
            errors="coerce",
        )

    if prepared_data.empty:
        raise ResultsDataError("Tahmin aralıklı rolling test dosyası boş.")
    if prepared_data.isna().any().any():
        raise ResultsDataError(
            "Tahmin aralıklı rolling test dosyasında geçersiz tarih veya "
            "sayısal değer bulunuyor."
        )
    if (prepared_data["Lower"] > prepared_data["Upper"]).any():
        raise ResultsDataError(
            "Tahmin aralıklı rolling test dosyasında alt sınırın üst sınırı "
            "aştığı satırlar bulunuyor."
        )

    return prepared_data.sort_values("Date").reset_index(drop=True)


def load_interval_future_forecast_data(file_path: str | Path) -> pd.DataFrame:
    """Tahmin aralığı içeren sabit gelecek tahmin CSV dosyasını doğrular."""
    forecast_data = _read_prediction_csv(file_path)
    normalized_columns = {
        str(column).strip().lower(): column for column in forecast_data.columns
    }
    required_columns = {
        "date": "Date",
        "predicted_close": "Predicted_Close",
        "lower": "Lower",
        "upper": "Upper",
    }
    missing_columns = [
        display_name
        for normalized_name, display_name in required_columns.items()
        if normalized_name not in normalized_columns
    ]
    if missing_columns:
        raise ResultsDataError(
            "Tahmin aralıklı gelecek tahmin dosyasında zorunlu sütunlar eksik: "
            + ", ".join(missing_columns)
        )

    prepared_data = forecast_data.rename(
        columns={
            normalized_columns[normalized_name]: display_name
            for normalized_name, display_name in required_columns.items()
        }
    )[["Date", "Predicted_Close", "Lower", "Upper"]].copy()
    prepared_data["Date"] = pd.to_datetime(prepared_data["Date"], errors="coerce")
    for column_name in ("Predicted_Close", "Lower", "Upper"):
        prepared_data[column_name] = pd.to_numeric(
            prepared_data[column_name],
            errors="coerce",
        )

    if prepared_data.empty:
        raise ResultsDataError("Tahmin aralıklı gelecek tahmin dosyası boş.")
    if prepared_data.isna().any().any():
        raise ResultsDataError(
            "Tahmin aralıklı gelecek tahmin dosyasında geçersiz tarih veya "
            "sayısal değer bulunuyor."
        )
    if (prepared_data["Lower"] > prepared_data["Upper"]).any():
        raise ResultsDataError(
            "Gelecek tahmin dosyasında alt sınırın üst sınırı aştığı "
            "satırlar bulunuyor."
        )

    return prepared_data.sort_values("Date").reset_index(drop=True)


def load_recent_close_data(file_path: str | Path = "recent_close.csv") -> pd.DataFrame:
    """Son dönem gerçek kapanış fiyatlarını güvenli tahmin klasöründen yükler."""
    recent_data = _read_prediction_csv(file_path)
    normalized_columns = {
        str(column).strip().lower(): column for column in recent_data.columns
    }
    required_columns = {"date": "Date", "close": "Close"}
    missing_columns = [
        display_name
        for normalized_name, display_name in required_columns.items()
        if normalized_name not in normalized_columns
    ]
    if missing_columns:
        raise ResultsDataError(
            "Geçmiş kapanış dosyasında zorunlu sütunlar eksik: "
            + ", ".join(missing_columns)
        )

    prepared_data = recent_data.rename(
        columns={
            normalized_columns[normalized_name]: display_name
            for normalized_name, display_name in required_columns.items()
        }
    )[["Date", "Close"]].copy()
    prepared_data["Date"] = pd.to_datetime(prepared_data["Date"], errors="coerce")
    prepared_data["Close"] = pd.to_numeric(
        prepared_data["Close"], errors="coerce"
    )

    if prepared_data.empty or prepared_data.isna().any().any():
        raise ResultsDataError(
            "Geçmiş kapanış dosyasında geçersiz tarih veya fiyat değeri bulunuyor."
        )

    return prepared_data.sort_values("Date").reset_index(drop=True)


def determine_best_model(
    results: dict,
    metric_name: str,
    higher_is_better: bool = False,
) -> dict:
    """En az iki geçerli sonuç varsa seçilen metriğe göre en iyi modeli belirler."""
    normalized_metric_name = metric_name.strip().lower()
    metric_aliases = {
        "mae": "mae",
        "rmse": "rmse",
        "r2": "r2",
        "r²": "r2",
        "direction_accuracy": "direction_accuracy",
        "yön doğruluğu": "direction_accuracy",
    }
    metric_key = metric_aliases.get(normalized_metric_name)

    if metric_key not in EXPECTED_METRICS:
        return {
            "success": False,
            "message": f"Desteklenmeyen karşılaştırma metriği: {metric_name}",
        }

    if metric_key in {"r2", "direction_accuracy"}:
        higher_is_better = True
    elif metric_key in {"mae", "rmse"}:
        higher_is_better = False

    candidates = []
    for model_key in EXPECTED_MODELS:
        model_result = get_model_result(results, model_key)
        value = model_result["metrics"].get(metric_key)
        if value is None:
            continue

        candidates.append(
            {
                "model_key": model_key,
                "display_name": model_result["display_name"],
                "value": _normalize_metric(metric_key, value),
            }
        )

    if len(candidates) < 2:
        return {
            "success": False,
            "message": "Karşılaştırma için yeterli model sonucu bulunmuyor.",
        }

    best_model = (
        max(candidates, key=lambda item: item["value"])
        if higher_is_better
        else min(candidates, key=lambda item: item["value"])
    )
    return {
        "success": True,
        "metric": metric_key,
        **best_model,
        "model_count": len(candidates),
    }
