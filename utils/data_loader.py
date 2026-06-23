"""Finansal verilerin yüklenmesi, doğrulanması ve hazırlanması için araçlar."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import gettempdir
from typing import Final

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


DEFAULT_SYMBOL: Final = "AAPL"
YFINANCE_CACHE_PATH: Final = Path(gettempdir()) / "capital_market_ai_yfinance"
REQUIRED_COLUMNS: Final = ("Date", "Open", "High", "Low", "Close", "Volume")
OPTIONAL_COLUMNS: Final = ("Adj Close",)
NUMERIC_COLUMNS: Final = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
INDICATOR_COLUMNS: Final = (
    "Daily_Return",
    "Log_Return",
    "MA_20",
    "MA_50",
    "Volatility_20",
    "Price_Change",
)

_CANONICAL_COLUMN_NAMES: Final = {
    "date": "Date",
    "datetime": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Adj Close",
    "adjclose": "Adj Close",
    "volume": "Volume",
}


class MarketDataError(RuntimeError):
    """Piyasa verisi işlemlerindeki kullanıcıya gösterilebilir hataları temsil eder."""


def _canonicalize_column_name(column: object) -> str:
    """Tek veya çok seviyeli bir sütun adını standart piyasa alanına dönüştürür."""
    if isinstance(column, tuple):
        parts = [
            str(part).strip()
            for part in column
            if str(part).strip() and not str(part).startswith("Unnamed")
        ]
    else:
        parts = [str(column).strip()]

    for part in parts:
        canonical_name = _CANONICAL_COLUMN_NAMES.get(part.lower())
        if canonical_name:
            return canonical_name

    return parts[0] if parts else ""


def _standardize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Sütunları ve olası tarih indeksini tek seviyeli standart yapıya getirir."""
    data = dataframe.copy()

    has_date_column = any(
        _canonicalize_column_name(column) == "Date" for column in data.columns
    )
    if not has_date_column:
        data = data.reset_index()

    data.columns = [_canonicalize_column_name(column) for column in data.columns]
    data = data.loc[:, ~data.columns.duplicated(keep="first")]
    return data


def _normalize_date(value: date | datetime | str, field_name: str) -> pd.Timestamp:
    """Tarih girdisini saat diliminden arındırılmış pandas zaman damgasına dönüştürür."""
    try:
        normalized = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise MarketDataError(f"{field_name} geçerli bir tarih olmalıdır.") from error

    if normalized.tzinfo is not None:
        normalized = normalized.tz_localize(None)

    return normalized.normalize()


def _filter_date_range(
    dataframe: pd.DataFrame,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
) -> pd.DataFrame:
    """Veriyi verilen başlangıç ve bitiş tarihleri arasında süzer."""
    start = _normalize_date(start_date, "Başlangıç tarihi")
    end = _normalize_date(end_date, "Bitiş tarihi")

    if start > end:
        raise MarketDataError("Başlangıç tarihi bitiş tarihinden sonra olamaz.")

    filtered_data = dataframe.loc[
        dataframe["Date"].between(start, end, inclusive="both")
    ].copy()

    if filtered_data.empty:
        raise MarketDataError(
            "Seçilen tarih aralığında kullanılabilir piyasa verisi bulunamadı."
        )

    return filtered_data.reset_index(drop=True)


def _configure_yfinance_cache() -> None:
    """yfinance önbelleğini proje içindeki yazılabilir klasöre yönlendirir."""
    try:
        YFINANCE_CACHE_PATH.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(YFINANCE_CACHE_PATH))
    except OSError as error:
        raise MarketDataError(
            f"Yahoo Finance önbellek klasörü hazırlanamadı: {error}"
        ) from error


@st.cache_data(ttl=3600, show_spinner=False)
def download_market_data(
    symbol: str = DEFAULT_SYMBOL,
    start_date: date | datetime | str = "2020-01-01",
    end_date: date | datetime | str | None = None,
) -> pd.DataFrame:
    """Yahoo Finance üzerinden belirtilen tarih aralığındaki piyasa verisini indirir."""
    normalized_symbol = symbol.strip().upper() if symbol else DEFAULT_SYMBOL
    start = _normalize_date(start_date, "Başlangıç tarihi")
    end = _normalize_date(end_date or date.today(), "Bitiş tarihi")

    if start > end:
        raise MarketDataError("Başlangıç tarihi bitiş tarihinden sonra olamaz.")

    _configure_yfinance_cache()

    # yfinance bitiş tarihini hariç tuttuğu için seçilen son güne bir gün eklenir.
    exclusive_end = end + timedelta(days=1)

    try:
        downloaded_data = yf.download(
            tickers=normalized_symbol,
            start=start.strftime("%Y-%m-%d"),
            end=exclusive_end.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )
    except Exception as error:
        raise MarketDataError(
            f"{normalized_symbol} verileri Yahoo Finance üzerinden indirilemedi: "
            f"{error}"
        ) from error

    if downloaded_data is None or downloaded_data.empty:
        raise MarketDataError(
            f"Yahoo Finance, {normalized_symbol} için seçilen tarih aralığında "
            "veri döndürmedi."
        )

    return _standardize_columns(downloaded_data)


def load_local_data(file_path: str | Path) -> pd.DataFrame:
    """Yerel CSV dosyasını okur, hazırlar ve doğrular."""
    path = Path(file_path)

    if not path.exists():
        raise MarketDataError(f"Yerel veri dosyası bulunamadı: {path}")

    try:
        local_data = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError) as error:
        raise MarketDataError(
            f"Yerel veri dosyası okunamadı ({path}): {error}"
        ) from error

    try:
        return prepare_market_data(local_data)
    except MarketDataError as error:
        raise MarketDataError(
            f"Yerel veri dosyası geçerli piyasa verisi içermiyor: {error}"
        ) from error


def save_local_data(dataframe: pd.DataFrame, file_path: str | Path) -> None:
    """Piyasa verisini güvenli biçimde yerel CSV dosyasına kaydeder."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data_to_save = dataframe.copy()
    if "Date" in data_to_save.columns:
        data_to_save["Date"] = pd.to_datetime(
            data_to_save["Date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    temporary_path = path.with_suffix(f"{path.suffix}.tmp")

    try:
        data_to_save.to_csv(temporary_path, index=False, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as error:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)
        raise MarketDataError(
            f"Piyasa verisi yerel dosyaya kaydedilemedi ({path}): {error}"
        ) from error


def validate_market_data(dataframe: pd.DataFrame) -> bool:
    """Piyasa verisinin zorunlu alanlarını ve kullanılabilirliğini doğrular."""
    if not isinstance(dataframe, pd.DataFrame):
        raise MarketDataError("Piyasa verisi pandas DataFrame biçiminde olmalıdır.")

    if dataframe.empty:
        raise MarketDataError("Piyasa verisi boş.")

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        raise MarketDataError(
            "Piyasa verisinde zorunlu sütunlar eksik: "
            + ", ".join(missing_columns)
        )

    if dataframe[list(REQUIRED_COLUMNS)].isna().any(axis=1).all():
        raise MarketDataError(
            "Piyasa verisinde zorunlu alanları eksiksiz olan satır bulunamadı."
        )

    return True


def prepare_market_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ham piyasa verisini temizler ve teknik göstergelerle zenginleştirir."""
    if not isinstance(dataframe, pd.DataFrame):
        raise MarketDataError("Hazırlanacak veri pandas DataFrame biçiminde olmalıdır.")

    if dataframe.empty:
        raise MarketDataError("Hazırlanacak piyasa verisi boş.")

    data = _standardize_columns(dataframe)
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise MarketDataError(
            "Piyasa verisinde zorunlu sütunlar eksik: "
            + ", ".join(missing_columns)
        )

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce", utc=True)
    data["Date"] = data["Date"].dt.tz_convert(None).dt.normalize()

    for column in NUMERIC_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    available_numeric_columns = [
        column for column in NUMERIC_COLUMNS if column in data.columns
    ]
    data[available_numeric_columns] = data[available_numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.dropna(subset=list(REQUIRED_COLUMNS))
    data = data.loc[
        (data[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
        & (data["Volume"] >= 0)
        & (data["High"] >= data["Low"])
    ].copy()

    data = (
        data.sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )

    validate_market_data(data)

    data["Daily_Return"] = data["Close"].pct_change(fill_method=None)
    data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
    data["MA_20"] = data["Close"].rolling(window=20, min_periods=20).mean()
    data["MA_50"] = data["Close"].rolling(window=50, min_periods=50).mean()
    data["Volatility_20"] = data["Log_Return"].rolling(
        window=20, min_periods=20
    ).std()
    data["Price_Change"] = data["Close"].diff().abs()

    data[list(INDICATOR_COLUMNS)] = data[list(INDICATOR_COLUMNS)].replace(
        [np.inf, -np.inf], np.nan
    )

    ordered_columns = [
        *REQUIRED_COLUMNS,
        *[column for column in OPTIONAL_COLUMNS if column in data.columns],
        *INDICATOR_COLUMNS,
    ]
    remaining_columns = [
        column for column in data.columns if column not in ordered_columns
    ]
    return data[[*ordered_columns, *remaining_columns]]


def load_market_data(
    symbol: str = DEFAULT_SYMBOL,
    start_date: date | datetime | str = "2020-01-01",
    end_date: date | datetime | str | None = None,
    local_file_path: str | Path = "data/aapl_history.csv",
) -> tuple[pd.DataFrame, str]:
    """Önce Yahoo Finance verisini, başarısızlıkta yerel CSV yedeğini yükler."""
    resolved_end_date = end_date or date.today()
    online_error: MarketDataError | None = None

    try:
        online_data = download_market_data(symbol, start_date, resolved_end_date)
        prepared_online_data = prepare_market_data(online_data)
        selected_online_data = _filter_date_range(
            prepared_online_data, start_date, resolved_end_date
        )
        save_local_data(prepared_online_data, local_file_path)
        return selected_online_data, "Yahoo Finance"
    except MarketDataError as error:
        online_error = error

    try:
        local_data = load_local_data(local_file_path)
        selected_local_data = _filter_date_range(
            local_data, start_date, resolved_end_date
        )
        return selected_local_data, "Yerel CSV"
    except MarketDataError as local_error:
        raise MarketDataError(
            "Piyasa verisi yüklenemedi. "
            f"Yahoo Finance hatası: {online_error} "
            f"Yerel CSV hatası: {local_error}"
        ) from local_error
