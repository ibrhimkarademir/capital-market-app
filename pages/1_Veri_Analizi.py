"""AAPL geçmiş piyasa verilerinin analiz edildiği Streamlit sayfası."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.chart_utils import (
    create_price_chart,
    create_return_chart,
    create_volatility_chart,
    create_volume_chart,
)
from utils.data_loader import (
    DEFAULT_SYMBOL,
    MarketDataError,
    download_market_data,
    load_local_data,
    load_market_data,
)
from utils.presentation_mode import (
    apply_presentation_mode_styles,
    render_presentation_toggle,
)
from utils.ui_components import load_css


LOCAL_DATA_PATH = Path("data/aapl_history.csv")
STYLE_PATH = Path(__file__).resolve().parents[1] / "assets" / "style.css"
DEFAULT_START_DATE = date(2020, 1, 1)

load_css(STYLE_PATH)
render_presentation_toggle()
apply_presentation_mode_styles()


def get_default_end_date(local_file_path: Path) -> date:
    """Varsayılan bitiş tarihini yerel veriden veya sistem tarihinden belirler."""
    if not local_file_path.exists():
        return date.today()

    try:
        local_data = load_local_data(local_file_path)
    except MarketDataError:
        return date.today()

    latest_local_date = pd.to_datetime(
        local_data["Date"], errors="coerce"
    ).max()
    if pd.isna(latest_local_date):
        return date.today()

    return min(latest_local_date.date(), date.today())


def format_currency(value: float) -> str:
    """Sayısal değeri ABD doları biçiminde gösterir."""
    return f"{value:,.2f} USD" if pd.notna(value) else "Veri yok"


def format_percentage(value: float) -> str:
    """Ondalık oranı yüzde biçiminde gösterir."""
    return f"%{value * 100:,.2f}" if pd.notna(value) else "Veri yok"


def format_volume(value: float) -> str:
    """İşlem hacmini okunabilir tam sayı biçiminde gösterir."""
    return f"{value:,.0f}" if pd.notna(value) else "Veri yok"


st.title("Piyasa Verilerinin Analizi")
st.write(
    "Bu sayfa, AAPL geçmiş piyasa verilerini, getirilerini, işlem hacmini "
    "ve volatilitesini incelemek için kullanılmaktadır."
)

default_end_date = get_default_end_date(LOCAL_DATA_PATH)

with st.sidebar:
    st.subheader("Veri Seçimi")
    selected_symbol = st.selectbox(
        "Hisse sembolü",
        options=[DEFAULT_SYMBOL],
        index=0,
        help="Bu çalışma kapsamında yalnızca Apple Inc. (AAPL) kullanılmaktadır.",
    )
    selected_start_date = st.date_input(
        "Başlangıç tarihi",
        value=DEFAULT_START_DATE,
        max_value=date.today(),
        format="DD/MM/YYYY",
    )
    selected_end_date = st.date_input(
        "Bitiş tarihi",
        value=default_end_date,
        max_value=date.today(),
        format="DD/MM/YYYY",
    )
    refresh_data = st.button("Verileri Yenile", width="stretch")

if refresh_data:
    download_market_data.clear()
    st.rerun()

if selected_start_date > selected_end_date:
    st.error("Başlangıç tarihi bitiş tarihinden sonra olamaz.")
    st.stop()

try:
    with st.spinner("AAPL piyasa verileri yükleniyor..."):
        market_data, data_source = load_market_data(
            symbol=selected_symbol,
            start_date=selected_start_date,
            end_date=selected_end_date,
            local_file_path=LOCAL_DATA_PATH,
        )
except MarketDataError as error:
    st.error(str(error))
    st.info(
        "İnternet bağlantısını ve yerel veri dosyasını kontrol ettikten sonra "
        "“Verileri Yenile” düğmesini kullanabilirsiniz."
    )
    st.warning(
        "Gösterilen veriler ve analizler yalnızca akademik amaçlıdır ve "
        "yatırım tavsiyesi niteliği taşımaz."
    )
    st.stop()

st.caption(f"Veri kaynağı: {data_source}")

if len(market_data) < 50:
    st.info(
        "Seçilen aralık 50 işlem gününden az veri içeriyor. Bu nedenle bazı "
        "hareketli ortalama ve volatilite değerleri eksik olabilir."
    )

latest_close = market_data["Close"].iloc[-1]
first_close = market_data["Close"].iloc[0]
total_change = (
    latest_close / first_close - 1
    if pd.notna(first_close) and first_close != 0
    else np.nan
)
average_volume = market_data["Volume"].mean()
annualized_volatility = market_data["Log_Return"].std() * np.sqrt(252)

close_card, change_card, volume_card, volatility_card = st.columns(4)
close_card.metric("Son Kapanış Fiyatı", format_currency(latest_close))
change_card.metric("Dönemsel Toplam Değişim", format_percentage(total_change))
volume_card.metric("Ortalama Günlük İşlem Hacmi", format_volume(average_volume))
volatility_card.metric(
    "Yıllıklandırılmış Volatilite",
    format_percentage(annualized_volatility),
)

price_tab, volume_tab, return_tab, volatility_tab, raw_data_tab = st.tabs(
    [
        "Fiyat ve Hareketli Ortalamalar",
        "İşlem Hacmi",
        "Getiri",
        "Volatilite",
        "Ham Veri",
    ]
)

with price_tab:
    st.plotly_chart(
        create_price_chart(market_data),
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )

with volume_tab:
    st.plotly_chart(
        create_volume_chart(market_data),
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )

with return_tab:
    st.plotly_chart(
        create_return_chart(market_data),
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )

with volatility_tab:
    st.plotly_chart(
        create_volatility_chart(market_data),
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )

with raw_data_tab:
    displayed_data = market_data.sort_values("Date", ascending=False).copy()
    displayed_data["Date"] = displayed_data["Date"].dt.date

    st.dataframe(
        displayed_data,
        width="stretch",
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Tarih", format="DD/MM/YYYY"),
            "Open": st.column_config.NumberColumn("Açılış", format="$%.2f"),
            "High": st.column_config.NumberColumn("En Yüksek", format="$%.2f"),
            "Low": st.column_config.NumberColumn("En Düşük", format="$%.2f"),
            "Close": st.column_config.NumberColumn("Kapanış", format="$%.2f"),
            "Adj Close": st.column_config.NumberColumn(
                "Düzeltilmiş Kapanış", format="$%.2f"
            ),
            "Volume": st.column_config.NumberColumn("İşlem Hacmi", format="%d"),
            "Daily_Return": st.column_config.NumberColumn(
                "Günlük Getiri", format="%.4f"
            ),
            "Log_Return": st.column_config.NumberColumn(
                "Log-Getiri", format="%.4f"
            ),
            "MA_20": st.column_config.NumberColumn(
                "20 Günlük Ortalama", format="$%.2f"
            ),
            "MA_50": st.column_config.NumberColumn(
                "50 Günlük Ortalama", format="$%.2f"
            ),
            "Volatility_20": st.column_config.NumberColumn(
                "20 Günlük Volatilite", format="%.4f"
            ),
            "Price_Change": st.column_config.NumberColumn(
                "Mutlak Fiyat Değişimi", format="$%.2f"
            ),
        },
    )

    csv_data = displayed_data.to_csv(
        index=False,
        date_format="%Y-%m-%d",
    ).encode("utf-8")
    st.download_button(
        label="Görüntülenen Veriyi CSV Olarak İndir",
        data=csv_data,
        file_name="aapl_analiz_verisi.csv",
        mime="text/csv",
    )

st.warning(
    "Gösterilen veriler ve analizler yalnızca akademik amaçlıdır ve "
    "yatırım tavsiyesi niteliği taşımaz."
)
