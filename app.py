"""Streamlit uygulamasının akademik ana dashboard sayfası."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import MarketDataError, load_market_data
from utils.presentation_mode import (
    apply_presentation_mode_styles,
    render_presentation_toggle,
)
from utils.ui_components import (
    load_css,
    render_academic_notice,
    render_footer,
    render_hero,
    render_info_box,
    render_metric_card,
    render_model_card,
    render_process_card,
    render_section_header,
)


PROJECT_ROOT = Path(__file__).resolve().parent
STYLE_PATH = PROJECT_ROOT / "assets" / "style.css"
LOCAL_DATA_PATH = PROJECT_ROOT / "data" / "aapl_history.csv"

PROJECT_TITLE = "Sermaye Piyasalarında Dinamik Fiyat Tahmini"
ASSET_SYMBOL = "AAPL"
INPUT_WINDOW = 30
FORECAST_HORIZON = 5
MODEL_NAMES = ("LSTM", "TFT Tabanlı Model", "Chronos")
DATA_START_DATE = date(2020, 1, 1)

MODEL_DESCRIPTIONS = (
    (
        "LSTM",
        "5 günlük Direct Multi-Output LSTM. Doğru dönem rolling test sonucu mevcut. "
        "Detaylar Model Karşılaştırma sayfasında.",
    ),
    (
        "TFT Tabanlı Model",
        "5 günlük TFT Tabanlı Model. Rolling test sonucu mevcut. "
        "Detaylar Model Karşılaştırma sayfasında.",
    ),
    (
        "Chronos",
        "Zero-shot Chronos modeli. 5 günlük medyan tahmin ve 0.10–0.90 "
        "tahmin aralığı mevcut.",
    ),
)

process_steps = [
    {
        "number": 1,
        "title": "Piyasa Verisi",
        "description": "AAPL geçmiş fiyat ve hacim verilerinin alınması.",
    },
    {
        "number": 2,
        "title": "Veri Ön İşleme",
        "description": (
            "Eksik değerlerin, log-getirilerin ve teknik göstergelerin "
            "hazırlanması."
        ),
    },
    {
        "number": 3,
        "title": "Girdi Penceresi",
        "description": (
            "Son 30 işlem gününün model girdisine dönüştürülmesi."
        ),
    },
    {
        "number": 4,
        "title": "Model Tahmini",
        "description": (
            "LSTM, TFT Tabanlı Model ve Chronos ile 5 günlük tahmin oluşturulması."
        ),
    },
    {
        "number": 5,
        "title": "Performans Analizi",
        "description": (
            "Gerçek ve tahmin edilen değerlerin karşılaştırılması."
        ),
    },
]


st.set_page_config(
    page_title=PROJECT_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css(STYLE_PATH)
render_presentation_toggle()
apply_presentation_mode_styles()

render_hero()

render_section_header(
    "Proje Parametreleri",
    "Tez kapsamında sabitlenen temel analiz ve tahmin yapılandırması.",
)

parameter_columns = st.columns(4)
with parameter_columns[0]:
    render_metric_card("Finansal varlık", ASSET_SYMBOL, "Apple Inc.")
with parameter_columns[1]:
    render_metric_card(
        "Girdi penceresi",
        f"{INPUT_WINDOW} işlem günü",
        "Her tahmin için kullanılan geçmiş dönem",
    )
with parameter_columns[2]:
    render_metric_card(
        "Tahmin ufku",
        f"{FORECAST_HORIZON} işlem günü",
        "Ana kısa vadeli tahmin aralığı",
    )
with parameter_columns[3]:
    render_metric_card(
        "Model sayısı",
        str(len(MODEL_NAMES)),
        "LSTM, TFT Tabanlı Model ve Chronos",
    )

render_section_header(
    "Güncel Veri Özeti",
    "AAPL geçmiş piyasa verilerinden hesaplanan güncel göstergeler.",
)

try:
    market_data, data_source = load_market_data(
        symbol=ASSET_SYMBOL,
        start_date=DATA_START_DATE,
        end_date=date.today(),
        local_file_path=LOCAL_DATA_PATH,
    )

    latest_close = market_data["Close"].iloc[-1]
    latest_date = pd.Timestamp(market_data["Date"].iloc[-1])
    recent_data = market_data.tail(INPUT_WINDOW)

    reference_close = recent_data["Close"].iloc[0]
    recent_change = (
        latest_close / reference_close - 1
        if pd.notna(reference_close) and reference_close != 0
        else np.nan
    )
    annualized_volatility = recent_data["Log_Return"].std() * np.sqrt(252)

    render_info_box(
        f"Veri kaynağı: {data_source}",
        "Göstergeler gerçek AAPL piyasa verilerinden hesaplanmıştır.",
    )

    summary_columns = st.columns(4)
    with summary_columns[0]:
        render_metric_card(
            "Son kapanış fiyatı",
            f"{latest_close:,.2f} USD",
            ASSET_SYMBOL,
        )
    with summary_columns[1]:
        render_metric_card(
            "Son işlem tarihi",
            latest_date.strftime("%d.%m.%Y"),
            "En güncel doğrulanmış gözlem",
        )
    with summary_columns[2]:
        change_value = (
            f"%{recent_change * 100:,.2f}"
            if pd.notna(recent_change)
            else "Hesaplanamadı"
        )
        render_metric_card(
            "Son 30 işlem günü değişimi",
            change_value,
            "Kapanış fiyatı bazında",
        )
    with summary_columns[3]:
        volatility_value = (
            f"%{annualized_volatility * 100:,.2f}"
            if pd.notna(annualized_volatility)
            else "Hesaplanamadı"
        )
        render_metric_card(
            "30 günlük yıllıklandırılmış volatilite",
            volatility_value,
            "Log-getiri standart sapması × √252",
        )
except MarketDataError as error:
    st.error(f"Güncel veri özeti yüklenemedi: {error}")
except (IndexError, KeyError, TypeError, ValueError) as error:
    st.error(
        "Güncel veri özeti hesaplanamadı. Piyasa verisi yapısını kontrol edin. "
        f"Ayrıntı: {error}"
    )

render_section_header(
    "Model Yaklaşımları",
    "Çalışmada değerlendirilecek yeni nesil finansal zaman serisi modelleri.",
)

model_columns = st.columns(3)
for model_column, (model_name, model_description) in zip(
    model_columns, MODEL_DESCRIPTIONS, strict=True
):
    with model_column:
        render_model_card(
            model_name=model_name,
            model_description=model_description,
            status=(
                "Tez sonucu entegre edildi"
                if model_name in {
                    "LSTM",
                    "TFT Tabanlı Model",
                    "Chronos",
                }
                else "Model entegrasyonu bekliyor"
            ),
        )

render_section_header(
    "Uygulama Akışı",
    "Verinin alınmasından model çıktılarının değerlendirilmesine uzanan süreç.",
)

first_process_row = st.columns(3, gap="medium")
for column, step in zip(first_process_row, process_steps[:3], strict=True):
    with column:
        render_process_card(
            step["number"],
            step["title"],
            step["description"],
        )

second_process_row = st.columns(2, gap="medium")
for column, step in zip(second_process_row, process_steps[3:], strict=True):
    with column:
        render_process_card(
            step["number"],
            step["title"],
            step["description"],
        )

render_section_header(
    "Hızlı Erişim",
    "Analiz, tahmin ve proje belgelerine doğrudan ulaşın.",
)

with st.container(key="quick_access_grid"):
    first_link_row = st.columns(3, gap="medium")
    with first_link_row[0]:
        st.page_link(
            "pages/1_Veri_Analizi.py",
            label="Veri Analizi",
            icon="📊",
            use_container_width=True,
        )
    with first_link_row[1]:
        st.page_link(
            "pages/2_Fiyat_Tahmini.py",
            label="Fiyat Tahmini",
            icon="🔭",
            use_container_width=True,
        )
    with first_link_row[2]:
        st.page_link(
            "pages/3_Model_Karsilastirma.py",
            label="Model Karşılaştırma",
            icon="⚖️",
            use_container_width=True,
        )

    second_link_row = st.columns([0.5, 1, 1, 0.5], gap="medium")
    with second_link_row[1]:
        st.page_link(
            "pages/4_Metodoloji.py",
            label="Metodoloji",
            icon="🧭",
            use_container_width=True,
        )
    with second_link_row[2]:
        st.page_link(
            "pages/5_Proje_Hakkinda.py",
            label="Proje Hakkında",
            icon="🎓",
            use_container_width=True,
        )

render_academic_notice()
render_footer()
