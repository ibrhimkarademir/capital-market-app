"""Tezde kaydedilen beş günlük LSTM, TFT ve Chronos tahminlerini gösterir."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.results_loader import (
    ResultsDataError,
    get_model_result,
    load_future_forecast_data,
    load_interval_future_forecast_data,
    load_model_results,
    load_recent_close_data,
)
from utils.presentation_mode import (
    apply_presentation_mode_styles,
    render_presentation_toggle,
)
from utils.ui_components import load_css


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE_PATH = PROJECT_ROOT / "data" / "model_results.json"
PLOTS_DIRECTORY = PROJECT_ROOT / "results" / "plots"
STYLE_PATH = PROJECT_ROOT / "assets" / "style.css"

MODEL_OPTIONS = {
    "LSTM": "LSTM",
    "TFT Tabanlı Model": "TFT",
    "Chronos": "Chronos",
}

PLOT_DEFINITIONS = {
    "LSTM": {
        "title": "LSTM Orijinal Grafikleri",
        "plots": (
            ("lstm_5day_loss.png", "LSTM Eğitim ve Doğrulama Kayıpları"),
            ("lstm_5day_rolling_test.png", "LSTM Rolling Test Grafiği"),
            ("lstm_5day_future_forecast.png", "LSTM Beş Günlük Gelecek Tahmini"),
        ),
        "note": (
            "Eğitim kaybı azalırken doğrulama kaybının ilerleyen epochlarda "
            "yükselmesi, genelleme performansının ayrıca test sonuçlarıyla "
            "değerlendirilmesini gerekli kılmaktadır."
        ),
        "note_after_first": True,
    },
    "TFT": {
        "title": "TFT Tabanlı Model Orijinal Grafikleri",
        "plots": (
            ("tft_5day_loss.png", "TFT Eğitim ve Doğrulama Kayıpları"),
            ("tft_5day_rolling_test.png", "TFT Rolling Test Grafiği"),
            ("tft_5day_future_forecast.png", "TFT Beş Günlük Gelecek Tahmini"),
        ),
        "note": (
            "Eğitim ve doğrulama kayıpları farklı seviyelerde seyretmektedir. "
            "Model başarısı yalnızca loss eğrileriyle değil, test performans "
            "metrikleriyle birlikte değerlendirilmiştir."
        ),
        "note_after_first": True,
    },
    "Chronos": {
        "title": "Chronos Orijinal Grafikleri",
        "plots": (
            (
                "chronos_5day_rolling_test.png",
                "Chronos Zero-Shot Rolling Test Grafiği",
            ),
            (
                "chronos_5day_future_forecast.png",
                "Chronos Beş Günlük Gelecek Tahmini",
            ),
        ),
        "note": (
            "Chronos zero-shot yaklaşımında proje verisi üzerinde model eğitimi "
            "yapılmadığı için eğitim ve doğrulama loss grafiği bulunmamaktadır."
        ),
        "note_after_first": False,
    },
}

load_css(STYLE_PATH)
render_presentation_toggle()
apply_presentation_mode_styles()


def render_status_card(model_name: str, status: str) -> None:
    """Model entegrasyon durumunu yerel Streamlit kartında gösterir."""
    with st.container(border=True):
        st.subheader(model_name)
        st.caption(status)


def render_summary_card(title: str, value: str, description: str) -> None:
    """Seçilen modelin tahmin özetini yerel Streamlit kartında gösterir."""
    with st.container(border=True):
        st.caption(title)
        st.markdown(f"### {value}")
        st.caption(description)


def prepare_point_forecast_table(forecast_data: pd.DataFrame) -> pd.DataFrame:
    """Log-getiri tabanlı model tahminlerini kullanıcı tablosuna dönüştürür."""
    return pd.DataFrame(
        {
            "Tarih": forecast_data["Date"].dt.strftime("%d.%m.%Y"),
            "Tahmin Edilen Log-Getiri": forecast_data["Predicted_Log_Return"],
            "Tahmin Edilen Kapanış Fiyatı (USD)": forecast_data[
                "Predicted_Close"
            ],
        }
    )


def prepare_interval_forecast_table(forecast_data: pd.DataFrame) -> pd.DataFrame:
    """Chronos medyan ve quantile tahminlerini kullanıcı tablosuna dönüştürür."""
    return pd.DataFrame(
        {
            "Tarih": forecast_data["Date"].dt.strftime("%d.%m.%Y"),
            "Medyan Kapanış Tahmini (USD)": forecast_data["Predicted_Close"],
            "Alt Tahmin Sınırı – 0.10 (USD)": forecast_data["Lower"],
            "Üst Tahmin Sınırı – 0.90 (USD)": forecast_data["Upper"],
        }
    )


def render_original_plots(model_key: str) -> None:
    """Bir modele ait değiştirilmemiş tez PNG dosyalarını gösterir."""
    plot_group = PLOT_DEFINITIONS[model_key]
    st.markdown(f"#### {plot_group['title']}")

    for index, (file_name, caption) in enumerate(plot_group["plots"]):
        plot_path = PLOTS_DIRECTORY / file_name
        if plot_path.exists() and plot_path.stat().st_size > 0:
            st.image(str(plot_path), caption=caption, width="stretch")
            if index == 0 and plot_group["note_after_first"]:
                st.caption(plot_group["note"])
        else:
            st.info(f"Orijinal tez grafiği bulunamadı: {caption}")

    if not plot_group["note_after_first"]:
        st.caption(plot_group["note"])


st.title("Beş Günlük Fiyat Tahmini")
st.write(
    "Bu sayfa, tez çalışması sırasında kaydedilen LSTM, TFT Tabanlı Model ve "
    "Chronos tahminlerini birlikte gösterir. Sonuçlar canlı olarak yeniden "
    "hesaplanmaz."
)

try:
    results = load_model_results(RESULTS_FILE_PATH)
except ResultsDataError as error:
    st.error(str(error))
    st.stop()

model_results = {
    model_key: get_model_result(results, model_key)
    for model_key in MODEL_OPTIONS.values()
}

st.subheader("Model Durumu")
status_columns = st.columns(3)
with status_columns[0]:
    render_status_card("LSTM", "Tez çıktısı mevcut")
with status_columns[1]:
    render_status_card("TFT Tabanlı Model", "Tez çıktısı mevcut")
with status_columns[2]:
    render_status_card("Chronos", "Zero-shot tez çıktısı mevcut")

st.warning(
    "Bu tahminler tez çalışması sırasında oluşturulan sabit model çıktılarıdır. "
    "Güncel piyasa verileri kullanılarak anlık olarak yeniden hesaplanmamaktadır."
)

try:
    forecast_by_model = {
        "LSTM": load_future_forecast_data(
            model_results["LSTM"]["future_forecast_file"]
        ),
        "TFT": load_future_forecast_data(
            model_results["TFT"]["future_forecast_file"]
        ),
        "Chronos": load_interval_future_forecast_data(
            model_results["Chronos"]["future_forecast_file"]
        ),
    }
    expected_dates = forecast_by_model["LSTM"]["Date"]
    if not expected_dates.equals(forecast_by_model["TFT"]["Date"]):
        raise ResultsDataError(
            "LSTM ve TFT gelecek tahmin tarihleri birbiriyle eşleşmiyor."
        )
    if not expected_dates.equals(forecast_by_model["Chronos"]["Date"]):
        raise ResultsDataError(
            "Chronos gelecek tahmin tarihleri diğer modellerle eşleşmiyor."
        )
except ResultsDataError as error:
    st.error(str(error))
    st.stop()

selected_model_name = st.selectbox(
    "Özet bilgileri gösterilecek model",
    options=list(MODEL_OPTIONS),
)
selected_model_key = MODEL_OPTIONS[selected_model_name]
selected_forecast = forecast_by_model[selected_model_key]

first_forecast = float(selected_forecast["Predicted_Close"].iloc[0])
last_forecast = float(selected_forecast["Predicted_Close"].iloc[-1])
forecast_change = (last_forecast / first_forecast - 1) * 100

st.subheader(f"{selected_model_name} Tahmin Özeti")
summary_columns = st.columns(5)
with summary_columns[0]:
    render_summary_card(
        "İlk tahmin tarihi",
        selected_forecast["Date"].iloc[0].strftime("%d.%m.%Y"),
        "Birinci tahmin günü",
    )
with summary_columns[1]:
    render_summary_card(
        "Son tahmin tarihi",
        selected_forecast["Date"].iloc[-1].strftime("%d.%m.%Y"),
        "Beşinci tahmin günü",
    )
with summary_columns[2]:
    render_summary_card(
        "İlk tahmin edilen kapanış",
        f"{first_forecast:,.2f} USD",
        "Birinci tahmin günü",
    )
with summary_columns[3]:
    render_summary_card(
        "Beşinci gün tahmini",
        f"{last_forecast:,.2f} USD",
        "Son tahmin günü",
    )
with summary_columns[4]:
    render_summary_card(
        "İlk tahminden beşinci tahmine değişim",
        f"%{forecast_change:.2f}",
        "Seçilen modelin tahmin dönemi",
    )

st.subheader("Model Bazlı Tahmin Tabloları")
lstm_tab, tft_tab, chronos_tab = st.tabs(
    ["LSTM", "TFT Tabanlı Model", "Chronos"]
)

with lstm_tab:
    st.dataframe(
        prepare_point_forecast_table(forecast_by_model["LSTM"]),
        width="stretch",
        hide_index=True,
    )
with tft_tab:
    st.dataframe(
        prepare_point_forecast_table(forecast_by_model["TFT"]),
        width="stretch",
        hide_index=True,
    )
with chronos_tab:
    st.dataframe(
        prepare_interval_forecast_table(forecast_by_model["Chronos"]),
        width="stretch",
        hide_index=True,
    )

forecast_figure = go.Figure()
try:
    recent_close_data = load_recent_close_data()
except ResultsDataError as error:
    recent_close_data = None
    st.info(f"Son 90 günlük kapanış verisi gösterilemedi: {error}")

if recent_close_data is not None:
    forecast_figure.add_trace(
        go.Scatter(
            x=recent_close_data["Date"],
            y=recent_close_data["Close"],
            mode="lines",
            name="Son 90 Gün Gerçek Kapanış",
        )
    )

forecast_figure.add_trace(
    go.Scatter(
        x=forecast_by_model["LSTM"]["Date"],
        y=forecast_by_model["LSTM"]["Predicted_Close"],
        mode="lines+markers",
        name="LSTM Tahmini",
    )
)
forecast_figure.add_trace(
    go.Scatter(
        x=forecast_by_model["TFT"]["Date"],
        y=forecast_by_model["TFT"]["Predicted_Close"],
        mode="lines+markers",
        name="TFT Tabanlı Model Tahmini",
    )
)
forecast_figure.add_trace(
    go.Scatter(
        x=forecast_by_model["Chronos"]["Date"],
        y=forecast_by_model["Chronos"]["Lower"],
        mode="lines",
        line={"width": 0},
        showlegend=False,
        hoverinfo="skip",
        name="Chronos Alt Sınır",
    )
)
forecast_figure.add_trace(
    go.Scatter(
        x=forecast_by_model["Chronos"]["Date"],
        y=forecast_by_model["Chronos"]["Upper"],
        mode="lines",
        line={"width": 0},
        fill="tonexty",
        fillcolor="rgba(171, 126, 255, 0.20)",
        name="Chronos 0.10–0.90 Tahmin Aralığı",
    )
)
forecast_figure.add_trace(
    go.Scatter(
        x=forecast_by_model["Chronos"]["Date"],
        y=forecast_by_model["Chronos"]["Predicted_Close"],
        mode="lines+markers",
        name="Chronos Medyan Tahmini",
    )
)
forecast_figure.update_layout(
    title="5 Günlük Model Tahminleri",
    xaxis_title="Tarih",
    yaxis_title="Kapanış Fiyatı (USD)",
    hovermode="x unified",
    autosize=True,
)
st.plotly_chart(
    forecast_figure,
    width="stretch",
    config={"displaylogo": False, "responsive": True},
)

st.info(
    "Gösterilen tarihler, tahmin değerleri ve tahmin aralıkları tez "
    "çalışmasında oluşturulan özgün deney dosyalarından değiştirilmeden "
    "aktarılmıştır."
)

with st.expander("Tezde Üretilen Orijinal Grafikler"):
    render_original_plots("LSTM")
    st.divider()
    render_original_plots("TFT")
    st.divider()
    render_original_plots("Chronos")

st.caption(
    "Sonuç türü: Sabit tez çıktısı · Üretim dönemi: 2025 Aralık · "
    "Modeller: LSTM, TFT Tabanlı Model ve Chronos"
)
st.warning(
    "Bu sayfadaki tahminler yalnızca akademik ve deneysel amaçlıdır; "
    "yatırım tavsiyesi niteliği taşımaz."
)
