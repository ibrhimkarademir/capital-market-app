"""Tez kapsamında doğrulanan LSTM, TFT ve Chronos sonuçlarını karşılaştırır."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.results_loader import (
    EXPECTED_MODELS,
    ResultsDataError,
    determine_best_model,
    get_model_result,
    load_interval_prediction_data,
    load_model_results,
    load_prediction_data,
)
from utils.presentation_mode import (
    apply_presentation_mode_styles,
    render_presentation_toggle,
)
from utils.ui_components import load_css


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE_PATH = PROJECT_ROOT / "data" / "model_results.json"
STYLE_PATH = PROJECT_ROOT / "assets" / "style.css"

METRIC_LABELS = {
    "mae": "MAE (USD)",
    "rmse": "RMSE (USD)",
    "r2": "R²",
    "direction_accuracy": "Yön Doğruluğu",
}

DETAIL_METRIC_LABELS = {
    "mse": "MSE",
    "rmse": "RMSE (USD)",
    "mae": "MAE (USD)",
    "mape": "MAPE",
    "r2": "R²",
    "direction_accuracy": "Yön Doğruluğu",
}

HYPERPARAMETER_LABELS = {
    "units": "Nöron sayısı",
    "hidden_size": "Hidden size",
    "dropout": "Dropout",
    "batch_size": "Batch size",
    "epochs": "Tanımlanan maksimum epoch",
    "patience": "Patience",
    "learning_rate": "Learning rate",
    "num_heads": "Attention head sayısı",
}

MODEL_HYPERPARAMETERS = {
    "LSTM": (
        "units",
        "dropout",
        "batch_size",
        "epochs",
        "patience",
        "learning_rate",
    ),
    "TFT": (
        "hidden_size",
        "dropout",
        "batch_size",
        "epochs",
        "patience",
        "learning_rate",
        "num_heads",
    ),
}

MODEL_APPROACHES = {
    "LSTM": "Eğitilmiş Direct Multi-Output model",
    "TFT": "Eğitilmiş TFT Tabanlı Model mimarisi",
    "Chronos": "Zero-shot önceden eğitilmiş model",
}

TEST_PERIODS = {
    "LSTM": "18.02.2025–23.12.2025",
    "TFT": "18.02.2025–23.12.2025",
    "Chronos": "05.02.2025–26.12.2025",
}

load_css(STYLE_PATH)
render_presentation_toggle()
apply_presentation_mode_styles()


def is_valid_number(value: object) -> bool:
    """Değerin bool olmayan sonlu bir sayı olup olmadığını denetler."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def normalize_direction_accuracy(value: object) -> float | None:
    """Yön doğruluğunu 0–100 yüzde ölçeğine güvenli biçimde dönüştürür."""
    if not is_valid_number(value):
        return None
    numeric_value = float(value)
    if not 0 <= numeric_value <= 100:
        return None
    return numeric_value * 100 if numeric_value <= 1 else numeric_value


def format_metric(
    metric_name: str,
    value: object,
    missing_text: str = "—",
) -> str:
    """Metrik değerini kullanıcı arayüzüne uygun biçimde gösterir."""
    if not is_valid_number(value):
        return missing_text
    if metric_name == "direction_accuracy":
        percentage = normalize_direction_accuracy(value)
        return f"%{percentage:.2f}" if percentage is not None else missing_text
    return f"{float(value):.4f}"


def format_detail_metric(metric_name: str, value: object) -> str:
    """Deney ayrıntısındaki metriği kaynak hassasiyetini koruyarak gösterir."""
    if not is_valid_number(value):
        return "—"
    if metric_name in {"mape", "direction_accuracy"}:
        return f"%{float(value):.6f}"
    return f"{float(value):.6f}"


def normalized_chart_value(metric_name: str, value: object) -> float | None:
    """Grafikte kullanılacak sayısal metrik değerini standartlaştırır."""
    if not is_valid_number(value):
        return None
    if metric_name == "direction_accuracy":
        return normalize_direction_accuracy(value)
    return float(value)


def render_summary_card(title: str, value: object) -> None:
    """Proje özet bilgisini yerel Streamlit kartında gösterir."""
    with st.container(border=True):
        st.caption(title)
        st.markdown(f"### {value if value is not None else 'Eklenmedi'}")


def render_model_status_card(model_key: str, model_result: dict) -> None:
    """Tek modelin durumunu ve doğrulanmış test metriklerini gösterir."""
    metrics = model_result["metrics"]
    with st.container(border=True):
        st.subheader(model_result["display_name"])
        st.caption(model_result["status"])
        if model_key == "Chronos":
            st.caption("Zero-shot tahmin · 0.10–0.90 tahmin aralığı mevcut")

        first_metric_row = st.columns(2)
        first_metric_row[0].metric(
            "MAE (USD)",
            format_metric("mae", metrics["mae"]),
        )
        first_metric_row[1].metric(
            "RMSE (USD)",
            format_metric("rmse", metrics["rmse"]),
        )
        second_metric_row = st.columns(2)
        second_metric_row[0].metric("R²", format_metric("r2", metrics["r2"]))
        second_metric_row[1].metric(
            "Yön Doğruluğu",
            format_metric("direction_accuracy", metrics["direction_accuracy"]),
        )

        if model_result.get("notes"):
            st.caption(f"Not: {model_result['notes']}")


def render_metrics(metrics: dict) -> None:
    """Test veya validation metriklerini iki satırlı kart düzeninde gösterir."""
    metric_names = tuple(DETAIL_METRIC_LABELS)
    for start_index in range(0, len(metric_names), 3):
        metric_columns = st.columns(3)
        for column, metric_name in zip(
            metric_columns,
            metric_names[start_index : start_index + 3],
            strict=True,
        ):
            column.metric(
                DETAIL_METRIC_LABELS[metric_name],
                format_detail_metric(metric_name, metrics.get(metric_name)),
            )


def render_hyperparameters(model_key: str, hyperparameters: dict) -> None:
    """Eğitilmiş modelin kaynak dosyadan aktarılan ayarlarını gösterir."""
    fields = MODEL_HYPERPARAMETERS[model_key]
    for start_index in range(0, len(fields), 3):
        visible_fields = fields[start_index : start_index + 3]
        columns = st.columns(len(visible_fields))
        for column, field_name in zip(columns, visible_fields, strict=True):
            value = hyperparameters.get(field_name)
            column.metric(
                HYPERPARAMETER_LABELS[field_name],
                str(value) if value is not None else "—",
            )


def render_experiment_details(model_key: str, model_result: dict) -> None:
    """LSTM veya TFT deney ayrıntılarını sekmeler içinde gösterir."""
    st.subheader(f"{model_result['display_name']} Deney Detayları")
    test_tab, validation_tab, hyperparameters_tab = st.tabs(
        ["Test Sonuçları", "Validation Sonuçları", "Hiperparametreler"]
    )
    with test_tab:
        render_metrics(model_result["test_metrics"])
    with validation_tab:
        render_metrics(model_result["validation_metrics"])
    with hyperparameters_tab:
        render_hyperparameters(model_key, model_result["hyperparameters"])


def render_chronos_details(model_result: dict) -> None:
    """Chronos test sonuçlarını ve zero-shot çıkarım ayarlarını gösterir."""
    st.subheader("Chronos Deney Detayları")
    test_tab, settings_tab, interval_tab = st.tabs(
        ["Test Sonuçları", "Model Ayarları", "Tahmin Aralığı"]
    )
    settings = model_result["model_settings"]

    with test_tab:
        render_metrics(model_result["test_metrics"])
    with settings_tab:
        setting_rows = (
            ("Model", model_result.get("model_id")),
            ("Yaklaşım", model_result.get("inference_type")),
            ("Bağlam uzunluğu", model_result.get("context_length")),
            ("Tahmin ufku", f"{settings.get('forecast_horizon')} işlem günü"),
            ("Rolling adımı", settings.get("rolling_step")),
        )
        for start_index in range(0, len(setting_rows), 3):
            visible_settings = setting_rows[start_index : start_index + 3]
            columns = st.columns(len(visible_settings))
            for column, (label, value) in zip(
                columns,
                visible_settings,
                strict=True,
            ):
                column.metric(label, str(value))
    with interval_tab:
        interval_rows = (
            ("Merkez tahmin", "Medyan / 0.50 quantile"),
            ("Alt sınır", "0.10 quantile"),
            ("Üst sınır", "0.90 quantile"),
        )
        columns = st.columns(3)
        for column, (label, value) in zip(
            columns,
            interval_rows,
            strict=True,
        ):
            column.metric(label, value)


def create_metric_chart(results: dict, metric_name: str) -> go.Figure | None:
    """En az iki gerçek değer varsa seçilen metrik için sütun grafik oluşturur."""
    chart_rows = []
    for model_key in EXPECTED_MODELS:
        model_result = get_model_result(results, model_key)
        value = normalized_chart_value(
            metric_name,
            model_result["metrics"].get(metric_name),
        )
        if value is not None:
            chart_rows.append(
                {
                    "Model": model_result["display_name"],
                    "Değer": value,
                }
            )

    if len(chart_rows) < 2:
        return None

    chart_data = pd.DataFrame(chart_rows)
    figure = go.Figure(
        data=[
            go.Bar(
                x=chart_data["Model"],
                y=chart_data["Değer"],
                name=METRIC_LABELS[metric_name],
            )
        ]
    )
    figure.update_layout(
        xaxis_title="Model",
        yaxis_title=(
            "Yüzde (%)"
            if metric_name == "direction_accuracy"
            else METRIC_LABELS[metric_name]
        ),
        autosize=True,
    )
    return figure


st.title("Model Performanslarının Karşılaştırılması")
st.write(
    "Bu sayfa, LSTM, TFT Tabanlı Model ve Chronos için tez kapsamında "
    "kaydedilen test sonuçlarını ortak ölçütlerle karşılaştırır."
)

try:
    results = load_model_results(RESULTS_FILE_PATH)
except ResultsDataError as error:
    st.error(str(error))
    st.stop()

project = results["project"]
model_results = {
    model_key: get_model_result(results, model_key)
    for model_key in EXPECTED_MODELS
}

st.subheader("Proje Özeti")
summary_columns = st.columns(4)
with summary_columns[0]:
    render_summary_card("Finansal varlık", project.get("symbol"))
with summary_columns[1]:
    render_summary_card(
        "Girdi penceresi",
        f"{project.get('input_window')} işlem günü",
    )
with summary_columns[2]:
    render_summary_card(
        "Tahmin ufku",
        f"{project.get('forecast_horizon')} işlem günü",
    )
with summary_columns[3]:
    render_summary_card("Hedef değişken", project.get("target_type"))

st.subheader("Model Durumları")
model_columns = st.columns(3)
for column, model_key in zip(model_columns, EXPECTED_MODELS, strict=True):
    with column:
        render_model_status_card(model_key, model_results[model_key])

st.info(
    "Akademik not: Gösterilen metrikler tez çalışmasındaki özgün deney "
    "çıktılarından alınmıştır. LSTM ve TFT Tabanlı Model aynı rolling test döneminde "
    "değerlendirilirken, Chronos zero-shot deneyi daha geniş bir test aralığını "
    "kapsamaktadır. Sonuçlar yeniden hesaplanmadan tezde raporlandığı biçimde "
    "sunulmuştur."
)

st.subheader("Karşılaştırma Tablosu")
comparison_rows = []
for model_key in EXPECTED_MODELS:
    model_result = model_results[model_key]
    metrics = model_result["metrics"]
    comparison_rows.append(
        {
            "Model": model_result["display_name"],
            "Yaklaşım": MODEL_APPROACHES[model_key],
            "Test dönemi": TEST_PERIODS[model_key],
            "MAE (USD)": format_metric("mae", metrics["mae"]),
            "RMSE (USD)": format_metric("rmse", metrics["rmse"]),
            "R²": format_metric("r2", metrics["r2"]),
            "Yön Doğruluğu": format_metric(
                "direction_accuracy",
                metrics["direction_accuracy"],
            ),
        }
    )

st.dataframe(
    pd.DataFrame(comparison_rows),
    width="stretch",
    hide_index=True,
)

render_experiment_details("LSTM", model_results["LSTM"])
render_experiment_details("TFT", model_results["TFT"])
render_chronos_details(model_results["Chronos"])

st.subheader("Metrik Grafikleri")
metric_tabs = st.tabs(["MAE (USD)", "RMSE (USD)", "R²", "Yön Doğruluğu"])
for tab, metric_name in zip(metric_tabs, METRIC_LABELS, strict=True):
    with tab:
        metric_chart = create_metric_chart(results, metric_name)
        if metric_chart is None:
            st.info("Karşılaştırmalı grafik için yeterli model sonucu bulunmuyor.")
        else:
            st.plotly_chart(
                metric_chart,
                width="stretch",
                config={"displaylogo": False, "responsive": True},
            )
            st.caption(
                "Grafik, tezde raporlanan özgün test metriklerini "
                "karşılaştırmaktadır. Chronos test dönemi diğer iki modelden "
                "bir miktar daha geniştir."
            )

st.subheader("LSTM ve TFT Rolling Test Karşılaştırması")
try:
    lstm_predictions = load_prediction_data(
        model_results["LSTM"]["prediction_file"]
    )
    tft_predictions = load_prediction_data(
        model_results["TFT"]["prediction_file"]
    )
    if not lstm_predictions["Date"].equals(tft_predictions["Date"]):
        raise ResultsDataError(
            "LSTM ve TFT rolling test tarihleri birbiriyle eşleşmiyor."
        )
    if not lstm_predictions["Actual"].equals(tft_predictions["Actual"]):
        raise ResultsDataError(
            "LSTM ve TFT rolling test gerçek kapanış serileri eşleşmiyor."
        )
except ResultsDataError as error:
    st.error(str(error))
else:
    shared_figure = go.Figure()
    shared_figure.add_trace(
        go.Scatter(
            x=lstm_predictions["Date"],
            y=lstm_predictions["Actual"],
            mode="lines",
            name="Gerçek Kapanış",
        )
    )
    shared_figure.add_trace(
        go.Scatter(
            x=lstm_predictions["Date"],
            y=lstm_predictions["Predicted"],
            mode="lines",
            name="LSTM Rolling Tahmin",
        )
    )
    shared_figure.add_trace(
        go.Scatter(
            x=tft_predictions["Date"],
            y=tft_predictions["Predicted"],
            mode="lines",
            name="TFT Rolling Tahmin",
        )
    )
    shared_figure.update_layout(
        xaxis_title="Tarih",
        yaxis_title="Kapanış Fiyatı (USD)",
        hovermode="x unified",
        autosize=True,
    )
    st.plotly_chart(
        shared_figure,
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )
    st.caption(
        "LSTM ve TFT Tabanlı Model aynı test dönemi ve aynı gerçek kapanış "
        "serisi üzerinde beş günlük bloklar hâlinde değerlendirilmiştir."
    )

st.subheader("Chronos Zero-Shot Rolling Test Sonuçları")
try:
    chronos_predictions = load_interval_prediction_data(
        model_results["Chronos"]["prediction_file"]
    )
except ResultsDataError as error:
    st.error(str(error))
else:
    chronos_figure = go.Figure()
    chronos_figure.add_trace(
        go.Scatter(
            x=chronos_predictions["Date"],
            y=chronos_predictions["Lower"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
            name="Chronos Alt Sınır",
        )
    )
    chronos_figure.add_trace(
        go.Scatter(
            x=chronos_predictions["Date"],
            y=chronos_predictions["Upper"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(171, 126, 255, 0.20)",
            name="Chronos 0.10–0.90 Tahmin Aralığı",
        )
    )
    chronos_figure.add_trace(
        go.Scatter(
            x=chronos_predictions["Date"],
            y=chronos_predictions["Actual"],
            mode="lines",
            name="Gerçek Kapanış",
        )
    )
    chronos_figure.add_trace(
        go.Scatter(
            x=chronos_predictions["Date"],
            y=chronos_predictions["Predicted"],
            mode="lines",
            name="Chronos Medyan Tahmini",
        )
    )
    chronos_figure.update_layout(
        xaxis_title="Tarih",
        yaxis_title="Kapanış Fiyatı (USD)",
        hovermode="x unified",
        autosize=True,
    )
    st.plotly_chart(
        chronos_figure,
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )
    st.caption(
        "Chronos rolling test çıktısı 225 satır ve 05.02.2025–26.12.2025 "
        "dönemini kapsar; daha kısa olan LSTM/TFT dönemine göre kırpılmamıştır."
    )

st.subheader("Raporlanan En İyi Test Değeri")
selected_metric_label = st.selectbox(
    "Karşılaştırma metriği",
    options=["MAE (USD)", "RMSE (USD)", "R²", "Yön Doğruluğu"],
)
selected_metric_key = {
    "MAE (USD)": "mae",
    "RMSE (USD)": "rmse",
    "R²": "r2",
    "Yön Doğruluğu": "direction_accuracy",
}[selected_metric_label]
best_model = determine_best_model(results, selected_metric_key)

if best_model["success"] and best_model["model_key"] == "TFT":
    st.success(
        "TFT Tabanlı Model, tezde raporlanan sonuçlar arasında bu metrik için "
        "en iyi değeri üretmiştir."
    )
else:
    st.info("Seçilen metrik için raporlanan sonuçlar karşılaştırılamadı.")

st.warning(
    "Chronos deneyinin test dönemi daha geniş olduğu için bu karşılaştırma "
    "sonuçları deney kapsamıyla birlikte değerlendirilmelidir."
)
