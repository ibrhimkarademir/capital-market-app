"""Bitirme projesinin akademik metodolojisini açıklayan Streamlit sayfası."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.presentation_mode import (
    apply_presentation_mode_styles,
    render_presentation_toggle,
)
from utils.ui_components import load_css


STYLE_PATH = Path(__file__).resolve().parents[1] / "assets" / "style.css"
DATA_SUMMARY = [
    ("Varlık", "AAPL"),
    ("Girdi uzunluğu", "30 işlem günü"),
    ("Tahmin ufku", "5 işlem günü"),
    ("Veri türü", "Günlük piyasa verileri"),
    ("Hedef dönüşüm", "Log-getiri"),
    ("Tahmin çıktısı", "Beş günlük çoklu çıktı"),
]

load_css(STYLE_PATH)
render_presentation_toggle()
apply_presentation_mode_styles()

PREPARATION_STEPS = [
    {
        "number": 1,
        "title": "Veri Alımı",
        "description": "Günlük piyasa verilerinin Yahoo Finance üzerinden alınması.",
    },
    {
        "number": 2,
        "title": "Yapı Standardizasyonu",
        "description": "Tarih ve sütun yapısının ortak formata dönüştürülmesi.",
    },
    {
        "number": 3,
        "title": "Veri Kontrolü",
        "description": "Eksik, tekrar eden ve geçersiz değerlerin denetlenmesi.",
    },
    {
        "number": 4,
        "title": "Özellik Hazırlama",
        "description": "Log-getiri ve doğrulanmış göstergelerin hesaplanması.",
    },
    {
        "number": 5,
        "title": "Veri Bölme",
        "description": "Verinin eğitim, doğrulama ve test bölümlerine ayrılması.",
    },
    {
        "number": 6,
        "title": "Ölçekleme",
        "description": "Model girdilerinin StandardScaler ile ölçeklenmesi.",
    },
    {
        "number": 7,
        "title": "Girdi Pencereleri",
        "description": "30 günlük kayan model girdilerinin hazırlanması.",
    },
    {
        "number": 8,
        "title": "Hedef Dizileri",
        "description": "Beş günlük çoklu hedef dizilerinin oluşturulması.",
    },
]

EXPERIMENTAL_FLOW = [
    "Geçmiş veri",
    "Ön işleme",
    "30 günlük giriş penceresi",
    "LSTM / TFT / Chronos",
    "5 günlük log-getiri tahmini",
    "Fiyata geri dönüşüm",
    "Gerçek değerlerle karşılaştırma",
]


def render_data_summary_card(title: str, value: str) -> None:
    """Veri setine ait temel bir bilgiyi yerel Streamlit kartında gösterir."""
    with st.container(border=True):
        st.caption(title)
        st.markdown(f"**{value}**")


def render_preparation_step(step: dict) -> None:
    """Veri hazırlama adımını güvenli yerel bileşenlerle gösterir."""
    with st.container(border=True):
        st.markdown(f"### {step['number']}")
        st.markdown(f"**{step['title']}**")
        st.caption(step["description"])


st.title("Çalışmanın Metodolojisi")
st.write(
    "Bu sayfa, çalışmanın veri hazırlama, modelleme, tahmin ve değerlendirme "
    "aşamalarını akademik bir çerçevede özetlemektedir."
)

st.header("Araştırma Amacı")
st.write(
    "Çalışmanın amacı, geçmiş piyasa verilerinden yararlanarak Apple Inc. "
    "(AAPL) hissesinin kısa vadeli fiyat hareketlerini farklı yapay zekâ "
    "yaklaşımlarıyla tahmin etmek ve modellerin performanslarını ortak "
    "ölçütlerle karşılaştırmaktır. Finansal piyasaların belirsiz yapısı "
    "nedeniyle çalışma, kesin veya garantili tahmin iddiası taşımamaktadır."
)

st.header("Veri Seti ve Hedef Değişken")
summary_rows = (DATA_SUMMARY[:3], DATA_SUMMARY[3:])
for summary_row in summary_rows:
    summary_columns = st.columns(3)
    for column, (title, value) in zip(
        summary_columns, summary_row, strict=True
    ):
        with column:
            render_data_summary_card(title, value)

st.info("Veri kaynağı: Yahoo Finance")
st.write(
    "Temel piyasa alanları açılış, en yüksek, en düşük, kapanış ve işlem "
    "hacmidir. Teknik göstergeler, model dosyalarındaki gerçek uygulamaya "
    "göre değişebilir; uygulamada yalnızca doğrulanmış özellikler kullanılacaktır."
)

st.header("Log-Getiri Dönüşümü")
st.latex(r"r_t = \ln\left(\frac{C_t}{C_{t-1}}\right)")
formula_columns = st.columns(3)
with formula_columns[0]:
    st.markdown(r"**\(C_t\)**")
    st.caption("t günündeki kapanış fiyatı")
with formula_columns[1]:
    st.markdown(r"**\(C_{t-1}\)**")
    st.caption("Önceki işlem günündeki kapanış fiyatı")
with formula_columns[2]:
    st.markdown(r"**\(r_t\)**")
    st.caption("Günlük log-getiri")

st.write(
    "Log-getiri dönüşümü, fiyat seviyelerindeki ölçek farklarını azaltmak ve "
    "zaman serisini modelleme sürecine daha uygun bir biçimde ifade etmek "
    "amacıyla kullanılmaktadır. Bu dönüşüm tek başına istatistiksel üstünlük "
    "veya tahmin başarısı garantisi sağlamaz."
)

st.header("Fiyata Geri Dönüşüm")
st.latex(
    r"C_{t+h} = C_t \times "
    r"\exp\left(\sum_{i=1}^{h}\hat{r}_{t+i}\right)"
)
st.write(
    "Modelin ürettiği tahmini log-getiriler ardışık olarak biriktirilir ve "
    "son bilinen kapanış fiyatı üzerinden yeniden fiyat seviyesine çevrilir. "
    "Burada h, ileriye dönük tahmin adımını; tahmin işaretli r değerleri ise "
    "modelin öngördüğü log-getirileri temsil eder."
)

st.header("Veri Hazırlama Süreci")
for start_index in range(0, len(PREPARATION_STEPS), 4):
    step_columns = st.columns(4)
    for column, step in zip(
        step_columns,
        PREPARATION_STEPS[start_index : start_index + 4],
        strict=True,
    ):
        with column:
            render_preparation_step(step)

st.header("Model Yaklaşımları")
with st.expander("Direct Multi-Output LSTM", expanded=True):
    st.write(
        "Ardışık finansal verilerde uzun ve kısa dönemli bağımlılıkları "
        "öğrenmeyi amaçlayan tekrarlayan sinir ağı yaklaşımıdır."
    )
    st.write(
        "Tek bir 30 günlük giriş penceresinden beş günlük çıktıyı doğrudan "
        "üretir. Böylece beş ayrı bağımsız model yerine tahmin ufkunun tamamı "
        "tek model içinde oluşturulur."
    )

with st.expander("TFT Tabanlı Model"):
    st.write(
        "Çalışmada sadeleştirilmiş değişken seçimi, LSTM tabanlı zamansal "
        "kodlayıcı, multi-head attention ve gated residual yapı içeren TFT "
        "tabanlı bir mimari kullanılmıştır."
    )
    st.write(
        "Bu uygulama, özgün Temporal Fusion Transformer mimarisinin tüm "
        "bileşenlerinin eksiksiz bir uygulaması olarak sunulmamaktadır. "
        "Yalnızca tez deneyinde doğrulanan sadeleştirilmiş yapı ve sonuçlar "
        "gösterilmektedir."
    )

with st.expander("Chronos"):
    st.write(
        "Tez deneyinde `amazon/chronos-t5-small` modeli zero-shot çıkarım "
        "yaklaşımıyla kullanılmıştır. Model proje verisi üzerinde yeniden "
        "eğitilmemiştir."
    )
    st.write(
        "Her tahminde 256 kapanış gözleminden oluşan bağlam kullanılmıştır. "
        "Merkez tahmin medyan, başka bir ifadeyle 0.50 quantile değeridir. "
        "0.10 ve 0.90 quantile değerleri ise gösterilen tahmin aralığının alt "
        "ve üst sınırlarını oluşturmaktadır."
    )

st.header("Değerlendirme Metrikleri")
metric_columns = st.columns(2)
with metric_columns[0]:
    with st.container(border=True):
        st.subheader("MAE")
        st.write(
            "Ortalama mutlak tahmin hatasıdır. Daha düşük değer daha iyi "
            "performansa işaret eder."
        )
    with st.container(border=True):
        st.subheader("R²")
        st.write(
            "Modelin hedef değişkendeki değişimi açıklama düzeyini gösterir. "
            "Daha yüksek değer genellikle daha iyidir; metrik negatif değer "
            "de alabilir."
        )

with metric_columns[1]:
    with st.container(border=True):
        st.subheader("RMSE")
        st.write(
            "Büyük hatalara daha fazla ağırlık veren hata ölçüsüdür. Daha "
            "düşük değer daha iyi performansa işaret eder."
        )
    with st.container(border=True):
        st.subheader("Yön Doğruluğu")
        st.write(
            "Modelin fiyat veya getiri yönünü doğru tahmin etme oranıdır. "
            "Fiyat hatası düşük bir modelin yön doğruluğu mutlaka yüksek "
            "olmayabilir."
        )

st.header("Deneysel Akış")
for start_index in range(0, len(EXPERIMENTAL_FLOW), 4):
    flow_items = EXPERIMENTAL_FLOW[start_index : start_index + 4]
    flow_columns = st.columns(len(flow_items))
    for column, flow_item in zip(flow_columns, flow_items, strict=True):
        with column:
            with st.container(border=True):
                st.markdown(f"**{flow_item}**")
                if flow_item != EXPERIMENTAL_FLOW[-1]:
                    st.caption("Sonraki aşamaya aktarılır →")

st.header("Sınırlılıklar")
st.markdown(
    """
    - Finansal piyasalarda yüksek düzeyde gürültü ve belirsizlik bulunur.
    - Geçmiş performans gelecekteki sonuçları garanti etmez.
    - Haberler, makroekonomik gelişmeler ve yatırımcı davranışları modele
      bütünüyle yansımayabilir.
    - Tahmin ufku uzadıkça ardışık hatalar birikebilir.
    - Sonuçlar kullanılan veri aralığına, özelliklere ve hiperparametrelere
      duyarlıdır.
    """
)

st.warning(
    "Bu metodoloji, akademik bir model karşılaştırma çalışmasına aittir. "
    "Modellerin ürettiği tahminler yatırım kararı vermek amacıyla "
    "kullanılmamalıdır."
)
