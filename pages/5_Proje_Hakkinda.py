"""Bitirme projesinin kapsamını ve geliştirme durumunu açıklayan sayfa."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.presentation_mode import (
    apply_presentation_mode_styles,
    render_presentation_toggle,
)
from utils.ui_components import load_css


STYLE_PATH = Path(__file__).resolve().parents[1] / "assets" / "style.css"
PROJECT_SCOPE = [
    "Geçmiş piyasa verilerinin analizi",
    "Finansal verilerin ön işlenmesi",
    "Log-getiri tabanlı hedef oluşturulması",
    "Beş günlük doğrudan tahmin",
    "LSTM, TFT Tabanlı Model ve Chronos karşılaştırması",
    "Tahmin sonuçlarının interaktif biçimde görselleştirilmesi",
]

load_css(STYLE_PATH)
render_presentation_toggle()
apply_presentation_mode_styles()

ARCHITECTURE_ITEMS = [
    ("pages", "Uygulamanın çok sayfalı kullanıcı arayüzü"),
    ("utils", "Veri, grafik, sonuç ve arayüz yardımcıları"),
    ("data", "Piyasa verileri ve merkezi sonuç dosyaları"),
    ("models", "Doğrulanmış eğitilmiş model dosyaları"),
    ("results", "Tahmin ve grafik çıktıları"),
    ("assets", "Görsel stil dosyaları"),
]

COMPLETED_ITEMS = [
    "Temel Streamlit iskeleti",
    "Veri yükleme ve doğrulama",
    "AAPL veri analizi",
    "Ana dashboard",
    "Model sonuç altyapısı",
    "Model karşılaştırma arayüzü",
    "Doğru dönemli LSTM sonuçlarının aktarılması",
    "TFT Tabanlı Model sonuçlarının aktarılması",
    "İki modelin rolling test karşılaştırması",
    "İki modelin ortak tahmin grafiği",
    "Chronos zero-shot sonuçlarının aktarılması",
    "Üç modelin test metriklerinin karşılaştırılması",
    "Chronos tahmin aralığının görselleştirilmesi",
    "Üç modelin beş günlük tahmin grafiği",
    "Metodoloji sayfası",
    "Proje hakkında sayfası",
    "Sunum modu",
    "Otomatik bütünlük testleri",
    "Production requirements hazırlığı",
    "Deployment kontrol listesi",
]

PENDING_ITEMS = [
    "Kullanıcı onayı sonrası gerçek deployment",
]


def render_list_card(title: str, items: list[str], icon: str) -> None:
    """Başlıklı bir madde listesini yerel Streamlit kartında gösterir."""
    with st.container(border=True):
        st.subheader(f"{icon} {title}")
        for item in items:
            st.markdown(f"- {item}")


st.title("Proje Hakkında")

st.header("Proje Özeti")
st.write(
    "Bu bitirme projesi, yeni nesil yapay zekâ yaklaşımlarını finansal zaman "
    "serisi tahmini bağlamında karşılaştırmak amacıyla geliştirilmiştir. "
    "Çalışmada Apple Inc. (AAPL) hissesine ait geçmiş piyasa verileri "
    "incelenmekte; LSTM, TFT Tabanlı Model ve Chronos yaklaşımlarının "
    "beş işlem günlük tahmin problemindeki sonuçlarının ortak bir akademik "
    "çerçevede değerlendirilmesi hedeflenmektedir."
)

st.header("Projenin Kapsamı")
scope_columns = st.columns(2)
for index, scope_item in enumerate(PROJECT_SCOPE):
    with scope_columns[index % 2]:
        with st.container(border=True):
            st.markdown(f"**{scope_item}**")

st.header("Uygulamanın Bölümleri")
first_page_row = st.columns(3)
with first_page_row[0]:
    st.page_link(
        "app.py",
        label="Ana Dashboard",
        icon="🏠",
        help="Proje parametreleri, güncel veri özeti ve genel uygulama akışı.",
        use_container_width=True,
    )
with first_page_row[1]:
    st.page_link(
        "pages/1_Veri_Analizi.py",
        label="Veri Analizi",
        icon="📊",
        help="AAPL fiyat, hacim, getiri ve volatilite incelemeleri.",
        use_container_width=True,
    )
with first_page_row[2]:
    st.page_link(
        "pages/2_Fiyat_Tahmini.py",
        label="Fiyat Tahmini",
        icon="🔭",
        help="LSTM, TFT Tabanlı Model ve Chronos sabit tez tahminleri.",
        use_container_width=True,
    )

second_page_row = st.columns(3)
with second_page_row[0]:
    st.page_link(
        "pages/3_Model_Karsilastirma.py",
        label="Model Karşılaştırma",
        icon="⚖️",
        help="Üç modelin test metrikleri ve rolling sonuçları.",
        use_container_width=True,
    )
with second_page_row[1]:
    st.page_link(
        "pages/4_Metodoloji.py",
        label="Metodoloji",
        icon="🧭",
        help="Veri hazırlama, modelleme ve değerlendirme yaklaşımı.",
        use_container_width=True,
    )
with second_page_row[2]:
    st.page_link(
        "pages/5_Proje_Hakkinda.py",
        label="Proje Hakkında",
        icon="🎓",
        help="Projenin kapsamı, mimarisi ve mevcut geliştirme durumu.",
        use_container_width=True,
    )

st.header("Teknolojiler")
technology_columns = st.columns(2)
with technology_columns[0]:
    render_list_card(
        "Mevcut Uygulama Altyapısı",
        [
            "Python",
            "Streamlit",
            "Pandas",
            "NumPy",
            "Plotly",
            "Yahoo Finance / yfinance",
        ],
        "🛠️",
    )
with technology_columns[1]:
    with st.container(border=True):
        st.subheader("🧠 Model Teknolojileri")
        st.write(
            "LSTM ve TFT Tabanlı Model'in doğrulanmış tez çıktıları ile Chronos "
            "zero-shot sonuçları sisteme aktarılmıştır. Uygulama kayıtlı tez "
            "çıktılarını gösterir; modelleri yeniden çalıştırmaz."
        )
        st.info("Durum: Üç modelin tez sonuçları entegre edildi")

st.header("Proje Dosya Mimarisi")
architecture_columns = st.columns(3)
for index, (folder_name, description) in enumerate(ARCHITECTURE_ITEMS):
    with architecture_columns[index % 3]:
        with st.container(border=True):
            st.markdown(f"**{folder_name}**")
            st.caption(description)

#st.header("Mevcut Geliştirme Durumu")
#status_columns = st.columns(2)
#with status_columns[0]:
#    render_list_card("Tamamlanan", COMPLETED_ITEMS, "✅")
#with status_columns[1]:
#    render_list_card("Bekleyen", PENDING_ITEMS, "⏳")

st.warning(
    "Bu uygulama bir lisans bitirme projesi kapsamında geliştirilmiştir. "
    "İçerdiği tahminler, grafikler ve analizler yalnızca eğitim ve araştırma "
    "amaçlıdır. Yatırım tavsiyesi değildir."
)
