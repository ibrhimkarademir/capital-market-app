"""Ana dashboard için tekrar kullanılabilir Streamlit arayüz bileşenleri."""

from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st


def _safe_text(value: object, fallback: str = "—") -> str:
    """HTML içeriğinde kullanılacak değeri güvenli ve okunabilir metne dönüştürür."""
    text = str(value).strip() if value is not None else ""
    return escape(text or fallback)


def load_css(css_file_path: str | Path) -> bool:
    """CSS dosyasını yükler; dosya yoksa uygulamayı durdurmadan devam eder."""
    path = Path(css_file_path)

    try:
        css_content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False

    if not css_content.strip():
        return False

    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    return True


def render_hero() -> None:
    """Projenin ana tanıtım alanını oluşturur."""
    st.markdown(
        """
        <section class="dashboard-hero">
            <div class="hero-eyebrow">
                MATEMATİK MÜHENDİSLİĞİ BİTİRME PROJESİ
            </div>
            <h1 class="hero-title">
                Sermaye Piyasalarında Dinamik Fiyat Tahmini
            </h1>
            <p class="hero-subtitle">
                LSTM, TFT tabanlı model ve Chronos yaklaşımlarıyla
                kısa vadeli finansal zaman serisi analizi
            </p>
            <p class="hero-description">
                Bu uygulama, geçmiş piyasa verilerinden yararlanarak Apple Inc.
                hissesi için beş işlem günlük fiyat tahmini yaklaşımını
                görselleştirmektedir.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    title: object,
    value: object,
    description: object | None = None,
) -> None:
    """Başlık, değer ve isteğe bağlı açıklama içeren özet kartı gösterir."""
    description_html = (
        f'<p class="metric-description">{_safe_text(description)}</p>'
        if description
        else ""
    )
    st.markdown(
        f"""
        <article class="metric-card">
            <p class="metric-title">{_safe_text(title)}</p>
            <p class="metric-value">{_safe_text(value)}</p>
            {description_html}
        </article>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(
    title: object,
    description: object | None = None,
) -> None:
    """Dashboard bölümleri için ortak başlık alanı oluşturur."""
    description_html = (
        f'<p class="section-description">{_safe_text(description)}</p>'
        if description
        else ""
    )
    st.markdown(
        f"""
        <header class="section-header">
            <h2 class="section-title">{_safe_text(title)}</h2>
            {description_html}
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_model_card(
    model_name: object,
    model_description: object,
    status: object,
) -> None:
    """Model açıklamasını ve entegrasyon durumunu sonuç üretmeden gösterir."""
    st.markdown(
        f"""
        <article class="model-card">
            <div class="status-badge">{_safe_text(status)}</div>
            <h3 class="model-name">{_safe_text(model_name)}</h3>
            <p class="model-description">{_safe_text(model_description)}</p>
        </article>
        """,
        unsafe_allow_html=True,
    )


def render_info_box(title: object, description: object | None = None) -> None:
    """Veri kaynağı gibi tamamlayıcı bilgileri sade bir kutuda gösterir."""
    description_html = (
        f'<p class="info-description">{_safe_text(description)}</p>'
        if description
        else ""
    )
    st.markdown(
        f"""
        <aside class="info-box">
            <p class="info-title">{_safe_text(title)}</p>
            {description_html}
        </aside>
        """,
        unsafe_allow_html=True,
    )


def render_process_card(
    number: int | str,
    title: str,
    description: str,
) -> None:
    """Bir süreç adımını yalnızca yerel Streamlit bileşenleriyle gösterir."""
    invalid_types = (tuple, list, dict, set)

    if isinstance(number, invalid_types) or number is None:
        number_text = "?"
    else:
        number_text = str(number).strip() or "?"

    if isinstance(title, invalid_types):
        title_text = "Geçersiz başlık verisi"
    elif title is None:
        title_text = "Başlık bulunamadı"
    else:
        title_text = str(title).strip() or "Başlık bulunamadı"

    if isinstance(description, invalid_types):
        description_text = "Geçersiz açıklama verisi."
    elif description is None:
        description_text = ""
    else:
        description_text = str(description).strip()

    with st.container(border=True):
        st.markdown(f"### {number_text}")
        st.markdown(f"**{title_text}**")
        st.caption(description_text)


def render_academic_notice() -> None:
    """Akademik kullanım ve yatırım tavsiyesi uyarısını gösterir."""
    st.markdown(
        """
        <aside class="academic-notice">
            <strong>Akademik kullanım uyarısı</strong>
            <p>
                Bu uygulama yalnızca akademik ve deneysel amaçlarla
                geliştirilmiştir. Gösterilen analizler ve gelecekte
                oluşturulacak model tahminleri yatırım tavsiyesi niteliği
                taşımaz.
            </p>
        </aside>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Ana sayfanın sade alt bilgi alanını oluşturur."""
    st.markdown(
        """
        <footer class="dashboard-footer">
            Yeni Nesil Yapay Zekâ Yaklaşımları ile Finansal Zaman Serisi Tahmini
        </footer>
        """,
        unsafe_allow_html=True,
    )
