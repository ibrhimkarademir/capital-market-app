"""Streamlit sayfaları için ortak sunum modu yardımcıları."""

from __future__ import annotations

import streamlit as st


PRESENTATION_MODE_KEY = "presentation_mode_enabled"


def is_presentation_mode() -> bool:
    """Sunum modunun mevcut oturumda açık olup olmadığını döndürür."""
    return bool(st.session_state.get(PRESENTATION_MODE_KEY, False))


def render_presentation_toggle() -> bool:
    """Kenar çubuğunda tüm sayfalarla ortak çalışan sunum modu anahtarını gösterir."""
    if PRESENTATION_MODE_KEY not in st.session_state:
        st.session_state[PRESENTATION_MODE_KEY] = False

    with st.sidebar:
        st.caption("Görünüm")
        st.toggle(
            "Sunum Modu",
            key=PRESENTATION_MODE_KEY,
            help=(
                "Başlıkları, kartları ve grafikleri geniş ekran sunumuna daha "
                "uygun hâle getirir. Akademik uyarılar gizlenmez."
            ),
        )

    return is_presentation_mode()


def apply_presentation_mode_styles() -> bool:
    """Sunum modu açıksa güvenli CSS dokunuşlarını uygular."""
    if not is_presentation_mode():
        return False

    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1680px;
                padding-top: 1.4rem;
                padding-bottom: 4rem;
            }

            h1 {
                font-size: clamp(2.2rem, 4vw, 3.6rem) !important;
                line-height: 1.08 !important;
            }

            h2,
            h3 {
                letter-spacing: -0.02em;
            }

            [data-testid="stMetric"],
            [data-testid="stVerticalBlockBorderWrapper"] {
                min-width: 0;
            }

            [data-testid="stMetric"] {
                padding: 0.35rem 0;
            }

            .metric-card,
            .model-card,
            .academic-notice,
            .info-box {
                box-shadow: 0 14px 34px rgba(0, 0, 0, 0.16);
            }

            .metric-card {
                min-height: 165px;
            }

            .model-card {
                min-height: 260px;
            }

            .stPlotlyChart {
                width: 100%;
                padding-top: 0.35rem;
                padding-bottom: 0.75rem;
            }

            [data-testid="stDataFrameResizable"] {
                width: 100%;
            }

            [data-testid="stToolbar"],
            [data-testid="stStatusWidget"],
            [data-testid="stDecoration"],
            [data-testid="stDeployButton"],
            #MainMenu,
            footer {
                visibility: hidden;
                height: 0;
            }

            [data-testid="stSidebar"] {
                min-width: 18rem;
            }

            @media (max-width: 760px) {
                .block-container {
                    padding-left: 0.8rem;
                    padding-right: 0.8rem;
                }

                .metric-card,
                .model-card {
                    min-height: auto;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    return True
