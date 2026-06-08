"""Brutalist "Swiss Industrial Print" theme for the Streamlit app.

Single source of truth for the look defined by the taste-skill `industrial-brutalist-ui`
skill: documentation-paper substrate, carbon ink, one acid-yellow accent, Archivo Black
headers + JetBrains Mono telemetry, hard 2px borders, square corners, faint grain.

Structural theme (colors, square corners, ink borders, named fonts, chart colors) lives
in .streamlit/config.toml. This module loads the fonts and applies everything config
cannot reach, via one injected <style> block, plus small HTML builders for the hero and
result cards. Import is presentation-only; no app logic depends on it.

CSS targets stable Streamlit `data-testid` attributes and baseweb hooks rather than
hashed classnames, so it survives Streamlit version bumps.
"""
from __future__ import annotations

import html

import streamlit as st

# Design tokens — mirror .streamlit/config.toml so the HTML builders stay in sync.
TOKENS = {
    "paper": "#F4F4F0",
    "paper2": "#ECEAE2",
    "ink": "#111111",
    "accent": "#E9FF3A",
    "muted": "#6B6B63",
}

# Faint fractal-noise grain, rendered once as a fixed, click-through overlay.
_GRAIN = (
    "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E"
    "%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.82' "
    "numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E"
    "%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")"
)

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

/* ---- faint analog grain over the whole viewport (clicks pass through) ---- */
[data-testid="stAppViewContainer"]::before {
  content: ""; position: fixed; inset: 0; z-index: 100;
  pointer-events: none; mix-blend-mode: multiply; opacity: .05;
  background-image: __GRAIN__;
}

/* ---- chrome: blend Streamlit's top bar into the paper ---- */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppViewContainer"] .block-container { padding-top: 2.4rem; max-width: 1180px; }

/* ---- macro typography: headings as solid uppercase blocks ---- */
h1, h2, h3, h4,
[data-testid="stHeading"] h1, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {
  font-family: 'Archivo', sans-serif !important;
  font-weight: 900 !important;
  text-transform: uppercase !important;
  letter-spacing: -0.02em !important;
  line-height: 1.0 !important;
}

/* ---- micro typography: captions read as telemetry ---- */
[data-testid="stCaptionContainer"], .stCaption, [data-testid="stCaptionContainer"] p {
  font-family: 'JetBrains Mono', monospace !important;
  text-transform: uppercase; letter-spacing: .06em;
  font-size: .72rem !important; color: #6B6B63 !important;
}

/* ---- hero block ---- */
.bx-hero { border: 2px solid #111; padding: 18px 20px 20px; margin: 0 0 14px;
  box-shadow: 6px 6px 0 #111; background: #F4F4F0; }
.bx-telemetry { display: flex; justify-content: space-between; gap: 12px;
  font-family: 'JetBrains Mono', monospace; font-size: .62rem; letter-spacing: .12em;
  text-transform: uppercase; border-bottom: 1px solid #111; padding-bottom: 7px; color: #111; }
.bx-title { font-family: 'Archivo', sans-serif; font-weight: 900; text-transform: uppercase;
  letter-spacing: -0.035em; line-height: .9; margin: 14px 0 6px;
  font-size: clamp(2.1rem, 6vw, 4rem); color: #111; }
.bx-sub { font-family: 'JetBrains Mono', monospace; font-size: .74rem; letter-spacing: .05em;
  text-transform: uppercase; color: #111; opacity: .85; }

/* ---- eyebrow label ([ SECTION ]) ---- */
.bx-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: .68rem; font-weight: 700;
  letter-spacing: .14em; text-transform: uppercase; color: #111;
  border-left: 4px solid #E9FF3A; padding-left: 9px; margin: 4px 0 2px; }

/* ---- sidebar as a hard compartment ---- */
[data-testid="stSidebar"] { border-right: 2px solid #111; }

/* ---- metrics ---- */
[data-testid="stMetricValue"] { font-family: 'Archivo', sans-serif !important;
  font-weight: 900 !important; letter-spacing: -.02em; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
  font-family: 'JetBrains Mono', monospace !important; text-transform: uppercase;
  letter-spacing: .08em; font-size: .68rem !important; }

/* ---- tabs: uppercase mono, yellow active underline ---- */
button[data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace !important;
  text-transform: uppercase; letter-spacing: .08em; font-weight: 700 !important; }
/* active tab text defaults to the yellow primary (invisible on paper) — force ink. */
button[data-baseweb="tab"], button[data-baseweb="tab"][aria-selected="true"],
button[data-baseweb="tab"][aria-selected="true"] * { color: #111 !important; }
[data-baseweb="tab-highlight"] { background-color: #E9FF3A !important; height: 4px !important; }
[data-baseweb="tab-border"] { background-color: #111 !important; }

/* ---- buttons: hard block + offset shadow, mechanical press ---- */
.stButton > button,
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {
  border: 2px solid #111 !important; border-radius: 0 !important;
  box-shadow: 4px 4px 0 #111 !important; color: #111 !important;
  font-family: 'JetBrains Mono', monospace !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: .06em !important;
  transition: transform .04s ease, box-shadow .04s ease; }
.stButton > button:hover { transform: translate(2px, 2px);
  box-shadow: 2px 2px 0 #111 !important; }
.stButton > button:active { transform: translate(4px, 4px);
  box-shadow: 0 0 0 #111 !important; }

/* ---- sliders: square thumb, ink outline; ink value labels (default yellow = invisible) ---- */
[data-testid="stSlider"] div[role="slider"] {
  border-radius: 0 !important; border: 2px solid #111 !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stThumbValue"], [data-testid="stSliderThumbValue"] {
  color: #111 !important; font-family: 'JetBrains Mono', monospace !important; }

/* ---- dataframe / heatmap: hard frame, mono numerals ---- */
[data-testid="stDataFrame"] { border: 2px solid #111 !important; }
[data-testid="stDataFrame"] [role="gridcell"], [data-testid="stDataFrame"] [role="columnheader"] {
  font-family: 'JetBrains Mono', monospace !important; }

/* ---- alerts / code: square, ink-bordered ---- */
[data-testid="stAlert"] { border-radius: 0 !important; border: 2px solid #111 !important; }
[data-testid="stCode"], pre { border-radius: 0 !important; border: 2px solid #111 !important; }

/* ---- bordered containers (result cards) ---- */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 0 !important; }

/* ---- result card header ---- */
.bx-res { display: flex; align-items: baseline; gap: 12px; }
.bx-res-rank { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.4rem;
  color: #111; background: #E9FF3A; padding: 1px 8px; border: 2px solid #111; }
.bx-res-title { font-family: 'Archivo', sans-serif; font-weight: 900; text-transform: uppercase;
  letter-spacing: -.01em; font-size: 1.02rem; line-height: 1.05; color: #111; }
.bx-res-sim { font-family: 'JetBrains Mono', monospace; font-size: .66rem; letter-spacing: .08em;
  text-transform: uppercase; color: #6B6B63; margin-top: 2px; }
"""


def inject_css() -> None:
    """Inject the brutalist stylesheet. Call once, right after st.set_page_config."""
    css = _CSS.replace("__GRAIN__", _GRAIN)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def hero() -> None:
    """Render the masthead hero (replaces the old emoji st.title)."""
    st.markdown(
        """
<div class="bx-hero">
  <div class="bx-telemetry"><span>POP-LYRICS-PROFILER &reg;</span><span>REV 2.6 &middot; 1965&ndash;2025 +</span></div>
  <div class="bx-title">Pop Lyrics<br>Taste Profiler</div>
  <div class="bx-sub">// set your taste &middot; the agent finds matching recent songs</div>
</div>
""",
        unsafe_allow_html=True,
    )


def eyebrow(label: str) -> None:
    """Small monospace `[ LABEL ]` section marker."""
    st.markdown(
        f'<div class="bx-eyebrow">[ {html.escape(label.upper())} ]</div>',
        unsafe_allow_html=True,
    )


def result_header(rank: int, artist: str, title: str, score: float) -> None:
    """Brutalist header row for one ranked recommendation."""
    st.markdown(
        f'<div class="bx-res">'
        f'<div class="bx-res-rank">{rank:02d}</div>'
        f'<div><div class="bx-res-title">{html.escape(artist)} &mdash; {html.escape(title)}</div>'
        f'<div class="bx-res-sim">similarity {score:.2f}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
