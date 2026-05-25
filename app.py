"""Streamlit UI for the lyrics recommendation agent."""
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.agent.ranking import load_stats, rank
from src.agent.runner import run

load_dotenv()
st.set_page_config(page_title="Pop Lyrics Taste Profiler", layout="wide")

SLIDERS = {
    "valence":             ("Emotional valence (sad ↔ happy)", -2.0, 2.0),
    "arousal":             ("Arousal (calm ↔ intense)", -2.0, 2.0),
    "chorus_repeat_ratio": ("Repetition (unique ↔ chant-like)", -2.0, 2.0),
    "rhyme_density":       ("Rhyme density (sparse ↔ dense)", -2.0, 2.0),
    "mean_concreteness":   ("Concreteness (abstract ↔ vivid)", -2.0, 2.0),
    "pronoun_i":           ("Self-focus (outward ↔ I-heavy)", -2.0, 2.0),
}

st.title("🎵 Pop Lyrics Taste Profiler")
st.caption("Set your lyrical preferences, and the agent will find recent songs that match.")


@st.cache_data
def load_features():
    return pd.read_parquet("data/song_features.parquet")


try:
    feat = load_features()
    stats = load_stats()
    data_loaded = True
except Exception as e:
    st.warning(f"Could not load corpus data: {e}. Run the offline pipeline first (see README).")
    feat = None
    stats = None
    data_loaded = False

with st.sidebar:
    st.header("Corpus overview")
    if data_loaded and feat is not None:
        st.metric("Songs", len(feat))
        st.metric("Years", f"{int(feat['year'].min())}–{int(feat['year'].max())}")
        st.bar_chart(feat.groupby("decade").size())
    else:
        st.info("Pipeline not yet run.")

st.subheader("Your taste profile")
cols = st.columns(2)
profile: dict[str, float] = {}
for i, (key, (label, lo, hi)) in enumerate(SLIDERS.items()):
    with cols[i % 2]:
        profile[key] = st.slider(label, lo, hi, 0.0, 0.1, key=key)

go = st.button("Find matching recent songs", type="primary", disabled=not data_loaded)

if go and data_loaded:
    profile_text = "\n".join(
        f"- {SLIDERS[k][0]}: {v:+.1f} z" for k, v in profile.items()
    )
    candidates: list[dict] = []
    trace_lines: list[str] = []

    st.subheader("Agent trace")
    trace_box = st.empty()

    def trace_sink(line: str) -> None:
        trace_lines.append(line)
        trace_box.code("\n".join(trace_lines[-30:]), language="text")

    with st.spinner("Agent is searching…"):
        final_msg = run(profile_text, candidate_sink=candidates, trace_sink=trace_sink)

    st.subheader("Agent reasoning")
    st.write(final_msg)

    st.subheader("Top recommendations")
    if not candidates:
        st.warning("Agent did not return any scored candidates. Try widening your profile.")
    else:
        ranked = rank(candidates, profile, stats)[:5]
        for i, c in enumerate(ranked, 1):
            with st.container():
                st.markdown(f"**{i}. {c['artist']} — {c['title']}**  (similarity {c['score']:.2f})")
                feature_keys = list(SLIDERS.keys())
                comparison = pd.DataFrame({
                    "feature": feature_keys,
                    "you": [profile[k] for k in feature_keys],
                    "song": [
                        (c["features"].get(k, 0.0) - stats[k]["mean"]) / max(stats[k]["std"], 1e-9)
                        for k in feature_keys
                    ],
                })
                st.bar_chart(comparison.set_index("feature"))
