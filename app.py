"""Streamlit UI: Recommend (taste profiler) + Analyze (decade/genre style report)."""
import hmac
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Pop Lyrics Taste Profiler", layout="wide")

# Brutalist "Swiss Industrial Print" theme (taste-skill: industrial-brutalist-ui).
# Inject before the gate so the password screen is styled too. Presentation only.
from src.theme import eyebrow, hero, inject_css, result_header

inject_css()


# ----------------------------------------------------------------------
# Password gate (protects OpenAI quota when deployed publicly).
# If APP_PASSWORD env var is unset, the gate is disabled (local dev).
# ----------------------------------------------------------------------
def _password_gate() -> None:
    expected = os.environ.get("APP_PASSWORD")
    if not expected:
        return  # gate disabled
    if st.session_state.get("auth_ok"):
        return

    eyebrow("Restricted Access")
    st.markdown("## Pop Lyrics Taste Profiler")
    st.caption("This demo is restricted to authorized viewers (password-gated to protect API quota).")
    pw = st.text_input("Password", type="password", key="_pw_input")
    if pw:
        if hmac.compare_digest(pw, expected):
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


_password_gate()

# Imports the agent (which initializes OpenAI client) happen AFTER the gate
# so unauthenticated visitors don't trigger client construction.
from src.agent.ranking import load_stats, rank
from src.agent.runner import run

# Slider config — each entry: (label, range_lo, range_hi, help_text_with_examples)
SLIDERS: dict[str, tuple[str, float, float, str]] = {
    "valence": (
        "Emotional valence (sad ↔ happy)", -2.0, 2.0,
        "How positive vs. negative the lyric mood feels.\n\n"
        "**Low (sad):** Adele — *Someone Like You*\n\n"
        "**High (happy):** Pharrell — *Happy*",
    ),
    "arousal": (
        "Arousal (calm ↔ intense)", -2.0, 2.0,
        "Energy / intensity of the emotional language.\n\n"
        "**Low (calm):** Billie Eilish — *ocean eyes*\n\n"
        "**High (intense):** Kendrick Lamar — *DNA.*",
    ),
    "chorus_repeat_ratio": (
        "Repetition (verse-driven ↔ chant-like)", -2.0, 2.0,
        "How much the same lines repeat across the song.\n\n"
        "**Low (verse-driven):** Bob Dylan — *Like a Rolling Stone* (every line different)\n\n"
        "**High (chant-like):** Daft Punk — *Around the World* (same phrase over and over)",
    ),
    "rhyme_density": (
        "Rhyme density (free-verse ↔ rhyme-packed)", -2.0, 2.0,
        "Fraction of line-ending words that rhyme with nearby lines.\n\n"
        "**Low (free-verse):** Joni Mitchell — *Both Sides Now*\n\n"
        "**High (rhyme-packed):** Eminem — *Lose Yourself*",
    ),
    "mean_concreteness": (
        "Concreteness (abstract ↔ vivid)", -2.0, 2.0,
        "Whether the lyrics talk about ideas/feelings or physical objects/scenes.\n\n"
        "**Low (abstract):** Sia — *Chandelier* (freedom, fear, control)\n\n"
        "**High (vivid):** Taylor Swift — *All Too Well* (scarves, kitchens, dancing in fridge light)",
    ),
    "pronoun_i": (
        "Self-focus (outward 'we/they' ↔ inward 'I/me')", -2.0, 2.0,
        "How much the singer talks about themselves vs. others.\n\n"
        "**Low (outward):** USA for Africa — *We Are the World* ('we', 'they', 'us')\n\n"
        "**High (inward):** Olivia Rodrigo — *drivers license* ('I', 'me' on every line)",
    ),
}

# Hand-curated era descriptions used in the Analyze tab — anchored in known music history.
DECADE_NOTES: dict[str, str] = {
    "1960s": "Folk-rock, soul, and the British Invasion. Lyrically lean (avg ~37 word-types per song), "
             "with strong outward orientation and abstract themes (peace, love, freedom). Bob Dylan, "
             "The Beatles, Aretha Franklin, Motown.",
    "1970s": "Singer-songwriter era plus disco's rise. Vocabulary diversity ticks up. Funkadelic, "
             "Stevie Wonder, Carole King, Bee Gees. Disco brings more body-oriented and dance-floor "
             "imagery.",
    "1980s": "Synth-pop, hair metal, early hip-hop. Production becomes louder; lyrically still verse-driven "
             "(low chorus-repeat). Michael Jackson, Madonna, Prince, Run-DMC introduces more concrete "
             "narrative.",
    "1990s": "Grunge, R&B ballads, and gangsta rap. Emotional intensity rises (TLC, Whitney Houston). "
             "Self-reference creeps up. Vocabulary diversity hits a local peak around 1997-99.",
    "2000s": "Hip-hop dominates the Hot 100; club anthems and post-9/11 introspection coexist. Lyrical "
             "complexity (MTLD ≈ 41) is the highest of any decade — rap pushed vocabulary up. 50 Cent, "
             "Beyoncé, Usher, Coldplay, Eminem.",
    "2010s": "Genre-blending era — EDM, trap, country crossover. First decade where chorus repetition "
             "becomes measurable in our corpus, reflecting hook-driven production. Drake, Taylor Swift, "
             "Adele, Pharrell.",
    "2020s": "Bedroom pop, hyperpop, post-pandemic introspection. The MOST distinctive decade by our "
             "model (AUC ≈ 0.98). Highest vocabulary diversity (MTLD ≈ 54), highest chorus-repeat ratio "
             "(short hooks), and most self-focused. Olivia Rodrigo, Billie Eilish, Sabrina Carpenter, "
             "Doja Cat.",
}

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "valence": "Valence (positive ↔ negative)",
    "arousal": "Arousal (energy)",
    "chorus_repeat_ratio": "Chorus repetition",
    "rhyme_density": "Rhyme density",
    "mean_concreteness": "Concreteness",
    "mtld": "Vocabulary diversity (MTLD)",
    "pronoun_i": "Self-focus (I/me)",
    "pronoun_we": "Group-focus (we/us)",
    "emo_joy": "Joy",
    "emo_sadness": "Sadness",
    "emo_anger": "Anger",
    "emo_fear": "Fear",
    "emo_anticipation": "Anticipation",
    "emo_trust": "Trust",
}


@st.cache_data
def load_features() -> pd.DataFrame | None:
    try:
        return pd.read_parquet("data/song_features.parquet")
    except Exception:
        return None


feat = load_features()
try:
    stats = load_stats()
except Exception:
    stats = None
data_loaded = feat is not None and stats is not None

hero()

with st.sidebar:
    eyebrow("Corpus")
    st.header("Corpus overview")
    if data_loaded:
        st.metric("Songs", f"{len(feat):,}")
        st.metric("Years", f"{int(feat['year'].min())}–{int(feat['year'].max())}")
        st.bar_chart(feat.groupby("decade").size())
        if "genre" in feat.columns and feat["genre"].nunique() > 1:
            st.subheader("By genre")
            st.bar_chart(feat["genre"].value_counts())
    else:
        st.warning("Pipeline data not loaded. Run the offline pipeline (see README).")

tab_rec, tab_analyze = st.tabs(["[ Recommend ]", "[ Analyze ]"])


# ----------------------------------------------------------------------
# Recommend tab
# ----------------------------------------------------------------------
with tab_rec:
    st.caption("Set your lyrical preferences, and the agent will find recent songs that match.")
    st.subheader("Your taste profile")

    cols = st.columns(2)
    profile: dict[str, float] = {}
    for i, (key, (label, lo, hi, help_text)) in enumerate(SLIDERS.items()):
        with cols[i % 2]:
            profile[key] = st.slider(label, lo, hi, 0.0, 0.1, key=f"slider_{key}", help=help_text)

    go = st.button("Find matching recent songs ▸", type="primary", disabled=not data_loaded)

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
                with st.container(border=True):
                    result_header(i, c["artist"], c["title"], c["score"])
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


# ----------------------------------------------------------------------
# Analyze tab
# ----------------------------------------------------------------------
with tab_analyze:
    st.caption("How lyrics changed across decades and genres — based on the 5,205-song corpus.")

    if not data_loaded:
        st.info("Pipeline data not loaded.")
    else:
        # --- 1. Per-decade style profile ---
        st.subheader("Decade-by-decade style")

        decade_choice = st.selectbox(
            "Pick a decade to inspect",
            options=sorted(feat["decade"].unique()),
            index=len(feat["decade"].unique()) - 1,  # default to most recent
        )

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown(f"**{decade_choice} — what it sounds like**")
            st.write(DECADE_NOTES.get(decade_choice, "No notes for this era yet."))

            # Standout features for this decade vs corpus-wide
            dec_means = feat[feat["decade"] == decade_choice][list(FEATURE_DISPLAY_NAMES.keys())].mean()
            corpus_means = feat[list(FEATURE_DISPLAY_NAMES.keys())].mean()
            corpus_stds = feat[list(FEATURE_DISPLAY_NAMES.keys())].std()
            z_scores = ((dec_means - corpus_means) / corpus_stds.replace(0, 1)).sort_values()
            standout_low = z_scores.head(3)
            standout_high = z_scores.tail(3).iloc[::-1]

            st.markdown(f"**{decade_choice} is unusually HIGH in:**")
            for name, z in standout_high.items():
                st.write(f"- {FEATURE_DISPLAY_NAMES[name]} (z = {z:+.2f})")

            st.markdown(f"**Unusually LOW in:**")
            for name, z in standout_low.items():
                st.write(f"- {FEATURE_DISPLAY_NAMES[name]} (z = {z:+.2f})")

        with col_right:
            st.markdown(f"**{decade_choice} feature averages**")
            comp = pd.DataFrame({
                "z-score vs corpus": z_scores.reindex(list(FEATURE_DISPLAY_NAMES.keys())).rename(
                    lambda k: FEATURE_DISPLAY_NAMES[k]
                ),
            })
            st.bar_chart(comp)

        # --- 2. Cross-decade comparison heatmap ---
        st.subheader("All decades, all features — at a glance")
        st.caption("Each cell is a z-score: how much that decade's feature value deviates from the corpus average.")

        decades = sorted(feat["decade"].unique())
        heat_features = ["mtld", "chorus_repeat_ratio", "rhyme_density", "mean_concreteness",
                         "valence", "arousal", "pronoun_i", "pronoun_we", "emo_joy", "emo_sadness"]
        rows = []
        for dec in decades:
            row = {"decade": dec}
            sub = feat[feat["decade"] == dec]
            for f in heat_features:
                mean = sub[f].mean()
                z = (mean - feat[f].mean()) / max(feat[f].std(), 1e-9)
                row[FEATURE_DISPLAY_NAMES.get(f, f)] = z
            rows.append(row)
        heat_df = pd.DataFrame(rows).set_index("decade")
        st.dataframe(heat_df.style.background_gradient(cmap="RdBu_r", axis=None, vmin=-2, vmax=2)
                     .format("{:+.2f}"), use_container_width=True)

        # --- 3. Genre breakdown (if tagger has run) ---
        if "genre" in feat.columns and feat["genre"].nunique() > 1:
            st.subheader("By genre")
            genre_counts = feat["genre"].value_counts()
            st.caption(f"Genres tagged via OpenAI batch classification. Distribution across {len(feat):,} songs.")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.bar_chart(genre_counts)
            with col2:
                # Feature means by genre
                genre_heat = feat.groupby("genre")[heat_features].mean()
                for f in heat_features:
                    genre_heat[f] = (genre_heat[f] - feat[f].mean()) / max(feat[f].std(), 1e-9)
                genre_heat = genre_heat.rename(columns=FEATURE_DISPLAY_NAMES)
                st.dataframe(
                    genre_heat.style.background_gradient(cmap="RdBu_r", axis=None, vmin=-2, vmax=2)
                    .format("{:+.2f}"),
                    use_container_width=True,
                )
        else:
            st.info("Genre breakdown will appear after the genre tagger completes (running now in background).")

        # --- 4. Trend plots ---
        st.subheader("How features have shifted, year by year")
        trends_dir = Path("output/trends")
        if trends_dir.exists():
            trend_files = sorted(trends_dir.glob("*.png"))
            # Show curated featured trends first, then a grid of the rest
            featured = ["mtld.png", "chorus_repeat_ratio.png", "valence.png", "pronoun_i.png"]
            featured_paths = [trends_dir / f for f in featured if (trends_dir / f).exists()]

            st.markdown("**Highlights**")
            cols = st.columns(2)
            for i, p in enumerate(featured_paths):
                with cols[i % 2]:
                    st.image(str(p), caption=p.stem.replace("_", " "), use_container_width=True)

            with st.expander("Show all 22 trend plots"):
                grid = st.columns(3)
                for i, p in enumerate(trend_files):
                    with grid[i % 3]:
                        st.image(str(p), caption=p.stem.replace("_", " "), use_container_width=True)
        else:
            st.info("Trend plots not generated yet. Run: `venv/bin/python -m src.trends`.")

        # --- 5. Classifier result ---
        st.subheader("Can a model tell decades apart from lyrics alone?")
        cv_path = Path("output/cv_results.json")
        if cv_path.exists():
            import json
            cv = json.loads(cv_path.read_text())
            st.markdown(
                f"**Yes, modestly.** A logistic regression on the 410-dim feature vector "
                f"(rhyme + repetition + emotion + concreteness + pronouns + 384-d SBERT) "
                f"hits **{cv['stratified']['accuracy_mean']:.1%}** accuracy under stratified 5-fold "
                f"CV — well above the **{1/len(decades):.1%}** random baseline. "
                f"The 2020s alone are detected with AUC {cv['stratified']['auc_per_decade'].get('2020s', 0):.2f}."
            )
            st.markdown(
                f"**No artist leakage**: holding artists out (GroupKFold-by-artist) gives "
                f"**{cv['grouped']['accuracy_mean']:.1%}** accuracy — virtually identical. The model "
                f"is learning era characteristics, not memorising artists."
            )
            cv_table = pd.DataFrame({
                "Stratified 5-fold AUC": cv["stratified"]["auc_per_decade"],
                "GroupKFold-by-artist AUC": cv["grouped"]["auc_per_decade"],
            })
            st.dataframe(cv_table.style.format("{:.3f}"), use_container_width=True)
        else:
            st.info("CV results not yet produced. Run: `venv/bin/python -m src.analyze`.")
