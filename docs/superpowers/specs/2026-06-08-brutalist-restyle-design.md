# Design — taste-skill install + brutalist Streamlit restyle

**Date:** 2026-06-08
**Status:** Approved (user set goal "implement this")
**Source:** Implement https://github.com/leonxlnx/taste-skill against the Pop Lyrics Taste Profiler.

## Goal

Two deliverables:

1. **Install** the `taste-skill` repo's agent skills into this environment so they are
   usable via the `Skill` tool.
2. **Restyle** the existing Streamlit app (Pop Lyrics Taste Profiler) using the repo's
   `brutalist-skill` design language — **presentation only**, no logic changes.

The user chose: install **and** redesign → redesign target = **restyle the existing
Streamlit app** (not a new landing page or a Streamlit replacement).

## Locked design decisions

| Decision | Value |
|---|---|
| Aesthetic | Brutalist — **Swiss Industrial Print** (light), per `brutalist-skill` |
| Substrate | Paper `#F4F4F0`, ink `#111111`, 2px ink hairlines |
| Accent | **Acid Yellow `#E9FF3A`** (ink text on yellow). Single accent. |
| Heatmap | Keep `RdBu_r` diverging (encodes z-score sign + magnitude — data-meaningful) |
| Type | **Archivo** (900, UPPERCASE, tight tracking) headers/hero; **JetBrains Mono** for all metadata/numbers/labels/buttons |
| Shape | `border-radius: 0`, hard 2px borders, 3–4px hard offset shadows |
| Texture | One faint fixed grain layer only (`pointer-events:none`, behind widgets) |
| Markers | ASCII only — `[ ]`, `//`, `▸`, `®`, `+`. **No emoji.** |
| Trend PNGs | Regenerate 22 plots in brutalist matplotlib style |
| Deploy | Apply + verify locally, update HF metadata, push to HF Space |

## Skill install

- Location: `~/.claude/skills/` (user-global; same place existing skills live; reusable).
- All 13 skills from the repo, each in a directory named by its frontmatter `name`
  (e.g. `design-taste-frontend/`, `industrial-brutalist-ui/`, …) containing `SKILL.md`
  plus any sibling assets.

## Architecture — where the style lives

- **`.streamlit/config.toml`** (new) — base light theme (paper bg, ink text, yellow
  primary) so Streamlit chrome matches even before injected CSS loads.
- **`src/theme.py`** (new) — single source of truth for the design:
  - `TOKENS` dict (colors, fonts).
  - `inject_css()` — the `<style>` block: Google-Fonts `@import` (Archivo + JetBrains
    Mono), grain layer, and **resilient selectors** for title/sidebar/tabs/sliders/
    buttons/inputs/dataframes/metrics.
  - `altair_theme()` — ink+yellow on paper theme for `st.bar_chart`.
  - HTML helpers: `hero()`, `section(label)`, `result_card(...)`.
- **`app.py`** (modify) — call `inject_css()` after `set_page_config`; brutalist hero
  block replacing emoji title; uppercase ASCII tab labels (`[ RECOMMEND ]` /
  `[ ANALYZE ]`); restyle password gate; wrap result cards in brutalist card helper.
  **Logic untouched.**
- **`src/trends.py`** (modify) — brutalist matplotlib rcParams (paper face, ink
  spines/text, **yellow** LOESS line, **DejaVu Sans Mono** — always present in the
  Docker image, no font install needed, square frame). Regenerate all 22 PNGs.
- **`README.md`** (modify) — HF frontmatter `colorFrom`/`colorTo` blue/purple → neutral.

## Resilience note

Streamlit auto-generates hashed CSS classnames that change between versions. Injected
CSS targets **stable `data-testid` attributes** (`stSlider`, `stButton`, `stTabs`,
`stDataFrame`, `stMetric`, `stAppViewContainer`, `stSidebar`) and semantic structure —
not hashed classes. The rendered DOM is verified with the `run-app` skill rather than
trusting selectors blind. This is the known-fragile part.

## Verification & deploy

- `run-app` skill: launch with the `APP_PASSWORD` gate; screenshot gate + Recommend +
  Analyze; confirm look and that widgets still function.
- Update HF frontmatter; push to the Hugging Face Space (`hf` CLI / existing remote).

## Out of scope

App logic, the agent, ranking, data pipeline, feature extraction. Presentation only,
plus regenerating trend PNGs from already-committed data (no API calls).
