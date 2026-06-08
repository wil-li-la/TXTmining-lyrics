---
name: run-app
description: Use when asked to run, launch, serve, smoke-test, or screenshot the Pop Lyrics Taste Profiler Streamlit app locally — covers Playwright setup, the APP_PASSWORD gate, and the verified screenshot tour.
---

# Run the Pop Lyrics Taste Profiler

A Streamlit app (`app.py`). Two tabs: **🎧 Recommend** (taste sliders → OpenAI agent)
and **📊 Analyze** (decade profiles, z-score heatmap, 22 trend plots, classifier).
A password gate (`APP_PASSWORD` in `.env`) protects the OpenAI/Genius quota; it is
**disabled when `APP_PASSWORD` is unset**.

## One-time setup

No `chromium-cli` here — use Playwright in the venv:

```bash
venv/bin/pip install playwright
venv/bin/python -m playwright install chromium
```

## Launch + stop

macOS has no `timeout`; poll the port instead. Use 8502 (8501 is often taken):

```bash
venv/bin/streamlit run app.py --server.port 8502 --server.headless true \
  --browser.gatherUsageStats false > /tmp/st.log 2>&1 &
echo $! > /tmp/st.pid
for i in $(seq 1 30); do curl -sf http://localhost:8502/ >/dev/null && break; sleep 1; done
# ... drive it ...
kill $(cat /tmp/st.pid)   # stop when done
```

`app.py` calls `load_dotenv()`, so the running app reads `APP_PASSWORD` from `.env`
and the gate is active. The driver below reads the same value — never hardcode it.

## Screenshot tour

```bash
venv/bin/python .claude/skills/run-app/capture_screenshots.py
# → output/screenshots/comprehensive/ (12 retina PNGs: gate, both tabs, sections)
```

For a one-shot smoke test, that the gate clears and `text=Your taste profile`
appears is enough proof the app is up.

## Gotchas (hit these, don't rediscover them)

- **`full_page=True` does NOT capture the whole Analyze tab.** Streamlit renders
  content in an internal scroll container. Scroll the target element into view
  (`scrollIntoView({block:'center'})`) and shoot the viewport instead.
- **Slider help** = `[data-testid="stTooltipIcon"]` (6, one per slider). Hover, then
  wait for `[data-testid="stTooltipContent"]`.
- **Decade selectbox is a type-to-filter combobox.** Older decades scroll out of the
  popover — click `[data-testid="stSelectbox"] input`, `type("1960")`, press Enter.
  Clicking a `role=option` for an off-screen value times out.
- **Sliders:** `eval value=` won't register. Click the `div[role="slider"]` thumb and
  press Arrow keys.
- **No genre breakdown.** `data/song_features.parquet` has one genre value, so the
  app shows "Genre breakdown will appear…". Nothing to screenshot there.
- **Do NOT click "Find matching recent songs"** unless asked — it runs the live
  OpenAI + Genius agent loop and spends the very quota the gate protects.
