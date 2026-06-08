#!/usr/bin/env python3
"""Comprehensive Playwright screenshot tour of the Pop Lyrics Taste Profiler.

Drives a headless Chromium through the password gate and both tabs, capturing a
verified set of screenshots. Run AFTER the Streamlit server is already serving.

The gate password is read from APP_PASSWORD (loaded from .env). If it is unset,
the app's gate is disabled and the auth steps are skipped automatically.

Usage:
    venv/bin/python .claude/skills/run-app/capture_screenshots.py \
        [--base http://localhost:8502] [--out output/screenshots/comprehensive]

Prereqs (one-time): venv/bin/pip install playwright && venv/bin/python -m playwright install chromium
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8502")
    ap.add_argument("--out", default="output/screenshots/comprehensive")
    args = ap.parse_args()

    pw_secret = os.environ.get("APP_PASSWORD")  # gate disabled if None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shots: list[tuple[str, int]] = []

    def snap(page, name, full=False):
        p = out / f"{name}.png"
        page.screenshot(path=str(p), full_page=full)
        shots.append((name, p.stat().st_size))
        print(f"  [shot] {name}.png  ({p.stat().st_size // 1024} KB)")

    def settle(page, ms=900):
        page.wait_for_timeout(ms)

    def scroll_to(page, text, pause=1000):
        el = page.query_selector(f"text={text}")
        if el:
            el.scroll_into_view_if_needed()
            settle(page, pause)
        return el

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--no-sandbox"])
        # device_scale_factor=2 → retina-sharp PNGs
        ctx = browser.new_context(viewport={"width": 1440, "height": 960}, device_scale_factor=2)
        page = ctx.new_page()
        page.goto(args.base, wait_until="networkidle")

        # ---- Password gate (only if APP_PASSWORD is set) ----
        if pw_secret:
            page.wait_for_selector("input[type=password]", timeout=30000)
            settle(page)
            snap(page, "01_password_gate")

            page.fill("input[type=password]", "wrongpass")
            page.press("input[type=password]", "Enter")
            page.wait_for_selector("text=Incorrect password", timeout=15000)
            settle(page)
            snap(page, "02_wrong_password")

            page.fill("input[type=password]", pw_secret)
            page.press("input[type=password]", "Enter")

        # ---- Recommend tab ----
        page.wait_for_selector("text=Your taste profile", timeout=30000)
        settle(page, 1500)
        snap(page, "03_recommend_tab")

        # Slider help tooltip — there are 6 stTooltipIcon, one per slider.
        try:
            icons = page.query_selector_all('[data-testid="stTooltipIcon"]')
            tgt = icons[3] if len(icons) > 3 else icons[0]
            tgt.scroll_into_view_if_needed()
            tgt.hover()
            page.wait_for_selector('[data-testid="stTooltipContent"]', timeout=8000)
            settle(page, 700)
            snap(page, "04_recommend_slider_help")
        except Exception as e:
            print("  [warn] help tooltip:", e)

        # Set a configured profile via keyboard on the slider thumbs.
        try:
            thumbs = page.query_selector_all('div[role="slider"]')
            moves = [("ArrowRight", 12), ("ArrowLeft", 8), ("ArrowRight", 15),
                     ("ArrowLeft", 5), ("ArrowRight", 10), ("ArrowRight", 6)]
            for thumb, (key, n) in zip(thumbs, moves):
                thumb.click()
                for _ in range(n):
                    page.keyboard.press(key)
                page.wait_for_timeout(80)
            page.keyboard.press("Escape")
            settle(page)
            snap(page, "05_recommend_profile_set")
        except Exception as e:
            print("  [warn] sliders:", e)

        # ---- Analyze tab — default decade (most recent) ----
        page.get_by_role("tab", name="Analyze").click()
        page.wait_for_selector("text=Decade-by-decade style", timeout=20000)
        settle(page, 1800)
        page.evaluate("window.scrollTo(0, 0)")
        settle(page, 500)
        snap(page, "06_analyze_decade_recent")

        # Switch decade — the selectbox is a type-to-filter combobox (older
        # decades are scrolled out of the popover, so type instead of click).
        try:
            box = page.query_selector('[data-testid="stSelectbox"] input')
            box.click()
            settle(page, 500)
            box.type("1960", delay=60)
            settle(page, 500)
            page.keyboard.press("Enter")
            page.wait_for_selector("text=1960s — what it sounds like", timeout=15000)
            page.keyboard.press("Escape")
            page.evaluate("window.scrollTo(0, 0)")
            settle(page, 1200)
            snap(page, "07_analyze_decade_1960s")
        except Exception as e:
            print("  [warn] decade switch:", e)
        page.keyboard.press("Escape")

        # Heatmap — center the first dataframe (Streamlit's internal scroll
        # container defeats full_page, so center the element and shoot viewport).
        try:
            dfs = page.query_selector_all('[data-testid="stDataFrame"]')
            dfs[0].evaluate("e => e.scrollIntoView({block:'center', behavior:'instant'})")
            settle(page, 1200)
            snap(page, "08_analyze_heatmap")
        except Exception as e:
            print("  [warn] heatmap:", e)

        scroll_to(page, "How features have shifted", 1100)
        snap(page, "09_analyze_trends_highlights")

        try:
            exp = page.query_selector('summary:has-text("Show all 22 trend plots")')
            if exp:
                exp.scroll_into_view_if_needed()
                exp.click()
                settle(page, 1600)
            snap(page, "10_analyze_all_trends_expanded")
        except Exception as e:
            print("  [warn] expander:", e)

        scroll_to(page, "Can a model tell decades apart", 1000)
        snap(page, "11_analyze_classifier")

        # Sidebar corpus overview crop.
        try:
            page.evaluate("window.scrollTo(0, 0)")
            settle(page, 500)
            sidebar = page.query_selector('[data-testid="stSidebar"]')
            sidebar.screenshot(path=str(out / "12_sidebar_corpus.png"))
            sz = (out / "12_sidebar_corpus.png").stat().st_size
            shots.append(("12_sidebar_corpus", sz))
            print(f"  [shot] 12_sidebar_corpus.png  ({sz // 1024} KB)")
        except Exception as e:
            print("  [warn] sidebar:", e)

        browser.close()

    print(f"\n=== Captured {len(shots)} screenshots to {out} ===")
    for name, size in shots:
        print(f"  {name:34s} {size // 1024:5d} KB")


if __name__ == "__main__":
    main()
