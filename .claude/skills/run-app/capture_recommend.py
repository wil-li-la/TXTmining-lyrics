#!/usr/bin/env python3
"""Capture the NEW retrieval-based recommendation flow (report §6.3).

Drives the Recommend tab through the password gate, sets a non-trivial taste
profile, triggers the recommender (deterministic retrieval + one LLM-written
explanation), and captures: profile, ranking progress, the 'why these match'
prose, and the top-5 results with per-feature comparison charts.

Run AFTER the Streamlit server is serving. Reads APP_PASSWORD from .env.
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
    ap.add_argument("--out", default="output/screenshots")
    args = ap.parse_args()

    pw_secret = os.environ.get("APP_PASSWORD")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shots = []

    def snap(page, name, full=False):
        p = out / f"{name}.png"
        page.screenshot(path=str(p), full_page=full)
        shots.append(name)
        print(f"  [shot] {name}.png  ({p.stat().st_size // 1024} KB)")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--no-sandbox"])
        ctx = browser.new_context(viewport={"width": 1440, "height": 1100},
                                  device_scale_factor=2)
        page = ctx.new_page()
        page.goto(args.base, wait_until="networkidle")

        if pw_secret:
            page.wait_for_selector("input[type=password]", timeout=30000)
            page.fill("input[type=password]", pw_secret)
            page.press("input[type=password]", "Enter")

        page.wait_for_selector("text=Your taste profile", timeout=30000)
        page.wait_for_timeout(1500)

        # Set a non-trivial profile via arrow keys on the slider thumbs.
        thumbs = page.query_selector_all('div[role="slider"]')
        moves = [("ArrowRight", 15), ("ArrowRight", 12), ("ArrowLeft", 6),
                 ("ArrowRight", 8), ("ArrowRight", 14), ("ArrowLeft", 10)]
        for thumb, (key, n) in zip(thumbs, moves):
            thumb.click()
            for _ in range(n):
                page.keyboard.press(key)
            page.wait_for_timeout(70)
        page.keyboard.press("Escape")
        page.wait_for_timeout(600)
        snap(page, "recommend_01_profile")

        # Trigger the recommender.
        page.get_by_role("button", name="Find matching songs ▸").click()

        # Wait for the deterministic ranking + LLM explanation to finish: the
        # 'Top recommendations' header, then either result rows or a warning.
        page.wait_for_selector("text=Top recommendations", timeout=90000)
        try:
            page.wait_for_selector(".bx-res", timeout=90000)
        except Exception as e:
            print("  [warn] no result rows (empty profile?):", e)
        n_rows = len(page.query_selector_all(".bx-res"))
        warn = page.query_selector("text=No matches found")
        print(f"  result rows: {n_rows}; warning present: {warn is not None}")
        page.wait_for_timeout(1500)

        # Ranking progress trace (scroll it into view).
        prog = page.query_selector("text=Ranking progress")
        if prog:
            prog.scroll_into_view_if_needed()
            page.wait_for_timeout(600)
            snap(page, "recommend_02_ranking_progress")

        why = page.query_selector("text=Why these match")
        if why:
            why.scroll_into_view_if_needed()
            page.wait_for_timeout(600)
            snap(page, "recommend_03_why_these_match")

        top = page.query_selector("text=Top recommendations")
        if top:
            top.scroll_into_view_if_needed()
            page.wait_for_timeout(800)
            snap(page, "recommend_04_top_results")

        # A full-page capture of the whole result for the record.
        snap(page, "recommend_05_full", full=True)

        browser.close()

    print(f"\n=== Captured {len(shots)} screenshots to {out} ===")


if __name__ == "__main__":
    main()
