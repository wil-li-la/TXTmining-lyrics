"""End-to-end verification of the Streamlit app via Playwright.

Captures 5 screenshots matching the assignment deliverable spec § 6.6:
  01_initial.png    - sliders at 0, sidebar visible
  02_profile_set.png - sliders adjusted
  03_searching.png  - agent trace mid-run
  04_scoring.png    - first candidate features visible
  05_ranked.png     - final ranked list

Run AFTER `streamlit run app.py` is up at http://localhost:8501.
"""
import os
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
OUT = Path("output/screenshots")
OUT.mkdir(parents=True, exist_ok=True)


def shot(page, name: str) -> None:
    path = OUT / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    print(f"  saved {path}")


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1400, "height": 1000})
        page = ctx.new_page()

        print("Loading app...")
        page.goto(URL, wait_until="networkidle", timeout=60_000)
        # Streamlit can take a few seconds to fully render after networkidle
        page.wait_for_selector("text=Pop Lyrics Taste Profiler", timeout=30_000)
        time.sleep(2)

        # 1. Initial state
        shot(page, "01_initial")

        # 2. Adjust sliders — Streamlit sliders have role="slider"; we use keyboard arrows
        sliders = page.locator('[role="slider"]')
        n = sliders.count()
        print(f"Found {n} sliders. Adjusting...")
        # Push a few sliders to non-zero positions
        for i in range(min(n, 6)):
            sliders.nth(i).focus()
            # Each ArrowRight press moves +0.1 in our slider config; press multiple times
            for _ in range(8 + i):  # varying amounts per slider
                page.keyboard.press("ArrowRight")
            time.sleep(0.1)
        time.sleep(1)
        shot(page, "02_profile_set")

        # 3. Click "Find matching recent songs"
        button = page.get_by_role("button", name="Find matching recent songs")
        button.click()
        # Wait briefly for trace to start
        time.sleep(5)
        shot(page, "03_searching")

        # 4. Wait a bit longer for some candidates to be scored
        time.sleep(20)
        shot(page, "04_scoring")

        # 5. Wait for the agent to finish (look for "Top recommendations" heading or spinner gone)
        try:
            page.wait_for_selector("text=Top recommendations", timeout=120_000)
            time.sleep(2)
        except Exception as e:
            print(f"  warn: Top recommendations not reached: {e}")
        shot(page, "05_ranked")

        browser.close()
    print("All 5 screenshots captured.")


if __name__ == "__main__":
    main()
