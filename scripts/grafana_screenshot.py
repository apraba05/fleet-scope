"""Headless screenshot of the provisioned Grafana dashboard (for docs).

    python scripts/grafana_screenshot.py http://localhost:3000 docs/img/grafana-pipeline.png \
        --from-ms 1792685700000 --to-ms 1792686500000
"""

import argparse

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("out")
ap.add_argument("--from-ms", default="now-30m")
ap.add_argument("--to-ms", default="now")
args = ap.parse_args()

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1600, "height": 1420})
    page.goto(f"{args.url}/d/fleetscope-pipeline?orgId=1&from={args.from_ms}&to={args.to_ms}&kiosk")
    page.wait_for_selector("canvas", timeout=60000)
    page.wait_for_timeout(15000)  # let every panel query resolve and draw
    # Grafana scrolls inside its own container, so size the viewport to the dashboard.
    page.screenshot(path=args.out)
    b.close()
