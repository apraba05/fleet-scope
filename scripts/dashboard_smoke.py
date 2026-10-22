"""Headless smoke test + screenshots of the dashboard.

    FLEETSCOPE_LAKE=lake streamlit run dashboard/app.py --server.headless true &
    python scripts/dashboard_smoke.py http://localhost:8501 docs/img

Fails (exit 1) if any tab renders a Streamlit exception or the page throws.
"""

import sys

from playwright.sync_api import sync_playwright

url, out = sys.argv[1], sys.argv[2]
tabs = ["Fleet", "Telemetry", "Scenarios", "Selection", "Pipeline health"]
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1500, "height": 1000})
    errors: list[str] = []
    failed = False
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url)
    page.wait_for_selector("text=FleetScope", timeout=60000)
    page.wait_for_timeout(6000)
    for t in tabs:
        page.get_by_role("tab", name=t).click()
        page.wait_for_timeout(5000)
        exc = page.locator("[data-testid='stException']")
        n = exc.count()
        print(t, "exceptions:", n, (exc.first.inner_text()[:800] if n else ""))
        failed |= n > 0
        page.screenshot(path=f"{out}/dashboard-{t.split()[0].lower()}.png", full_page=True)
    print("page errors:", errors)
    b.close()
    if failed or errors:
        sys.exit(1)
