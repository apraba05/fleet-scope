"""Generate dashboards/fleetscope-pipeline.json (run: python infra/grafana/build_dashboard.py)."""

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
panels = []


def panel(title, exprs, unit="short", kind="timeseries", w=8, h=7, desc=""):
    n = len(panels)
    panels.append(
        {
            "id": n + 1,
            "type": kind,
            "title": title,
            "description": desc,
            "datasource": DS,
            "gridPos": {"x": (n * w) % 24, "y": (n * w) // 24 * h, "w": w, "h": h},
            "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
            "options": {"legend": {"displayMode": "list", "placement": "bottom"}},
            "targets": [
                {"refId": chr(65 + i), "datasource": DS, "expr": e, "legendFormat": lf}
                for i, (e, lf) in enumerate(exprs)
            ],
        }
    )


r = "[1m]"


def pct(metric, q):
    return f"histogram_quantile({q}, sum by (le) (rate({metric}_bucket{r})))"


panel("Records consumed / s", [(f"rate(fleetscope_records_consumed_total{r})", "records")], "ops")
panel("Ingest MB / s", [(f"rate(fleetscope_bytes_consumed_total{r}) / 1e6", "MB/s")], "none")
panel("Consumer lag (records)", [("fleetscope_consumer_lag_records", "p{{partition}}")])
panel(
    "Rows by validation status / s",
    [(f"sum by (status) (rate(fleetscope_rows_total{r}))", "{{status}}")],
    "ops",
)
panel(
    "Quarantined rows by rule / s",
    [(f"sum by (rule) (rate(fleetscope_invalid_rows_total{r}))", "{{rule}}")],
    "ops",
)
panel(
    "Chunks by outcome / s",
    [(f"sum by (outcome) (rate(fleetscope_chunks_total{r}))", "{{outcome}}")],
    "ops",
    desc="accepted, duplicate (dedup), dlq (undecodable / bad header / expired)",
)
panel(
    "DLQ by reason / s", [(f"sum by (reason) (rate(fleetscope_dlq_total{r}))", "{{reason}}")], "ops"
)
panel(
    "Ingest latency (broker append -> processed)",
    [
        (
            pct("fleetscope_ingest_latency_seconds", q),
            f"p{int(float(q) * 100)}",
        )
        for q in ("0.5", "0.99")
    ],
    "s",
)
panel(
    "Commit latency (broker append -> in lake)",
    [
        (
            pct("fleetscope_commit_latency_seconds", q),
            f"p{int(float(q) * 100)}",
        )
        for q in ("0.5", "0.99")
    ],
    "s",
    desc="End-to-end freshness: oldest record of an epoch until its files are visible to SQL.",
)
panel(
    "Batch processing time p99",
    [
        (
            pct("fleetscope_batch_process_seconds", 0.99),
            "p99",
        )
    ],
    "s",
)
panel(
    "Late rows / s",
    [(f"rate(fleetscope_late_rows_total{r})", "late")],
    "ops",
    desc="Rows kept in the raw lake but past the window watermark (not aggregated).",
)
panel(
    "Checkpoints & windows / s",
    [
        (f"rate(fleetscope_checkpoints_total{r})", "checkpoints"),
        (f"rate(fleetscope_windows_emitted_total{r})", "windows"),
    ],
    "ops",
)
panel(
    "Lake writes MB / s",
    [(f"sum by (table) (rate(fleetscope_file_bytes_written_total{r})) / 1e6", "{{table}}")],
    "none",
)
panel(
    "Open state",
    [
        ("fleetscope_open_streams", "dedup streams"),
        ("fleetscope_open_window_streams", "window streams"),
        ("fleetscope_buffered_rows", "buffered rows"),
    ],
)
panel(
    "Totals",
    [
        ("fleetscope_rows_total", "rows {{status}}"),
        ("sum(fleetscope_dlq_total)", "dlq"),
        ("sum(fleetscope_chunks_total{outcome='duplicate'})", "duplicates"),
    ],
    kind="stat",
)

dash = {
    "uid": "fleetscope-pipeline",
    "title": "FleetScope pipeline health",
    "tags": ["fleetscope"],
    "timezone": "utc",
    "schemaVersion": 39,
    "refresh": "5s",
    "time": {"from": "now-15m", "to": "now"},
    "panels": panels,
}
out = Path(__file__).parent / "dashboards" / "fleetscope-pipeline.json"
out.write_text(json.dumps(dash, indent=2) + "\n")
print(f"wrote {out} ({len(panels)} panels)")
