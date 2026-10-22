"""FleetScope dashboard: ``FLEETSCOPE_LAKE=lake streamlit run dashboard/app.py``.

Reads only the lake (committed files) and the processors' health snapshots, so it can
run next to a live pipeline: "Refresh" rebinds the views to the newest committed epochs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

from fleetscope.lake.sql import Lake
from fleetscope.metrics.driving import event_rates, fleet_hourly, fleet_summary
from fleetscope.scenarios.selection import SelectionConfig, compare

LAKE_ROOT = Path(os.environ.get("FLEETSCOPE_LAKE", "lake"))

st.set_page_config(page_title="FleetScope", layout="wide")


@st.cache_resource
def lake() -> Lake:
    return Lake(LAKE_ROOT)


def q(sql: str, params: list | None = None) -> pd.DataFrame:
    return lake().df(sql, params)


def has(table: str) -> bool:
    return lake().has(table)


st.sidebar.title("FleetScope")
st.sidebar.caption(f"lake: `{LAKE_ROOT}`")
if st.sidebar.button("Refresh"):
    lake().refresh()
if not has("telemetry") or q("SELECT count(*) AS n FROM telemetry").n[0] == 0:
    st.warning("The lake is empty. Run `make demo` (or `fleetscope local`) first.")
    st.stop()

tabs = st.tabs(["Fleet", "Telemetry", "Scenarios", "Selection", "Pipeline health"])

# --- fleet overview ----------------------------------------------------------------------
with tabs[0]:
    if has("trip_metrics"):
        s = fleet_summary(lake())
        c = st.columns(6)
        c[0].metric("Vehicles", f"{s['vehicles']:.0f}")
        c[1].metric("Distance", f"{s['distance_km']:,.0f} km")
        c[2].metric("Driving", f"{s['driving_hours']:,.1f} h")
        c[3].metric("Safety events / 1000 km", f"{s['safety_events_per_1000km']:.1f}")
        c[4].metric("Completeness", f"{s['completeness']:.2%}")
        c[5].metric("Quarantine rate", f"{s['quarantine_rate']:.3%}")
    else:
        st.info("Run `fleetscope mine` for trip metrics and scenarios.")
    left, right = st.columns(2)
    hourly = fleet_hourly(lake()).to_pandas()
    if len(hourly):
        hourly["hour"] = hourly.hour.dt.strftime("%Y-%m-%d %H:00")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=hourly.hour, y=hourly.distance_km, name="distance (km)")
        fig.add_scatter(
            x=hourly.hour, y=hourly.active_vehicles, name="active vehicles", secondary_y=True
        )
        # Categorical axis: real drives (2018) and synthetic fleets (2026) share one chart.
        fig.update_xaxes(type="category")
        fig.update_layout(title="Fleet activity per hour (streaming 10 s windows)", height=380)
        left.plotly_chart(fig, use_container_width=True)
    if has("scenarios") and has("trip_metrics"):
        rates = event_rates(lake()).to_pandas()
        fig = px.bar(
            rates,
            x="scenario_type",
            y="per_1000km",
            color="source",
            barmode="group",
            title="Scenario rate per 1000 km",
        )
        fig.update_layout(height=380)
        right.plotly_chart(fig, use_container_width=True)
    if has("trip_metrics"):
        tm = q("SELECT * FROM trip_metrics")
        fig = px.scatter(
            tm,
            x="accel_long_abs_p95",
            y="safety_events_per_100km",
            color="source",
            size="distance_km",
            hover_data=["vehicle_id", "trip_id"],
            title="Driving smoothness vs. safety events (per trip)",
        )
        st.plotly_chart(fig, use_container_width=True)

# --- telemetry explorer ------------------------------------------------------------------
with tabs[1]:
    trips = q(
        "SELECT vehicle_id, trip_id, any_value(source) AS source, count(*) AS samples "
        "FROM telemetry GROUP BY 1, 2 ORDER BY 1, 2"
    )
    # open on the trip of the top-ranked scenario
    top = (
        q("SELECT vehicle_id, trip_id, source FROM scenarios ORDER BY rank LIMIT 1")
        if has("scenarios")
        else pd.DataFrame()
    )
    sources = sorted(trips.source.unique())
    c1, c2 = st.columns([1, 3])
    source = c1.selectbox("Source", sources, index=sources.index(top.source[0]) if len(top) else 0)
    sub = trips[trips.source == source]
    labels = (sub.vehicle_id + "  |  " + sub.trip_id).tolist()
    want = f"{top.vehicle_id[0]}  |  {top.trip_id[0]}" if len(top) else ""
    label = c2.selectbox(
        "Trip", labels, index=labels.index(want) if want in labels else 0, help="vehicle | trip"
    )
    vehicle, trip = (x.strip() for x in label.split("|", 1))
    tel = q(
        "SELECT ts, speed, accel_long, accel_lat, yaw_rate, lead_dist, lead_rel_speed, lat, lon "
        "FROM telemetry WHERE vehicle_id = ? AND trip_id = ? ORDER BY ts",
        [vehicle, trip],
    )
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04)
    fig.add_scatter(x=tel.ts, y=tel.speed, name="speed (m/s)", row=1, col=1)
    fig.add_scatter(x=tel.ts, y=tel.accel_long, name="a_long (m/s²)", row=2, col=1)
    fig.add_scatter(x=tel.ts, y=tel.accel_lat, name="a_lat (m/s²)", row=2, col=1)
    fig.add_scatter(x=tel.ts, y=tel.lead_dist, name="lead gap (m)", row=3, col=1)
    if has("scenarios"):
        ev = q(
            "SELECT scenario_type, start_ts, end_ts, severity FROM scenarios "
            "WHERE vehicle_id = ? AND trip_id = ?",
            [vehicle, trip],
        )
        for r in ev.itertuples():
            fig.add_vrect(
                x0=r.start_ts,
                x1=r.end_ts,
                fillcolor="red",
                opacity=0.15 + 0.3 * r.severity,
                line_width=0,
                annotation_text=r.scenario_type,
                annotation_position="top left",
            )
    fig.update_layout(height=650, title=f"{vehicle} — {trip}")
    st.plotly_chart(fig, use_container_width=True)
    gps = tel.dropna(subset=["lat", "lon"])
    if len(gps):
        st.map(gps.iloc[:: max(1, len(gps) // 2000)], latitude="lat", longitude="lon", size=2)

# --- scenarios ---------------------------------------------------------------------------
with tabs[2]:
    if not has("scenarios"):
        st.info("Run `fleetscope mine` first.")
    else:
        sc = q("SELECT * FROM scenarios ORDER BY rank")
        types = st.multiselect("Types", sorted(sc.scenario_type.unique()))
        min_sev = st.slider("Min severity", 0.0, 1.0, 0.0, 0.05)
        view = sc[(sc.severity >= min_sev) & (sc.scenario_type.isin(types) if types else True)]
        fig = px.scatter(
            view,
            x="rarity",
            y="severity",
            color="scenario_type",
            size="priority",
            hover_data=["scenario_id", "vehicle_id", "peak", "speed_start", "rank"],
            title=f"{len(view)} scenarios: severity vs. rarity (size = priority)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            view[
                [
                    "rank",
                    "scenario_type",
                    "severity",
                    "rarity",
                    "priority",
                    "peak",
                    "speed_start",
                    "duration_s",
                    "vehicle_id",
                    "start_ts",
                    "source",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        if has("ground_truth"):
            from fleetscope.scenarios.pipeline import evaluate

            ev = evaluate(lake()).to_pandas()
            ev["recall"] = ev.tp_labels / ev.labels.where(ev.labels > 0)
            ev["precision"] = ev.tp_detections / ev.detections.where(ev.detections > 0)
            st.subheader("Detector accuracy vs. synthetic ground truth")
            st.dataframe(ev, use_container_width=True, hide_index=True)

# --- selection ---------------------------------------------------------------------------
with tabs[3]:
    if not has("scenarios"):
        st.info("Run `fleetscope mine` first.")
    else:
        budget = st.slider("Upload budget (GB)", 0.2, 50.0, 5.0, 0.2)
        table = lake().arrow("SELECT * FROM scenarios ORDER BY rank")
        cmp = compare(table, SelectionConfig(budget_bytes=int(budget * 1e9))).to_pandas()
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            px.bar(cmp, x="method", y="coverage", title="Diversity coverage f(S) / f(all)"),
            use_container_width=True,
        )
        c2.plotly_chart(
            px.bar(cmp, x="method", y="types_covered", title="Scenario types represented"),
            use_container_width=True,
        )
        st.dataframe(cmp, use_container_width=True, hide_index=True)
        if has("selection"):
            st.subheader("Persisted selection (`fleetscope select`)")
            st.dataframe(q("SELECT * FROM selection ORDER BY pick_order"), hide_index=True)

# --- pipeline health ---------------------------------------------------------------------
with tabs[4]:
    rows = []
    for f in sorted((LAKE_ROOT / "_ops").glob("health-*.jsonl")):
        for line in f.read_text().splitlines():
            rows.append({"process": f.stem.removeprefix("health-"), **json.loads(line)})
    if rows:
        h = pd.DataFrame(rows).fillna(0)
        h["time"] = pd.to_datetime(h.ts, unit="s")
        h = h.sort_values("time")
        for col in ("rows.valid", "records.consumed"):
            h[col + "/s"] = h.groupby("process")[col].diff() / h.groupby("process").ts.diff()
        st.plotly_chart(
            px.line(h, x="time", y="rows.valid/s", color="process", title="Valid rows / s"),
            use_container_width=True,
        )
        last = h.groupby("process").tail(1).set_index("process")
        keys = [c for c in last.columns if c.split(".")[0] in ("dlq", "invalid", "chunks")]
        st.subheader("Latest cumulative counters")
        st.dataframe(last[[c for c in last.columns if c not in ("ts", "time")]].T)
        if keys:
            st.plotly_chart(
                px.bar(
                    last[keys].T.reset_index(),
                    x="index",
                    y=list(last.index),
                    title="Rejected / duplicate data by reason",
                ),
                use_container_width=True,
            )
    else:
        st.info("No health snapshots (written by `fleetscope process`).")
    files = q(
        "SELECT 'telemetry' AS t, count(*) AS rows FROM telemetry UNION ALL "
        "SELECT 'quarantine', count(*) FROM quarantine UNION ALL "
        "SELECT 'dlq', count(*) FROM dlq UNION ALL SELECT 'windows_10s', count(*) FROM windows_10s"
    )
    sizes = []
    for t in files.t:
        paths = list((LAKE_ROOT / t).rglob("*.parquet"))
        sizes.append(
            {"table": t, "files": len(paths), "mb": sum(p.stat().st_size for p in paths) / 1e6}
        )
    st.subheader("Lake tables")
    st.dataframe(
        files.rename(columns={"t": "table"}).merge(pd.DataFrame(sizes), on="table"),
        hide_index=True,
    )
    lat = q("SELECT epoch_ns(max(ts)) / 1e9 AS newest_event_s FROM telemetry").newest_event_s[0]
    st.caption(f"Newest committed event time: {pd.to_datetime(lat, unit='s')} UTC")
