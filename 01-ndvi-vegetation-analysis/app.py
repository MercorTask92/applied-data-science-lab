"""Interactive assessment of vegetation changes using synthetic NDVI data.

Phone-friendly Streamlit app. No external datasets, credentials, or APIs needed
for the calculations. The optional interactive map uses online basemap tiles.
"""

from __future__ import annotations

from io import BytesIO, StringIO
from hashlib import sha256
from datetime import datetime, timezone
import secrets
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

from PIL import Image, UnidentifiedImageError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import streamlit as st


# Fixed, fully synthetic observations ensure reproducible training results.
SYNTHETIC_SITES = [
    ("Forest A",    39.282, -76.818, 0.77, 0.60,  5),
    ("Cropland B",  39.166, -76.939, 0.64, 0.57, 10),
    ("Wetland C",   39.059, -76.708, 0.70, 0.44, 35),
    ("Grassland D", 39.094, -76.835, 0.41, 0.49,  5),
    ("Forest E",    39.244, -76.980, 0.73, 0.61, 12),
    ("Orchard F",   39.020, -76.947, 0.66, 0.64,  8),
    ("Shrubland G", 39.201, -76.741, 0.48, 0.39, 26),
    ("Riparian H",  39.128, -76.752, 0.68, 0.59, 18),
]
FIELDS = ["site", "latitude", "longitude", "ndvi_2024", "ndvi_2025", "cloud_cover_pct"]
SIGNIFICANT = "Significant decline"
MINOR = "Minor decline"
NO_DECLINE = "No decline / improvement"
UNRELIABLE = "Unreliable — recheck"
PALETTE = {
    SIGNIFICANT: "#C34B3C",
    MINOR: "#E6A33D",
    NO_DECLINE: "#248D78",
    UNRELIABLE: "#77818D",
}


def synthetic_data() -> pd.DataFrame:
    """Return a new copy of the fixed synthetic training dataset."""
    return pd.DataFrame(SYNTHETIC_SITES, columns=FIELDS)


def classify_site(change: float, cloud_pct: float, threshold: float, cloud_limit: int) -> str:
    """Quality screening takes precedence over NDVI-based classification."""
    if cloud_pct > cloud_limit:
        return UNRELIABLE
    if change <= threshold:
        return SIGNIFICANT
    if change < 0:
        return MINOR
    return NO_DECLINE


def analyze_data(data: pd.DataFrame, threshold: float = -0.10, cloud_limit: int = 20) -> pd.DataFrame:
    """Analyze a synthetic NDVI dataset with explicit, reproducible rules."""
    if not (-0.30 <= threshold < 0):
        raise ValueError("Significant-decline threshold must be between -0.30 and 0.")
    if not (0 <= cloud_limit <= 100):
        raise ValueError("Cloud threshold must be between 0% and 100%.")
    result = data.copy(deep=True)
    result["ndvi_change"] = (result["ndvi_2025"] - result["ndvi_2024"]).round(3)
    result["classification"] = [
        classify_site(delta, cloud, threshold, cloud_limit)
        for delta, cloud in zip(result["ndvi_change"], result["cloud_cover_pct"])
    ]
    result["map_color"] = result["classification"].map(PALETTE)
    return result


def select_priority(results: pd.DataFrame) -> pd.Series | None:
    """Choose the largest significant decline among non-cloud-flagged sites."""
    candidates = results.loc[results["classification"] == SIGNIFICANT]
    if candidates.empty:
        return None
    return candidates.sort_values(["ndvi_change", "site"]).iloc[0]


def comparison_table(results: pd.DataFrame) -> pd.DataFrame:
    """Show changes relative to the predefined baseline scenario."""
    baseline = analyze_data(synthetic_data(), threshold=-0.10, cloud_limit=20)
    merged = results[["site", "classification"]].merge(
        baseline[["site", "classification"]].rename(columns={"classification": "baseline_classification"}),
        on="site", validate="one_to_one"
    )
    merged["changed"] = merged["classification"] != merged["baseline_classification"]
    return merged



def analysis_identifier(raw: pd.DataFrame, threshold: float, cloud_limit: int) -> str:
    """Identify an exact set of synthetic inputs and currently selected settings."""
    payload = (f"cutoff={threshold:.2f};cloud={cloud_limit}\n" +
               raw[FIELDS].to_csv(index=False, float_format="%.5f"))
    return sha256(payload.encode("utf-8")).hexdigest()[:12]


def verify_analysis(raw: pd.DataFrame, results: pd.DataFrame, threshold: float,
                    cloud_limit: int, csv_bytes: bytes, png_bytes: bytes,
                    analysis_id: str, run_id: str, run_time: str) -> pd.DataFrame:
    """Automatically verify both computed values and exported CSV and PNG contents."""
    checks = []

    def add(deliverable: str, name: str, passed: bool, evidence: str) -> None:
        checks.append({"Deliverable": deliverable, "Check": name,
                       "Status": "PASS" if passed else "FAIL", "Evidence": evidence})

    expected_columns = set(FIELDS)
    integrity = (expected_columns.issubset(raw.columns)
                 and len(raw) == len(results) == 8
                 and raw["site"].is_unique
                 and raw[["ndvi_2024", "ndvi_2025"]].ge(-1).all().all()
                 and raw[["ndvi_2024", "ndvi_2025"]].le(1).all().all()
                 and raw["cloud_cover_pct"].between(0, 100).all())
    add("CSV", "Synthetic input integrity", bool(integrity),
        "8 distinct sites; NDVI in [−1, 1]; cloud cover in [0, 100]%")

    joined = raw.merge(results[["site", "ndvi_change", "classification"]],
                       on="site", validate="one_to_one")
    arithmetic_ok = all(abs((float(r.ndvi_2025) - float(r.ndvi_2024)) - float(r.ndvi_change)) < 0.00051
                        for r in joined.itertuples(index=False))
    forest = joined.loc[joined["site"] == "Forest A", "ndvi_change"].iloc[0]
    grass = joined.loc[joined["site"] == "Grassland D", "ndvi_change"].iloc[0]
    known_cases_ok = abs(forest + 0.17) < 0.00051 and abs(grass - 0.08) < 0.00051
    add("CSV", "NDVI calculations", arithmetic_ok and known_cases_ok,
        "NDVI 2025 − NDVI 2024; Forest A −0.17; Grassland D +0.08")

    flagged = joined["cloud_cover_pct"] > cloud_limit
    add("CSV", "Cloud-quality screening",
        bool((flagged == (joined["classification"] == UNRELIABLE)).all()),
        f"Every observation exceeding {cloud_limit}% is flagged")

    expected_labels = []
    for r in joined.itertuples(index=False):
        if float(r.cloud_cover_pct) > cloud_limit:
            label = UNRELIABLE
        elif float(r.ndvi_change) <= threshold + 1e-9:
            label = SIGNIFICANT
        elif float(r.ndvi_change) < 0:
            label = MINOR
        else:
            label = NO_DECLINE
        expected_labels.append(label)
    add("CSV", "Classification thresholds", joined["classification"].tolist() == expected_labels,
        "All classifications independently rechecked against the selected controls")

    eligible = joined.loc[joined["classification"] == SIGNIFICANT].sort_values(["ndvi_change", "site"])
    expected_priority = eligible.iloc[0]["site"] if len(eligible) else None
    selected = select_priority(results)
    actual_priority = selected["site"] if selected is not None else None
    add("Report", "Priority location", expected_priority == actual_priority,
        f"Independently selected priority: {expected_priority or 'none'}")

    try:
        exported = pd.read_csv(BytesIO(csv_bytes))
        exact_columns = list(results[["site", "ndvi_2024", "ndvi_2025", "ndvi_change",
                                      "cloud_cover_pct", "classification", "latitude", "longitude"]].columns)
        expected_numeric = ["ndvi_2024", "ndvi_2025", "ndvi_change",
                            "cloud_cover_pct", "latitude", "longitude"]
        csv_ok = (len(exported) == len(results)
                  and exported["site"].tolist() == results["site"].tolist()
                  and exported["classification"].tolist() == results["classification"].tolist()
                  and all((exported[column] - results[column]).abs().le(0.00051).all()
                          for column in expected_numeric)
                  and (exported["analysis_id"] == analysis_id).all()
                  and (exported["run_id"] == run_id).all()
                  and (exported["run_time_utc"] == run_time).all()
                  and exported["decline_cutoff"].eq(threshold).all()
                  and exported["cloud_limit_pct"].eq(cloud_limit).all()
                  and set(exact_columns).issubset(exported.columns))
    except (ValueError, KeyError, UnicodeDecodeError):
        csv_ok = False
    add("CSV", "Export matches live dashboard", bool(csv_ok),
        "All 8 site values, classifications, thresholds, run ID, and UTC run time match")

    try:
        with Image.open(BytesIO(png_bytes)) as img:
            png_ok = (img.format == "PNG"
                      and img.width >= 800 and img.height >= 400
                      and img.info.get("Description") ==
                      f"analysis_id={analysis_id};run_id={run_id};cutoff={threshold:.2f};cloud={cloud_limit}")
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        png_ok = False
    add("PNG", "Image integrity and current analysis ID", bool(png_ok),
        "Decoded PNG contains the current analysis ID, completed run ID and settings")
    return pd.DataFrame(checks)


def verify_report(report: str, results: pd.DataFrame, threshold: float,
                  cloud_limit: int, analysis_id: str, run_id: str,
                  run_time: str, run_log: list[dict]) -> pd.DataFrame:
    """Check that the report matches the same controls, records and conclusions."""
    priority = select_priority(results)
    priority_name = priority["site"] if priority is not None else "No reliable site"
    expected_snippets = [
        f"**Analysis ID:** `{analysis_id}`",
        f"**Final completed run:** `{run_id}`",
        f"**Run time (UTC):** `{run_time}`",
        f"significant decline if ΔNDVI ≤ {threshold:.2f}",
        f"cloud cover > {cloud_limit}%",
        f"Sites analyzed: **{len(results)}**",
        priority_name,
        *results["site"].tolist(),
        *(f"| {item['run_id']} | {item['threshold']:.2f} | {item['cloud_limit']}% |"
          for item in run_log),
    ]
    ok = all(phrase in report for phrase in expected_snippets)
    return pd.DataFrame([{
        "Deliverable": "Report", "Check": "Report matches live settings and results",
        "Status": "PASS" if ok else "FAIL",
        "Evidence": "Run ID, UTC time, settings, run history, eight sites and priority match",
    }])


def verify_run_history(raw: pd.DataFrame, run_log: list[dict], active: dict) -> pd.DataFrame:
    """Recompute run history independently; verify final item is the displayed run."""
    try:
        ok = bool(run_log) and run_log[-1] == active
        seen = set()
        for sequence, entry in enumerate(run_log, start=1):
            run_result = analyze_data(raw, entry["threshold"], entry["cloud_limit"])
            expected = {
                "reliable": int((run_result["classification"] != UNRELIABLE).sum()),
                "significant": int((run_result["classification"] == SIGNIFICANT).sum()),
                "cloud_flagged": int((run_result["classification"] == UNRELIABLE).sum()),
                "analysis_id": analysis_identifier(raw, entry["threshold"], entry["cloud_limit"]),
            }
            if any(entry[key] != value for key, value in expected.items()):
                ok = False
            if entry["sequence"] != sequence or entry["run_id"] in seen:
                ok = False
            seen.add(entry["run_id"])
        if len(seen) != len(run_log):
            ok = False
    except (KeyError, ValueError, TypeError):
        ok = False
    return pd.DataFrame([{
        "Deliverable": "Report", "Check": "All recorded analysis runs are consistent",
        "Status": "PASS" if ok else "FAIL",
        "Evidence": "Recorded button clicks, run sequence, thresholds and all counts recomputed",
    }])


def make_bundle(csv_bytes: bytes, png_bytes: bytes, report_bytes: bytes) -> bytes:
    """Package exactly the three user-requested deliverables into one ZIP."""
    buffer = BytesIO()
    files = {
        "01_ndvi_change_results.csv": csv_bytes,
        "02_vegetation_change_visualizations.png": png_bytes,
        "03_environmental_assessment_summary.md": report_bytes,
    }
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        for name, content in files.items():
            # A fixed ZIP entry timestamp prevents the lock hash from changing
            # merely because Streamlit rerendered the same verified files.
            info = ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content)
    return buffer.getvalue()


def verify_bundle(bundle: bytes, csv_bytes: bytes, png_bytes: bytes,
                  report_bytes: bytes) -> pd.DataFrame:
    """Confirm the ZIP opens and contains the three exact verified files."""
    expected = {
        "01_ndvi_change_results.csv": csv_bytes,
        "02_vegetation_change_visualizations.png": png_bytes,
        "03_environmental_assessment_summary.md": report_bytes,
    }
    try:
        with ZipFile(BytesIO(bundle), "r") as archive:
            passed = (archive.testzip() is None
                      and set(archive.namelist()) == set(expected)
                      and all(archive.read(name) == contents for name, contents in expected.items()))
    except (OSError, ValueError, KeyError):
        passed = False
    return pd.DataFrame([{
        "Deliverable": "ZIP package", "Check": "All 3 files are complete and unchanged",
        "Status": "PASS" if passed else "FAIL",
        "Evidence": "ZIP is readable and contains exactly the matching CSV, PNG, and Markdown files",
    }])


def make_figure(results: pd.DataFrame, analysis_id: str, run_id: str,
                threshold: float, cloud_limit: int) -> bytes:
    """Export both a location diagram and NDVI bar chart as one PNG image."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax_map, ax_bar = axes
    for _, site in results.iterrows():
        ax_map.scatter(
            float(site["longitude"]), float(site["latitude"]),
            s=170, c=site["map_color"], edgecolors="#FFFFFF", linewidths=1.5,
            zorder=3
        )
        short_name = site["site"].split()[-1]
        ax_map.annotate(short_name, (site["longitude"], site["latitude"]),
                        xytext=(6, 6), textcoords="offset points", fontsize=10)
    ax_map.set_title("Synthetic monitoring locations", fontsize=13, weight="bold", pad=12)
    ax_map.set_xlabel("Illustrative longitude")
    ax_map.set_ylabel("Illustrative latitude")
    ax_map.grid(alpha=0.18)
    ax_map.set_aspect("equal", adjustable="datalim")
    for side in ("top", "right"):
        ax_map.spines[side].set_visible(False)

    values = results["ndvi_change"].tolist()
    colors = results["map_color"].tolist()
    indices = list(range(len(results)))
    ax_bar.barh(indices, values, color=colors, edgecolor="none", height=0.65)
    ax_bar.set_yticks(indices, labels=results["site"].tolist(), fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.axvline(0, color="#283645", linewidth=1)
    ax_bar.set_xlabel("ΔNDVI = NDVI 2025 − NDVI 2024")
    ax_bar.set_title("Observed synthetic NDVI change", fontsize=13, weight="bold", pad=12)
    ax_bar.grid(axis="x", alpha=0.15)
    ax_bar.set_axisbelow(True)
    for side in ("top", "right"):
        ax_bar.spines[side].set_visible(False)
    for index, value in enumerate(values):
        ax_bar.text(value + (0.008 if value >= 0 else -0.008), index,
                    f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    ax_bar.set_xlim(min(-0.35, min(values) - 0.07), max(0.16, max(values) + 0.08))
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=clr,
               markeredgecolor="white", markersize=10, label=kind)
        for kind, clr in PALETTE.items()
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.05))
    fig.suptitle("Vegetation change assessment  |  synthetic training data", x=0.52,
                 y=1.02, fontsize=16, weight="bold", color="#173C3A")
    fig.tight_layout(rect=(0, 0.12, 1, 0.97), w_pad=3)
    fig.text(0.5, 0.02,
             f"FINAL RECORDED RUN {run_id}  |  cutoff {threshold:.2f}  |  cloud limit {cloud_limit}%  |  analysis {analysis_id}",
             ha="center", fontsize=9, color="#283645", weight="bold")
    image = BytesIO()
    fig.savefig(image, format="png", dpi=150, bbox_inches="tight", facecolor="white",
                metadata={"Software": "Synthetic NDVI training",
                          "Description": f"analysis_id={analysis_id};run_id={run_id};cutoff={threshold:.2f};cloud={cloud_limit}"})
    plt.close(fig)
    return image.getvalue()


def make_report(results: pd.DataFrame, threshold: float, cloud_limit: int,
                analysis_id: str, run_id: str, run_time: str,
                run_log: list[dict], verification: pd.DataFrame | None = None) -> str:
    """Create a concise, reproducible assessment summary in Markdown."""
    n = len(results)
    reliable = results.loc[results["classification"] != UNRELIABLE]
    flagged = results.loc[results["classification"] == UNRELIABLE]
    significant = results.loc[results["classification"] == SIGNIFICANT]
    minor = results.loc[results["classification"] == MINOR]
    nondecline = results.loc[results["classification"] == NO_DECLINE]
    priority = select_priority(results)
    changed = int(comparison_table(results)["changed"].sum())
    if priority is None:
        priority_text = "No reliable site meets the selected significant-decline threshold."
    else:
        priority_text = (
            f"{priority['site']} is the first follow-up priority among reliable observations "
            f"(ΔNDVI {priority['ndvi_change']:+.2f}; cloud cover {priority['cloud_cover_pct']:.0f}%)."
        )
    flagged_text = ", ".join(flagged["site"].tolist()) if not flagged.empty else "None"
    lines = [
        "# Vegetation Change Assessment — Synthetic NDVI Data",
        "",
        "**Purpose:** Demonstrate educational NDVI change screening for 2024 and 2025.",
        "**Data origin:** Entirely synthetic; coordinates are illustrative and not surveyed.",
        f"**Analysis ID:** `{analysis_id}`",
        f"**Final completed run:** `{run_id}`",
        f"**Run time (UTC):** `{run_time}`",
        "**Scope:** CSV and PNG show the final completed run; this report also records each clicked analysis run.",
        f"**Analysis settings:** significant decline if ΔNDVI ≤ {threshold:.2f}; unreliable if cloud cover > {cloud_limit}%.",
        "",
        "## Results",
        f"- Sites analyzed: **{n}**; reliable: **{len(reliable)}**; cloud-flagged: **{len(flagged)}**.",
        f"- Significant decline: **{len(significant)}**; minor decline: **{len(minor)}**; no decline/improvement: **{len(nondecline)}**.",
        f"- Classifications different from the baseline settings (−0.10; 20% cloud): **{changed}**.",
        "",
        "| Location | ΔNDVI | Cloud cover | Classification |",
        "|---|---:|---:|---|",
    ]
    lines.extend(
        f"| {r.site} | {r.ndvi_change:+.2f} | {r.cloud_cover_pct:.0f}% | {r.classification} |"
        for r in results.itertuples(index=False)
    )
    lines.extend([
        "",
        "## Priority and interpretation",
        priority_text,
        "",
        f"Locations requiring clearer observations because of cloud cover: **{flagged_text}**.",
        "",
        "## Recommended follow-up",
        "1. Recheck cloud-flagged sites with cloud-free, season-matched observations.",
        "2. Verify significant declines using independent imagery or field observations before attributing causes.",
        "",
        "## Important limitations",
        "NDVI is an indicator of vegetation greenness, not proof of deforestation. ",
        "This exercise does **not** process real satellite imagery. NDVI and cloud cover are synthetic inputs. ",
        "Seasonal timing, weather, agricultural practices, sensor differences, and measurement uncertainty ",
        "are not modeled. The simple site-level cloud percentage is an illustrative training rule.",
    ])
    lines.extend([
        "", "## Recording run history",
        "Each row was created only when **Run NDVI Analysis** was clicked in this browser session.",
        "", "| Completed run | NDVI cutoff | Cloud limit | Reliable | Significant decline | Cloud-flagged |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    lines.extend(
        f"| {item['run_id']} | {item['threshold']:.2f} | {item['cloud_limit']}% | "
        f"{item['reliable']} | {item['significant']} | {item['cloud_flagged']} |"
        for item in run_log
    )
    lines.extend([
        "", "**Recording consistency:** The run ID printed on the dashboard, CSV, PNG and this report must match.",
        "Run history applies only to the current browser session; refreshing or restarting creates a new session.",
        "", "## Verification", "Internal consistency checks only; no external satellite-data validation.",
    ])
    if verification is None:
        lines.append("Verification was not run for these analysis settings.")
    else:
        for item in verification.itertuples(index=False):
            lines.append(f"- **{item.Status}** — {item.Check}: {item.Evidence}")
    return "\n".join(lines) + "\n"


st.set_page_config(
    page_title="Synthetic NDVI | Vegetation Change Assessment",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container { max-width: 1040px; padding-top: 1.2rem; padding-bottom: 3rem; }
h1, h2, h3 { letter-spacing: -0.025em; }
/* Dark dashboard: visible metrics on both mobile and desktop. */
[data-testid="stMetric"] {
    background: #182638;
    border: 1px solid #32465A;
    padding: 16px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.12);
    color: #F8FAFC !important;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
    color: #CAD6E5 !important;
    opacity: 1 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricValue"] *,
[data-testid="stMetric"] [data-testid="stMetricDelta"],
[data-testid="stMetric"] [data-testid="stMetricDelta"] * {
    color: #FFFFFF !important;
    opacity: 1 !important;
}
.stButton button { min-height: 2.9rem; font-weight: 650; }
.small-note { color: #A9BAC9; font-size: 0.91rem; }
</style>
""", unsafe_allow_html=True)

st.title("🌿 Vegetation Change Assessment")
st.caption("Interactive GIS & Remote Sensing Training  ·  2024 vs 2025  ·  100% synthetic data")
st.write(
    "Use this dashboard to investigate vegetation decline, test classification "
    "thresholds, screen cloud-affected observations, and export a short assessment."
)

with st.expander("📘 What does NDVI change mean?", expanded=False):
    st.markdown(
        "**NDVI** measures vegetation greenness from red and near-infrared reflectance. "
        "This demonstration starts with two synthetic NDVI values for each site; "
        "it does not retrieve or process satellite imagery."
    )
    st.latex(r"\Delta NDVI = NDVI_{2025} - NDVI_{2024}")
    st.write(
        "Negative values suggest reduced greenness; positive values suggest an increase. "
        "Cloud contamination makes a simple comparison less reliable. "
        "A change alone cannot establish deforestation."
    )

if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False
if "run_log" not in st.session_state:
    st.session_state.run_log = []
if "recording_session" not in st.session_state:
    st.session_state.recording_session = secrets.token_hex(4).upper()
if "active_run" not in st.session_state:
    st.session_state.active_run = None
if "controls_dirty" not in st.session_state:
    st.session_state.controls_dirty = False
if "verified_run_id" not in st.session_state:
    st.session_state.verified_run_id = None
if "verified_zip_sha" not in st.session_state:
    st.session_state.verified_zip_sha = None


def invalidate_verification() -> None:
    """Any edit to a slider revokes verification until a new analysis run."""
    st.session_state.controls_dirty = True
    st.session_state.verified_run_id = None
    st.session_state.verified_zip_sha = None


st.subheader("01 · Load synthetic data")
if st.button("📂 Load Synthetic Dataset", type="primary", use_container_width=True):
    st.session_state.dataset_loaded = True
    st.session_state.run_log = []
    st.session_state.active_run = None
    st.session_state.controls_dirty = False
    st.session_state.verified_run_id = None
    st.session_state.verified_zip_sha = None
    st.session_state.recording_session = secrets.token_hex(4).upper()

if not st.session_state.dataset_loaded:
    st.info("Tap **Load Synthetic Dataset** to begin the exercise.")
    st.stop()

raw = synthetic_data()
st.success(f"Dataset loaded: {len(raw)} fictional monitoring sites across two simulated periods.")
st.dataframe(
    raw.rename(columns={
        "site": "Location", "latitude": "Latitude", "longitude": "Longitude",
        "ndvi_2024": "NDVI 2024", "ndvi_2025": "NDVI 2025",
        "cloud_cover_pct": "Cloud cover (%)",
    }),
    hide_index=True, use_container_width=True,
)
st.download_button(
    "Download synthetic input CSV (optional)",
    data=raw.to_csv(index=False).encode("utf-8"),
    file_name="synthetic_ndvi_input.csv", mime="text/csv", use_container_width=True,
)

st.subheader("02 · Analyze vegetation change")
left, right = st.columns(2)
with left:
    threshold = st.slider(
        "Significant decline cutoff (ΔNDVI)",
        min_value=-0.30, max_value=-0.05, value=-0.10, step=0.01, format="%.2f",
        help="Sites at or below this change are considered significantly declining if cloud cover is acceptable.",
        on_change=invalidate_verification,
    )
with right:
    cloud_limit = st.slider(
        "Maximum acceptable cloud cover (%)",
        min_value=0, max_value=50, value=20, step=5,
        help="A site with cloud cover above this value is marked unreliable, regardless of NDVI change.",
        on_change=invalidate_verification,
    )
st.caption("Reference settings: significant decline ≤ −0.10 and cloud-flagged when cloud cover > 20%. Current slider choices override these defaults.")
if st.button("▶ Run NDVI Analysis", type="primary", use_container_width=True):
    completed_threshold = round(threshold, 2)
    new_results = analyze_data(raw, completed_threshold, cloud_limit)
    sequence = len(st.session_state.run_log) + 1
    record = {
        "sequence": sequence,
        "run_id": f"{st.session_state.recording_session}-R{sequence:02d}",
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": completed_threshold,
        "cloud_limit": cloud_limit,
        "reliable": int((new_results["classification"] != UNRELIABLE).sum()),
        "significant": int((new_results["classification"] == SIGNIFICANT).sum()),
        "cloud_flagged": int((new_results["classification"] == UNRELIABLE).sum()),
        "analysis_id": analysis_identifier(raw, completed_threshold, cloud_limit),
    }
    st.session_state.run_log.append(record)
    st.session_state.active_run = record
    st.session_state.controls_dirty = False
    st.session_state.verified_run_id = None
    st.session_state.verified_zip_sha = None
if st.session_state.active_run is None:
    st.info("Tap **Run NDVI Analysis** to calculate changes, classifications, and recommendations.")
    st.stop()

active = st.session_state.active_run
# Charts and exports always use the last clicked Run, NEVER the uncommitted slider values.
current_threshold = active["threshold"]
current_cloud_limit = active["cloud_limit"]
pending_changes = (st.session_state.controls_dirty or
                   round(threshold, 2) != current_threshold or
                   cloud_limit != current_cloud_limit)
if pending_changes:
    st.warning("⚠️ Slider settings have changed since the last analysis. "
               "Click **Run NDVI Analysis** before verifying or downloading. "
               "The results below still show the last completed run.")
else:
    st.success(f"Completed run **{active['run_id']}** · NDVI cutoff **{current_threshold:.2f}** "
               f"· cloud limit **{current_cloud_limit}%**")
results = analyze_data(raw, threshold=current_threshold, cloud_limit=current_cloud_limit)
reliable = results.loc[results["classification"] != UNRELIABLE]
priority = select_priority(results)
comparison = comparison_table(results)

m1, m2 = st.columns(2)
m1.metric("Reliable locations", f"{len(reliable)} / {len(results)}")
m2.metric("Significant declines", int((results["classification"] == SIGNIFICANT).sum()))
m3, m4 = st.columns(2)
m3.metric("Cloud-flagged", int((results["classification"] == UNRELIABLE).sum()))
m4.metric("Changes vs default", int(comparison["changed"].sum()))

st.subheader("03 · Explore the map and NDVI results")
st.markdown(
    "🔴 Significant decline　🟠 Minor decline　🟢 No decline / improvement　⚪ Unreliable"
)
st.caption("Site coordinates and measurements are synthetic. The interactive map uses online basemap tiles.")
st.map(
    results[["latitude", "longitude", "map_color"]],
    latitude="latitude", longitude="longitude", color="map_color", size=500,
    height=320,
)

st.write("**NDVI change by location**")
st.bar_chart(results, x="site", y="ndvi_change", color="#167C73", height=290)
st.caption("Cloud-flagged sites remain visible for illustration but should not be interpreted as verified change.")

st.write("**Classification table**")
table_columns = ["site", "ndvi_2024", "ndvi_2025", "ndvi_change", "cloud_cover_pct", "classification"]
st.dataframe(
    results[table_columns].rename(columns={
        "site": "Location", "ndvi_2024": "NDVI 2024", "ndvi_2025": "NDVI 2025",
        "ndvi_change": "ΔNDVI", "cloud_cover_pct": "Cloud (%)",
        "classification": "Quality-aware classification",
    }),
    hide_index=True, use_container_width=True,
)

st.subheader("04 · Examine threshold sensitivity and cloud quality")
changed = comparison.loc[comparison["changed"]]
if changed.empty:
    st.success("No locations changed classification compared with the default settings.")
else:
    st.warning(f"{len(changed)} location(s) changed classification compared with the default settings.")
    st.dataframe(
        changed[["site", "baseline_classification", "classification"]].rename(columns={
            "site": "Location", "baseline_classification": "Default classification",
            "classification": "Current classification",
        }),
        hide_index=True, use_container_width=True,
    )
cloud_sites = results.loc[results["classification"] == UNRELIABLE, "site"].tolist()
if cloud_sites:
    st.info("Cloud-flagged for recheck: " + ", ".join(cloud_sites) + ".")
else:
    st.info("No sites exceed the selected cloud-cover limit.")

st.subheader("05 · Choose an evidence-based follow-up")
if priority is None:
    st.info("No reliable site meets the current significant-decline threshold. Try adjusting the sliders.")
else:
    st.success(
        f"**Priority: {priority['site']}** — ΔNDVI {priority['ndvi_change']:+.2f}; "
        f"cloud cover {priority['cloud_cover_pct']:.0f}%. "
        "This is the largest significant decline among the sites passing the cloud screen."
    )
st.caption(
    "Follow-up action: obtain clearer, season-matched imagery and verify any apparent decline "
    "before inferring a specific cause."
)

st.subheader("06 · Preview, verify and download your 3 deliverables")
st.write(
    "**Recording-matched exports:** The three files reflect the **last completed Run NDVI Analysis click**, "
    "not unsubmitted slider changes. You must verify that run before downloading. "
    "The report also documents every completed run from this browser session."
)

# The run ID in each of the three exports must match the recording's on-screen run ID.
analysis_id = active["analysis_id"]
run_id = active["run_id"]
run_time = active["time_utc"]
export_frame = results[table_columns + ["latitude", "longitude"]].copy()
export_frame["decline_cutoff"] = current_threshold
export_frame["cloud_limit_pct"] = current_cloud_limit
export_frame["analysis_id"] = analysis_id
export_frame["run_id"] = run_id
export_frame["run_time_utc"] = run_time
export_csv = export_frame.to_csv(index=False).encode("utf-8")
export_png = make_figure(results, analysis_id, run_id, current_threshold, current_cloud_limit)
base_checks = verify_analysis(raw, results, current_threshold, current_cloud_limit,
                              export_csv, export_png, analysis_id, run_id, run_time)
history_check = verify_run_history(raw, st.session_state.run_log, active)
export_report = make_report(results, threshold=current_threshold, cloud_limit=current_cloud_limit,
                            analysis_id=analysis_id, run_id=run_id, run_time=run_time,
                            run_log=st.session_state.run_log,
                            verification=pd.concat([base_checks, history_check], ignore_index=True))
export_md = export_report.encode("utf-8")
report_check = verify_report(export_report, results, current_threshold,
                             current_cloud_limit, analysis_id, run_id, run_time,
                             st.session_state.run_log)
export_zip = make_bundle(export_csv, export_png, export_md)
zip_check = verify_bundle(export_zip, export_csv, export_png, export_md)
checks = pd.concat([base_checks, history_check, report_check, zip_check], ignore_index=True)
all_pass = bool(checks["Status"].eq("PASS").all())
archive_sha = sha256(export_zip).hexdigest()
ready = (all_pass and not pending_changes and
         st.session_state.verified_run_id == run_id and
         st.session_state.verified_zip_sha == archive_sha)

st.caption(
    f"Displayed FINAL run: `{run_id}` · ΔNDVI cutoff {current_threshold:.2f} "
    f"· Cloud limit {current_cloud_limit}% · Analysis ID `{analysis_id}`"
)
st.info("**How to match your recording:** Keep the visible run ID on-screen while reviewing the results. "
        "The CSV, exported PNG and Markdown report repeat this same ID and final-run settings. "
        "The report records your earlier Run clicks as a compact audit trail.")
if pending_changes:
    st.error("Unapplied slider edits detected. Download is locked until you click **Run NDVI Analysis**.")
for file_name, display_name in [
    ("CSV", "1 · NDVI results (.csv)"),
    ("PNG", "2 · Visualizations (.png)"),
    ("Report", "3 · Assessment summary (.md)"),
]:
    matching = checks.loc[checks["Deliverable"] == file_name]
    passed = int(matching["Status"].eq("PASS").sum())
    if passed == len(matching):
        st.success(f"✅ {display_name}: VERIFIED ({passed}/{len(matching)} checks)")
    else:
        st.error(f"❌ {display_name}: VERIFICATION FAILED ({passed}/{len(matching)} checks passed)")

if st.button("🔎 Verify and lock this completed run for export", use_container_width=True,
             disabled=pending_changes):
    if all_pass:
        st.session_state.verified_run_id = run_id
        st.session_state.verified_zip_sha = archive_sha
        st.success(f"Verified and locked run {run_id}. The ZIP will contain these exact results.")
    else:
        st.session_state.verified_run_id = None
        st.session_state.verified_zip_sha = None
        st.error("Verification failed. Downloads remain locked; inspect the checks below.")
ready = (all_pass and not pending_changes and
         st.session_state.verified_run_id == run_id and
         st.session_state.verified_zip_sha == archive_sha)
if ready:
    st.success(f"🔒 READY: run {run_id} matches all 3 deliverables and the final ZIP.")
else:
    st.warning("🔒 Downloads locked. Run the analysis with your final slider choices, then click Verify and lock.")

with st.expander("Show detailed verification evidence", expanded=False):
    st.dataframe(checks, hide_index=True, use_container_width=True)
    st.caption("Verification proves internal consistency of synthetic outputs, not real-world ground truth.")

with st.expander("Show recorded analysis runs", expanded=True):
    st.dataframe(pd.DataFrame(st.session_state.run_log)[[
        "run_id", "time_utc", "threshold", "cloud_limit", "reliable",
        "significant", "cloud_flagged"]].rename(columns={
            "run_id": "Run ID", "time_utc": "UTC time", "threshold": "NDVI cutoff",
            "cloud_limit": "Cloud limit (%)", "reliable": "Reliable",
            "significant": "Significant", "cloud_flagged": "Cloud-flagged",
        }), hide_index=True, use_container_width=True)
    st.caption("Only analysis button clicks are captured. Moving a slider without running the analysis is not a completed run.")

st.markdown("#### View and inspect the three deliverables")
tab_csv, tab_png, tab_report = st.tabs(["1 · CSV results", "2 · PNG visualization", "3 · Markdown report"])
with tab_csv:
    st.write("**Actual CSV file preview** — including analysis ID and the selected thresholds.")
    st.dataframe(pd.read_csv(BytesIO(export_csv)), hide_index=True, use_container_width=True)
    st.download_button(
        "⬇ Download CSV", data=export_csv, file_name="01_ndvi_change_results.csv",
        mime="text/csv", use_container_width=True, disabled=not ready,
        key="csv_export", on_click="ignore",
    )
with tab_png:
    st.write("**Actual PNG file preview** — this image is the one included in your ZIP.")
    st.image(export_png, caption=f"Synthetic monitoring locations and NDVI changes · Analysis {analysis_id}",
             use_container_width=True)
    st.download_button(
        "⬇ Download PNG", data=export_png, file_name="02_vegetation_change_visualizations.png",
        mime="image/png", use_container_width=True, disabled=not ready,
        key="png_export", on_click="ignore",
    )
with tab_report:
    st.write("**Actual report preview** — conclusions and verification notes for the current settings.")
    st.markdown(export_report)
    st.download_button(
        "⬇ Download Markdown report", data=export_md,
        file_name="03_environmental_assessment_summary.md", mime="text/markdown",
        use_container_width=True, disabled=not ready,
        key="md_export", on_click="ignore",
    )

st.divider()
st.markdown("#### Export all three in one ZIP file")
if ready:
    st.success(f"✅ All 3 deliverables and ZIP match completed recording run {run_id}.")
else:
    st.error("ZIP export is disabled until the final recorded run has been verified and locked.")
st.download_button(
    "📦 Download all 3 verified deliverables (.zip)",
    data=export_zip,
    file_name=f"ndvi_assessment_{run_id}.zip",
    mime="application/zip", type="primary", use_container_width=True,
    disabled=not ready, key="zip_export", on_click="ignore",
)
st.caption(f"Final run ID: `{run_id}` (shown in the CSV, PNG and report). "
           "The ZIP contains the final run only; the report also includes the prior run history.")
st.caption("ZIP contents: 01_ndvi_change_results.csv · "
           "02_vegetation_change_visualizations.png · "
           "03_environmental_assessment_summary.md")

st.divider()
st.caption(
    "Training demonstration only. All coordinates, NDVI values and cloud percentages "
    "are synthetic. Map tiles require internet access; analysis and exports require no external data."
)
