"""Interactive assessment of vegetation changes using synthetic NDVI data.

Phone-friendly Streamlit app. No external datasets, credentials, or APIs needed
for the calculations. The optional interactive map uses online basemap tiles.
"""

from __future__ import annotations

from io import BytesIO

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


def make_figure(results: pd.DataFrame) -> bytes:
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
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Vegetation change assessment  |  synthetic training data", x=0.52,
                 y=1.02, fontsize=16, weight="bold", color="#173C3A")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97), w_pad=3)
    image = BytesIO()
    fig.savefig(image, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return image.getvalue()


def make_report(results: pd.DataFrame, threshold: float, cloud_limit: int) -> str:
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
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

st.subheader("01 · Load synthetic data")
if st.button("📂 Load Synthetic Dataset", type="primary", use_container_width=True):
    st.session_state.dataset_loaded = True
    st.session_state.analysis_started = False

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
    )
with right:
    cloud_limit = st.slider(
        "Maximum acceptable cloud cover (%)",
        min_value=0, max_value=50, value=20, step=5,
        help="A site with cloud cover above this value is marked unreliable, regardless of NDVI change.",
    )
st.caption("Reference settings: significant decline ≤ −0.10 and cloud-flagged when cloud cover > 20%. Current slider choices override these defaults.")
if st.button("▶ Run NDVI Analysis", type="primary", use_container_width=True):
    st.session_state.analysis_started = True
if not st.session_state.analysis_started:
    st.info("Tap **Run NDVI Analysis** to calculate changes, classifications, and recommendations.")
    st.stop()

results = analyze_data(raw, threshold=round(threshold, 2), cloud_limit=cloud_limit)
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

st.subheader("06 · Export your three final deliverables")
st.write("Download the files produced from the current slider settings.")
export_csv = results[table_columns + ["latitude", "longitude"]].to_csv(index=False).encode("utf-8")
export_png = make_figure(results)
export_report = make_report(results, threshold=round(threshold, 2), cloud_limit=cloud_limit)

st.download_button(
    "⬇ 1. Download NDVI change results (.csv)",
    data=export_csv, file_name="01_ndvi_change_results.csv", mime="text/csv",
    use_container_width=True,
)
st.download_button(
    "⬇ 2. Download vegetation visualizations (.png)",
    data=export_png, file_name="02_vegetation_change_visualizations.png", mime="image/png",
    use_container_width=True,
)
st.download_button(
    "⬇ 3. Download assessment summary (.md)",
    data=export_report.encode("utf-8"), file_name="03_environmental_assessment_summary.md",
    mime="text/markdown", use_container_width=True,
)
with st.expander("Preview the assessment summary"):
    st.markdown(export_report)

st.divider()
st.caption(
    "Training demonstration only. All coordinates, NDVI values and cloud percentages "
    "are synthetic. Map tiles require internet access; analysis and exports require no external data."
)
