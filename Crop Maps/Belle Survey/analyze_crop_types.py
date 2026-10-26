from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Crop_Data_Cleaned"
OUT_DIR = SCRIPT_DIR / "crop_type_analysis"
OUTPUT = OUT_DIR / "type_composition_stacked.png"
SUMMARY_OUTPUT = OUT_DIR / "harvested_crop_type_summary.csv"
AREA_OUTPUT = OUT_DIR / "harvested_area_share_by_season.csv"
PIVOT_OUTPUT = OUT_DIR / "harvested_crop_acreage_by_season.csv"

TOP_N = 50
SEASON_ORDER = {"Kharif": 0, "Rabi": 1, "Summer": 2}  # chronological within a year
NON_CROP_CATEGORIES = ["Fallow / Non-Agricultural"]
DOMINANT_SHARE = 0.70
INPUT_COLUMNS = [
    "year", 
    "season", 
    "survey_num", 
    "area_ha", 
    "crop_name", 
    "crop_detail",
    "crop_category", 
    "kgis_village_id",
]

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def load_crops():
    frames = [
        pd.read_csv(
            path,
            usecols=INPUT_COLUMNS,
            dtype={"survey_num": "string", "kgis_village_id": "string"},
        )
        for path in sorted(DATA_DIR.glob("*_crop_data_cleaned.csv"))
    ]
    if not frames:
        raise FileNotFoundError(f"No cleaned crop data CSV files found in {DATA_DIR}")

    crops = pd.concat(frames, ignore_index=True)
    crops["crop_detail"] = crops["crop_detail"].astype("string").str.strip()
    crops["crop_name"] = crops["crop_name"].astype("string").str.strip()
    crops["season"] = crops["season"].str.strip().str.capitalize()
    crops["period_idx"] = crops["year"] * 3 + crops["season"].map(SEASON_ORDER)
    crops["harvested"] = crops["crop_detail"].str.match(r"(?i)^harvested\s*-", na=False)
    crops["type"] = crops["crop_detail"].str.replace(
        r"(?i)^harvested\s*-\s*", "", regex=True
    ).str.strip()
    crops["plot_id"] = crops["kgis_village_id"] + "|" + crops["survey_num"]
    return crops


def composition_stats(crops):
    """Share of crop names with a single type / one dominant type (non-harvested records)."""
    standard = crops.loc[~crops["harvested"]]
    areas = standard.groupby(["crop_name", "type"])["area_ha"].sum()
    areas = areas[areas > 0].reset_index()
    areas["share"] = areas["area_ha"] / areas.groupby("crop_name")["area_ha"].transform("sum")
    per_crop = areas.groupby("crop_name").agg(n_types=("type", "size"), top_share=("share", "max"))

    n_crops = len(per_crop)
    single = (per_crop["n_types"] == 1).sum()
    multi = per_crop[per_crop["n_types"] > 1]
    dominant = (multi["top_share"] > DOMINANT_SHARE).sum()
    print(f"Crop names (non-harvested): {n_crops}")
    print(f"  Single type only: {single} ({single / n_crops:.1%})")
    print(f"  Multiple types: {len(multi)}")
    print(f"  Multi-type with one type > {DOMINANT_SHARE:.0%} of area: "
          f"{dominant}/{len(multi)} ({dominant / len(multi):.1%})")
    print(multi.sort_values("top_share", ascending=False).to_string())


def harvested_area_share(crops):
    """Harvested share of total and of cropped (non fallow/non-agricultural) area per season."""
    rows = []
    for (year, season), g in crops.groupby(["year", "season"]):
        harv = g.loc[g["harvested"], "area_ha"].sum()
        total = g["area_ha"].sum()
        cropped = g.loc[~g["crop_category"].isin(NON_CROP_CATEGORIES)]
        cropped_total = cropped["area_ha"].sum()
        harv_cropped = cropped.loc[cropped["harvested"], "area_ha"].sum()
        harvested_types = (
            g.loc[g["harvested"]]
            .groupby("type")["area_ha"]
            .sum()
            .sort_values(ascending=False)
        )
        crop_composition = ", ".join(
            f"{crop_type} ({100 * area / harv:.2f}%)"
            for crop_type, area in harvested_types.items()
        ) if harv else ""
        rows.append({
            "year": year,
            "season": season,
            "harvested_area_ha": harv,
            "total_surveyed_area_ha": total,
            "harvested_pct_of_total_area": 100 * harv / total,
            "cropped_area_ha": cropped_total,
            "harvested_pct_of_cropped_area": 100 * harv_cropped / cropped_total,
            "harvested_crop_composition": crop_composition,
        })
    result = pd.DataFrame(rows)
    result["_order"] = result["season"].map(SEASON_ORDER)
    result = result.sort_values(["year", "_order"]).drop(columns="_order")
    result.to_csv(AREA_OUTPUT, index=False)
    print(f"\nSaved {AREA_OUTPUT}")
    print(result.to_string(index=False))


def harvested_acreage_pivot(crops):
    """Write harvested crop-type area with crop types as rows and seasons as columns."""
    harvested = crops.loc[crops["harvested"]].copy()
    harvested["season_label"] = harvested["year"].astype(str) + " " + harvested["season"]
    season_order = (
        harvested[["season_label", "period_idx"]]
        .drop_duplicates()
        .sort_values("period_idx")["season_label"]
    )
    pivot = (
        harvested.groupby(["type", "season_label"])["area_ha"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=season_order, fill_value=0)
    )
    pivot["total_harvested_area_ha"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total_harvested_area_ha", ascending=False)
    pivot.index.name = "crop_type"
    pivot.to_csv(PIVOT_OUTPUT)
    print(f"Saved {PIVOT_OUTPUT} ({len(pivot)} crop types)")


def harvested_summary(crops):
    """One row per detailed harvested crop type with prior-season persistence on the same plot."""
    valid = crops["plot_id"].notna()
    harvested = (
        crops.loc[valid & crops["harvested"], ["plot_id", "period_idx", "year", "season", "crop_name", "type"]]
        .drop_duplicates(["plot_id", "period_idx", "crop_name", "type"])
    )
    standard = (
        crops.loc[valid & ~crops["harvested"], ["plot_id", "period_idx", "crop_name", "type"]]
        .drop_duplicates()
    )

    previous = standard.assign(period_idx=standard["period_idx"] + 1, preceded=True)
    harvested = harvested.merge(previous, on=["plot_id", "period_idx", "crop_name", "type"], how="left")
    harvested["preceded"] = harvested["preceded"].eq(True)
    harvested["eligible"] = harvested["period_idx"] > crops["period_idx"].min()

    eligible_all = harvested[harvested["eligible"]]
    print(f"\nHarvested plot-season records: {len(harvested)} "
          f"({len(eligible_all)} with a previous season in the data)")
    print(f"  Preceded by same crop type on same plot in previous season: "
          f"{eligible_all['preceded'].sum()} ({eligible_all['preceded'].mean():.1%})")

    harvested["season_label"] = harvested["year"].astype(str) + " " + harvested["season"]
    harvested_area = (
        crops.loc[crops["harvested"]].groupby(["crop_name", "type"])["area_ha"].sum()
    )
    rows = []
    for (crop_name, crop_type), g in harvested.groupby(["crop_name", "type"]):
        eligible = g[g["eligible"]]
        seasons = g.sort_values("period_idx")["season_label"].unique()
        rows.append({
            "crop_name": crop_name,
            "crop_type": crop_type,
            "harvested_plot_season_records": len(g),
            "harvested_area_ha": harvested_area[(crop_name, crop_type)],
            "seasons": ", ".join(seasons),
            "records_with_previous_season_data": len(eligible),
            "records_preceded_by_same_crop_prev_season": int(eligible["preceded"].sum()),
            "pct_preceded_by_same_crop_prev_season": (
                100 * eligible["preceded"].mean() if len(eligible) else pd.NA
            ),
        })
    summary = pd.DataFrame(rows).sort_values("harvested_plot_season_records", ascending=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    print(f"Saved {SUMMARY_OUTPUT} ({len(summary)} rows)")


def plot_composition(crops):
    top_crops = crops.groupby("crop_name")["area_ha"].sum().nlargest(TOP_N).index
    standard = crops.loc[~crops["harvested"] & crops["crop_name"].isin(top_crops)]
    areas = standard.groupby(["crop_name", "type"], as_index=False)["area_ha"].sum()
    areas = areas[areas["area_ha"] > 0]
    areas["share"] = areas["area_ha"] / areas.groupby("crop_name")["area_ha"].transform("sum")
    order = standard.groupby("crop_name")["area_ha"].sum().reindex(top_crops).sort_values().index
    areas["crop_name"] = pd.Categorical(areas["crop_name"], categories=order, ordered=True)
    areas = areas.sort_values(["crop_name", "share"])

    width = 12
    row_heights = []
    for crop in order:
        types = areas.loc[areas["crop_name"] == crop, "type"].astype(str)
        row_heights.append(max(0.6, max((len(name) * 5 / 72 for name in types), default=0.6)))

    fig, ax = plt.subplots(figsize=(width, sum(row_heights) + 1.6))
    colors = plt.get_cmap("tab20")
    y_positions = []
    y = 0
    for crop, row_height in zip(order, row_heights):
        y_positions.append(y + row_height / 2)
        left = 0.0
        parts = areas.loc[areas["crop_name"] == crop]
        for idx, part in enumerate(parts.itertuples(index=False)):
            share = part.share
            ax.add_patch(Rectangle(
                (left, y), share, row_height,
                facecolor=colors(idx % 20), edgecolor="white", linewidth=0.7,
            ))
            label = str(part.type)
            label_size = 7
            rotate = share * width < len(label) * label_size * 0.55 / 72
            if rotate:
                label_size = min(5, max(3, share * width * 72 * 0.8))
            ax.text(
                left + share / 2, y + row_height / 2, label,
                ha="center", va="center", fontsize=label_size,
                rotation=90 if rotate else 0, clip_on=True,
            )
            left += share
        y += row_height

    ax.set_xlim(0, 1)
    ax.set_ylim(0, y)
    ax.set_yticks(y_positions, [str(crop) for crop in order])
    ax.invert_yaxis()
    ax.set_xlabel("Share of standard crop area")
    ax.set_title(f"Crop-type composition of the {len(order)} largest crop names")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT}")

# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
    crops = load_crops()
    composition_stats(crops)
    harvested_area_share(crops)
    harvested_acreage_pivot(crops)
    harvested_summary(crops)
    plot_composition(crops)
