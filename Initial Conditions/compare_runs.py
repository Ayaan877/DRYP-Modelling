import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from tqdm import tqdm

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
VARIABLE = "gdh"
# Each dataset: plot label, GeoTIFF directory, and filename prefix before the timestamp.
INPUT_DATASETS: list[dict[str, Any]] = [
    {"label": "run 2", "directory": SCRIPT_DIR / f"{VARIABLE}_run2", "filename_prefix": f"{VARIABLE}_run2_"},
    {"label": "run 3", "directory": SCRIPT_DIR / f"{VARIABLE}_run3", "filename_prefix": f"{VARIABLE}_run3_"},
]
COMPARISON_PAIRS: list[tuple[str, str]] = [("run 2", "run 3")]
OUTPUT_DIRECTORY = SCRIPT_DIR / "run comparison"
START_DATE: Optional[datetime] = None
END_DATE: Optional[datetime] = None
TIME_STEP = 1
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------


def slugify(label: str) -> str:
    return re.sub(r"\W+", "_", label).strip("_").lower()


def index_dataset(dataset: dict[str, Any]) -> dict[datetime, Path]:
    label = str(dataset["label"])
    directory = Path(dataset["directory"])
    pattern = re.compile(rf"^{re.escape(str(dataset['filename_prefix']))}(.+)\.tif$")

    index: dict[datetime, Path] = {}
    for path in tqdm(sorted(directory.glob("*.tif")), desc=f"Indexing {label}", unit="file"):
        match = pattern.match(path.name)
        if not match:
            continue
        try:
            timestamp = datetime.strptime(match.group(1), TIMESTAMP_FORMAT)
        except ValueError:
            continue
        index[timestamp] = path

    if not index:
        raise FileNotFoundError(f"No matching GeoTIFFs for {label!r} in {directory}")
    return index


def select_timestamps(common: list[datetime]) -> list[datetime]:
    selected = common
    if START_DATE is not None:
        selected = [t for t in selected if t.date() >= START_DATE.date()]
    if END_DATE is not None:
        selected = [t for t in selected if t.date() <= END_DATE.date()]
    return selected[::TIME_STEP]


def read_field(path: Path) -> tuple[np.ndarray, rasterio.Affine, tuple[int, int]]:
    with rasterio.open(path) as src:
        data = src.read(1, masked=True).astype(np.float64).filled(np.nan)
        data[~np.isfinite(data)] = np.nan
        return data, src.transform, (src.height, src.width)


def plot_aggregate(
    times: np.ndarray,
    mean_a: np.ndarray,
    mean_b: np.ndarray,
    label_a: str,
    label_b: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, mean_a, lw=0.8, label=label_a)
    ax.plot(times, mean_b, lw=0.8, label=label_b)
    ax.set_ylabel(f"Spatial mean {VARIABLE}")
    ax.set_title(f"{VARIABLE}: {label_a} vs {label_b} (aggregate over valid pixels)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_difference_series(
    times: np.ndarray,
    msd: np.ndarray,
    mean_diff: np.ndarray,
    label_a: str,
    label_b: str,
    path: Path,
) -> None:
    fig, (ax_msd, ax_bias) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_msd.plot(times, msd, lw=0.8, color="tab:red")
    ax_msd.set_ylabel("MSD")
    ax_msd.set_title(f"{VARIABLE}: {label_a} minus {label_b}")
    ax_msd.grid(alpha=0.3)
    ax_bias.plot(times, mean_diff, lw=0.8, color="tab:gray")
    ax_bias.axhline(0, color="k", lw=0.6)
    ax_bias.set_ylabel("Mean difference")
    ax_bias.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pixel_maps(
    msd_map: np.ndarray,
    bias_map: np.ndarray,
    transform: Any,
    label_a: str,
    label_b: str,
    path: Path,
) -> None:
    height, width = msd_map.shape
    left = transform.c
    top = transform.f
    extent = (left, left + width * transform.a, top + height * transform.e, top)

    fig, (ax_msd, ax_bias) = plt.subplots(1, 2, figsize=(14, 6))
    image_msd = ax_msd.imshow(msd_map, extent=extent, origin="upper", cmap="viridis")
    ax_msd.set_title("Time-mean MSD per pixel")
    fig.colorbar(image_msd, ax=ax_msd, shrink=0.8)

    limit = np.nanmax(np.abs(bias_map))
    image_bias = ax_bias.imshow(
        bias_map, extent=extent, origin="upper", cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    ax_bias.set_title(f"Time-mean difference per pixel ({label_a} - {label_b})")
    fig.colorbar(image_bias, ax=ax_bias, shrink=0.8)

    fig.suptitle(f"{VARIABLE}: pixel-level comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare_pair(
    label_a: str,
    index_a: dict[datetime, Path],
    label_b: str,
    index_b: dict[datetime, Path],
    output_directory: Path,
) -> None:
    common = select_timestamps(sorted(set(index_a) & set(index_b)))
    if not common:
        raise ValueError(f"No common timestamps for {label_a!r} and {label_b!r} in the selected range.")
    print(f"{label_a} vs {label_b}: {len(common)} common timestamps.")

    output_directory.mkdir(parents=True, exist_ok=True)
    pair_prefix = f"{slugify(VARIABLE)}_R2vsR3"

    _, transform_a, shape_a = read_field(index_a[common[0]])
    _, transform_b, shape_b = read_field(index_b[common[0]])
    if shape_a != shape_b or transform_a != transform_b:
        raise ValueError(f"{label_a!r} and {label_b!r} are not on the same grid; resample before comparing.")

    count = np.zeros(shape_a)
    sum_diff = np.zeros(shape_a)
    sum_sq = np.zeros(shape_a)
    rows: list[tuple[datetime, float, float, float, float, float, int]] = []

    for timestamp in tqdm(common, desc=f"{label_a} vs {label_b}", unit="timestep"):
        a, _, _ = read_field(index_a[timestamp])
        b, _, _ = read_field(index_b[timestamp])
        valid = np.isfinite(a) & np.isfinite(b)
        n_valid = int(valid.sum())
        if n_valid == 0:
            rows.append((timestamp, np.nan, np.nan, np.nan, np.nan, np.nan, 0))
            continue

        a_values = a[valid]
        b_values = b[valid]
        diff = a_values - b_values
        msd = float(np.mean(diff**2))
        rows.append(
            (
                timestamp,
                float(a_values.mean()),
                float(b_values.mean()),
                float(diff.mean()),
                msd,
                float(np.sqrt(msd)),
                n_valid,
            )
        )
        count[valid] += 1
        sum_diff[valid] += diff
        sum_sq[valid] += diff**2

    with np.errstate(invalid="ignore", divide="ignore"):
        msd_map = np.where(count > 0, sum_sq / count, np.nan)
        bias_map = np.where(count > 0, sum_diff / count, np.nan)

    csv_path = output_directory / f"{pair_prefix}_timeseries_stats.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["time", f"mean_{slugify(label_a)}", f"mean_{slugify(label_b)}", "mean_difference", "msd", "rmse", "n_valid_pixels"]
        )
        for row in rows:
            writer.writerow([row[0].isoformat(sep=" ")] + list(row[1:]))

    times = np.array([row[0] for row in rows], dtype=object)
    mean_a = np.array([row[1] for row in rows], dtype=float)
    mean_b = np.array([row[2] for row in rows], dtype=float)
    mean_diff = np.array([row[3] for row in rows], dtype=float)
    msd_series = np.array([row[4] for row in rows], dtype=float)

    plot_aggregate(
        times,
        mean_a,
        mean_b,
        label_a,
        label_b,
        output_directory / f"{pair_prefix}_aggregate_timeseries.png",
    )
    plot_difference_series(
        times,
        msd_series,
        mean_diff,
        label_a,
        label_b,
        output_directory / f"{pair_prefix}_difference_timeseries.png",
    )
    plot_pixel_maps(
        msd_map,
        bias_map,
        transform_a,
        label_a,
        label_b,
        output_directory / f"{pair_prefix}_pixel_maps.png",
    )

    print(f"Mean MSD over the period: {np.nanmean(msd_series):.4e}")
    print(f"Outputs written to {output_directory}")


# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------


def main() -> None:
    indexes = {
        str(dataset["label"]): index_dataset(dataset)
        for dataset in tqdm(INPUT_DATASETS, desc="Datasets", unit="dataset")
    }
    for label_a, label_b in tqdm(COMPARISON_PAIRS, desc="Comparisons", unit="pair"):
        compare_pair(label_a, indexes[label_a], label_b, indexes[label_b], OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
