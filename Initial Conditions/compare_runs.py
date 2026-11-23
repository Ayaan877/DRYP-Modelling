import csv
import re
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date
from tqdm import tqdm
from netcdf_to_geotiffs import (
    coordinate_variable_name,
    make_grid,
    resolve_crs,
    spatial_dimensions,
)

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
VARIABLES = ["wte"]
INPUT_DATASETS: list[tuple[str, Path]] = [
    ("run 1", SCRIPT_DIR / "OBF_hourly_grid_r1.nc"),
    ("run 2", SCRIPT_DIR / "OBF_hourly_grid_r2.nc"),
    ("run 3", SCRIPT_DIR / "OBF_hourly_grid_r3.nc")
]
COMPARISON_PAIRS: list[tuple[str, str]] = [("run 1", "run 2"), ("run 2", "run 3")]
OUTPUT_DIRECTORY = SCRIPT_DIR / "run comparison"
START_DATE: Optional[datetime] = None
END_DATE: Optional[datetime] = None
TIME_STEP = 1
TIME_VARIABLE = "time"

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------


def slugify(label: str) -> str:
    return re.sub(r"\W+", "_", label).strip("_").lower()


def index_dataset(dataset: Dataset) -> dict[datetime, int]:
    if TIME_VARIABLE not in dataset.variables:
        raise ValueError(f"Time variable {TIME_VARIABLE!r} not found in {dataset.filepath()}.")
    time_variable = dataset.variables[TIME_VARIABLE]
    if time_variable.ndim != 1 or not getattr(time_variable, "units", None):
        raise ValueError("The time variable must be one-dimensional and have units.")
    calendar = getattr(time_variable, "calendar", "standard")
    if calendar.lower() not in ("standard", "gregorian", "proleptic_gregorian"):
        raise ValueError(f"Unsupported calendar {calendar!r}; expected a Gregorian calendar.")
    timestamps = np.asarray(
        num2date(
            time_variable[:], time_variable.units, calendar=calendar,
            only_use_cftime_datetimes=False,
        ), dtype=object,
    ).reshape(-1)
    index: dict[datetime, int] = {}
    for position, value in enumerate(timestamps):
        timestamp = datetime(value.year, value.month, value.day, value.hour, value.minute, value.second, value.microsecond)
        if timestamp in index:
            raise ValueError(f"Duplicate timestamp {timestamp} in {dataset.filepath()}.")
        index[timestamp] = position
    return index


def select_timestamps(common: list[datetime]) -> list[datetime]:
    if TIME_STEP < 1:
        raise ValueError("TIME_STEP must be a positive integer.")
    selected = common
    if START_DATE is not None:
        selected = [t for t in selected if t.date() >= START_DATE.date()]
    if END_DATE is not None:
        selected = [t for t in selected if t.date() <= END_DATE.date()]
    return selected[::TIME_STEP]


def prepare_layer(dataset: Dataset, name: str) -> tuple[Any, Any, Any, tuple[int, int], bool, bool, tuple[int, int]]:
    if name not in dataset.variables:
        raise ValueError(f"Variable {name!r} not found in {dataset.filepath()}.")
    variable = dataset.variables[name]
    time_dimension = dataset.variables[TIME_VARIABLE].dimensions[0]
    if time_dimension not in variable.dimensions or variable.ndim != 3:
        raise ValueError(f"Variable {name!r} must have one time and two spatial dimensions.")
    y_dimension, x_dimension = spatial_dimensions(dataset, variable, time_dimension)
    x_name = coordinate_variable_name(dataset, x_dimension, None)
    y_name = coordinate_variable_name(dataset, y_dimension, None)
    x_coordinate = dataset.variables[x_name]
    y_coordinate = dataset.variables[y_name]
    x_values = np.asarray(x_coordinate[:], dtype=np.float64)
    y_values = np.asarray(y_coordinate[:], dtype=np.float64)
    transform, reverse_x, reverse_y = make_grid(x_values, y_values)
    crs = resolve_crs(dataset, variable, x_coordinate, y_coordinate)
    remaining = [dimension for dimension in variable.dimensions if dimension != time_dimension]
    transpose_order = (remaining.index(y_dimension), remaining.index(x_dimension))
    return variable, transform, crs, (len(y_values), len(x_values)), reverse_x, reverse_y, transpose_order


def read_field(layer: tuple[Any, Any, Any, tuple[int, int], bool, bool, tuple[int, int]], time_dimension: str, position: int) -> np.ndarray:
    variable, _, _, _, reverse_x, reverse_y, transpose_order = layer
    indexers = tuple(position if dimension == time_dimension else slice(None) for dimension in variable.dimensions)
    data = np.ma.filled(
        np.ma.transpose(np.ma.asarray(variable[indexers]), axes=transpose_order).astype(np.float64),
        np.nan,
    )
    if reverse_y:
        data = data[::-1, :]
    if reverse_x:
        data = data[:, ::-1]
    data[~np.isfinite(data)] = np.nan
    return data


def plot_aggregate(
    variable: str,
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
    ax.set_ylabel(f"Spatial mean {variable}")
    ax.set_title(f"{variable}: {label_a} vs {label_b} (aggregate over valid pixels)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_difference_series(
    variable: str,
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
    ax_msd.set_title(f"{variable}: {label_a} minus {label_b}")
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
    variable: str,
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

    finite_bias = np.abs(bias_map[np.isfinite(bias_map)])
    limit = float(finite_bias.max()) if finite_bias.size else 0.0
    image_bias = ax_bias.imshow(
        bias_map, extent=extent, origin="upper", cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    ax_bias.set_title(f"Time-mean difference per pixel ({label_a} - {label_b})")
    fig.colorbar(image_bias, ax=ax_bias, shrink=0.8)

    fig.suptitle(f"{variable}: pixel-level comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare_pair(
    variable: str,
    label_a: str,
    dataset_a: Dataset,
    index_a: dict[datetime, int],
    label_b: str,
    dataset_b: Dataset,
    index_b: dict[datetime, int],
    output_directory: Path,
) -> None:
    common = select_timestamps(sorted(set(index_a) & set(index_b)))
    if not common:
        raise ValueError(f"No common timestamps for {label_a!r} and {label_b!r} in the selected range.")
    print(f"{variable}: {label_a} vs {label_b}: {len(common)} common timestamps.")

    layer_a = prepare_layer(dataset_a, variable)
    layer_b = prepare_layer(dataset_b, variable)
    _, transform_a, crs_a, shape_a, _, _, _ = layer_a
    _, transform_b, crs_b, shape_b, _, _, _ = layer_b
    if shape_a != shape_b or transform_a != transform_b or crs_a != crs_b:
        raise ValueError(f"{label_a!r} and {label_b!r} are not on the same grid; resample before comparing.")
    output_directory.mkdir(parents=True, exist_ok=True)
    pair_prefix = f"{slugify(variable)}_{slugify(label_a)}vs{slugify(label_b)}"
    time_dimension_a = dataset_a.variables[TIME_VARIABLE].dimensions[0]
    time_dimension_b = dataset_b.variables[TIME_VARIABLE].dimensions[0]

    count = np.zeros(shape_a)
    sum_diff = np.zeros(shape_a)
    sum_sq = np.zeros(shape_a)
    rows: list[tuple[datetime, float, float, float, float, float, int]] = []

    for timestamp in tqdm(common, desc=f"{variable}: {label_a} vs {label_b}", unit="timestep"):
        a = read_field(layer_a, time_dimension_a, index_a[timestamp])
        b = read_field(layer_b, time_dimension_b, index_b[timestamp])
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
        variable,
        times,
        mean_a,
        mean_b,
        label_a,
        label_b,
        output_directory / f"{pair_prefix}_aggregate_timeseries.png",
    )
    plot_difference_series(
        variable,
        times,
        msd_series,
        mean_diff,
        label_a,
        label_b,
        output_directory / f"{pair_prefix}_difference_timeseries.png",
    )
    plot_pixel_maps(
        variable,
        msd_map,
        bias_map,
        transform_a,
        label_a,
        label_b,
        output_directory / f"{pair_prefix}_pixel_maps.png",
    )

    finite_msd = msd_series[np.isfinite(msd_series)]
    print(f"Mean MSD over the period: {finite_msd.mean():.4e}" if finite_msd.size else "No valid pixels over the selected period.")
    print(f"Outputs written to {output_directory}")


# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------


def main() -> None:
    with ExitStack() as stack:
        datasets = {label: stack.enter_context(Dataset(path, "r")) for label, path in INPUT_DATASETS}
        indexes = {label: index_dataset(dataset) for label, dataset in datasets.items()}
        for variable in VARIABLES:
            for label_a, label_b in COMPARISON_PAIRS:
                compare_pair(
                    variable, label_a, datasets[label_a], indexes[label_a],
                    label_b, datasets[label_b], indexes[label_b], OUTPUT_DIRECTORY,
                )


if __name__ == "__main__":
    main()
