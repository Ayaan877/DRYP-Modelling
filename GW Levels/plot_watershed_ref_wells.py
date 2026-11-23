from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "watershed_ref_well_data.csv"
DRYP_DATA_PATH = SCRIPT_DIR / "DRYP_ref_well_data.csv"
PLOT_PATH = SCRIPT_DIR / "watershed_ref_wells.png"
SAVE_PLOT = True
PLOT_OVERLAP_ONLY = True

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def load_data(path: Path):
	"""Read the well data with numeric levels and parsed dates, dropping unusable rows."""
	frame = pd.read_csv(path, dtype={"well ID": str}, keep_default_na=False)
	frame["water level"] = -pd.to_numeric(frame["water level"], errors="coerce")
	frame["collection date"] = pd.to_datetime(frame["collection date"], errors="coerce")
	return frame.dropna(subset=["water level", "collection date"])


def plot_wells(frame: pd.DataFrame, dryp: pd.DataFrame):
	"""Plot observed and DRYP water level time series per well as stacked subplots."""
	well_ids = list(dict.fromkeys(frame["well ID"]))
	fig, axes = plt.subplots(len(well_ids), 1, figsize=(8, min(10, 2 * len(well_ids) + 1)), squeeze=False)
	for ax, well_id in zip(axes[:, 0], well_ids):
		well = frame[frame["well ID"] == well_id].sort_values("collection date")
		village = well["village"].iloc[0]
		well_depth = well["well depth"].iloc[0]
		dryp_well = dryp[dryp["well ID"] == well_id].sort_values("collection date")
		if PLOT_OVERLAP_ONLY:
			start = max(well["collection date"].min(), dryp_well["collection date"].min())
			end = min(well["collection date"].max(), dryp_well["collection date"].max())
			well = well[well["collection date"].between(start, end)]
			dryp_well = dryp_well[dryp_well["collection date"].between(start, end)]
		ax.plot(dryp_well["collection date"], dryp_well["water level"], "-", linewidth=0.8, color="tab:orange", label="DRYP")
		ax.plot(well["collection date"], well["water level"], "o-", markersize=3, linewidth=1, color="tab:blue", label="Observed")
		ax.legend(loc="best")
		ax.set_title(f"{well_id} - {village} ({int(well_depth)}m)")
		ax.set_ylabel("Water Level Depth (m)")
		ax.invert_yaxis()
		ax.set_xlabel("Date")
		ax.xaxis.set_major_locator(mdates.YearLocator())
		ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
		ax.tick_params(axis="x", labelbottom=True)
		ax.xaxis.set_minor_locator(mdates.MonthLocator())
		ax.grid(True, which="major", alpha=0.8)
		ax.grid(True, which="minor", axis="x", alpha=0.3)
	fig.tight_layout()
	return fig


def main():
	frame = load_data(DATA_PATH)
	dryp = load_data(DRYP_DATA_PATH)
	fig = plot_wells(frame, dryp)
	if SAVE_PLOT:
		fig.savefig(PLOT_PATH, dpi=150)
		print(f"Saved plot to {PLOT_PATH.name}")
	else:
		plt.show()


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	main()
