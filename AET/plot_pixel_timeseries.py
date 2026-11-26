import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import watershed_operations

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

# Names: "<data> <source> <timescale> <resolution> [gapfilling] [version]"
# data: AET, ETf, Irr, GW, PET, refET
# source: bhuvan, ssebop, ssebop_era5, rainfed, bhuvan_era5, converted, agera5, era5
# timescale: hourly, daily, dekadal
# resolution: 30m, 250m, 750m, 1000m, 5500m, 10km, native
# gapfilling: temporal, spatiotemporal (ETf only), redistributed (GW only)
# version: v1, v2 (hourly and gapfilled datasets only)
# Combine with + - * /, e.g. "AET ssebop daily 30m/refET era5 daily 10km"
# Select one variable to show pixel traces.
VARIABLES = ["Irr ssebop_era5 hourly 250m v2"]
YEARS = list(range(2015, 2026))
N_COLUMNS = 3
SAVE_PLOT = False
SHOW_PIXEL_SERIES = False

PIXEL_ALPHA = 0.1
PIXEL_LINEWIDTH = 0.35
MAX_PIXEL_SERIES = 5000
MEAN_LINEWIDTH = 2
MEAN_LINE_STYLE = "-"  # e.g. "o", "o-", "--", "-o", ":"
SD_ALPHA = 0.5
SINGLE_MEAN_COLOR = "crimson"  # single dataset: contrasts with the pixel traces
SINGLE_SD_COLOR = "gold"
SUBPLOT_SIZE = (5, 2.5)  # width, height of each panel
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def load_series(variable, year):
	"""Return dates and a (time, row, column) cube for a dataset or expression."""
	series = watershed_operations.resolve_request(variable, year)
	return series.dates, series.values, series.units


def parse_line_style(style):
	"""Split a format string such as "o-" into (linestyle, marker)."""
	linestyle = "None"
	for candidate in ("--", "-.", ":", "-"):
		if candidate in style:
			linestyle = candidate
			style = style.replace(candidate, "", 1)
			break
	return linestyle, style or "None"


def series_colors(color, show_pixels):
	"""Return (mean, sd) colours; pixel mode uses contrasting colours."""
	if show_pixels:
		return SINGLE_MEAN_COLOR, SINGLE_SD_COLOR
	return color, color


def draw_variable(ax, variable, year, color, show_pixels):
	"""Draw a request's pixel traces when singular, then its mean and standard deviation."""
	dates, data, units = load_series(variable, year)
	series = data.reshape(len(dates), -1)
	series = series[:, np.isfinite(series).any(axis=0)]
	x = mdates.date2num(dates)
	if show_pixels and series.shape[1]:
		pixel_indexes = np.flatnonzero(np.isfinite(series).any(axis=0))
		if len(pixel_indexes) > MAX_PIXEL_SERIES:
			pixel_indexes = pixel_indexes[np.linspace(0, len(pixel_indexes) - 1, MAX_PIXEL_SERIES, dtype=int)]
		segments = []
		for pixel_index in pixel_indexes:
			valid = np.isfinite(series[:, pixel_index])
			if valid.sum() > 1:
				segments.append(np.column_stack((x[valid], series[valid, pixel_index])))
		if segments:
			ax.add_collection(LineCollection(
				segments, colors=color, alpha=PIXEL_ALPHA,
				linewidths=PIXEL_LINEWIDTH, zorder=1,
			), autolim=True)
			ax.autoscale_view()

	n_valid = np.isfinite(series).sum(axis=1)
	has_data = n_valid > 0
	mean = np.nansum(series[has_data], axis=1) / n_valid[has_data]
	std = np.zeros(len(mean))
	multiple_pixels = n_valid[has_data] > 1
	if multiple_pixels.any():
		std[multiple_pixels] = np.nanstd(series[has_data][multiple_pixels], axis=1, ddof=1)
	xs = x[has_data]
	mean_color, sd_color = series_colors(color, show_pixels)
	ax.fill_between(xs, mean - std, mean + std, color=sd_color, alpha=SD_ALPHA, zorder=2)
	linestyle, marker = parse_line_style(MEAN_LINE_STYLE)
	ax.plot(xs, mean, color=mean_color, linewidth=MEAN_LINEWIDTH, linestyle=linestyle, marker=marker, zorder=3)
	return series.shape[1], units


def plot_year(ax, year):
	"""Overlay the requested datasets or expressions on one set of axes."""
	show_pixels = SHOW_PIXEL_SERIES and len(VARIABLES) == 1
	results = [
		draw_variable(ax, variable, year, COLORS[index % len(COLORS)], show_pixels)
		for index, variable in enumerate(VARIABLES)
	]
	ax.set_xlabel("Date")
	date_locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
	ax.xaxis.set_major_locator(date_locator)
	ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(date_locator))
	units = {unit for _, unit in results}
	ax.set_ylabel(units.pop() if len(units) == 1 else "Value (mixed units)")
	counts = ", ".join(f"{count} px" for count, _ in results)
	ax.set_title(f"{year} ({counts})")


def plot_years(years):
	n_cols = min(N_COLUMNS, len(years))
	n_rows = -(-len(years) // n_cols)
	fig, axes = plt.subplots(
		n_rows, n_cols,
		figsize=(SUBPLOT_SIZE[0] * n_cols, SUBPLOT_SIZE[1] * n_rows),
		squeeze=False,
		sharey=True,
	)
	flat_axes = axes.ravel()
	for ax, year in zip(flat_axes, years):
		plot_year(ax, year)
	for ax in flat_axes[len(years):]:
		ax.set_visible(False)
	single = len(VARIABLES) == 1
	show_pixels = SHOW_PIXEL_SERIES and single
	handles = []
	linestyle, marker = parse_line_style(MEAN_LINE_STYLE)
	for index, variable in enumerate(VARIABLES):
		mean_color, sd_color = series_colors(COLORS[index % len(COLORS)], show_pixels)
		mean_label = "Mean" if single else variable
		sd_label = "±1 SD" if single else f"{variable} ±1 SD"
		handles.extend([
			Line2D([0], [0], color=mean_color, linewidth=MEAN_LINEWIDTH, linestyle=linestyle, marker=marker, label=mean_label),
			Patch(facecolor=sd_color, alpha=SD_ALPHA, label=sd_label),
		])
	if show_pixels:
		handles.append(Line2D([0], [0], color=COLORS[0], alpha=PIXEL_ALPHA, label="Pixel series (sample)"))
	title = fig.suptitle(f"{', '.join(VARIABLES)}", fontsize=12, va="top", y=0.995)
	title_bottom = title.get_window_extent().transformed(fig.transFigure.inverted()).y0
	legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, title_bottom - 0.005), ncol=1, frameon=False)
	inverse = fig.transFigure.inverted()
	legend_bottom = legend.get_window_extent().transformed(inverse).y0
	top = legend_bottom - 0.01
	fig.tight_layout(rect=(0, 0, 1, top))
	# tight_layout leaves a gap above the axes titles; a second pass corrects it
	gap = top - flat_axes[0].title.get_window_extent().transformed(inverse).y1
	fig.tight_layout(rect=(0, 0, 1, top + gap))
	if SAVE_PLOT:
		fig.savefig(Path(__file__).with_suffix(".png"), dpi=300, bbox_inches="tight")
		plt.close(fig)
	else:
		plt.show()


if __name__ == "__main__":
	plot_years(YEARS)
