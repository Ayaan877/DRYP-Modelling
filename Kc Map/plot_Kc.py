import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

# Configuration
VERSION = "v2.0"
TIME_ORIGIN = "2015-01-01"
crop_curves = pd.read_csv(f"cropwise_Kc_curves_{VERSION}.csv")
class_curves = pd.read_csv(f"classwise_Kc_curves_{VERSION}.csv")
crop_metadata = pd.read_csv(f"cropwise_Kc.csv")

crop_classes = crop_metadata.set_index("Crop_type")["Class"]
colors = ["C1", "C4", "C2", "C5", "C3", "C0"]
class_values = list(range(2, 8))
class_colors = {
		crop_class: colors[index % len(colors)]
		for index, crop_class in enumerate(class_values)}

max_time = max(crop_curves["Time"].max(), class_curves["Time"].max())
time_origin = pd.Timestamp(TIME_ORIGIN)
end_date = time_origin + pd.Timedelta(days=int(max_time))
season_colors = {
	"Kharif": "#f6cf71",
	"Rabi": "#66c5cc",
	"Summer": "#f89c74",
}


def day_offset(date):
	return (pd.Timestamp(date) - time_origin).days


def date_ranges(start_month, start_day, end_month, end_day):
	"""Return yearly calendar-season ranges clipped to the plotted period."""
	ranges = []
	for year in range(time_origin.year - 1, end_date.year + 1):
		start = pd.Timestamp(year=year, month=start_month, day=start_day)
		end_year = (end_month, end_day) <= (start_month, start_day)
		end = pd.Timestamp(
			year=year + int(end_year), month=end_month, day=end_day)
		if end > time_origin and start < end_date + pd.Timedelta(days=1):
			ranges.append((max(day_offset(start), 0), min(day_offset(end), max_time)))
	return ranges


season_ranges = {
	"Kharif": date_ranges(6, 1, 12, 1),
	"Rabi": date_ranges(12, 1, 5, 1),
	"Summer": date_ranges(5, 1, 6, 1),
}

year_starts = [
	day_offset(pd.Timestamp(year=year, month=1, day=1))
	for year in range(time_origin.year, end_date.year + 1)
	if 0 <= day_offset(pd.Timestamp(year=year, month=1, day=1)) <= max_time
]
month_starts = [
	day_offset(date)
	for date in pd.date_range(
		time_origin.replace(day=1), end_date + pd.offsets.MonthBegin(1), freq="MS"
	)
	if 0 <= day_offset(date) <= max_time
]
figure, axes = plt.subplots(
	3,
	2,
	figsize=(10, 6),
	sharex=True,
	squeeze=False,
)
axes = axes.ravel()

line_styles = ["--", ":", "-."]

for axis, crop_class in zip(axes, class_values):
	for season_name, ranges in season_ranges.items():
		for band_start, band_end in ranges:
			axis.axvspan(
				band_start,
				band_end,
				color=season_colors[season_name],
				alpha=0.4,
				label="_nolegend_",
				zorder=-2,
			)

	crop_names = [
		crop_name
		for crop_name in crop_curves.columns[1:]
		if crop_name in crop_classes
		and crop_classes[crop_name] == crop_class
	]

	for crop_index, crop_name in enumerate(crop_names):
		axis.plot(
			crop_curves["Time"],
			crop_curves[crop_name],
			label=crop_name,
			color=class_colors[crop_class],
			linestyle=line_styles[crop_index % len(line_styles)],
			linewidth=1.5,
			alpha=0.75,
		)

	if crop_class == 7:
		axis.plot(
			crop_curves["Time"],
			crop_curves["Finger millet"],
			label="Finger millet",
			color=class_colors[crop_class],
			linestyle="--",
			linewidth=1.5,
			alpha=0.75,
		)

		axis.plot(
			class_curves["Time"],
			class_curves["Class_4"],
			label="Class_4",
			color=class_colors[crop_class],
			linestyle=":",
			linewidth=1.5,
			alpha=0.75,
		)

	class_name = f"Class_{int(crop_class)}"
	if class_name in class_curves:
		axis.plot(
			class_curves["Time"],
			class_curves[class_name],
			label=f"{class_name} avg.",
			color=class_colors[crop_class],
			linewidth=2,
		)

	axis.set_title(f"Kc Curves (Class {int(crop_class)})")
	axis.set_ylabel("Kc Value")
	axis.grid(True, which="both", axis="y")

	for year_start in year_starts:
		axis.axvline(
			year_start,
			color="black",
			linewidth=1.2,
			alpha=0.55,
			zorder=0,
		)

	for month_start in month_starts:
		axis.axvline(
			month_start,
			color="gray",
			linewidth=0.6,
			alpha=0.35,
			zorder=0,
		)

	axis.legend(loc="upper right", framealpha=0.5, fontsize="small")

for axis in axes[len(class_values):]:
	axis.set_visible(False)

axes[4].set_xlabel(f"Days since {time_origin.date()}")
axes[5].set_xlabel(f"Days since {time_origin.date()}")

season_handles = [
	Patch(facecolor=band_color, edgecolor="none", alpha=0.4, label=season_name)
	for season_name, band_color in season_colors.items()
]
figure.legend(
	handles=season_handles,
	loc="upper center",
	ncol=3,
	bbox_to_anchor=(0.5, 0.995),
	title="Seasons",
	fontsize="medium",
)

plt.show()