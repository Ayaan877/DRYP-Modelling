import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

crop_curves = pd.read_csv("cropwise_Kc_curves_v2.0.csv")
time = crop_curves["Time"]
max_time = int(time.max())
year_length = 365
month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
season_bands = {
	"Kharif": (151, 334, "#f6cf71"),
	"Rabi": (334, 365, "#66c5cc"),
	"Summer": (120, 151, "#f89c74"),
}

figure, axis = plt.subplots(figsize=(12, 4))

for year_start in range(0, max_time + 1, year_length):
	for season_name, (band_start, band_end, band_color) in season_bands.items():
		if season_name == "Rabi":
			band_ranges = [
				(year_start + band_start, year_start + band_end),
				(year_start, year_start + 120),
			]
		else:
			band_ranges = [(year_start + band_start, year_start + band_end)]

		for band_start_value, band_end_value in band_ranges:
			if band_start_value > max_time:
				continue

			axis.axvspan(
				band_start_value,
				min(band_end_value, max_time),
				color=band_color,
				alpha=0.4,
				zorder=-2,
			)

axis.plot(
	time,
	crop_curves["Onion"],
	label="Onion",
color="C5",
linewidth=2,
)

for year_start in range(0, max_time + 1, year_length):
	axis.axvline(
		year_start,
		color="black",
		linewidth=1.2,
		alpha=0.55,
		zorder=0,
	)

	for month_start in month_starts[1:]:
		month_day = year_start + month_start
		if month_day <= max_time:
			axis.axvline(
				month_day,
				color="gray",
				linewidth=0.6,
				alpha=0.35,
				zorder=0,
			)

axis.set_xlabel("Days (starting from January 1)")
axis.set_ylabel("Kc Value")
axis.set_title("Onion Kc Curve")
axis.grid(True, which="both", axis="y")

season_handles = [
	Patch(facecolor=band_color, edgecolor="none", alpha=0.4, label=season_name)
	for season_name, (_, _, band_color) in season_bands.items()
]
curve_handles, curve_labels = axis.get_legend_handles_labels()
axis.legend(
	handles=curve_handles + season_handles,
	labels=curve_labels + [handle.get_label() for handle in season_handles],
	loc="center left",
	framealpha=0.5,
	bbox_to_anchor=(1.02, 0.5),
)
figure.subplots_adjust(right=0.75)
plt.show()
