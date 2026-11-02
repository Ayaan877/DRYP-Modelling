import json
import os
import re
import time
import zipfile
from datetime import date, datetime, timedelta
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------\
RESOLUTION = "750m"
START_DATE = "2020-01-01"
END_DATE = "2020-03-31"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_BASE_DIR = SCRIPT_DIR / "Bhuvan AET Raw"
OVERWRITE_EXISTING = True
RETRIES_PER_REQUEST = 2
REQUEST_TIMEOUT_SECONDS = 900
RETRY_DELAY_SECONDS = 10
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
USER_AGENT = "BhuvanAETDownloader/1.0"
DOWNLOAD_PURPOSE = "Crop water requirement estimation"

DB_SERVER = "https://bhuvan.nrsc.gov.in/nhpds"
FILE_SERVER = "https://bhuvan.nrsc.gov.in/nhpfs"

DAILY_INVENTORY_ENDPOINTS = {
	"750m": f"{DB_SERVER}/get-avail-dates-et-evapotranspiration750",
	"5500m": f"{DB_SERVER}/get-avail-dates-et-evapotranspiration",
}

DOWNLOAD_PRODUCT_CODES = {
	"750m": "EB750",
	"5500m": "EB",
}

FILENAME_PATTERNS = {
	"750m": re.compile(r"^AET_SNPP_N_EB_DY_(\d{8})\.tif$"),
	"5500m": re.compile(r"^NHP_ISRO_AET_EB_DY_(\d{8})\.tif$"),
}

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def parse_dates(value, name):
	try:
		return datetime.strptime(value, "%Y-%m-%d").date()
	except ValueError as exc:
		raise ValueError(f"{name} must use YYYY-MM-DD format: {value!r}") from exc


def validate_configuration():
	if RESOLUTION not in DAILY_INVENTORY_ENDPOINTS:
		raise ValueError('RESOLUTION must be set to "750m" or "5500m"')

	start_date = parse_dates(START_DATE, "START_DATE")
	end_date = parse_dates(END_DATE, "END_DATE")
	if start_date > end_date:
		raise ValueError("START_DATE must be on or before END_DATE")

	if RETRIES_PER_REQUEST < 0:
		raise ValueError("RETRIES_PER_REQUEST must be at least 0")
	if REQUEST_TIMEOUT_SECONDS <= 0:
		raise ValueError("REQUEST_TIMEOUT_SECONDS must be greater than 0")

	return start_date, end_date


def post_json(url, payload):
	request = Request(
		url,
		data=json.dumps(payload).encode("utf-8"),
		headers={
			"Content-Type": "application/json",
			"Accept": "application/json",
			"User-Agent": USER_AGENT,
		},
		method="POST",
	)
	with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
		return json.loads(response.read().decode("utf-8"))


def request_json(url):
	return post_json(url, {})


def fetch_available_dates(resolution):
	url = DAILY_INVENTORY_ENDPOINTS[resolution]
	last_error = None

	for attempt in range(RETRIES_PER_REQUEST + 1):
		try:
			items = request_json(url)
			if not isinstance(items, list):
				raise RuntimeError(f"Unexpected inventory response from {url}")

			pattern = FILENAME_PATTERNS[resolution]
			available = {}
			for item in items:
				filename = item.get("filename") if isinstance(item, dict) else None
				if not filename:
					continue
				match = pattern.fullmatch(filename)
				if match:
					available[datetime.strptime(match.group(1), "%Y%m%d").date()] = filename
			return available
		except (HTTPError, URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
			last_error = exc
			if attempt < RETRIES_PER_REQUEST:
				time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))

	raise RuntimeError(f"Could not retrieve available {resolution} AET dates: {last_error}")


def iter_dates(start_date, end_date):
	current = start_date
	while current <= end_date:
		yield current
		current += timedelta(days=1)


def iter_download_ranges(start_date, end_date):
	current = start_date
	while current <= end_date:
		if current.month <= 3:
			quarter_end = date(current.year, 3, 31)
		elif current.month <= 6:
			quarter_end = date(current.year, 6, 30)
		elif current.month <= 9:
			quarter_end = date(current.year, 9, 30)
		else:
			quarter_end = date(current.year, 12, 31)

		range_end = min(quarter_end, end_date)
		yield current, range_end
		current = range_end + timedelta(days=1)


def register_download():
	result = post_json(
		f"{DB_SERVER}/save-user-download-data",
		{
			"theme": "ET",
			# This is the product value used by the portal's own daily AET flow.
			"product": "EB",
			"subproduct": None,
			"user": None,
			"purpose": DOWNLOAD_PURPOSE,
		},
	)
	if not isinstance(result, dict) or not result.get("requestid"):
		raise RuntimeError(
			"Portal did not return a download request ID. "
			"Response: "
			f"{result!r}"
		)
	print(f"  Download request registered: {result['requestid']}", flush=True)


def download_range(
	resolution,
	start_date,
	end_date,
	destination,
	expected_tif_names,
):
	payload = json.dumps(
		{
			"theme": "ET",
			"product": DOWNLOAD_PRODUCT_CODES[resolution],
			# The portal sends selected dates to the file server as YYYYMMDD.
			"d1": start_date.strftime("%Y%m%d"),
			"d2": end_date.strftime("%Y%m%d"),
			"user": None,
			"date_data": None,
		}
	).encode("utf-8")
	request = Request(
		f"{FILE_SERVER}/download_multiple",
		data=payload,
		headers={
			"Content-Type": "application/json",
			"Accept": "application/zip, application/octet-stream",
			"User-Agent": USER_AGENT,
		},
		method="POST",
	)
	temp_path = destination.with_suffix(destination.suffix + ".part")
	register_download()

	for attempt in range(RETRIES_PER_REQUEST + 1):
		try:
			print(
				f"  Downloading AET {resolution}, {start_date} to {end_date} "
				f"(attempt {attempt + 1}/{RETRIES_PER_REQUEST + 1})",
				flush=True,
			)
			with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
				downloaded_bytes = 0
				started_at = time.monotonic()
				last_update = started_at
				with temp_path.open("wb") as output:
					while True:
						chunk = response.read(DOWNLOAD_CHUNK_SIZE)
						if not chunk:
							break
						output.write(chunk)
						downloaded_bytes += len(chunk)
						now = time.monotonic()
						if now - last_update >= 0.25:
							elapsed = now - started_at
							speed = downloaded_bytes / 1024 / 1024 / elapsed
							elapsed_text = time.strftime(
								"%H:%M:%S", time.gmtime(elapsed)
							)
							progress = (
								f"{start_date} to {end_date} "
								f"[{downloaded_bytes / 1024 / 1024:.1f}MB, "
								f"{speed:.1f}MB/s, {elapsed_text}]"
							)
							print(f"\r{progress}", end="", flush=True)
							last_update = now

				elapsed = time.monotonic() - started_at
				speed = downloaded_bytes / 1024 / 1024 / elapsed
				elapsed_text = time.strftime("%H:%M:%S", time.gmtime(elapsed))
				progress = (
					f"{start_date} to {end_date} "
					f"[{downloaded_bytes / 1024 / 1024:.1f}MB, "
					f"{speed:.1f}MB/s, {elapsed_text}]"
				)
				print(f"\r{progress}", flush=True)

			with zipfile.ZipFile(temp_path) as archive:
				if archive.testzip() is not None:
					raise zipfile.BadZipFile("ZIP contains a corrupt member")
				archive_tif_names = {
					Path(name).name
					for name in archive.namelist()
					if name.lower().endswith(".tif")
				}
				if not archive_tif_names:
					raise zipfile.BadZipFile(
						"ZIP contains no TIFF data files; it may contain metadata only"
					)
				missing_tif_names = expected_tif_names - archive_tif_names
				if missing_tif_names:
					raise zipfile.BadZipFile(
						f"ZIP is missing {len(missing_tif_names)} expected TIFF "
						"data file(s)"
					)
				saved_tif_count = len(expected_tif_names & archive_tif_names)
			os.replace(temp_path, destination)
			requested_days = (end_date - start_date).days + 1
			print(
				f"  ZIP validation passed "
				f"({saved_tif_count}/{requested_days} dates saved).",
				flush=True,
			)
			return
		except (
			HTTPError,
			URLError,
			TimeoutError,
			OSError,
			IncompleteRead,
			zipfile.BadZipFile,
			RuntimeError,
		) as exc:
			if temp_path.exists():
				temp_path.unlink()
			print(flush=True)
			if attempt == RETRIES_PER_REQUEST:
				raise RuntimeError(
					f"Download failed for {start_date} to {end_date}: {exc}"
				) from exc
			print(
				f"  Download failed: {exc}. Retrying in "
				f"{RETRY_DELAY_SECONDS * (attempt + 1)} seconds...",
				flush=True,
			)
			time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))

# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	start_date, end_date = validate_configuration()
	available_dates = fetch_available_dates(RESOLUTION)
	requested_dates = list(iter_dates(start_date, end_date))
	dates_to_download = [
		day for day in requested_dates if day in available_dates
	]
	missing_dates = [
		day for day in requested_dates if day not in available_dates
	]

	output_dir = OUTPUT_BASE_DIR / f"AET {RESOLUTION}"
	output_dir.mkdir(parents=True, exist_ok=True)
	prefix = "NHP_NRSC_ET_EB" if RESOLUTION == "5500m" else "AET_SNPP_N_EB_DY"
	download_ranges = [
		(range_start, range_end)
		for range_start, range_end in iter_download_ranges(start_date, end_date)
		if any(
			range_start <= day <= range_end
			for day in dates_to_download
		)
	]

	print(
		f"Found {len(dates_to_download)}/{len(requested_dates)} available dates in the requested "
		f"{start_date} to {end_date} range."
	)
	if missing_dates:
		print(f"Skipping {len(missing_dates)} dates with no available data.")
	print(f"Downloading {len(download_ranges)} ZIP file(s).")

	failures = []
	for range_start, range_end in download_ranges:
		archive_stem = (
			f"{prefix}_{range_start.strftime('%Y%m%d')}_"
			f"{range_end.strftime('%Y%m%d')}"
		)
		archive_path = output_dir / f"{archive_stem}.zip"
		expected_tif_names = {
			available_dates[day]
			for day in dates_to_download
			if range_start <= day <= range_end
		}
		print(f"Saving raw ZIP to: .../AET {RESOLUTION}/{archive_path.name}")

		try:
			if archive_path.exists() and not OVERWRITE_EXISTING:
				print(f"SKIP existing archive: {archive_path.name}")
			else:
				download_range(
					RESOLUTION,
					range_start,
					range_end,
					archive_path,
					expected_tif_names,
				)
		except (RuntimeError, OSError, zipfile.BadZipFile) as exc:
			failures.append((range_start, range_end, str(exc)))
			print(f"FAIL {exc}")

	print(f"Output directory: {output_dir}")
	if missing_dates:
		missing_fraction = len(missing_dates) / len(requested_dates)
		print(
			f"Unavailable dates: {len(missing_dates)}/{len(requested_dates)} "
			f"({missing_fraction:.1%})"
		)
	if failures:
		print("Failed downloads:")
		for failed_start, failed_end, error in failures:
			print(f"{failed_start} to {failed_end}: {error}")
		raise RuntimeError(f"{len(failures)} download(s) failed")
