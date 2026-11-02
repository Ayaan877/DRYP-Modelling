import concurrent.futures
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_URL = (
	"https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/"
	"dekadal/etav61/downloads/dekadal/"
)
OUTPUT_DIR = SCRIPT_DIR / "ssebop_dekadal_zip"

START_YEAR = 2015
END_YEAR = 2025

LIST_ONLY = False
OVERWRITE_EXISTING = True
MAX_WORKERS = 3
RETRIES_PER_FILE = 3

INDEX_TIMEOUT_SECONDS = 60
DOWNLOAD_TIMEOUT_SECONDS = 180
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
USER_AGENT = "Mozilla/5.0 (compatible; SSEBopDekadalDownloader/1.0)"

# Matches: dYYYYMMK.zip where MM=01..12 and K=1/2/3
ZIP_NAME_PATTERN = re.compile(r"\bd(?P<year>\d{4})(?P<month>\d{2})(?P<dekad>[1-3])\.zip\b")


# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def validate_configuration():
	if START_YEAR > END_YEAR:
		raise ValueError("START_YEAR must be less than or equal to END_YEAR")
	if MAX_WORKERS < 1:
		raise ValueError("MAX_WORKERS must be at least 1")
	if RETRIES_PER_FILE < 0:
		raise ValueError("RETRIES_PER_FILE must be >= 0")


def fetch_index_html(base_url):
	request = Request(base_url, headers={"User-Agent": USER_AGENT})
	with urlopen(request, timeout=INDEX_TIMEOUT_SECONDS) as response:
		return response.read().decode("utf-8", errors="replace")


def extract_zip_names(html, start_year, end_year):
	valid = set()
	for match in ZIP_NAME_PATTERN.finditer(html):
		year = int(match.group("year"))
		month = int(match.group("month"))
		dekad = int(match.group("dekad"))
		if start_year <= year <= end_year and 1 <= month <= 12 and 1 <= dekad <= 3:
			valid.add(match.group(0))
	return sorted(valid)


def download_single_file(file_url, destination, overwrite, retries):
	if destination.exists() and not overwrite and destination.stat().st_size > 0:
		return f"SKIP  {destination.name} (already exists)"

	temp_path = destination.with_suffix(destination.suffix + ".part")
	attempt = 0
	while True:
		try:
			request = Request(file_url, headers={"User-Agent": USER_AGENT})
			with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, open(temp_path, "wb") as handle:
				while True:
					chunk = response.read(DOWNLOAD_CHUNK_SIZE)
					if not chunk:
						break
					handle.write(chunk)
			os.replace(temp_path, destination)
			return f"DONE  {destination.name}"
		except (HTTPError, URLError, TimeoutError, OSError) as exc:
			attempt += 1
			if temp_path.exists():
				try:
					temp_path.unlink()
				except OSError:
					pass
			if attempt > retries:
				return f"FAIL  {destination.name} ({exc})"
			time.sleep(min(20, 2**attempt))


def download_all_files(base_url, output_dir, zip_names, overwrite, workers, retries):
	output_dir.mkdir(parents=True, exist_ok=True)
	total = len(zip_names)
	completed = 0
	failures = 0

	with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
		future_map = {}
		for file_name in zip_names:
			file_url = urljoin(base_url, file_name)
			destination = output_dir / file_name
			future = executor.submit(
				download_single_file,
				file_url,
				destination,
				overwrite,
				retries,
			)
			future_map[future] = file_name

		for future in concurrent.futures.as_completed(future_map):
			completed += 1
			result = future.result()
			if result.startswith("FAIL"):
				failures += 1
			print(f"[{completed}/{total}] {result}")

	return failures


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

def run_ssebop_scraper():
	validate_configuration()
	print(f"Reading index: {BASE_URL}")
	html = fetch_index_html(BASE_URL)
	zip_names = extract_zip_names(html, START_YEAR, END_YEAR)

	if not zip_names:
		raise RuntimeError(f"No ZIP files found for years {START_YEAR}-{END_YEAR}")

	print(f"Found {len(zip_names)} ZIP files for {START_YEAR}-{END_YEAR}.")

	if LIST_ONLY:
		for name in zip_names:
			print(name)
		return

	print(f"Downloading to: {OUTPUT_DIR}")
	failures = download_all_files(
		BASE_URL,
		OUTPUT_DIR,
		zip_names,
		OVERWRITE_EXISTING,
		MAX_WORKERS,
		RETRIES_PER_FILE,
	)

	if failures:
		raise RuntimeError(f"Finished with {failures} failed downloads")

	print("All downloads completed successfully.")


if __name__ == "__main__":
	run_ssebop_scraper()
