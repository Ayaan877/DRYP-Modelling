import csv
import re
from pathlib import Path
import pdfplumber

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PDF_PATH = SCRIPT_DIR / "Raw Data" / "KWRIS_GW_Level_Report.pdf"
OUT_PATH = SCRIPT_DIR / "Raw Data" / "KWRIS_Table.csv"
PAGE_INDICES = range(13, 30)

MONTHS = {
	"January": 1, 
	"February": 2, 
	"March": 3, 
	"April": 4,
	"May": 5, 
	"June": 6, 
	"July": 7, 
	"August": 8,
	"September": 9, 
	"October": 10, 
	"November": 11, 
	"December": 12,
}
ROW_RE = re.compile(
	r"^(\d+)\s+(\d{6})\s+(.+?)\s+(\d{4})\s+(" + "|".join(MONTHS) + r")\s+(\d{1,2})\s+(\S+)$"
)
HEADERS = [
	"source", 
	"latitude", 
	"longitude", 
	"well ID", 
	"village", 
	"block", 
	"well type",
	"status", 
	"well depth", 
	"well depth units", 
	"water level", 
	"water level units", 
	"collection date", 
	"collection time",
]
SOURCE = "KWRIS"
BLOCK = "Sidlaghatta"
DEPTH_UNITS = "mbgl"
LEVEL_UNITS = "mbgl"

# well ID -> (latitude, longitude, well type, well depth) from Table-1
WELL_INFO = {
	"131013": ("13.486", "77.8648", "Bore Well", "194"),
	"131002": ("13.5582", "77.9045", "Bore Well", "65"),
	"131004": ("13.49", "77.9827", "Bore Well", "73.7"),
	"131009": ("13.3323", "77.9998", "Bore Well", "70"),
	"131012": ("13.3937", "77.8655", "Bore Well", "65"),
	"131016": ("13.6263", "77.8759", "Bore Well", "162"),
	"131008": ("13.2226", "77.9298", "Bore Well", "49"),
	"131015": ("13.2622", "77.8467", "Bore Well", "215"),
	"131014": ("13.5903", "77.9267", "Bore Well", "243"),
}

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def parse_page(text):
	rows = []
	unmatched = []
	for line in text.splitlines():
		line = line.strip()
		if not line:
			continue
		m = ROW_RE.match(line)
		if not m:
			if re.match(r"^\d+\s+\d{6}\s", line):
				unmatched.append(line)
			continue
		_, well_id, village, year, month, day, level = m.groups()
		lat, lon, well_type, depth = WELL_INFO[well_id]
		reading_date = f"{int(year):04d}-{MONTHS[month]:02d}-{int(day):02d}"
		if re.fullmatch(r"\d+(\.\d+)?", level):
			status, units = "wet", LEVEL_UNITS
		else:
			status, level, units = "dry", "", ""
		rows.append([SOURCE, lat, lon, well_id, village, BLOCK, well_type, status,
			depth, DEPTH_UNITS, level, units, reading_date, ""])
	return rows, unmatched


def extract_rows(pdf_path):
	rows = []
	unmatched = []
	with pdfplumber.open(pdf_path) as pdf:
		for idx in PAGE_INDICES:
			page_rows, page_unmatched = parse_page(pdf.pages[idx].extract_text() or "")
			rows.extend(page_rows)
			unmatched.extend(page_unmatched)
	return rows, unmatched


def write_csv(rows, out_path):
	with open(out_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(HEADERS)
		writer.writerows(rows)


# ---------------------------------------------------------------------------
# Run Script
# ---------------------------------------------------------------------------

def main():
	rows, unmatched = extract_rows(PDF_PATH)
	write_csv(rows, OUT_PATH)
	print(f"rows written: {len(rows)}")
	print(f"unmatched data lines: {len(unmatched)}")
	for line in unmatched[:20]:
		print(f"  {line}")

if __name__ == "__main__":
	main()
