import re
import sys
import time
from pathlib import Path
import requests

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
BBOX = [77.5, 13.0, 78.5, 14.0]
SENSORS = ("LE07", "LC08", "LC09")
TIERS = ("T1", "T2")
OUTPUT_FILE = Path(__file__).resolve().parent / "landsat_scene_ids.txt"

STAC_URL = "https://landsatlook.usgs.gov/stac-server/search"
SCENE_RE = re.compile(r"^L[CET][0-9]{2}_L2SP_\d{6}_\d{8}_\d{8}_02_T[12]$")


def stac_post(body, tries=6):
    for attempt in range(1, tries + 1):
        try:
            resp = requests.post(STAC_URL, json=body, timeout=120)
            if resp.status_code < 500:
                resp.raise_for_status()
                return resp.json()
            err = f"HTTP {resp.status_code}"
        except (requests.ConnectionError, requests.Timeout) as exc:
            err = str(exc)
        if attempt < tries:
            print(f"STAC search error ({err}); retrying in {5 * attempt}s")
            time.sleep(5 * attempt)
    sys.exit(f"STAC search failed after {tries} tries: {err}")


def save_scene_ids():
    body = {
        "collections": ["landsat-c2l2-sr"],
        "bbox": BBOX,
        "datetime": f"{START_DATE}T00:00:00Z/{END_DATE}T23:59:59Z",
        "limit": 100,
    }
    ids = set()
    while body:
        page = stac_post(body)
        for feat in page.get("features", []):
            sid = feat["id"].removesuffix("_SR")
            if SCENE_RE.match(sid) and sid[:4] in SENSORS and sid[-2:] in TIERS:
                ids.add(sid)
        nxt = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
        body = nxt.get("body") if nxt else None  # the next link's body is a complete request

    if not ids:
        sys.exit("Search found no scenes.")
    ids = sorted(ids, key=lambda i: i.split("_")[3])  # by acquisition date
    OUTPUT_FILE.write_text("\n".join(ids) + "\n")
    print(f"Found {len(ids)} scenes, saved to {OUTPUT_FILE.name}")

if __name__ == "__main__":
    save_scene_ids()