# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 23:53:16 2026

@author: karan

Village lookup table for the crop survey site, Chikkaballapura district.

Step 1  READ THE SITE   Reads every taluk, hobli and village from the site's dropdowns
                        for each season (no survey numbers queried; about 40 page loads
                        per season) into one table: village_lookup/site_villages_all.csv.
                        Seasons already in the table are skipped.
Step 2-3  MATCH         For each cadastral village and season:
          1. Own code   the village's own code is on the site.
                        Name score >= NAME_OK -> "own code"; otherwise the code is kept
                        but flagged "own code, CHECK name differs".
          2. Old hobli  villages whose code is not on the site (new taluks/hoblis the
                        site had not adopted yet) are searched in the old site hobli(s)
                        they came from (HOBLI_SOURCE), among site villages no other
                        village already claims by its own code:
                 2a. same village number in the old hobli, if the name agrees
                 2b. otherwise best name match; best pairs first, each site village
                     used once; "clear" if score >= NAME_SURE and ahead of the
                     runner-up by MARGIN, else "CHECK"
                 2c. still unmatched: same village number even though the name
                     differs (flagged CHECK; e.g. Kondakonda -> Kondikonda)
          3. MANUAL     your corrections override everything.
          Site codes used by two villages in one season are flagged DUPLICATE.
Step 4  FLAG            needs_check / check_seasons / check_reason: rows with any
                        CHECK, DUPLICATE or not found in any season.
Step 5  SAVE            village_lookup/village_lookup.csv, one row per village with the
                        site code, name, score and method for every season.
"""

#%% CONFIGURATION
import os, re, time, difflib
from pathlib import Path
import pandas as pd
import geopandas as gpd

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException

SEASONS   = ["2022", "2022_Rabi", "2022_Summer",
             "2023", "2023_Rabi", "2023_Summer",
             "2024", "2024_Rabi", "2024_Summer",
             "2025", "2025_Rabi", "2025_Summer"]
SCRIPT_DIR = Path(__file__).resolve().parent
CADASTRAL = SCRIPT_DIR / "Chikkaballapura_surveys.gpkg"
CADASTRAL_LAYER = "cadastral_Chikkaballapura_district"
DISTRICT  = "28"              # Chikkaballapur in the site's district dropdown
HEADLESS  = False
PAUSE     = 0.1               # extra seconds after each dropdown change
NAME_OK   = 0.55              # name score that confirms a code
NAME_SURE = 0.85              # name score for a "clear" name-only match
MARGIN    = 0.10              # lead over the runner-up for a "clear" name-only match

OUT_DIR    = SCRIPT_DIR / "village_lookup"
SITE_TABLE = OUT_DIR / "site_villages_all.csv"
LOOKUP     = OUT_DIR / "village_lookup.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# KGIS hobli (first 6 digits of the village code) -> site hobli(s) it came from
# before the new taluks were adopted on the site. Based on the site's hobli names,
# which did not change, and on the OLD_CODE map (former Chilakalanerpu hobli).
HOBLI_SOURCE = {
    "280603": ["280604"],                       # Ambajidurga villages, formerly Chilakalanerpu (Chintamani)
    "280704": ["280104"],                       # Manchenahalli <- Gauribidanur / Manchenahalli
    "280803": ["280403"],                       # Patapalya     <- Bagepalli / Patapalya
    "280805": ["280405"],                       # Mittemari     <- Bagepalli / Mittemari
    "280804": ["280404", "280403", "280405",    # Cheluru       <- Bagepalli / Cheluru, Patapalya, Mittemari
               "280803", "280805"],             #                  (and Cheluru taluk's Patapalya, Mittemari)
    "280807": ["280604"],                       # Chilakalanerpu <- Chintamani / Chilakalanerpu
}

# Your corrections after checking: {(kgis_code, season): site_code}
MANUAL = {
    # 2806030090 Mittahalli
    ("2806030090", "2022"): "2806040027",   # Mittahalli
    ("2806030090", "2022_Rabi"): "2806040027",   # Mittahalli
    ("2806030090", "2023"): "2806040027",   # MITTAHALLI
    ("2806030090", "2023_Rabi"): "2806040027",   # MITTAHALLI
    ("2806030090", "2023_Summer"): "2806040027",   # MITTAHALLI
    ("2806030090", "2024"): "2806040027",   # MITTAHALLI
    ("2806030090", "2024_Rabi"): "2806040027",   # MITTAHALLI
    # 2806030093 J.I.Goramapalli
    ("2806030093", "2022"): "2806040030",   # Gurrampalli
    ("2806030093", "2022_Rabi"): "2806040030",   # Gurrampalli
    ("2806030093", "2023"): "2806040030",   # GURRAMPALLI
    ("2806030093", "2023_Rabi"): "2806040030",   # GURRAMPALLI
    ("2806030093", "2023_Summer"): "2806040030",   # GURRAMPALLI
    ("2806030093", "2024"): "2806040030",   # GURRAMPALLI
    ("2806030093", "2024_Rabi"): "2806040030",   # GURRAMPALLI
    # 2806030096 Amitahalli
    ("2806030096", "2022"): "2806040033",   # Amita halli
    ("2806030096", "2022_Rabi"): "2806040033",   # Amita halli
    ("2806030096", "2023"): "2806040033",   # AMITA HALLI
    ("2806030096", "2023_Rabi"): "2806040033",   # AMITA HALLI
    ("2806030096", "2023_Summer"): "2806040033",   # AMITA HALLI
    ("2806030096", "2024"): "2806040033",   # AMITA HALLI
    ("2806030096", "2024_Rabi"): "2806040033",   # AMITA HALLI
    # 2806030109 Manira Muddenahalli
    ("2806030109", "2022"): "2806040046",   # Em muddalahalli
    ("2806030109", "2022_Rabi"): "2806040046",   # Em muddalahalli
    ("2806030109", "2023"): "2806040046",   # EM MUDDALAHALLI
    ("2806030109", "2023_Rabi"): "2806040046",   # EM MUDDALAHALLI
    ("2806030109", "2023_Summer"): "2806040046",   # EM MUDDALAHALLI
    ("2806030109", "2024"): "2806040046",   # EM MUDDALAHALLI
    ("2806030109", "2024_Rabi"): "2806040046",   # EM MUDDALAHALLI
    # 2808040026 Kondakonda
    ("2808040026", "2022"): "2804040026",   # Kondikonda
    ("2808040026", "2022_Rabi"): "2804040026",   # Kondikonda
    ("2808040026", "2023"): "2804040026",   # KONDIKONDA
    ("2808040026", "2023_Rabi"): "2804040026",   # KONDIKONDA
    ("2808040026", "2023_Summer"): "2804040026",   # KONDIKONDA
    ("2808040026", "2024"): "2804040026",   # KONDIKONDA
    ("2808040026", "2024_Rabi"): "2808040026",   # KONDIKONDA
    ("2808040026", "2025"): "2808040026",   # KONDIKONDA
    ("2808040026", "2025_Rabi"): "2808040026",   # KONDIKONDA
    # 2808040061 Dugginayakanapalli
    ("2808040061", "2022"): "2804030023",   # Dugginayakanapalli
    ("2808040061", "2022_Rabi"): "2804030023",   # Dugginayakanapalli
    ("2808040061", "2023"): "2804030023",   # DUGGINAYAKANAPALLI
    ("2808040061", "2023_Rabi"): "2804030023",   # DUGGINAYAKANAPALLI
    ("2808040061", "2023_Summer"): "2804030023",   # DUGGINAYAKANAPALLI
    ("2808040061", "2024"): "2804030023",   # DUGGINAYAKANAPALLI
    ("2808040061", "2024_Rabi"): "2808030023",   # DUGGINAYAKANAPALLI
    # 2808040062 Golavarapalli
    ("2808040062", "2022"): "2804030024",   # Gollavarapalli
    ("2808040062", "2022_Rabi"): "2804030024",   # Gollavarapalli
    ("2808040062", "2023"): "2804030024",   # GOLLAVARAPALLI
    ("2808040062", "2023_Rabi"): "2804030024",   # GOLLAVARAPALLI
    ("2808040062", "2023_Summer"): "2804030024",   # GOLLAVARAPALLI
    ("2808040062", "2024"): "2804030024",   # GOLLAVARAPALLI
    ("2808040062", "2024_Rabi"): "2808030024",   # GOLLAVARAPALLI
    # 2808040067 Kolavarapalli
    ("2808040067", "2022"): "2804030029",   # Kollavarapalli
    ("2808040067", "2022_Rabi"): "2804030029",   # Kollavarapalli
    ("2808040067", "2023"): "2804030029",   # KOLLAVARAPALLI
    ("2808040067", "2023_Rabi"): "2804030029",   # KOLLAVARAPALLI
    ("2808040067", "2023_Summer"): "2804030029",   # KOLLAVARAPALLI
    ("2808040067", "2024"): "2804030029",   # KOLLAVARAPALLI
    ("2808040067", "2024_Rabi"): "2808030029",   # KOLLAVARAPALLI
    # 2808040079 Chimannagaripalli
    ("2808040079", "2022"): "2804030044",   # Cheemannagarapalli
    ("2808040079", "2022_Rabi"): "2804030044",   # Cheemannagarapalli
    ("2808040079", "2023"): "2804030044",   # CHEEMANNAGARAPALLI
    ("2808040079", "2023_Rabi"): "2804030044",   # CHEEMANNAGARAPALLI
    ("2808040079", "2023_Summer"): "2804030044",   # CHEEMANNAGARAPALLI
    ("2808040079", "2024"): "2804030044",   # CHEEMANNAGARAPALLI
    ("2808040079", "2024_Rabi"): "2808030044",   # CHEEMANNAGARAPALLI
    # 2808040082 Ramanapade
    ("2808040082", "2022"): "2804030048",   # Ramanupadi
    ("2808040082", "2022_Rabi"): "2804030048",   # Ramanupadi
    ("2808040082", "2023"): "2804030048",   # RAMANUPADI
    ("2808040082", "2023_Rabi"): "2804030048",   # RAMANUPADI
    ("2808040082", "2023_Summer"): "2804030048",   # RAMANUPADI
    ("2808040082", "2024"): "2804030048",   # RAMANUPADI
    ("2808040082", "2024_Rabi"): "2808030048",   # RAMANUPADI
    # 2808040084 Dugginaykanapalli
    ("2808040084", "2022"): "2804050030",   # Dugginaykanapalli
    ("2808040084", "2022_Rabi"): "2804050030",   # Dugginaykanapalli
    ("2808040084", "2023"): "2804050030",   # DUGGINAYKANAPALLI
    ("2808040084", "2023_Rabi"): "2804050030",   # DUGGINAYKANAPALLI
    ("2808040084", "2023_Summer"): "2804050030",   # DUGGINAYKANAPALLI
    ("2808040084", "2024"): "2804050030",   # DUGGINAYKANAPALLI
    ("2808040084", "2024_Rabi"): "2808050030",   # DUGGINAYKANAPALLI
    # 2808040085 Poolakuntlapalli
    ("2808040085", "2022"): "2804050032",   # Pulakuntlapalli
    ("2808040085", "2022_Rabi"): "2804050032",   # Pulakuntlapalli
    ("2808040085", "2023"): "2804050032",   # PULAKUNTLAPALLI
    ("2808040085", "2023_Rabi"): "2804050032",   # PULAKUNTLAPALLI
    ("2808040085", "2023_Summer"): "2804050032",   # PULAKUNTLAPALLI
    ("2808040085", "2024"): "2804050032",   # PULAKUNTLAPALLI
    ("2808040085", "2024_Rabi"): "2808050032",   # PULAKUNTLAPALLI
}

#%% STEP 1  READ THE SITE'S DROPDOWNS INTO ONE TABLE
# Seasons already in the table are skipped, so a failed or interrupted run
# can simply be rerun.
def url(season): return f"https://cropsurvey.karnataka.gov.in/{season}/MISReports/frmGetFruitDetails.aspx"

def start_browser():
    options = Options()
    if HEADLESS:
        options.add_argument("-headless")
    driver = webdriver.Firefox(options=options)
    if not HEADLESS:
        driver.maximize_window()
    return driver, WebDriverWait(driver, 20)

def select(driver, wait, element_id, value, tries=5):
    """Select a dropdown value and wait for the page to reload."""
    for _ in range(tries):
        try:
            Select(wait.until(EC.presence_of_element_located((By.ID, element_id)))).select_by_value(value)
            time.sleep(2 + PAUSE)
            dd = Select(wait.until(EC.presence_of_element_located((By.ID, element_id))))
            if dd.first_selected_option.get_attribute("value") == value:
                return True
        except StaleElementReferenceException:
            pass
        time.sleep(1)
    return False

def options(driver, wait, element_id, tries=5):
    """All (value, text) pairs of a dropdown, without '--select--'."""
    for _ in range(tries):
        try:
            dd = Select(wait.until(EC.presence_of_element_located((By.ID, element_id))))
            return [(o.get_attribute("value"), o.text.strip()) for o in dd.options
                    if o.get_attribute("value") not in ("0", "")]
        except StaleElementReferenceException:
            time.sleep(1)
    return []

def read_season(season):
    driver, wait = start_browser()
    rows = []
    try:
        driver.get(url(season))
        if not select(driver, wait, "ddlDistrict", DISTRICT):
            raise RuntimeError("district not found")
        for t, t_name in options(driver, wait, "drpTaluk"):
            select(driver, wait, "drpTaluk", t)
            for h, h_name in options(driver, wait, "drpHobli"):
                select(driver, wait, "drpHobli", h)
                villages = options(driver, wait, "ddlVillage")
                rows += [{"season": season, "taluk_name": t_name, "hobli_name": h_name,
                          "site_code": f"{DISTRICT}{int(t):02d}{int(h):02d}{int(v):04d}",
                          "site_name": v_name} for v, v_name in villages]
                print(f"  [{season}] {t_name} / {h_name}: {len(villages)} villages")
    finally:
        driver.quit()
    return pd.DataFrame(rows)

site = pd.read_csv(SITE_TABLE, dtype=str) if os.path.exists(SITE_TABLE) else pd.DataFrame()
for season in SEASONS:
    if len(site) and season in set(site.season):
        continue
    print(f"Reading {season}")
    try:
        site = pd.concat([site, read_season(season)], ignore_index=True)
        site.to_csv(SITE_TABLE, index=False)          # saved after every season
    except Exception as e:
        print(f"  [{season}] failed: {e} (rerun to retry)")

print(site.groupby("season").agg(taluks=("taluk_name", "nunique"), villages=("site_code", "size")))

#%% NAME SCORE (letters only, case ignored, word order ignored)
def name_score(a, b):
    words = lambda s: re.sub(r"[^a-z ]", "", str(s).lower()).split()
    wa, wb = words(a), words(b)
    return max(difflib.SequenceMatcher(None, "".join(wa), "".join(wb)).ratio(),
               difflib.SequenceMatcher(None, "".join(sorted(wa)), "".join(sorted(wb))).ratio())

#%% STEP 2  INPUTS FOR MATCHING
cad = gpd.read_file(CADASTRAL, layer=CADASTRAL_LAYER, ignore_geometry=True,
                    columns=["KGIS_village_code", "village", "surveynumberi"])
vil = (cad[cad.surveynumberi > 0].groupby("KGIS_village_code").village.first()
          .reset_index().rename(columns={"KGIS_village_code": "kgis_code", "village": "kgis_name"}))
vil = vil[vil.kgis_code.str.len() == 10].reset_index(drop=True)   # 16-digit codes can't be placed on the site
print(f"{len(vil)} cadastral villages, {site.season.nunique()} seasons in the site table")

#%% STEP 3  MATCH, SEASON BY SEASON
table = vil.copy()
for season in SEASONS:
    s = site[site.season == season]
    if s.empty:
        print(f"[{season}] not in site table, skipped")
        continue
    names = dict(zip(s.site_code, s.site_name))
    code, how = {}, {}

    # 1. own code
    for v in vil.itertuples():
        if v.kgis_code in names:
            code[v.kgis_code] = v.kgis_code
            how[v.kgis_code] = "own code" if name_score(names[v.kgis_code], v.kgis_name) >= NAME_OK \
                               else "own code, CHECK name differs"

    # 2. villages not on the site: search their old hobli(s) among unclaimed site villages
    used = set(code.values())
    free = {c: n for c, n in names.items() if c not in used}
    todo = vil[~vil.kgis_code.isin(code) & vil.kgis_code.str[:6].isin(HOBLI_SOURCE)]
    src  = lambda kc: HOBLI_SOURCE[kc[:6]]

    for v in todo.itertuples():                                   # 2a. same number
        for h in src(v.kgis_code):
            c = h + v.kgis_code[6:]
            if c in free and c not in used and name_score(free[c], v.kgis_name) >= NAME_OK:
                code[v.kgis_code], how[v.kgis_code] = c, "old hobli, same number"
                used.add(c)
                break

    left  = todo[~todo.kgis_code.isin(code)]                      # 2b. best name, best pairs first
    pairs = sorted([(name_score(n, v.kgis_name), v.kgis_code, c)
                    for v in left.itertuples() for c, n in free.items()
                    if c[:6] in src(v.kgis_code) and c not in used], reverse=True)
    scores = {}
    for sc, kc, c in pairs:
        scores.setdefault(kc, []).append(sc)
    for sc, kc, c in pairs:
        if kc in code or c in used or sc < NAME_OK:
            continue
        runner = scores[kc][1] if len(scores[kc]) > 1 else 0
        code[kc] = c
        how[kc] = "old hobli, name" if sc >= NAME_SURE and sc - runner >= MARGIN else "old hobli, name, CHECK"
        used.add(c)

    for v in todo[~todo.kgis_code.isin(code)].itertuples():       # 2c. same number, name differs
        for h in src(v.kgis_code):
            c = h + v.kgis_code[6:]
            if c in free and c not in used:
                code[v.kgis_code], how[v.kgis_code] = c, "old hobli, same number, CHECK name differs"
                used.add(c)
                break

    # 3. manual corrections
    for (kc, se), c in MANUAL.items():
        if se == season:
            code[kc], how[kc] = c, "manual"

    col = lambda x: f"{x}_{season}"
    table[col("code")]  = table.kgis_code.map(code)
    table[col("name")]  = table[col("code")].map(names)
    table[col("score")] = [round(name_score(a, b), 2) if isinstance(b, str) else None
                           for a, b in zip(table.kgis_name, table[col("name")])]
    table[col("how")]   = table.kgis_code.map(how).fillna("not found")
    dup = table[col("code")].notna() & table[col("code")].duplicated(keep=False)
    table.loc[dup, col("how")] += " | DUPLICATE"

#%% STEP 4  FLAG ROWS TO CHECK
# A village needs checking if, in any season, its match is flagged CHECK,
# DUPLICATE or not found. check_seasons lists those seasons; check_reason the
# distinct reasons. Both columns sit right after the village name.
hows = [f"how_{se}" for se in SEASONS if f"how_{se}" in table]
bad  = table[hows].apply(lambda col: col.str.contains("CHECK|DUPLICATE|not found"))
table.insert(2, "needs_check", bad.any(axis=1))
table.insert(3, "check_seasons", bad.apply(lambda r: ", ".join(c[4:] for c in hows if r[c]), axis=1))
table.insert(4, "check_reason", table[hows].where(bad.values).apply(
    lambda r: "; ".join(sorted(set(r.dropna()))), axis=1))

#%% STEP 5  SAVE AND SUMMARISE
table.to_csv(LOOKUP, index=False)
print(f"{table.needs_check.sum()} villages need checking in at least one season")
print(f"\n{len(table)} villages -> {LOOKUP}\n")
summary = pd.DataFrame({se: table[f"how_{se}"].value_counts()
                        for se in SEASONS if f"how_{se}" in table}).fillna(0).astype(int)
print(summary.to_string())