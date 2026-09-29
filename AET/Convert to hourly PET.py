#!/usr/bin/env python
# coding: utf-8
"""
Convert_daily_to_hourly_PET_improved.py
========================================
Converts daily PET to hourly PET using an improved physically-based
disaggregation that better matches the original hPET dataset pattern.

Key improvements over the original sinusoidal method:
  1. Actual solar geometry (hour angle, declination, extraterrestrial radiation)
  2. Radiation-weighted disaggregation instead of simple sine curve
  3. Twilight zone — gradual PET ramp-up/ramp-down near sunrise/sunset
  4. Monsoon cloud factor — reduces midday peak during wet season (Jun–Sep)
  5. Temperature correction — PET weighted toward afternoon (heat lag effect)
  6. Strict mass conservation — hourly values always sum to daily total
"""

import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — change paths here
# ─────────────────────────────────────────────────────────────────────────────
INPUT_PATH  = r"E:\Well Labs\Extracted PET_daily\2015_daily_pet_Badenhalli_subwatershed.nc"
OUTPUT_PATH = r"E:\Well Labs\PET_hourly_converted\2015_hourly_converted_pet_Badenhalli_subwatershed.nc"
YEAR        = 2015

# Monsoon months for Karnataka (cloud correction applied)
MONSOON_MONTHS = [6, 7, 8, 9]   # Jun, Jul, Aug, Sep

# Afternoon heat lag: shift PET peak slightly after solar noon
# 0.0 = peak at solar noon; 1.0 = peak 1 hour after noon
HEAT_LAG_HOURS = 0.75

# Twilight transition width (hours) — gradual ramp at sunrise/sunset
TWILIGHT_HOURS = 0.5
# ─────────────────────────────────────────────────────────────────────────────


def solar_geometry(lat_deg, doy):
    """
    Compute solar geometry parameters for a given latitude and day of year.

    Returns
    -------
    decl      : solar declination (radians)
    ws        : sunset hour angle (radians)
    N         : daylight hours
    sunrise   : sunrise time (decimal hours from midnight)
    sunset    : sunset time (decimal hours from midnight)
    """
    # Solar declination — Spencer (1971) formula (more accurate than 0.409*sin)
    B = 2 * np.pi * (doy - 1) / 365
    decl = (0.006918
            - 0.399912 * np.cos(B)
            + 0.070257 * np.sin(B)
            - 0.006758 * np.cos(2*B)
            + 0.000907 * np.sin(2*B)
            - 0.002697 * np.cos(3*B)
            + 0.00148  * np.sin(3*B))

    lat_rad = np.radians(lat_deg)

    # Sunset hour angle
    cos_ws = -np.tan(lat_rad) * np.tan(decl)
    cos_ws = np.clip(cos_ws, -1.0, 1.0)   # avoid arccos domain error
    ws = np.arccos(cos_ws)

    # Daylight hours
    N = 24 / np.pi * ws

    sunrise = 12.0 - N / 2.0
    sunset  = 12.0 + N / 2.0

    return decl, ws, N, sunrise, sunset


def extraterrestrial_radiation_hourly(lat_deg, doy, hour_mid):
    """
    Compute hourly extraterrestrial radiation (Ra) in MJ/m²/hour.
    Based on FAO-56 equations 28 & 29.

    This gives a physically correct radiation shape that the hPET
    dataset uses internally — much better than a simple sine curve.

    Parameters
    ----------
    lat_deg  : latitude in degrees
    doy      : day of year (1–366)
    hour_mid : midpoint of hour (e.g. 6.5 for 06:00–07:00)

    Returns
    -------
    Ra : extraterrestrial radiation (MJ/m²/hour), >= 0
    """
    lat_rad = np.radians(lat_deg)

    # Solar declination (Spencer 1971)
    B = 2 * np.pi * (doy - 1) / 365
    decl = (0.006918
            - 0.399912 * np.cos(B)
            + 0.070257 * np.sin(B)
            - 0.006758 * np.cos(2*B)
            + 0.000907 * np.sin(2*B)
            - 0.002697 * np.cos(3*B)
            + 0.00148  * np.sin(3*B))

    # Inverse relative distance Earth–Sun
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)

    # Solar time angle at midpoint of hour (FAO-56 Eq. 29)
    # Assumes longitude correction = 0 (local solar time)
    omega = (np.pi / 12) * (hour_mid - 12)

    # Hour angle at start and end of hour
    omega1 = omega - np.pi / 24
    omega2 = omega + np.pi / 24

    # FAO-56 Equation 28
    Gsc = 4.92   # solar constant (MJ/m²/hour)
    Ra = (12 / np.pi) * Gsc * dr * (
        (omega2 - omega1) * np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * (np.sin(omega2) - np.sin(omega1))
    )

    return max(Ra, 0.0)


def monsoon_cloud_factor(doy):
    """
    Returns a multiplicative correction factor (0–1) to reduce midday PET
    during monsoon months, accounting for cloud cover over Karnataka.

    During monsoon (Jun–Sep), cloud cover reduces actual ET compared
    to clear-sky assumptions. This brings the converted hourly pattern
    closer to what hPET computes from real radiation data.

    The factor reduces the midday peak and redistributes to morning/evening,
    mimicking the effect of afternoon convective cloud build-up.
    """
    # Convert doy to approximate month
    month = datetime(YEAR, 1, 1) + timedelta(days=doy - 1)
    m = month.month

    if m in MONSOON_MONTHS:
        # Stronger cloud effect in peak monsoon (Jul–Aug)
        if m in [7, 8]:
            return 0.75   # reduce peak by 25%
        else:
            return 0.85   # reduce peak by 15% (Jun, Sep)
    return 1.0


def pet_hourly_weights(lat_deg, doy):
    """
    Compute 24 hourly PET distribution weights using:
      1. Physically correct extraterrestrial radiation (FAO-56)
      2. Afternoon heat lag correction
      3. Twilight zone (gradual ramp at sunrise/sunset)
      4. Monsoon cloud correction

    Weights sum to 1.0 — multiply by daily PET to get hourly values.
    """
    _, ws, N, sunrise, sunset = solar_geometry(lat_deg, doy)
    cloud_factor = monsoon_cloud_factor(doy)

    weights = np.zeros(24)

    for h in range(24):
        hour_mid = h + 0.5

        # Get raw extraterrestrial radiation for this hour
        Ra = extraterrestrial_radiation_hourly(lat_deg, doy, hour_mid)

        if Ra <= 0:
            weights[h] = 0.0
            continue

        # ── Afternoon heat lag correction ──────────────────────────────────
        # Shift the effective solar time to peak slightly after noon
        # This mimics temperature-driven ET peaking after solar noon
        hour_adj = hour_mid - HEAT_LAG_HOURS
        Ra_adj   = extraterrestrial_radiation_hourly(lat_deg, doy, hour_adj)
        Ra_adj   = max(Ra_adj, 0.0)

        # Blend: 60% actual radiation + 40% lagged radiation
        Ra_blended = 0.60 * Ra + 0.40 * Ra_adj

        # ── Twilight zone correction ───────────────────────────────────────
        # Gradual ramp-up after sunrise and ramp-down before sunset
        # Avoids the abrupt zero cutoff of the simple sinusoidal method
        twilight_factor = 1.0
        if hour_mid < sunrise + TWILIGHT_HOURS:
            # Morning twilight ramp-up
            twilight_factor = max(0, (hour_mid - (sunrise - TWILIGHT_HOURS))
                                  / (2 * TWILIGHT_HOURS))
        elif hour_mid > sunset - TWILIGHT_HOURS:
            # Evening twilight ramp-down
            twilight_factor = max(0, ((sunset + TWILIGHT_HOURS) - hour_mid)
                                  / (2 * TWILIGHT_HOURS))

        # ── Monsoon cloud factor ───────────────────────────────────────────
        # Only apply cloud reduction to hours near solar noon (peak hours)
        # Morning and evening less affected by afternoon cloud build-up
        if 10 <= hour_mid <= 16:
            effective_cloud = cloud_factor
        else:
            effective_cloud = 1.0 - (1.0 - cloud_factor) * 0.3

        weights[h] = Ra_blended * twilight_factor * effective_cloud

    # ── Normalise weights to sum = 1 ──────────────────────────────────────
    total = weights.sum()
    if total > 0:
        weights /= total
    else:
        # Polar night or no daylight — return zeros
        weights[:] = 0.0

    return weights


def disaggregate_pet(pet_daily_value, lat_deg, doy):
    """
    Disaggregate one day's PET to 24 hourly values.

    Parameters
    ----------
    pet_daily_value : float — daily PET (mm/day)
    lat_deg         : float — latitude (degrees)
    doy             : int   — day of year (1–366)

    Returns
    -------
    hourly_pet : np.array shape (24,) — hourly PET (mm/hr)
                 guaranteed to sum to pet_daily_value
    """
    if pet_daily_value <= 0:
        return np.zeros(24, dtype=np.float32)

    weights = pet_hourly_weights(lat_deg, doy)
    hourly  = weights * float(pet_daily_value)

    # ── Strict mass conservation ───────────────────────────────────────────
    # Rescale to ensure sum == daily total exactly (eliminates floating point drift)
    s = hourly.sum()
    if s > 0:
        hourly = hourly * (float(pet_daily_value) / s)

    return hourly.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN — Read daily PET, disaggregate, write hourly NetCDF
# ═════════════════════════════════════════════════════════════════════════════
print("Reading daily PET file...")
ds_in     = nc.Dataset(INPUT_PATH, 'r')
pet_daily = ds_in.variables['pet'][:]
lat       = ds_in.variables['latitude'][:]
lon       = ds_in.variables['longitude'][:]
ds_in.close()

n_days, n_lat, n_lon = pet_daily.shape
n_hours = n_days * 24

print(f"  Input shape  : {pet_daily.shape}  ({n_days} days × {n_lat} lat × {n_lon} lon)")
print(f"  Output shape : ({n_hours}, {n_lat}, {n_lon})  ({n_hours} hours)")

# ── Allocate output array ──────────────────────────────────────────────────
pet_hourly = np.zeros((n_hours, n_lat, n_lon), dtype=np.float32)

print("\nDisaggregating daily → hourly...")
for d in range(n_days):
    doy = d + 1
    if d % 30 == 0:
        approx_month = (datetime(YEAR, 1, 1) + timedelta(days=d)).strftime('%b')
        print(f"  Day {d+1:3d} ({approx_month})")

    for i in range(n_lat):
        for j in range(n_lon):
            val = pet_daily[d, i, j]

            if np.ma.is_masked(val) or np.isnan(float(val)):
                pet_hourly[d*24 : d*24+24, i, j] = 0.0
            else:
                hourly = disaggregate_pet(float(val), float(lat[i]), doy)
                pet_hourly[d*24 : d*24+24, i, j] = hourly

print("Disaggregation complete.")

# ── Mass conservation check ────────────────────────────────────────────────
print("\nVerifying mass conservation...")
max_err  = 0.0
mean_err = 0.0
n_checks = 0
for d in range(n_days):
    for i in range(n_lat):
        for j in range(n_lon):
            daily_val = float(pet_daily[d, i, j])
            if np.ma.is_masked(pet_daily[d, i, j]):
                continue
            hourly_sum = float(pet_hourly[d*24:d*24+24, i, j].sum())
            err = abs(hourly_sum - daily_val)
            max_err   = max(max_err, err)
            mean_err += err
            n_checks += 1

mean_err /= max(n_checks, 1)
print(f"  Max error  : {max_err:.2e} mm  (should be < 1e-4)")
print(f"  Mean error : {mean_err:.2e} mm")

# ── Write output NetCDF ────────────────────────────────────────────────────
print(f"\nWriting output to:\n  {OUTPUT_PATH}")
ds_out = nc.Dataset(OUTPUT_PATH, 'w', format='NETCDF4')

ds_out.createDimension('time',      n_hours)
ds_out.createDimension('latitude',  n_lat)
ds_out.createDimension('longitude', n_lon)

# Time variable — correct year
time_out          = ds_out.createVariable('time', 'f8', ('time',))
time_out.units    = f'hours since {YEAR}-01-01 00:00:00'   # ← fixed year
time_out.calendar = 'proleptic_gregorian'
time_out[:]       = np.arange(n_hours, dtype=np.float64)

# Coordinates
lat_out    = ds_out.createVariable('latitude',  'f4', ('latitude',))
lat_out[:] = lat
lat_out.units     = 'degrees_north'
lat_out.long_name = 'Latitude'

lon_out    = ds_out.createVariable('longitude', 'f4', ('longitude',))
lon_out[:] = lon
lon_out.units     = 'degrees_east'
lon_out.long_name = 'Longitude'

# PET variable
pet_out           = ds_out.createVariable(
    'pet', 'f4', ('time', 'latitude', 'longitude'),
    zlib=True, complevel=4
)
pet_out.units     = 'mm/hr'
pet_out.long_name = 'Potential Evapotranspiration (hourly)'
pet_out[:]        = pet_hourly

# Global attributes
ds_out.description  = ('Hourly PET disaggregated from daily using improved '
                       'physically-based method (FAO-56 Ra + heat lag + '
                       'monsoon cloud correction)')
ds_out.source       = INPUT_PATH
ds_out.method       = ('FAO-56 extraterrestrial radiation weighting, '
                       f'{HEAT_LAG_HOURS}h afternoon heat lag, '
                       f'{TWILIGHT_HOURS}h twilight zone, '
                       'monsoon cloud correction (Jun-Sep Karnataka)')
ds_out.year         = str(YEAR)
ds_out.created      = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
ds_out.close()

print("\n=== Summary ===")
print(f"  Output shape       : {pet_hourly.shape}")
print(f"  Max hourly PET     : {pet_hourly.max():.4f} mm/hr")
print(f"  Mean hourly PET    : {pet_hourly[pet_hourly>0].mean():.4f} mm/hr")
print(f"  Zero hours         : {(pet_hourly==0).sum()} ({100*(pet_hourly==0).sum()/pet_hourly.size:.1f}%)")
print(f"  Mass conservation  : max error = {max_err:.2e} mm")
print(f"\nDay 1, cell[0,0] check:")
print(f"  Daily original     : {float(pet_daily[0,0,0]):.4f} mm/day")
print(f"  Sum of 24 hourly   : {pet_hourly[:24,0,0].sum():.4f} mm")
print(f"\nDone! File saved to:\n  {OUTPUT_PATH}")

