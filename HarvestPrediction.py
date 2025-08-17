#!/usr/bin/env python3
"""
dssat_orchestrator_minimal.py

Minimal orchestrator: runs DSSAT once and prints ONLY:
  Predicted maturity date: YYYY-MM-DD
  Predicted yield (kg/ha): <value>

No files/folders are created and no other terminal output is produced.
"""

import sys
import argparse
import time
import traceback
from datetime import datetime, timedelta
import re
import json

import requests
import numpy as np
import pandas as pd

# plotting and file IO removed intentionally
# DSSATTools binding (must be on PYTHONPATH)
try:
    from DSSATTools import DSSAT, Crop, SoilProfile, Weather, Management
except Exception:
    # If DSSATTools isn't importable, we cannot proceed.
    # We intentionally do not print diagnostic lines here (per user's request).
    raise

# optional geocoding
try:
    from geopy.geocoders import Nominatim
except Exception:
    Nominatim = None

# ---------- CONFIG ----------
WARMUP_DAYS = 30
DEFAULT_GROWTH_DAYS = 160
SOILGRIDS_DEPTHS = ["0-5cm", "5-15cm", "15-30cm"]

OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
SOILGRIDS_BASE = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# ---------- Helpers (soil) ----------
def geocode_pin(pin):
    """Return (lat, lon) for given Indian PIN using Nominatim (if available)."""
    if Nominatim is None:
        raise RuntimeError("geopy not installed; cannot geocode.")
    geolocator = Nominatim(user_agent="dssat_pin_geocoder")
    loc = geolocator.geocode(f"{pin}, India", timeout=15)
    if loc is None:
        raise RuntimeError(f"Geocoding failed for PIN {pin}.")
    return float(loc.latitude), float(loc.longitude)

def _parse_depth_label_to_bottom_mm(label):
    m = re.match(r"^\s*\d+\s*-\s*(\d+)\s*cm\s*$", label)
    if m:
        bottom_cm = int(m.group(1))
        return bottom_cm * 10  # cm -> mm
    return None

def fetch_soilgrids_properties(lat, lon, properties=("sand","silt","clay","soc","bdod"), depths=SOILGRIDS_DEPTHS):
    """Query SoilGrids; returns list of dicts per depth (properties as floats)."""
    params = {
        "lon": lon,
        "lat": lat,
        "depths": ",".join(depths),
        "properties": ",".join(properties)
    }
    r = requests.get(SOILGRIDS_BASE, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    props_obj = j.get("properties", {}) or {}

    prop_entries = {}
    if isinstance(props_obj, dict):
        for p in properties:
            if p in props_obj:
                prop_entries[p] = props_obj[p]
    if not prop_entries and isinstance(props_obj.get("layers"), list):
        for entry in props_obj["layers"]:
            name = entry.get("name")
            if name:
                prop_entries[name] = entry

    layers_out = []
    for depth_label in depths:
        row = {p: float("nan") for p in properties}
        for p in properties:
            info = prop_entries.get(p)
            if not info:
                continue
            d_factor = None
            try:
                d_factor = info.get("unit_measure", {}).get("d_factor")
            except Exception:
                d_factor = None
            depth_list = info.get("depths") or info.get("layers") or []
            if isinstance(depth_list, list):
                for entry in depth_list:
                    label = entry.get("label")
                    if not label and entry.get("range"):
                        rng = entry["range"]
                        top = int(rng.get("top_depth", 0))
                        bottom = int(rng.get("bottom_depth", 0))
                        label = f"{top}-{bottom}cm"
                    if label == depth_label:
                        vals = entry.get("values", {}) or {}
                        meanv = None
                        if isinstance(vals, dict):
                            meanv = vals.get("mean") or vals.get("m") or None
                        elif isinstance(vals, (int, float, str)):
                            meanv = vals
                        if meanv is not None:
                            try:
                                val = float(meanv)
                                if d_factor:
                                    val = val / float(d_factor)
                                row[p] = val
                                break
                            except Exception:
                                pass
        for k in list(row.keys()):
            try:
                row[k] = float(row[k])
            except Exception:
                row[k] = float("nan")
        layers_out.append(row)

    # heuristic: scale down weird large numbers
    for layer in layers_out:
        for k in list(layer.keys()):
            v = layer[k]
            if not (isinstance(v, float) and np.isnan(v)) and v > 100:
                layer[k] = v / 10.0

    return layers_out

def texture_to_dssat_class(sand, silt, clay):
    if np.isnan(sand) or np.isnan(silt) or np.isnan(clay):
        return "SIL"
    if clay >= 35:
        return "CL"
    if sand >= 70:
        return "SL"
    if silt >= 50:
        return "SIL"
    if sand > clay and sand > silt:
        return "SL"
    return "L"

# ---------- Helpers (weather) ----------
def sanitize_rhum(s):
    s2 = s.copy()
    s2 = s2.ffill().bfill().fillna(60.0)
    s2 = s2.clip(0.0, 100.0)
    return s2

def fetch_open_meteo_range(lat, lon, start_date, end_date, timezone="Asia/Kolkata"):
    sd = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    ed = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": sd,
        "end_date": ed,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,shortwave_radiation_sum",
        "hourly": "relativehumidity_2m",
        "timezone": timezone
    }
    r = requests.get(OPENMETEO_ARCHIVE, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    daily = j.get("daily", {})
    if not daily:
        return pd.DataFrame(columns=["TMIN","TMAX","RAIN","SRAD","RHUM"])
    dates = pd.to_datetime(daily["time"])
    df = pd.DataFrame(index=dates)
    df["TMAX"] = np.array(daily.get("temperature_2m_max", [np.nan]*len(dates)))
    df["TMIN"] = np.array(daily.get("temperature_2m_min", [np.nan]*len(dates)))
    df["RAIN"] = np.array(daily.get("precipitation_sum", [0.0]*len(dates)))
    df["SRAD"] = np.array(daily.get("shortwave_radiation_sum", [np.nan]*len(dates)))
    rh_hourly = j.get("hourly", {}).get("relativehumidity_2m")
    hour_times = j.get("hourly", {}).get("time")
    if rh_hourly and hour_times:
        try:
            rh_ser = pd.Series(rh_hourly, index=pd.to_datetime(hour_times))
            rh_daily = rh_ser.resample("D").mean()
            df["RHUM"] = rh_daily.reindex(df.index).values
        except Exception:
            df["RHUM"] = np.nan
    else:
        df["RHUM"] = np.nan
    df["RHUM"] = sanitize_rhum(df["RHUM"])
    return df

def fetch_open_meteo_forecast(lat, lon, days=7, timezone="Asia/Kolkata"):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,shortwave_radiation_sum",
        "hourly": "relativehumidity_2m",
        "timezone": timezone,
    }
    r = requests.get(OPENMETEO_FORECAST, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    daily = j.get("daily", {})
    if not daily:
        return pd.DataFrame(columns=["TMIN","TMAX","RAIN","SRAD","RHUM"])
    dates = pd.to_datetime(daily["time"])
    df = pd.DataFrame(index=dates)
    df["TMAX"] = np.array(daily.get("temperature_2m_max", [np.nan]*len(dates)))
    df["TMIN"] = np.array(daily.get("temperature_2m_min", [np.nan]*len(dates)))
    df["RAIN"] = np.array(daily.get("precipitation_sum", [0.0]*len(dates)))
    df["SRAD"] = np.array(daily.get("shortwave_radiation_sum", [np.nan]*len(dates)))
    rh_hourly = j.get("hourly", {}).get("relativehumidity_2m")
    hour_times = j.get("hourly", {}).get("time")
    if rh_hourly and hour_times:
        try:
            rh_ser = pd.Series(rh_hourly, index=pd.to_datetime(hour_times))
            rh_daily = rh_ser.resample("D").mean()
            df["RHUM"] = rh_daily.reindex(df.index).values
        except Exception:
            df["RHUM"] = np.nan
    else:
        df["RHUM"] = np.nan
    df["RHUM"] = sanitize_rhum(df["RHUM"])
    if len(df) > days:
        df = df.iloc[:days]
    return df

def build_tmy_from_past_years(lat, lon, start_date, end_date, years=5):
    frames = []
    today = datetime.now().date()
    for y in range(today.year - years, today.year):
        try:
            def safe_replace(d, ynew):
                try:
                    return d.replace(year=ynew)
                except ValueError:
                    return d.replace(year=ynew, day=28)
            start_y = safe_replace(start_date, y)
            end_y = safe_replace(end_date, y)
            df = fetch_open_meteo_range(lat, lon, start_y, end_y)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        days = (end_date - start_date).days + 1
        dates = pd.date_range(start_date, periods=days, freq="D")
        df = pd.DataFrame({
            "TMIN": np.repeat(18.0, len(dates)),
            "TMAX": np.repeat(30.0, len(dates)),
            "RAIN": np.repeat(0.1, len(dates)),
            "SRAD": np.repeat(18.0, len(dates)),
            "RHUM": np.repeat(60.0, len(dates)),
        }, index=dates)
        return df
    target_index = pd.date_range(start_date, end_date, freq="D")
    stacked = []
    for f in frames:
        vals = {}
        for col in ["TMIN","TMAX","RAIN","SRAD","RHUM"]:
            arr = f[col].to_numpy()
            tlen = len(target_index)
            alen = len(arr)
            if alen >= tlen:
                arr2 = arr[:tlen]
            else:
                arr2 = np.concatenate([arr, np.repeat(arr[-1], tlen - alen)])
            vals[col] = arr2
        df2 = pd.DataFrame(vals, index=target_index)
        stacked.append(df2)
    avg = pd.concat(stacked).groupby(level=0).mean()
    avg["RHUM"] = sanitize_rhum(avg["RHUM"])
    return avg

def stitch_weather(past_df, forecast_df, tmy_df):
    parts = []
    if past_df is not None and not past_df.empty:
        parts.append(past_df)
    if forecast_df is not None and not forecast_df.empty:
        parts.append(forecast_df)
    if tmy_df is not None and not tmy_df.empty:
        parts.append(tmy_df)
    if not parts:
        raise RuntimeError("No weather data available to stitch.")
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    df = df.asfreq("D")
    df = df.ffill().bfill()
    df["RHUM"] = sanitize_rhum(df["RHUM"])
    return df

# ---------- Management & maturity helpers ----------
def random_management(planting_date, crop_name=None, defaults=None):
    """
    Create a Management object with small randomization and ensure crop-specific
    required planting params (e.g. PLWT and SPRL for potato) are present.
    No prints, no file writes.
    """
    defaults = defaults or {}
    if isinstance(planting_date, str):
        try:
            planting_date = datetime.strptime(planting_date, "%Y-%m-%d")
        except Exception:
            pass

    m = Management(planting_date=planting_date)

    # Ensure a dict-like planting_details
    try:
        if not hasattr(m, "planting_details") or m.planting_details is None:
            m.planting_details = {}
    except Exception:
        try:
            setattr(m, "planting_details", {})
        except Exception:
            # if unable to set, proceed — DSSATTools may still accept Management as-is
            pass

    # Randomize PLRS if possible
    try:
        from numpy.random import default_rng
        rng = default_rng()
        rs = int(defaults.get("PLRS", rng.integers(20, 46)))
        if isinstance(m.planting_details, dict):
            m.planting_details.setdefault("PLRS", rs)
        else:
            try:
                m.planting_details["PLRS"] = rs
            except Exception:
                pass
    except Exception:
        pass

    # Crop-specific mandatory parameters
    if crop_name and isinstance(crop_name, str) and crop_name.strip().lower() in ("potato", "potatoes"):
        plwt_default = defaults.get("PLWT", 2.5)
        sprl_default = defaults.get("SPRL", 3.0)
        try:
            if isinstance(m.planting_details, dict):
                m.planting_details.setdefault("PLWT", float(plwt_default))
                m.planting_details.setdefault("SPRL", float(sprl_default))
            else:
                try:
                    m.planting_details["PLWT"] = float(plwt_default)
                    m.planting_details["SPRL"] = float(sprl_default)
                except Exception:
                    try:
                        setattr(m, "planting_details", {"PLWT": float(plwt_default), "SPRL": float(sprl_default)})
                    except Exception:
                        pass
        except Exception:
            pass

    # Apply any other defaults (without overwriting)
    if isinstance(m.planting_details, dict):
        for k, v in defaults.items():
            m.planting_details.setdefault(k, v)

    return m

def parse_mdat_from_summary(summary_df, planting_date):
    try:
        if summary_df is None or summary_df.empty:
            return None
        if "MDAT" not in summary_df.columns:
            return None
        raw = summary_df["MDAT"].iloc[0]
        if pd.isna(raw):
            return None
        m = int(raw)
        if 1 <= m <= 366:
            return datetime(planting_date.year, 1, 1) + timedelta(days=m-1)
        s = str(m)
        if len(s) == 8:
            try:
                return datetime.strptime(s, "%Y%m%d")
            except Exception:
                pass
    except Exception:
        pass
    return None

def detect_maturity_from_plantgro(plantgro_df, planting_date, crop_name):
    df = plantgro_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        if "DATE" in df.columns:
            df.index = pd.to_datetime(df["DATE"])
        else:
            df.index = pd.date_range(planting_date - timedelta(days=WARMUP_DAYS), periods=len(df), freq="D")
    if "GSTD" in df.columns:
        gst = df["GSTD"].dropna()
        if not gst.empty:
            max_stage = gst.max()
            candidates = df[df["GSTD"] == max_stage]
            if not candidates.empty:
                return (candidates.index[-1], candidates.iloc[-1])
    target_col = "HWAD" if crop_name.lower() == "wheat" else "GWAD"
    if target_col in df.columns:
        s = df[target_col].dropna()
        if not s.empty:
            final_val = float(s.iloc[-1])
            if final_val > 0:
                threshold = final_val * 0.995
                idxs = s[s >= threshold].index
                if len(idxs) > 0:
                    idx = idxs[0]
                    return (idx, df.loc[idx])
                else:
                    return (s.index[-1], df.loc[s.index[-1]])
    if not df.empty:
        idx = df.dropna(how="all").index[-1]
        return (idx, df.loc[idx])
    return (None, None)

# ---------- Orchestrator ----------
def run_for_farmer(pin=None, latlon=None, planting_date_str=None, crop_name="wheat"):
    """
    Run the pipeline and return a dict with keys:
      - predicted_maturity_date (str or None)
      - predicted_yield_kg_ha (float or None)
      - ok (bool)
      - error (str) optional if not ok
    No file writes. No prints.
    """
    try:
        # 1) geocode or use latlon
        if pin:
            try:
                lat, lon = geocode_pin(pin)
            except Exception:
                # fallback coordinate if geocoding fails
                lat, lon = 26.90, 75.90
        elif latlon:
            lat, lon = float(latlon[0]), float(latlon[1])
        else:
            raise ValueError("Either pin or latlon must be provided.")

        # 2) soil (SoilGrids)
        try:
            layers = fetch_soilgrids_properties(lat, lon, properties=("sand","silt","clay","soc","bdod"), depths=SOILGRIDS_DEPTHS)
            topvals = layers[0] if layers else None
            if topvals is None or all(np.isnan(v) for v in topvals.values()):
                raise RuntimeError("SoilGrids returned empty for top layer.")
        except Exception:
            # fallback generic soil
            layers = [{"sand":30.0,"silt":40.0,"clay":30.0,"bdod":1.35,"soc":0.6} for _ in range(3)]

        # 3) weather: past (planting-warmup -> yesterday), forecast (7d), then TMY to cover growth
        planting_date = datetime.strptime(planting_date_str, "%Y-%m-%d")
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        past_start = (planting_date.date() - timedelta(days=WARMUP_DAYS))

        past_df = pd.DataFrame()
        if past_start <= yesterday:
            try:
                past_df = fetch_open_meteo_range(lat, lon, past_start, yesterday)
            except Exception:
                past_df = pd.DataFrame()

        try:
            forecast_df = fetch_open_meteo_forecast(lat, lon, days=7)
        except Exception:
            forecast_df = pd.DataFrame()

        forecast_end = forecast_df.index.max().date() if not forecast_df.empty else (datetime.now().date() + timedelta(days=6))
        tmy_start = forecast_end + timedelta(days=1)
        tmy_end = planting_date.date() + timedelta(days=DEFAULT_GROWTH_DAYS)
        tmy_df = pd.DataFrame()
        if tmy_start <= tmy_end:
            try:
                tmy_df = build_tmy_from_past_years(lat, lon, datetime.combine(tmy_start, datetime.min.time()), datetime.combine(tmy_end, datetime.min.time()))
            except Exception:
                tmy_df = pd.DataFrame()

        # stitch weather
        try:
            stitched = stitch_weather(past_df, forecast_df, tmy_df)
            # Ensure coverage at least planting -> planting+DEFAULT_GROWTH_DAYS
            needed_end = planting_date.date() + timedelta(days=DEFAULT_GROWTH_DAYS)
            if stitched.index.max().date() < needed_end:
                add_days = (needed_end - stitched.index.max().date()).days
                extra_idx = pd.date_range(stitched.index.max() + timedelta(days=1), periods=add_days, freq="D")
                last_row = stitched.iloc[-1]
                extra = pd.DataFrame([last_row.values]*add_days, index=extra_idx, columns=stitched.columns)
                stitched = pd.concat([stitched, extra])
        except Exception:
            # fallback to generated random weather if stitching fails
            total_days = WARMUP_DAYS + DEFAULT_GROWTH_DAYS
            start_date = planting_date - timedelta(days=WARMUP_DAYS)
            dates = pd.date_range(start_date, periods=total_days, freq="D")
            rng = np.random.default_rng()
            stitched = pd.DataFrame({
                "TMIN": np.round(rng.uniform(12,20,size=total_days),2),
                "TMAX": np.round(rng.uniform(25,35,size=total_days),2),
                "SRAD": np.round(rng.uniform(12,28,size=total_days),2),
                "RAIN": np.round(np.where(rng.random(size=total_days) < 0.25, rng.exponential(scale=6.0,size=total_days), 0.0),3),
                "RHUM": np.round(rng.uniform(40,95,size=total_days),1)
            }, index=dates)

        # 4) prepare DSSAT objects
        mapping = {"TMIN":"TMIN","TMAX":"TMAX","RAIN":"RAIN","SRAD":"SRAD","RHUM":"RHUM"}
        weather_obj = Weather(stitched, mapping, lat, lon, elev=0)

        # SoilProfile
        try:
            top = layers[0] if layers else {"sand":30,"silt":40,"clay":30}
            dssat_soil_class = texture_to_dssat_class(top.get("sand",float("nan")), top.get("silt",float("nan")), top.get("clay",float("nan")))
            soil = SoilProfile(default_class=dssat_soil_class)
        except Exception:
            soil = SoilProfile(default_class="SIL")

        management = random_management(planting_date, crop_name=crop_name)

        # 5) run DSSAT once
        dssat = DSSAT()
        try:
            dssat.run(soil, weather_obj, Crop(crop_name), management)
        except Exception:
            try:
                dssat.close()
            except Exception:
                pass
            return {"ok": False, "error": "DSSAT run failed."}

        # 6) parse outputs
        summary_df = dssat.output.get("Summary")
        plantgro = dssat.output.get("PlantGro")
        maturity_date = None
        maturity_row = None
        try:
            mdate_from_summary = parse_mdat_from_summary(summary_df, planting_date)
        except Exception:
            mdate_from_summary = None

        if mdate_from_summary is not None:
            maturity_date = mdate_from_summary
            try:
                if plantgro is not None and not plantgro.empty:
                    doy = maturity_date.timetuple().tm_yday
                    matches = [idx for idx in plantgro.index if idx.timetuple().tm_yday == doy]
                    if matches:
                        maturity_row = plantgro.loc[matches[-1]]
                    else:
                        diffs = np.abs((plantgro.index - maturity_date).astype('timedelta64[D]'))
                        pos = int(diffs.argmin())
                        maturity_row = plantgro.iloc[pos]
            except Exception:
                maturity_row = None
        else:
            if plantgro is not None:
                maturity_date, maturity_row = detect_maturity_from_plantgro(plantgro, planting_date, crop_name)

        predicted_yield = None
        target_col = "HWAD" if crop_name.lower() == "wheat" else "GWAD"
        if plantgro is not None and target_col in plantgro.columns:
            try:
                s = plantgro[target_col].dropna()
                if not s.empty:
                    predicted_yield = float(s.iloc[-1])
            except Exception:
                predicted_yield = None

        if predicted_yield is None and summary_df is not None:
            for k in ("YIELD","YLD","GYLD","GWAD","HWAD"):
                if k in summary_df.columns:
                    try:
                        v = summary_df[k].iloc[0]
                        predicted_yield = float(v)
                        break
                    except Exception:
                        pass

        try:
            dssat.close()
        except Exception:
            pass

        return {
            "ok": True,
            "predicted_maturity_date": maturity_date.strftime("%Y-%m-%d") if maturity_date is not None else None,
            "predicted_yield_kg_ha": predicted_yield
        }

    except Exception as e:
        # Do not print; return error in structure
        return {"ok": False, "error": "Pipeline exception."}

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="DSSAT minimal orchestrator (prints only final results)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pin", help="Indian PIN code to geocode")
    group.add_argument("--latlon", nargs=2, type=float, metavar=("LAT","LON"), help="Latitude and longitude")
    parser.add_argument("--crop", required=True, help="Crop name recognizable by DSSATTools (e.g. wheat, rice)")
    parser.add_argument("--planting", required=True, help="Planting date YYYY-MM-DD")
    args = parser.parse_args()

    if args.pin:
        summary = run_for_farmer(pin=args.pin, planting_date_str=args.planting, crop_name=args.crop)
    else:
        summary = run_for_farmer(latlon=args.latlon, planting_date_str=args.planting, crop_name=args.crop)

    # Print only two lines and nothing else
    if summary.get("ok"):
        print(f"Predicted maturity date: {summary.get('predicted_maturity_date')}")
        py = summary.get("predicted_yield_kg_ha")
        if py is None:
            print("Predicted yield (kg/ha): None")
        else:
            # Format numeric yield to 2 decimal places
            try:
                print(f"Predicted yield (kg/ha): {float(py):.2f}")
            except Exception:
                print(f"Predicted yield (kg/ha): {py}")
    else:
        # If pipeline failed, still print the two lines as None to conform to requirement of "only two lines"
        print("Predicted maturity date: None")
        print("Predicted yield (kg/ha): None")

if __name__ == "__main__":
    main()
