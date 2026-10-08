"""Live weather from Open-Meteo (free, no API key) for dynamic watering advice."""
import json
import os
import time
import urllib.parse
import urllib.request

import config

GEO = "https://geocoding-api.open-meteo.com/v1/search?name={}&count=1&language=en"
FORECAST = ("https://api.open-meteo.com/v1/forecast"
            "?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,precipitation,weather_code"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
            "precipitation_sum,sunshine_duration"
            "&timezone=auto&forecast_days=1")
CACHE_PATH = os.path.join(config.MODELS_DIR, "weather_cache.json")
TTL_SEC = 15 * 60


def _get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "ai-urban-farming/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _read_cache():
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def _write_cache(data):
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
    except Exception:  # noqa: BLE001
        pass


def geocode(city: str):
    cache = _read_cache()
    key = f"geo:{city.lower()}"
    if key in cache and time.time() - cache[key]["t"] < 86400:
        return cache[key]["v"]
    data = _get(GEO.format(urllib.parse.quote(city)))
    results = data.get("results") or []
    if not results:
        raise ValueError(f"City not found: {city}")
    r0 = results[0]
    v = {"name": f"{r0.get('name')}, {r0.get('country', '')}".strip(", "),
         "lat": r0["latitude"], "lon": r0["longitude"]}
    cache[key] = {"t": time.time(), "v": v}
    _write_cache(cache)
    return v


def current_weather(lat: float, lon: float) -> dict:
    cache = _read_cache()
    key = f"w:{round(float(lat), 3)}:{round(float(lon), 3)}"
    hit = cache.get(key)
    if hit and time.time() - hit["t"] < TTL_SEC:
        return hit["v"]

    d = _get(FORECAST.format(lat=lat, lon=lon))
    cur = d.get("current", {})
    daily = d.get("daily", {})
    sun_sec = (daily.get("sunshine_duration") or [8 * 3600])[0] or 8 * 3600
    out = {
        "temperature_c": float(cur.get("temperature_2m", 22.0)),
        "humidity_pct": float(cur.get("relative_humidity_2m", 55.0)),
        "rainfall_mm": float(daily.get("precipitation_sum", [0.0])[0] or 0.0),
        "rainfall_probability_pct": float(daily.get("precipitation_probability_max", [0.0])[0] or 0.0),
        "sunlight_hours": round(float(sun_sec) / 3600.0, 1),
        "temp_max_c": float((daily.get("temperature_2m_max") or [25.0])[0]),
        "temp_min_c": float((daily.get("temperature_2m_min") or [15.0])[0]),
        "source": "open-meteo.com",
    }
    cache[key] = {"t": time.time(), "v": out}
    _write_cache(cache)
    return out


def weather_for_city(city: str) -> dict:
    g = geocode(city)
    w = current_weather(g["lat"], g["lon"])
    w["location"] = g["name"]
    return w


def weather_for_input(city: str = None, lat: float = None, lon: float = None) -> dict:
    """Weather dict, or None when nothing can be resolved (offline demo still works)."""
    try:
        if city:
            return weather_for_city(city)
        if lat is not None and lon is not None:
            w = current_weather(lat, lon)
            w["location"] = f"{lat:.3f}, {lon:.3f}"
            return w
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "source": "offline"}
    return None
