"""Watering recommendation: XGBoost (trained) + optional live weather.

  recommend(species="Tomato", pot_size_l=12, soil_moisture_pct=55, city="Berlin")
  -> {action, confidence, hours_until_watering, water_amount_l, ...}

Falls back to the plant profile's climate when weather is unavailable (offline demo).
"""
import json
import os
from typing import Optional

import numpy as np
import pandas as pd

import config
from knowledge import PLANTS
from train_watering import feature_matrix
from watering_data import LATER_WATER_H, NOW_WATER_H, SOILS, daily_use_l, pot_area_m2

MODEL_DIR = config.MODELS_DIR
_FEATURES_PATH = os.path.join(MODEL_DIR, "watering_features.json")
_REG_PATH = os.path.join(MODEL_DIR, "watering_regressor.json")
_CLF_PATH = os.path.join(MODEL_DIR, "watering_classifier.json")

_BUNDLE = None


def load():
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    import xgboost as xgb

    if not (os.path.isfile(_REG_PATH) and os.path.isfile(_CLF_PATH)):
        raise SystemExit("FATAL: watering models missing. Run: python train_watering.py")
    with open(_FEATURES_PATH, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    reg = xgb.XGBRegressor()
    reg.load_model(_REG_PATH)
    clf = xgb.XGBClassifier()
    clf.load_model(_CLF_PATH)
    _BUNDLE = (reg, clf, meta)
    return _BUNDLE


def default_weather(species: str) -> dict:
    p = PLANTS.get(species, PLANTS["Tomato"])
    lo, hi = p["temperature_c"]
    return {
        "temperature_c": round((lo + hi) / 2.0, 1),
        "humidity_pct": round((p["humidity_pct"][0] + p["humidity_pct"][1]) / 2.0, 1),
        "rainfall_probability_pct": 20.0,
        "rainfall_mm": 0.0,
        "sunlight_hours": float(p["sunlight_hours"]),
        "source": "plant profile default",
        "location": "not set",
    }


def estimate_moisture(species, stage, pot_l, age_days, soil_type, hours_since, weather):
    use = daily_use_l(species, stage, pot_l, age_days, weather["temperature_c"],
                      weather["humidity_pct"], weather["sunlight_hours"])
    avail = pot_l * SOILS[soil_type]["avail_frac"]
    loss_h = max(use / 24.0, 0.01) / max(avail, 0.05) * 100.0
    return float(np.clip(100.0 - loss_h * max(hours_since, 0.0), 5.0, 100.0))


def estimate_hours_since(species, stage, pot_l, age_days, soil_type, moisture, weather):
    use = daily_use_l(species, stage, pot_l, age_days, weather["temperature_c"],
                      weather["humidity_pct"], weather["sunlight_hours"])
    avail = pot_l * SOILS[soil_type]["avail_frac"]
    loss_h = max(use / 24.0, 0.01) / max(avail, 0.05) * 100.0
    return float(np.clip((100.0 - moisture) / max(loss_h, 0.05), 0.0, 168.0))


def water_amount_l(pot_size_l, moisture) -> float:
    """Litres to bring the pot back to field capacity (plus a small flush factor)."""
    need = pot_size_l * SOILS.get("loam", {"avail_frac": 0.2})["avail_frac"]
    litres = (100.0 - moisture) / 100.0 * need * 1.15
    return round(max(0.15, min(litres, pot_size_l * 0.35)), 2)


def recommend(species: str,
              growth_stage: str = "vegetative",
              pot_size_l: Optional[float] = None,
              plant_age_days: int = 45,
              soil_type: str = "loam",
              soil_moisture_pct: Optional[float] = None,
              hours_since_last_watering: Optional[float] = None,
              weather: Optional[dict] = None,
              city: Optional[str] = None,
              lat: Optional[float] = None,
              lon: Optional[float] = None) -> dict:
    reg, clf, meta = load()

    if species not in PLANTS:
        species = "Tomato"
    profile = PLANTS[species]
    pot_size_l = float(pot_size_l or profile["optimal_pot_l"])
    growth_stage = growth_stage if growth_stage in profile["growth_stages"] else "vegetative"
    soil_type = soil_type if soil_type in SOILS else "loam"

    # ---- weather: live if asked for, otherwise the plant's comfort zone
    if weather is None:
        from weather import weather_for_input
        weather = weather_for_input(city=city, lat=lat, lon=lon)
        if not weather or weather.get("error"):
            weather = default_weather(species)
    w = dict(weather)
    for k, v in default_weather(species).items():
        w.setdefault(k, v)

    # ---- fill the missing soil reading (only one of the two is usually known)
    if soil_moisture_pct is None and hours_since_last_watering is None:
        soil_moisture_pct = 65.0
    if soil_moisture_pct is None:
        soil_moisture_pct = estimate_moisture(species, growth_stage, pot_size_l, plant_age_days,
                                              soil_type, hours_since_last_watering, w)
    if hours_since_last_watering is None:
        hours_since_last_watering = estimate_hours_since(species, growth_stage, pot_size_l,
                                                         plant_age_days, soil_type,
                                                         soil_moisture_pct, w)

    row = pd.DataFrame([{
        "species": species if species in meta["categories"]["species"] else "Tomato",
        "growth_stage": growth_stage,
        "pot_size_l": float(pot_size_l),
        "plant_age_days": int(plant_age_days),
        "soil_type": soil_type,
        "soil_moisture_pct": float(soil_moisture_pct),
        "hours_since_last_watering": float(hours_since_last_watering),
        "temperature_c": float(w["temperature_c"]),
        "humidity_pct": float(w["humidity_pct"]),
        "rainfall_probability_pct": float(w["rainfall_probability_pct"]),
        "rainfall_mm": float(w["rainfall_mm"]),
        "sunlight_hours": float(w["sunlight_hours"]),
    }])
    X = feature_matrix(row)
    for col in meta["feature_names"]:
        if col not in X.columns:
            X[col] = 0.0
    X = X[meta["feature_names"]]

    hours = float(np.clip(reg.predict(X)[0], 0, 168))
    probs = clf.predict_proba(X)[0]
    classes = list(meta["label_classes"])
    best = classes[int(np.argmax(probs))]
    conf = float(np.max(probs))

    # the regressor can disagree with the classifier near the boundaries -> reconcile
    reg_action = ("water_now" if hours <= NOW_WATER_H else
                  "water_later" if hours <= LATER_WATER_H else "no_watering")
    action = best if best == reg_action else reg_action

    amount = water_amount_l(pot_size_l, soil_moisture_pct)
    typical = dry_down_hours(species, growth_stage, pot_size_l, plant_age_days, soil_type, w)
    if action == "no_watering" and w.get("rainfall_probability_pct", 0) >= 60:
        note = "Rain is expected today - skip watering unless the soil is already dry."
    elif action == "water_now":
        note = "Soil is drying out - water today, at the base, until it drains."
    elif action == "water_later":
        note = "Check again tomorrow morning; water within the next 2 days."
    else:
        note = "Moisture is adequate - no watering needed right now."

    return {
        "action": action,
        "action_confidence": round(conf, 4),
        "action_probs": {c: round(float(p), 4) for c, p in zip(classes, probs)},
        "hours_until_watering": round(hours, 1),
        "regressed_action": reg_action,
        "water_amount_l": amount,
        "next_check_hours": min(max(hours * 0.6, 4), 48),
        "soil_moisture_pct": round(float(soil_moisture_pct), 1),
        "hours_since_last_watering": round(float(hours_since_last_watering), 1),
        "weather": w,
        "note": note,
        "inputs": {
            "species": species, "growth_stage": growth_stage, "pot_size_l": pot_size_l,
            "plant_age_days": int(plant_age_days), "soil_type": soil_type,
        },
        "schedule_hint": schedule_hint(typical),
        "typical_dry_down_hours": typical,
    }


def dry_down_hours(species, stage, pot_l, age_days, soil_type, weather) -> float:
    """Hours to go from field capacity (100%) to the stress point (30%)."""
    use = daily_use_l(species, stage, pot_l, age_days, weather["temperature_c"],
                      weather["humidity_pct"], weather["sunlight_hours"])
    avail = pot_l * SOILS[soil_type]["avail_frac"]
    loss_h = max(use / 24.0, 0.01) / max(avail, 0.05) * 100.0
    return round((100.0 - 30.0) / max(loss_h, 0.05), 1)


def schedule_hint(hours: float) -> str:
    if hours >= 168:
        return "roughly once a week"
    days = max(hours, 1.0) / 24.0
    if days < 1.4:
        return f"about every {max(hours, 1.0):.0f} hours"
    if days < 7:
        return f"about every {days:.1f} days"
    return "about once a week"


def main():
    import argparse

    ap = argparse.ArgumentParser(description="XGBoost watering recommendation")
    ap.add_argument("--species", default="Tomato")
    ap.add_argument("--stage", default="fruiting")
    ap.add_argument("--pot", type=float, default=12)
    ap.add_argument("--age", type=int, default=60)
    ap.add_argument("--soil", default="loam", choices=list(SOILS))
    ap.add_argument("--moisture", type=float, default=None)
    ap.add_argument("--hours-since", type=float, default=None)
    ap.add_argument("--city", default=None, help="live weather for this city (Open-Meteo)")
    a = ap.parse_args()
    out = recommend(a.species, a.stage, a.pot, a.age, a.soil, a.moisture, a.hours_since,
                    city=a.city)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
