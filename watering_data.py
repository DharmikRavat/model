"""Generate a labelled watering dataset with a physical soil water-balance simulator.

There is no public tabular dataset for "hours until a potted plant needs water", so
ground truth comes from a simple FAO-56-style bucket model:

  daily use  = ET0(weather) * Kc(growth stage) * pot geometry * plant age factor
  rain input = rainfall_mm * P(rain) * catchment efficiency
  soil       = pot available-water volume, drained at the plant's daily use

Each sample simulates 168 hours forward and labels the first hour at which soil
moisture crosses the stress point.

Run:  python watering_data.py          -> models/watering_dataset.csv
"""
import math
import os
import random

import pandas as pd

import config
from knowledge import PLANTS

SEED = 42
N = 12000
HORIZON_H = 168
STRESS_MOISTURE = 30.0     # % of available water -> plant starts to suffer
WATER_TO = 100.0
NOW_WATER_H = 12.0         # <= 12 h until stress -> water now
LATER_WATER_H = 48.0       # <= 48 h -> water later, otherwise no watering needed

SOILS = {
    # available water as a fraction of pot volume, drainage coefficient
    "sandy": {"avail_frac": 0.14, "drain": 1.35},
    "loam": {"avail_frac": 0.20, "drain": 1.00},
    "clay": {"avail_frac": 0.26, "drain": 0.75},
}
POT_EVAP_FACTOR = 6.0      # soil-surface evaporation + edge effects of a container


def et0(temperature_c, humidity_pct, sunlight_hours):
    """Hargreaves reference evapotranspiration (mm/day)."""
    sun = max(sunlight_hours, 0.5)
    ra = 10.0 + 0.9 * sun                                  # extraterrestrial radiation MJ/m2/day
    diurnal = 4.0 + 1.1 * sun                              # day-night range on clear days
    base = 0.0023 * (temperature_c + 17.8) * math.sqrt(diurnal) * ra
    humidity_adj = 1.0 + (45.0 - humidity_pct) / 400.0     # dry air evaporates a bit more
    heat_adj = 1.0 + max(temperature_c - 25.0, 0) * 0.02
    return max(0.5, base * humidity_adj * heat_adj)


def pot_area_m2(pot_l):
    """Cylinder, height = 1.2 * diameter. Pot volume is litres (dm3) -> area in m2."""
    d_dm = (pot_l / 0.9426) ** (1.0 / 3.0)
    return math.pi * (d_dm / 10.0) ** 2 / 4.0, d_dm / 10.0


def daily_use_l(species, stage, pot_l, age_days, temperature_c, humidity_pct, sunlight_hours):
    p = PLANTS[species]
    area, _ = pot_area_m2(pot_l)
    kc = p["stage_factor"][stage]
    age_f = min(1.0, 0.35 + 0.65 * min(age_days, 75) / 75.0)
    size_f = min(1.15, 0.75 + 0.25 * pot_l / max(p["optimal_pot_l"], 1))
    use = et0(temperature_c, humidity_pct, sunlight_hours) * kc * area * 1000.0 * POT_EVAP_FACTOR
    return use * age_f * size_f / 1000.0                     # litres/day


def hourly_shape(hour, sunlight_hours):
    """Fraction of daily use lost in each hour (daytime peak)."""
    day_start = max(6, 12 - int(sunlight_hours / 2) - 1)
    day_end = min(21, day_start + int(sunlight_hours) + 2)
    if day_start <= hour <= day_end:
        peak = 1.0 + 0.35 * math.sin((hour - day_start) / max(day_end - day_start, 1) * math.pi)
        return 1.6 * peak
    return 0.15


def simulate(row):
    """Return hours until soil moisture crosses the stress point (capped at horizon)."""
    total_avail_l = row["pot_size_l"] * SOILS[row["soil_type"]]["avail_frac"]
    drain = SOILS[row["soil_type"]]["drain"]
    use_l_day = daily_use_l(row["species"], row["growth_stage"], row["pot_size_l"],
                            row["plant_age_days"], row["temperature_c"],
                            row["humidity_pct"], row["sunlight_hours"]) * drain
    loss_per_h = use_l_day / 24.0
    moisture = row["soil_moisture_pct"]
    rain = row["rainfall_probability_pct"] / 100.0
    rain_mm = row["rainfall_mm"]
    rng = random.Random(row["_seed"])

    for h in range(1, HORIZON_H + 1):
        moisture -= (loss_per_h / max(total_avail_l, 0.05)) * 100.0 * hourly_shape(h % 24,
                                                                                  row["sunlight_hours"])
        if rng.random() < rain * 0.09:                       # ~one chance per rainy day-hour
            area, _ = pot_area_m2(row["pot_size_l"])
            moisture += (rain_mm * 0.6 * area) / max(total_avail_l, 0.05) * 100.0
        if moisture <= STRESS_MOISTURE:
            return float(h)
    return float(HORIZON_H)


def label_for(hours):
    if hours <= NOW_WATER_H:
        return "water_now"
    if hours <= LATER_WATER_H:
        return "water_later"
    return "no_watering"


def build(n=N, seed=SEED):
    rng = random.Random(seed)
    rows = []
    species = list(PLANTS.keys())
    for i in range(n):
        sp = rng.choice(species)
        prof = PLANTS[sp]
        stage = rng.choice(prof["growth_stages"])
        pot = rng.uniform(prof["min_pot_l"], prof["optimal_pot_l"] * 1.6)
        temp = rng.uniform(prof["temperature_c"][0] - 3, prof["temperature_c"][1] + 4)
        hum = rng.uniform(25, 95)
        sun = rng.uniform(2, 12)
        moisture = rng.uniform(35, 100)
        soil = rng.choice(list(SOILS))
        loss_est = daily_use_l(sp, stage, pot, rng.uniform(1, 120), temp, hum, sun)
        avail = pot * SOILS[soil]["avail_frac"]
        avg_loss_h = max(loss_est / 24.0, 0.01) / max(avail, 0.05) * 100.0
        hours_since = min(168.0, (100.0 - moisture) / max(avg_loss_h, 0.05))
        row = {
            "species": sp,
            "growth_stage": stage,
            "pot_size_l": round(pot, 1),
            "plant_age_days": int(rng.uniform(1, 150)),
            "soil_type": soil,
            "soil_moisture_pct": round(moisture, 1),
            "hours_since_last_watering": round(hours_since, 1),
            "temperature_c": round(temp, 1),
            "humidity_pct": round(hum, 1),
            "rainfall_probability_pct": round(rng.uniform(0, 90), 1),
            "rainfall_mm": round(rng.uniform(0, 25), 1),
            "sunlight_hours": round(sun, 1),
            "_seed": rng.randrange(10 ** 9),
        }
        hrs = simulate(row)
        row.pop("_seed")
        row["hours_until_watering"] = hrs
        row["action"] = label_for(hrs)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    config.ensure_dirs()
    df = build()
    path = os.path.join(config.MODELS_DIR, "watering_dataset.csv")
    df.to_csv(path, index=False)
    print(f"{len(df)} samples -> {path}")
    print(df["action"].value_counts().to_string())
    print("\nhours_until_watering:")
    print(df["hours_until_watering"].describe().to_string())
    corr = df[["hours_until_watering", "soil_moisture_pct", "temperature_c",
               "humidity_pct", "sunlight_hours", "pot_size_l"]].corr()["hours_until_watering"]
    print("\ncorrelation with target:")
    print(corr.round(3).to_string())


if __name__ == "__main__":
    main()
