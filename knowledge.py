"""Treatment + care knowledge base for the 10 supported classes and plant profiles.

Everything the dashboard shows (status / treatment / care / prevention) comes from here.
Keys must match models/class_names.json exactly.
"""

TREATMENTS = {
    "Tomato___healthy": {
        "status": "Healthy",
        "severity": "none",
        "treatment": "No treatment needed.",
        "care": "Keep a steady watering schedule, feed every 2 weeks during fruiting and give 6-8 hours of sun.",
        "prevention": "Inspect leaves weekly and remove old yellow leaves early.",
        "urgent": False,
    },
    "Tomato___Early_blight": {
        "status": "Early Blight (Alternaria)",
        "severity": "moderate",
        "treatment": ("Remove infected lower leaves and bin them. Apply a copper-based or "
                      "chlorothalonil fungicide every 7-10 days, 2-3 sprays."),
        "care": ("Mulch to stop soil splashing, water the soil only (never the leaves) and "
                 "feed with a balanced fertiliser to keep the plant vigorous."),
        "prevention": "Rotate crops, space plants for airflow, pinch lower branches 25 cm above soil.",
        "urgent": False,
    },
    "Tomato___Late_blight": {
        "status": "Late Blight (Phytophthora)",
        "severity": "high",
        "treatment": ("Remove and destroy infected foliage immediately. Apply a labelled "
                      "fungicide (mefenoxam, chlorothalonil or copper) - repeat every 5-7 days."),
        "care": ("Stop overhead watering, improve airflow between plants and avoid working "
                 "with wet foliage. Isolate healthy plants nearby."),
        "prevention": "Use resistant varieties, keep foliage dry, remove volunteer plants.",
        "urgent": True,
    },
    "Tomato___Septoria_leaf_spot": {
        "status": "Septoria Leaf Spot",
        "severity": "moderate",
        "treatment": ("Remove spotted lower leaves and bin them. Apply copper or a "
                      "protectant fungicide every 7 days until new growth is clean."),
        "care": ("Water at the base only, mulch, and stake plants so leaves dry quickly. "
                 "Avoid high-nitrogen feeds that produce soft growth."),
        "prevention": "Rotate yearly, clean stakes and support ties, remove plant debris.",
        "urgent": False,
    },
    "Tomato___Leaf_Mold": {
        "status": "Leaf Mold (Passalora)",
        "severity": "moderate",
        "treatment": ("Remove badly affected leaves. Apply copper-based fungicide; improve "
                      "ventilation - humidity above 85% is the main trigger."),
        "care": ("Reduce watering frequency on leaves, open the canopy and keep greenhouse "
                 "vents open. Remove lower leaves touching the soil."),
        "prevention": "Resistant varieties, wide spacing, morning watering only.",
        "urgent": False,
    },
    "Potato___healthy": {
        "status": "Healthy",
        "severity": "none",
        "treatment": "No treatment needed.",
        "care": "Hill soil around stems, keep evenly moist and harvest when foliage yellows.",
        "prevention": "Inspect weekly and remove any yellowing leaves early.",
        "urgent": False,
    },
    "Potato___Early_blight": {
        "status": "Potato Early Blight",
        "severity": "moderate",
        "treatment": ("Remove affected leaves and apply a protectant fungicide "
                      "(chlorothalonil or copper) every 7-10 days."),
        "care": "Hill soil, avoid water stress and remove crop debris after harvest.",
        "prevention": "Rotate every 3 years, balance fertilisation, use certified seed.",
        "urgent": False,
    },
    "Potato___Late_blight": {
        "status": "Potato Late Blight",
        "severity": "high",
        "treatment": ("Remove and destroy infected foliage. Apply a blight fungicide "
                      "immediately and repeat at the label interval. Do not compost haulms."),
        "care": ("Keep foliage dry, irrigate only at the root and lift tubers as soon as "
                 "haulms die back to avoid rot."),
        "prevention": "Resistant varieties, wide spacing, destroy volunteer plants.",
        "urgent": True,
    },
    "Pepper,_bell___healthy": {
        "status": "Healthy",
        "severity": "none",
        "treatment": "No treatment needed.",
        "care": "Even moisture, warm position and a low-nitrogen feed once fruit sets.",
        "prevention": "Check leaf undersides weekly for mites and aphids.",
        "urgent": False,
    },
    "Pepper,_bell___Bacterial_spot": {
        "status": "Bacterial Spot (Xanthomonas)",
        "severity": "high",
        "treatment": ("Remove badly infected leaves. Apply fixed copper spray every 7-10 days; "
                      "avoid working on plants when wet."),
        "care": ("Keep foliage dry, space plants, use drip irrigation and rotate where they "
                 "are grown. Fruit lesions are not curable - pick affected fruit off."),
        "prevention": "Certified seed, clean tools, crop rotation, no overhead watering.",
        "urgent": True,
    },
}

# ---------------------------------------------------------------- plant profiles
# baseline_evapotranspiration_l_per_day is the reference water use of a mature plant
# in a 5 L pot at 25 C / 60% RH / 8 sun-hours; the simulator scales it by weather.
PLANTS = {
    "Tomato": {
        "class_prefix": "Tomato",
        "water_profile": "thirsty",
        "base_l_per_day": 0.85,
        "min_pot_l": 8,
        "optimal_pot_l": 15,
        "growth_stages": ["seedling", "vegetative", "flowering", "fruiting"],
        "stage_factor": {"seedling": 0.35, "vegetative": 0.75, "flowering": 1.0, "fruiting": 1.25},
        "sunlight_hours": 6,
        "temperature_c": [18, 30],
        "humidity_pct": [40, 80],
        "fertiliser_interval_days": 14,
        "notes": "Needs steady moisture - drought causes blossom end rot and split fruit.",
    },
    "Pepper": {
        "class_prefix": "Pepper,_bell",
        "water_profile": "medium",
        "base_l_per_day": 0.55,
        "min_pot_l": 5,
        "optimal_pot_l": 10,
        "growth_stages": ["seedling", "vegetative", "flowering", "fruiting"],
        "stage_factor": {"seedling": 0.3, "vegetative": 0.7, "flowering": 0.95, "fruiting": 1.15},
        "sunlight_hours": 6,
        "temperature_c": [18, 32],
        "humidity_pct": [40, 75],
        "fertiliser_interval_days": 14,
        "notes": "Dislikes waterlogging; let the top 2 cm dry between waterings.",
    },
    "Potato": {
        "class_prefix": "Potato",
        "water_profile": "medium",
        "base_l_per_day": 0.65,
        "min_pot_l": 10,
        "optimal_pot_l": 20,
        "growth_stages": ["seedling", "vegetative", "flowering", "fruiting"],
        "stage_factor": {"seedling": 0.4, "vegetative": 0.9, "flowering": 1.1, "fruiting": 1.0},
        "sunlight_hours": 6,
        "temperature_c": [15, 25],
        "humidity_pct": [50, 85],
        "fertiliser_interval_days": 21,
        "notes": "Keep even moisture during tuber bulking; drought reduces yield.",
    },
}


def species_from_class(class_name: str) -> str:
    prefix = class_name.split("___")[0]
    for name, prof in PLANTS.items():
        if prof["class_prefix"] == prefix:
            return name
    return prefix


def disease_info(class_name: str) -> dict:
    return TREATMENTS.get(class_name, {
        "status": class_name.replace("___", " - ").replace("_", " "),
        "severity": "unknown",
        "treatment": "Consult a local agriculture expert.",
        "care": "Monitor the plant and keep it well watered and stress free.",
        "prevention": "Inspect the plant regularly.",
        "urgent": False,
    })


def all_classes():
    return list(TREATMENTS.keys())
