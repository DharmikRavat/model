"""FastAPI backend for the AI Urban Farming Assistant.

Run:  uvicorn api:app --port 8000     (from this folder)
Docs: http://127.0.0.1:8000/docs

Endpoints
  GET  /api/health
  POST /api/diagnose            multipart image (+ optional plant_id) -> diagnosis + treatment + Grad-CAM
  POST /api/gradcam             multipart image -> base64 Grad-CAM only
  POST /api/watering            JSON -> XGBoost watering recommendation (+ live weather)
  GET  /api/weather?city=...    live weather (Open-Meteo)
  GET  /api/knowledge           treatment knowledge base
  GET|POST /api/plants          plant tracker
  GET|DELETE /api/plants/{id}
  POST /api/plants/{id}/water   mark as watered

Short aliases are also available at /predict, /gradcam, /watering, and /plants.
"""
import base64
import io
import json
import os
import threading
import time
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from knowledge import TREATMENTS, disease_info, species_from_class
from predict_b2 import gradcam as gradcam_fn
from predict_b2 import load_bundle, predict

PLANTS_PATH = os.path.join(config.MODELS_DIR, "plants.json")
_STORE_LOCK = threading.Lock()

app = FastAPI(title="AI Urban Farming Assistant", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- store
def _load_plants() -> dict:
    if not os.path.isfile(PLANTS_PATH):
        return {}
    with open(PLANTS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_plants(data: dict) -> None:
    os.makedirs(os.path.dirname(PLANTS_PATH), exist_ok=True)
    with open(PLANTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- models
class PlantIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)
    species: str = "Tomato"
    growth_stage: str = "vegetative"
    pot_size_l: float = Field(12.0, gt=0, le=500)
    plant_age_days: int = Field(30, ge=0, le=2000)
    soil_type: str = "loam"
    city: Optional[str] = None
    notes: str = ""


class WateringIn(BaseModel):
    species: Optional[str] = None
    growth_stage: Optional[str] = None
    pot_size_l: Optional[float] = None
    plant_age_days: Optional[int] = None
    soil_type: Optional[str] = None
    soil_moisture_pct: Optional[float] = None
    hours_since_last_watering: Optional[float] = None
    city: Optional[str] = None
    plant_id: Optional[str] = None


# ---------------------------------------------------------------- startup
@app.on_event("startup")
def _startup():
    try:
        load_bundle()
        print("[startup] EfficientNet-B2 disease model loaded")
    except Exception as e:  # noqa: BLE001
        print(f"[startup] disease model NOT loaded: {e}")
    try:
        from watering import load as load_watering
        load_watering()
        print("[startup] XGBoost watering models loaded")
    except Exception as e:  # noqa: BLE001
        print(f"[startup] watering models NOT loaded: {e}")


@app.get("/api/health")
def health():
    disease_ok = os.path.isfile(os.path.join(config.MODELS_DIR, "plant_disease_b2.pt"))
    water_ok = os.path.isfile(os.path.join(config.MODELS_DIR, "watering_regressor.json"))
    return {"status": "ok", "disease_model": disease_ok, "watering_model": water_ok,
            "classes": list(TREATMENTS.keys()), "time": _now()}


@app.get("/api/knowledge")
def knowledge():
    return {"treatments": TREATMENTS}


# ---------------------------------------------------------------- diagnose
def _b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/api/diagnose")
async def diagnose(file: UploadFile = File(...), plant_id: Optional[str] = Form(None),
                   cam: bool = Form(True)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"cannot read image: {e}")

    started = time.time()
    res = predict(img)
    res["image_name"] = file.filename or "upload"

    payload = {
        "prediction": res["prediction"],
        "confidence": res["confidence"],
        "top_predictions": res["top_predictions"],
        "health_status": res["health_status"],
        "image_quality": res["quality"],
        "quality_issues": res["quality_issues"],
        "plant_detected": res["plant_detected"],
        "plant_species": species_from_class(res["prediction"]) if res["prediction"] else None,
        "status": res["status"],
        "message": res["message"],
        "treatment": disease_info(res["prediction"]) if res["prediction"] else None,
        "gradcam": None,
        "elapsed_sec": round(time.time() - started, 3),
    }

    if cam and res["plant_detected"] and res["prediction"]:
        try:
            classes = load_bundle()[1]["class_names"]
            idx = classes.index(res["prediction"])
            overlay, _ = gradcam_fn(img, class_index=idx)
            payload["gradcam"] = _b64(overlay)
        except Exception as e:  # noqa: BLE001
            payload["gradcam_error"] = str(e)

    if plant_id:
        with _STORE_LOCK:
            plants = _load_plants()
            if plant_id in plants:
                plants[plant_id].setdefault("health_log", []).append({
                    "at": _now(),
                    "prediction": res["prediction"],
                    "confidence": res["confidence"],
                    "health_status": res["health_status"],
                    "quality": res["quality"],
                })
                plants[plant_id]["health_status"] = (
                    "HEALTHY" if res["health_status"] == "HEALTHY" else "ATTENTION")
                _save_plants(plants)
                payload["logged_for"] = plants[plant_id]["name"]
    return payload


@app.post("/api/gradcam")
async def gradcam_endpoint(file: UploadFile = File(...)):
    from PIL import Image

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    res = predict(img)
    if not res["plant_detected"] or not res["prediction"]:
        raise HTTPException(422, res["message"])
    classes = load_bundle()[1]["class_names"]
    idx = classes.index(res["prediction"])
    overlay, heat = gradcam_fn(img, class_index=idx)
    return {"gradcam": _b64(overlay), "prediction": res["prediction"],
            "confidence": res["confidence"],
            "heatmap_max": round(float(heat.max()), 4)}


# ---------------------------------------------------------------- watering
@app.post("/api/watering")
def watering_endpoint(body: WateringIn):
    from watering import recommend

    plant = None
    if body.plant_id:
        plants = _load_plants()
        plant = plants.get(body.plant_id)
        if not plant:
            raise HTTPException(404, "plant not found")
    src = plant or {}
    out = recommend(
        species=body.species or src.get("species") or "Tomato",
        growth_stage=body.growth_stage or src.get("growth_stage") or "vegetative",
        pot_size_l=body.pot_size_l or src.get("pot_size_l"),
        plant_age_days=body.plant_age_days or src.get("plant_age_days") or 30,
        soil_type=body.soil_type or src.get("soil_type") or "loam",
        soil_moisture_pct=body.soil_moisture_pct,
        hours_since_last_watering=body.hours_since_last_watering,
        city=body.city or src.get("city"),
    )
    if plant and plant.get("last_watered_at"):
        last = datetime.fromisoformat(plant["last_watered_at"])
        out["hours_since_last_watering"] = round(
            (datetime.now() - last).total_seconds() / 3600.0, 1)
    return out


@app.get("/api/weather")
def weather_endpoint(city: str):
    from weather import weather_for_city

    try:
        return weather_for_city(city)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, str(e))


# ---------------------------------------------------------------- plants
@app.get("/api/plants")
def list_plants():
    plants = _load_plants()
    out = []
    from watering import recommend

    for pid, p in plants.items():
        entry = dict(p, id=pid)
        try:
            rec = recommend(species=p.get("species", "Tomato"),
                            growth_stage=p.get("growth_stage", "vegetative"),
                            pot_size_l=p.get("pot_size_l"),
                            plant_age_days=p.get("plant_age_days", 30),
                            soil_type=p.get("soil_type", "loam"),
                            city=p.get("city"))
            entry["watering"] = {"action": rec["action"],
                                 "hours_until_watering": rec["hours_until_watering"],
                                 "schedule_hint": rec["schedule_hint"]}
        except Exception as e:  # noqa: BLE001
            entry["watering"] = {"error": str(e)}
        out.append(entry)
    return sorted(out, key=lambda x: x.get("added_at", ""), reverse=True)


@app.post("/api/plants", status_code=201)
def add_plant(body: PlantIn):
    if body.species not in ("Tomato", "Pepper", "Potato"):
        raise HTTPException(400, "species must be one of Tomato, Pepper, Potato")
    pid = uuid.uuid4().hex[:12]
    with _STORE_LOCK:
        plants = _load_plants()
        plants[pid] = {
            "name": body.name,
            "species": body.species,
            "growth_stage": body.growth_stage,
            "pot_size_l": body.pot_size_l,
            "plant_age_days": body.plant_age_days,
            "soil_type": body.soil_type,
            "city": body.city,
            "notes": body.notes,
            "health_status": "UNKNOWN",
            "health_log": [],
            "watered_log": [],
            "last_watered_at": None,
            "added_at": _now(),
        }
        _save_plants(plants)
    return {"id": pid, **plants[pid]}


@app.get("/api/plants/{pid}")
def get_plant(pid: str):
    plants = _load_plants()
    if pid not in plants:
        raise HTTPException(404, "plant not found")
    return {"id": pid, **plants[pid]}


@app.delete("/api/plants/{pid}")
def delete_plant(pid: str):
    with _STORE_LOCK:
        plants = _load_plants()
        if pid not in plants:
            raise HTTPException(404, "plant not found")
        plants.pop(pid)
        _save_plants(plants)
    return {"deleted": pid}


@app.post("/api/plants/{pid}/water")
def water_plant(pid: str):
    with _STORE_LOCK:
        plants = _load_plants()
        if pid not in plants:
            raise HTTPException(404, "plant not found")
        plants[pid]["last_watered_at"] = _now()
        plants[pid].setdefault("watered_log", []).append(_now())
        _save_plants(plants)
    return {"id": pid, "last_watered_at": plants[pid]["last_watered_at"]}


@app.post("/api/plants/{pid}/stage")
def update_stage(pid: str, stage: str):
    with _STORE_LOCK:
        plants = _load_plants()
        if pid not in plants:
            raise HTTPException(404, "plant not found")
        plants[pid]["growth_stage"] = stage
        _save_plants(plants)
    return {"id": pid, "growth_stage": stage}


app.add_api_route("/predict", diagnose, methods=["POST"], name="predict_alias")
app.add_api_route("/api/predict", diagnose, methods=["POST"], name="api_predict_alias")
app.add_api_route("/gradcam", gradcam_endpoint, methods=["POST"], name="gradcam_alias")
app.add_api_route("/watering", watering_endpoint, methods=["POST"], name="watering_alias")
app.add_api_route("/plants", list_plants, methods=["GET"], name="plants_list_alias")
app.add_api_route("/plants", add_plant, methods=["POST"], name="plants_add_alias")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
