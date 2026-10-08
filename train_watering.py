"""Train the XGBoost watering models (regression + action classification).

Run:  python train_watering.py
Reads : models/watering_dataset.csv   (from watering_data.py)
Writes: models/watering_regressor.json, models/watering_classifier.json,
        models/watering_features.json, outputs/watering_report.txt
"""
import json
import os

import numpy as np
import pandas as pd

import config
from watering_data import STRESS_MOISTURE  # noqa: F401

CATEGORICALS = ["species", "growth_stage", "soil_type"]
NUMERICS = ["pot_size_l", "plant_age_days", "soil_moisture_pct", "hours_since_last_watering",
            "temperature_c", "humidity_pct", "rainfall_probability_pct", "rainfall_mm",
            "sunlight_hours"]
CATEGORIES = {
    "species": ["Tomato", "Pepper", "Potato"],
    "growth_stage": ["seedling", "vegetative", "flowering", "fruiting"],
    "soil_type": ["sandy", "loam", "clay"],
}
PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=4,
)


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    parts = [df[NUMERICS].astype(float)]
    for col in CATEGORICALS:
        one = pd.DataFrame(
            {f"{col}__{v}": (df[col] == v).astype(float) for v in CATEGORIES[col]},
            index=df.index)
        parts.append(one)
    return pd.concat(parts, axis=1)


def main():
    import xgboost as xgb
    from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                                 mean_squared_error, r2_score)
    from sklearn.model_selection import cross_val_score, train_test_split

    config.ensure_dirs()
    path = os.path.join(config.MODELS_DIR, "watering_dataset.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"FATAL: {path} missing. Run: python watering_data.py")

    df = pd.read_csv(path)
    X = feature_matrix(df)
    y_reg = df["hours_until_watering"].astype(float)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_cls = le.fit_transform(df["action"].astype(str))

    X_tr, X_te, y_tr, y_te, yc_tr, yc_te = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

    # ------------------------------------------------------------- regressor
    reg = xgb.XGBRegressor(**PARAMS)
    reg.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    pred = np.clip(reg.predict(X_te), 0, 168)
    reg_metrics = {
        "r2": round(float(r2_score(y_te, pred)), 4),
        "mae_hours": round(float(mean_absolute_error(y_te, pred)), 2),
        "rmse_hours": round(float(np.sqrt(mean_squared_error(y_te, pred))), 2),
        "cv_r2": [round(float(v), 4) for v in cross_val_score(
            xgb.XGBRegressor(**PARAMS), X, y_reg, cv=5, scoring="r2", n_jobs=4)],
    }

    # ------------------------------------------------------------- classifier
    clf_params = dict(PARAMS)
    clf_params["objective"] = "multi:softprob"
    clf_params["eval_metric"] = "mlogloss"
    clf = xgb.XGBClassifier(**clf_params, num_class=len(le.classes_))
    clf.fit(X_tr, yc_tr, eval_set=[(X_te, yc_te)], verbose=False)
    cpred = clf.predict(X_te)
    clf_metrics = {
        "accuracy": round(float(accuracy_score(yc_te, cpred)), 4),
        "f1_macro": round(float(f1_score(yc_te, cpred, average="macro")), 4),
        "f1_weighted": round(float(f1_score(yc_te, cpred, average="weighted")), 4),
        "cv_accuracy": [round(float(v), 4) for v in cross_val_score(
            xgb.XGBClassifier(**clf_params, num_class=len(le.classes_)), X, y_cls, cv=5,
            scoring="accuracy", n_jobs=4)],
        "classes": list(le.classes_),
    }

    # ------------------------------------------------------------- save
    reg.save_model(os.path.join(config.MODELS_DIR, "watering_regressor.json"))
    clf.save_model(os.path.join(config.MODELS_DIR, "watering_classifier.json"))
    features = {
        "numeric": NUMERICS,
        "categoricals": CATEGORICALS,
        "categories": CATEGORIES,
        "feature_names": list(X.columns),
        "params": {k: v for k, v in PARAMS.items()},
        "regressor_metrics": reg_metrics,
        "classifier_metrics": clf_metrics,
        "n_samples": int(len(df)),
        "label_classes": list(le.classes_),
        "target": ("hours_until_watering (0-168) ; action in "
                   "water_now<=12h, water_later<=48h, no_watering>48h"),
    }
    with open(os.path.join(config.MODELS_DIR, "watering_features.json"), "w", encoding="utf-8") as fh:
        json.dump(features, fh, indent=2)

    report = [
        "WATERING XGBOOST REPORT", "=" * 46,
        f"samples           : {len(df)}",
        f"features          : {X.shape[1]}",
        "", "REGRESSOR (hours until watering)",
        f"  R2              : {reg_metrics['r2']}",
        f"  MAE (hours)     : {reg_metrics['mae_hours']}",
        f"  RMSE (hours)    : {reg_metrics['rmse_hours']}",
        f"  5-fold R2       : {reg_metrics['cv_r2']}",
        "", "CLASSIFIER (water_now / water_later / no_watering)",
        f"  accuracy        : {clf_metrics['accuracy']}",
        f"  macro F1        : {clf_metrics['f1_macro']}",
        f"  weighted F1     : {clf_metrics['f1_weighted']}",
        f"  5-fold accuracy : {clf_metrics['cv_accuracy']}",
        "", "action distribution:",
        df["action"].value_counts().to_string(),
    ]
    txt = "\n".join(report)
    with open(os.path.join(config.OUTPUTS_DIR, "watering_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
