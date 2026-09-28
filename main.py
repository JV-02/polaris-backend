from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, json, os
import numpy as np
import shap
from openai import OpenAI

app = FastAPI(title="POLARIS EMS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("rf_load_model.joblib")
with open("features.json") as f:
    FEATURES = json.load(f)

explainer = shap.TreeExplainer(model)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

GENERATOR_CAPACITY = 100  # kW
CRITICAL_FRACTION = 0.60


class StationInput(BaseModel):
    temp_c: float
    wind_speed_ms: float
    rh_pct: float = 70
    pressure_hpa: float = 980
    ghi_w_m2: float = 150
    hour: int = 12
    month: int = 7


def build_features(d: StationInput):
    return np.array([[
        d.temp_c,
        d.wind_speed_ms,
        d.rh_pct,
        d.pressure_hpa,
        d.ghi_w_m2,
        np.sin(2*np.pi*d.hour/24),
        np.cos(2*np.pi*d.hour/24),
        np.sin(2*np.pi*d.month/12),
        np.cos(2*np.pi*d.month/12),
        max(0, 20 - d.temp_c)
    ]])


def wind_power(speed):
    if speed < 3: return 0.0
    if speed < 12: return 30 * ((speed-3)/9)**3
    if speed <= 45: return 30.0
    return 0.0  # hard cut-out


def pv_power(ghi):
    return float(np.clip(40 * (ghi/1000), 0, 40))


@app.get("/")
def root():
    return {"status": "POLARIS EMS API online"}


@app.post("/ems")
def ems(data: StationInput):
    X = build_features(data)
    load = float(model.predict(X)[0])

    pv = pv_power(data.ghi_w_m2)
    wind = wind_power(data.wind_speed_ms)
    renewable = pv + wind

    net = max(0, load - renewable)
    diesel = min(net, GENERATOR_CAPACITY)
    loading = (diesel / GENERATOR_CAPACITY) * 100

    critical = load * CRITICAL_FRACTION
    flexible = load - critical
    available = renewable + GENERATOR_CAPACITY
    shed = max(0, load - available)
    critical_unserved = max(0, critical - available)

    # Risk
    if loading >= 80 or data.wind_speed_ms > 45 or data.temp_c <= -25 or critical_unserved > 0:
        risk = "High"
    elif loading >= 65 or data.wind_speed_ms >= 12:
        risk = "Moderate"
    else:
        risk = "Low"

    return {
        "kpis": {
            "predicted_load_kw": round(load, 2),
            "renewable_kw": round(renewable, 2),
            "pv_kw": round(pv, 2),
            "wind_kw": round(wind, 2),
            "diesel_kw": round(diesel, 2),
            "generator_loading_pct": round(loading, 1),
            "critical_load_kw": round(critical, 2),
            "flexible_load_kw": round(flexible, 2),
            "shed_kw": round(shed, 2),
            "risk": risk
        },
        "energy_flow": {
            "pv": round(pv, 2),
            "wind": round(wind, 2),
            "diesel": round(diesel, 2),
            "load": round(load, 2),
            "renewable": round(renewable, 2)
        }
    }


@app.post("/explain")
def explain(data: StationInput):
    X = build_features(data)
    sv = explainer.shap_values(X)
    pairs = sorted(
        [(FEATURES[i], float(sv[0][i])) for i in range(len(FEATURES))],
        key=lambda x: abs(x[1]),
        reverse=True
    )
    return {
        "prediction": float(model.predict(X)[0]),
        "top_factors": [{"feature": k, "impact": round(v, 3)} for k, v in pairs[:5]]
    }


@app.post("/recommend")
def recommend(data: StationInput):
    ems_data = ems(data)
    k = ems_data["kpis"]

    prompt = f"""
You are POLARIS, AI energy officer for an Antarctic research station.
Situation:
- Load: {k['predicted_load_kw']} kW
- Renewable: {k['renewable_kw']} kW (PV {k['pv_kw']}, Wind {k['wind_kw']})
- Diesel: {k['diesel_kw']} kW at {k['generator_loading_pct']}% loading
- Temp: {data.temp_c} C, Wind: {data.wind_speed_ms} m/s
- Risk: {k['risk']}
- Shed needed: {k['shed_kw']} kW

Give 3 short operational recommendations for the station engineer.
"""

    if not os.getenv("OPENAI_API_KEY"):
        return {"recommendations": [
            f"Dispatch diesel near {k['diesel_kw']} kW for net demand.",
            "Keep critical loads (heating/comms/life-support) protected.",
            "If wind > 45 m/s, wind power drops to 0 — prepare diesel."
        ]}

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return {"recommendations": resp.choices[0].message.content}