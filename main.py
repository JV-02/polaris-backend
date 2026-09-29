import json, math, os
import numpy as np
import pandas as pd
import joblib
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Polaris EMS API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = joblib.load("rf_load_model.joblib")
FEATURES = json.load(open("features.json"))

GENERATOR_CAPACITY_KW = 150
SFC_POINTS = {40: 0.36, 60: 0.33, 85: 0.285, 95: 0.30, 100: 0.34}

def sfc_curve(load_pct: float) -> float:
    keys = np.array(list(SFC_POINTS.keys())); vals = np.array(list(SFC_POINTS.values()))
    return float(np.interp(np.clip(load_pct, 40, 100), keys, vals))

def wind_power_kw(speed_ms: float, rated_kw: float = 30) -> float:
    CUT_IN, RATED, CUT_OUT = 3.0, 12.0, 25.0
    if speed_ms < CUT_IN or speed_ms >= CUT_OUT: return 0.0
    if speed_ms < RATED: return rated_kw * ((speed_ms - CUT_IN) / (RATED - CUT_IN)) ** 3
    return rated_kw

def solar_power_kw(ghi_w_m2: float, rated_kwp: float = 40) -> float:
    return max(0.0, rated_kwp * (ghi_w_m2 / 1000.0))

def build_features(temp_c, wind_speed_ms, rh_pct, pressure_hpa, ghi_w_m2, hour, month) -> dict:
    return {
        "temp_c": temp_c, "wind_speed_ms": wind_speed_ms, "rh_pct": rh_pct,
        "pressure_hpa": pressure_hpa, "ghi_est_w_m2": ghi_w_m2,
        "hour_sin": math.sin(2*math.pi*hour/24), "hour_cos": math.cos(2*math.pi*hour/24),
        "month_sin": math.sin(2*math.pi*month/12), "month_cos": math.cos(2*math.pi*month/12),
        "heating_index": max(0.0, 18.0 - temp_c),
    }

class ConditionsIn(BaseModel):
    temp_c: float; wind_speed_ms: float; rh_pct: float
    pressure_hpa: float; ghi_w_m2: float; hour: int; month: int

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/api/dashboard")
def dashboard(inp: ConditionsIn):
    feat = build_features(inp.temp_c, inp.wind_speed_ms, inp.rh_pct,
                           inp.pressure_hpa, inp.ghi_w_m2, inp.hour, inp.month)
    row = pd.DataFrame([feat])[FEATURES]
    load_kw = float(model.predict(row)[0])

    wind_kw = wind_power_kw(inp.wind_speed_ms)
    solar_kw = solar_power_kw(inp.ghi_w_m2)
    renewable_kw = wind_kw + solar_kw
    diesel_kw = max(0.0, load_kw - renewable_kw)

    loading_pct = min(100.0, (diesel_kw / GENERATOR_CAPACITY_KW) * 100)
    optimized_liters = diesel_kw * sfc_curve(loading_pct) if diesel_kw > 0 else 0.0
    baseline_liters = load_kw * sfc_curve(min(100, (load_kw/GENERATOR_CAPACITY_KW)*100))
    fuel_saved_pct = (100*(baseline_liters-optimized_liters)/baseline_liters) if baseline_liters>0 else 0.0

    forecast_24h = []
    for h in range(24):
        f = build_features(inp.temp_c, inp.wind_speed_ms, inp.rh_pct,
                            inp.pressure_hpa, inp.ghi_w_m2, h, inp.month)
        r = pd.DataFrame([f])[FEATURES]
        l = float(model.predict(r)[0])
        forecast_24h.append({"hour": h, "load_kw": round(l,1), "renewable_kw": round(renewable_kw,1)})

    alerts = []
    if inp.wind_speed_ms >= 20: alerts.append({"level":"high","message":"High wind — turbines may cut out above 25 m/s."})
    if inp.temp_c <= -25: alerts.append({"level":"medium","message":"Extreme cold — heating load elevated."})
    if diesel_kw >= GENERATOR_CAPACITY_KW * 0.9: alerts.append({"level":"high","message":"Generator loading above 90% — near capacity."})
    if not alerts: alerts.append({"level":"info","message":"All systems nominal."})

    return {
        "kpis": {"total_load_kw": round(load_kw,1), "renewable_kw": round(renewable_kw,1),
                 "diesel_kw": round(diesel_kw,1),
                 "renewable_pct": round(100*renewable_kw/load_kw,1) if load_kw>0 else 0,
                 "fuel_saved_pct": round(fuel_saved_pct,1)},
        "generation_mix": {"solar_kw": round(solar_kw,1), "wind_kw": round(wind_kw,1), "diesel_kw": round(diesel_kw,1)},
        "forecast_24h": forecast_24h,
        "fuel_comparison": {"baseline_liters": round(baseline_liters,2), "optimized_liters": round(optimized_liters,2), "saved_pct": round(fuel_saved_pct,1)},
        "alerts": alerts,
    }

# ---- AI Insight (Groq, free tier — https://console.groq.com) ----
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

class InsightIn(BaseModel):
    kpis: dict
    alerts: list

@app.post("/api/ai-insight")
def ai_insight(inp: InsightIn):
    prompt = (f"Station load is {inp.kpis['total_load_kw']} kW, renewables covering "
              f"{inp.kpis['renewable_pct']}%, diesel at {inp.kpis['diesel_kw']} kW. "
              f"Alerts: {', '.join(a['message'] for a in inp.alerts)}. "
              "In 2 short sentences, tell the station operator what to do next.")
    if GROQ_API_KEY:
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={"model": "llama-3.1-8b-instant", "messages": [{"role":"user","content":prompt}],
                      "max_tokens": 120, "temperature": 0.4}, timeout=8)
            if r.status_code == 200:
                return {"insight": r.json()["choices"][0]["message"]["content"].strip()}
        except Exception:
            pass
    # Fallback — never breaks the demo if no key / call fails
    if inp.kpis["renewable_pct"] >= 70:
        return {"insight": "Renewables are covering most of the load. Good window to top up the battery."}
    if inp.kpis["diesel_kw"] > 100:
        return {"insight": "Diesel load is high. Consider shedding non-critical loads if this persists."}
    return {"insight": "Station operating within normal parameters. No action required."}