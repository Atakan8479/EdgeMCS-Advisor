# api/app.py
from __future__ import annotations

import json, math, logging, os, time, uuid, sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ------------------------- Paths / Globals -------------------------

APP_DIR       = Path(__file__).parent
PROJECT_ROOT  = APP_DIR.parent
MODELS_DIR    = PROJECT_ROOT / "models"
MODEL_PATH    = MODELS_DIR / "best_model.joblib"
SCHEMA_PATH   = MODELS_DIR / "schema.json"

ADVISOR_LOG   = os.getenv("ADVISOR_LOG", str(PROJECT_ROOT / "advisor_logs.jsonl"))
DB_PATH       = os.getenv("ADVISOR_DB", str(PROJECT_ROOT / "advisor.db"))

# Policy thresholds
EXTREME_BLER      = 0.50
MCS_SAFE_FALLBACK = "16QAM"
MCS_WEIGHTS       = {"QPSK":1.0, "16QAM":2.0, "64QAM":3.0, "256QAM":4.0}

# ------------------------- FastAPI app -------------------------

app = FastAPI(title="MCS_CWD0 Predictor", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5001", "http://localhost:5001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------- Load schema & model -------------------------

schema: Dict[str, Any] = {"features": {"numeric": [], "categorical": []}, "target": "MCS_CWD0", "classes": []}
if SCHEMA_PATH.exists():
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read schema.json: {e}") from e

model = None
try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded: %s", MODEL_PATH)
    else:
        logger.warning("Model file not found: %s (API will run without a model)", MODEL_PATH)
except Exception:
    logger.exception("Failed to load model; continuing without it")
    model = None

# ------------------------- SQLite persistence -------------------------

_conn = None
def _db():
    """singleton SQLite connection with FK and WAL"""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.execute("PRAGMA foreign_keys=ON;")
        _conn.execute("PRAGMA journal_mode=WAL;")
        _conn.execute("PRAGMA synchronous=NORMAL;")
    return _conn

def _ensure_tables():
    con = _db()
    # 1) decisions (1 row per recommendation)
    con.execute("""
    CREATE TABLE IF NOT EXISTS decisions (
        decision_id TEXT PRIMARY KEY,
        ts REAL NOT NULL,
        mcs TEXT,
        confidence REAL,
        risk TEXT,
        bitrate_hint REAL,
        packet_size INTEGER,
        explored INTEGER,
        traffic_type TEXT,
        features TEXT NOT NULL,   -- JSON snapshot
        qos TEXT,                 -- JSON snapshot
        hosts TEXT                -- JSON snapshot (array)
    );
    """)
    # 2) mec_hosts (0–1 row per decision; separate table keeps decisions lean)
    con.execute("""
    CREATE TABLE IF NOT EXISTS mec_hosts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_id TEXT NOT NULL,
        name TEXT,
        rtt_ms REAL,
        cpu_util REAL,
        queue_len REAL,
        FOREIGN KEY(decision_id) REFERENCES decisions(decision_id) ON DELETE CASCADE
    );
    """)
    # 3) class_probabilities (exactly 1 row per decision; fixed 4-class columns)
    con.execute("""
    CREATE TABLE IF NOT EXISTS class_probabilities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_id TEXT NOT NULL,
        prob_qpsk REAL,
        prob_16qam REAL,
        prob_64qam REAL,
        prob_256qam REAL,
        FOREIGN KEY(decision_id) REFERENCES decisions(decision_id) ON DELETE CASCADE
    );
    """)
    con.commit()

def _json_dumps(obj) -> str:
    try:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return json.dumps(_to_jsonable(obj), separators=(",", ":"), ensure_ascii=False)

def save_decision_row(
    decision_id: str,
    features: dict,
    qos: dict | None,
    hosts: list | None,
    traffic_type: Optional[str],
    mcs: str,
    confidence: float,
    risk: str,
    bitrate_hint: float,
    packet_size: int,
    explored: bool
):
    con = _db()
    con.execute(
        """
        INSERT OR REPLACE INTO decisions (
            decision_id, ts, mcs, confidence, risk, bitrate_hint, packet_size, explored,
            traffic_type, features, qos, hosts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            decision_id, time.time(), mcs, float(confidence), risk,
            float(bitrate_hint), int(packet_size), 1 if explored else 0,
            traffic_type or None,
            _json_dumps(features),
            _json_dumps(qos or {}),
            _json_dumps(hosts or [])
        )
    )
    con.commit()

def save_mec_host_row(decision_id: str, mec_host: Optional[dict]):
    if not mec_host:
        return
    con = _db()
    con.execute(
        """
        INSERT INTO mec_hosts (decision_id, name, rtt_ms, cpu_util, queue_len)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            decision_id,
            mec_host.get("name"),
            _safe_float(mec_host.get("rtt_ms")),
            _safe_float(mec_host.get("cpu_util")),
            _safe_float(mec_host.get("queue_len")),
        )
    )
    con.commit()

def save_class_prob_row(decision_id: str, pmap: Dict[str, float]):
    # Map to fixed columns; missing keys → NULL
    def g(key): return float(pmap[key]) if key in pmap else None
    con = _db()
    con.execute(
        """
        INSERT INTO class_probabilities (decision_id, prob_qpsk, prob_16qam, prob_64qam, prob_256qam)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            decision_id,
            g("QPSK") if "QPSK" in pmap else g("qpsk"),
            g("16QAM") if "16QAM" in pmap else g("16qam"),
            g("64QAM") if "64QAM" in pmap else g("64qam"),
            g("256QAM") if "256QAM" in pmap else g("256qam"),
        )
    )
    con.commit()

@app.on_event("startup")
def _startup_init_db():
    _ensure_tables()

# ------------------------- Helpers & policy -------------------------

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _ensure_model_ready():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not found. Train and save models/best_model.joblib first.")

def _to_jsonable(v):
    if isinstance(v, (np.generic,)):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_jsonable(x) for k, x in v.items()}
    return v

def _model_classes() -> List[str]:
    classes = schema.get("classes") or []
    if classes:
        return [str(c) for c in classes]
    try:
        return [str(c) for c in getattr(model, "classes_", [])]
    except Exception:
        return []

def _align_features(raw: Dict[str, Any], user_order: Optional[List[str]] = None) -> pd.DataFrame:
    safe = dict(raw or {})
    feats = schema.get("features", {}) or {}
    nums  = list(feats.get("numeric", []) or [])
    cats  = list(feats.get("categorical", []) or [])
    expected = nums + cats

    # Normalize BLER
    if "BLER" in expected and "BLER" in safe:
        try:
            b = float(safe["BLER"])
            if b > 1.0:
                b = b / 100.0
            safe["BLER"] = max(0.0, min(1.0, b))
        except Exception:
            safe["BLER"] = None

    # Derived TH_SUM
    if "TH_SUM" in expected and "TH_SUM" not in safe:
        try:
            c0 = float(safe.get("TH_CWD0", 0) or 0)
            c1 = float(safe.get("TH_CWD1", 0) or 0)
            safe["TH_SUM"] = c0 + c1
        except Exception:
            pass

    # Derived CQI_MEAN
    if "CQI_MEAN" in expected and "CQI_MEAN" not in safe:
        try:
            parts = []
            for k in ("CQI_CWD0","CQI_CWD1"):
                if k in safe and safe[k] not in (None, "", "null"):
                    v = float(safe[k])
                    parts.append(None if v == 0 else v)
            vals = [p for p in parts if p is not None]
            if vals:
                safe["CQI_MEAN"] = sum(vals)/len(vals)
        except Exception:
            pass

    df = pd.DataFrame([safe])
    for col in expected:
        if col not in df.columns:
            df[col] = None
    for col in nums:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            df[col] = pd.NA
    order = user_order if user_order else expected
    return df.reindex(columns=order)

def _proba_from_model(X: pd.DataFrame) -> np.ndarray:
    # A) predict_proba
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        p = np.asarray(p)
        if p.ndim == 2:
            return p
        raise ValueError("predict_proba returned unexpected shape")
    # B) decision_function -> sigmoid/softmax
    if hasattr(model, "decision_function"):
        df = model.decision_function(X)
        df = np.asarray(df)
        if df.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-df))
            return np.stack([1.0 - p1, p1], axis=1)
        if df.ndim == 2:
            z = df - df.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)
        raise ValueError("decision_function returned unexpected shape")
    raise HTTPException(
        status_code=400,
        detail="Model exposes neither predict_proba nor decision_function; cannot compute probabilities."
    )

def _norm_bler(val) -> float:
    try:
        x = float(val)
    except Exception:
        return 0.05
    if x > 1.0:
        x = x / 100.0
    return max(0.0, min(1.0, x))

def choose_packet_size(mcs: str, bler: float,
                       p95_target: Optional[float] = None,
                       traffic: Optional[str] = None) -> int:
    if traffic == "control":
        base = 600
    elif traffic == "telemetry":
        base = 900
    else:
        base = 1200
    if mcs in {"QPSK","16QAM"} or bler > 0.10:
        base = min(base, 600)
    elif mcs == "64QAM":
        base = min(base, 900)
    if p95_target is not None and p95_target <= 40:
        base = max(600, int(base * 0.8))
    return base

def score_host(h: Dict[str, Any], rtt_w: float = 0.6, cpu_w: float = 0.3, q_w: float = 0.1) -> float:
    rtt = float(h.get("rtt_ms", 5.0))
    cpu = float(h.get("cpu_util", 0.50))
    q   = float(h.get("queue_len", 0.0))
    return rtt_w*rtt + cpu_w*cpu + q_w*q

def choose_mec_host(hosts: List[Dict[str, Any]], p95_target: float | None = None) -> Optional[Dict[str, Any]]:
    if not hosts:
        return None
    if p95_target is not None and p95_target <= 40:
        return min(hosts, key=lambda h: score_host(h, rtt_w=0.8, cpu_w=0.15, q_w=0.05))
    return min(hosts, key=score_host)

def max_prob_conf(proba: List[float]) -> float:
    return float(max(proba)) if proba else 0.0

def risk_level(conf: float) -> str:
    if conf >= 0.85:
        return "low"
    if conf >= 0.60:
        return "medium"
    return "high"

def bitrate_hint(features: Dict[str, Any], mcs: str, conf: float) -> float:
    th_sum = float(features.get("TH_SUM", 0.0))
    th_c0  = float(features.get("TH_CWD0", 0.0))
    th_c1  = float(features.get("TH_CWD1", 0.0))
    base   = th_sum if th_sum > 0 else (th_c0 + th_c1)
    if base <= 0:
        base = 1.0
    bler = _norm_bler(features.get("BLER", 0.05))
    safe = max(0.5, 1.0 - bler - (0.20 if conf < 0.60 else 0.0))
    norm = MCS_WEIGHTS.get(mcs, 2.0) / MCS_WEIGHTS["64QAM"]
    return max(0.1, 0.90 * base * norm * safe)

def log_advisor(payload: Dict[str, Any]) -> None:
    try:
        with open(ADVISOR_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ------------------------- Pydantic models -------------------------

class HostStats(BaseModel):
    name: str
    rtt_ms: float
    cpu_util: float = Field(ge=0.0, le=1.0)
    queue_len: float = 0.0

class QoS(BaseModel):
    target_p95_latency_ms: Optional[float] = None
    min_bitrate: Optional[float] = None
    max_bitrate: Optional[float] = None
    traffic_type: Optional[Literal["telemetry","video","control"]] = None

class RecommendRequest(BaseModel):
    features: Dict[str, Any]
    qos: Optional[QoS] = None
    hosts: Optional[List[HostStats]] = None

class RecommendResponse(BaseModel):
    decision_id: str
    mcs: str
    probabilities: Dict[str, float]
    confidence: float
    risk: str
    bitrate_hint: float
    packet_size: int
    mec_host: Optional[Dict[str, Any]] = None
    rationale: Dict[str, Any]

class Feedback(BaseModel):
    decision_id: str
    realized_bitrate: Optional[float] = None
    realized_latency_ms: Optional[float] = None
    realized_bler: Optional[float] = None
    success: Optional[bool] = None
    notes: Optional[str] = None

class PredictRequest(BaseModel):
    features: Dict[str, Any]
    feature_order: Optional[List[str]] = None

# ------------------------- Routes -------------------------

@app.get("/")
def root():
    return {"ok": True, "service": app.title, "version": app.version}

@app.get("/health")
def health():
    feats = schema.get("features", {}) if isinstance(schema, dict) else {}
    return JSONResponse(
        content={
            "ok": model is not None,
            "has_model": MODEL_PATH.exists(),
            "has_schema": SCHEMA_PATH.exists(),
            "n_features_numeric": len(feats.get("numeric", []) or []),
            "n_features_categorical": len(feats.get("categorical", []) or []),
            "model_path": str(MODEL_PATH),
            "schema_path": str(SCHEMA_PATH),
        },
        status_code=200,
    )

@app.get("/schema")
def get_schema():
    return JSONResponse(content=schema, status_code=200)

@app.get("/features")
def features():
    return JSONResponse(content=schema.get("features", {}), status_code=200)

@app.get("/example_payload")
def example_payload():
    feats = schema.get("features", {})
    numeric = {k: 0 for k in feats.get("numeric", []) or []}
    categorical = {k: None for k in feats.get("categorical", []) or []}
    return JSONResponse(content={"features": {**numeric, **categorical}}, status_code=200)

@app.get("/model_metadata")
def model_metadata():
    info = {
        "ok": model is not None,
        "model_path": str(MODEL_PATH),
        "model_name": None,
        "trained_on": None,
        "metrics": {}
    }
    # name from loaded model
    try:
        last = model[-1] if hasattr(model, "__getitem__") else model
        info["model_name"] = type(last).__name__
    except Exception:
        pass

    # merge metadata.json and metrics.json if present
    meta_path = MODELS_DIR / "metadata.json"
    metr_path = MODELS_DIR / "metrics.json"
    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text("utf-8"))
            info["model_name"] = meta.get("model_name") or info["model_name"]
            info["model_path"] = meta.get("model_path") or info["model_path"]
            info["trained_on"] = meta.get("trained_on") or info.get("trained_on")
            if isinstance(meta.get("metrics"), dict):
                info["metrics"].update(meta["metrics"])
        if metr_path.exists():
            m = json.loads(metr_path.read_text("utf-8"))
            if isinstance(m, dict):
                info["metrics"].update(m)
    except Exception:
        pass

    return JSONResponse(content=info, status_code=200)

@app.post("/predict")
def predict(req: PredictRequest):
    _ensure_model_ready()
    X = _align_features(req.features, req.feature_order)
    try:
        yhat = model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")
    out: Dict[str, Any] = {"prediction": str(_to_jsonable(yhat))}
    try:
        p = _proba_from_model(X)[0]
        classes = _model_classes()
        if len(classes) != len(p):
            classes = [str(i) for i in range(len(p))]
        out["proba"] = {str(c): float(pp) for c, pp in zip(classes, p)}
    except Exception:
        out["proba"] = None
    return JSONResponse(content=out, status_code=200)

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    _ensure_model_ready()

    # ------ Inputs & normalization ------
    X = _align_features(req.features)
    bler = _norm_bler(req.features.get("BLER", 0.05))
    p95  = req.qos.target_p95_latency_ms if req.qos else None
    traffic = req.qos.traffic_type if req.qos else None

    # ------ Model inference ------
    try:
        proba = _proba_from_model(X)[0]      # np.array shape (C,)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Recommendation inference failed: {e}")

    labels = _model_classes()
    if len(labels) != len(proba):
        labels = [str(i) for i in range(len(proba))]

    pmap = {str(labels[i]): float(proba[i]) for i in range(len(proba))}
    mcs  = max(pmap.items(), key=lambda kv: kv[1])[0]
    conf = max_prob_conf(list(pmap.values()))
    risk = risk_level(conf)

    # ------ Guardrails & exploration ------
    explored = False
    forced_safe = False

    if bler > EXTREME_BLER:
        mcs = "QPSK"
        forced_safe = True
    else:
        if conf < 0.50:
            mcs = MCS_SAFE_FALLBACK
        else:
            EPS_DEFAULT = 0.0
            EPS = float(os.getenv("ADVISOR_EPS", EPS_DEFAULT))
            EXP_MIN_CONF = float(os.getenv("EXPLORE_MIN_CONF", 0.55))
            EXP_MAX_CONF = float(os.getenv("EXPLORE_MAX_CONF", 0.85))
            if EPS > 0 and EXP_MIN_CONF <= conf <= EXP_MAX_CONF and len(labels) > 1:
                probs = np.array(list(pmap.values()), dtype=float)
                top_idx = int(np.argmax(probs))
                mask = np.ones_like(probs, dtype=bool)
                mask[top_idx] = False
                weights = probs[mask]
                weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights)/weights.size
                alt_indices = np.arange(len(probs))[mask]
                import random as _rnd
                if _rnd.random() < EPS:
                    idx = int(np.random.choice(alt_indices, p=weights))
                    mcs = str(labels[idx])
                    explored = True

    # ------ Packet size & bitrate hint ------
    psize = choose_packet_size(mcs, bler, p95, traffic)
    br = bitrate_hint(req.features, mcs, conf)

    # Traffic nudges
    if traffic == "video":
        br *= 1.05
    elif traffic == "control":
        br *= 0.85

    # QoS bitrate clamps
    if req.qos and req.qos.min_bitrate is not None:
        br = max(br, float(req.qos.min_bitrate))
    if req.qos and req.qos.max_bitrate is not None:
        br = min(br, float(req.qos.max_bitrate))

    # ------ MEC host selection ------
    host_dicts = [h.dict() for h in (req.hosts or [])]
    chosen_host = choose_mec_host(host_dicts, p95)

    # ------ Rationale ------
    rationale = {
        "bler": bler,
        "conf": conf,
        "risk": risk,
        "traffic_type": traffic,
        "rule": "confidence+BLER+QoS policy (+ ε-greedy if enabled)",
        "explored": explored,
    }
    if p95 is not None:
        rationale["target_p95_ms"] = p95
    if forced_safe:
        rationale["warning"] = "High BLER (>50%): using safe MCS and small packets."

    # ------ Logging (file) ------
    decision_id = str(uuid.uuid4())
    payload = {
        "decision_id": decision_id,
        "inputs": {"features": req.features, "qos": req.qos.dict() if req.qos else None, "hosts": host_dicts},
        "outputs": {
            "mcs": mcs,
            "pmap": pmap,
            "confidence": conf,
            "risk": risk,
            "bitrate_hint": br,
            "packet_size": psize,
            "mec_host": chosen_host,
            "explored": explored,
        },
        "ts": time.time(),
    }
    log_advisor(payload)

    # ------ Persist to SQLite across 3 tables ------
    try:
        save_decision_row(
            decision_id=decision_id,
            features=req.features,
            qos=(req.qos.dict() if req.qos else None),
            hosts=host_dicts,
            traffic_type=traffic,
            mcs=mcs,
            confidence=conf,
            risk=risk,
            bitrate_hint=br,
            packet_size=psize,
            explored=explored
        )
        save_mec_host_row(decision_id, chosen_host)
        save_class_prob_row(decision_id, pmap)
    except Exception as e:
        logger.warning("SQLite save (3 tables) failed: %s", e)

    # ------ Response ------
    return RecommendResponse(
        decision_id=decision_id,
        mcs=mcs,
        probabilities=pmap,
        confidence=conf,
        risk=risk,
        bitrate_hint=br,
        packet_size=psize,
        mec_host=chosen_host,
        rationale=rationale
    )

@app.post("/feedback")
def feedback(fb: Feedback):
    log_advisor({"feedback": fb.dict(), "ts": time.time()})
    return {"ok": True}

@app.get("/metrics_window")
def metrics_window(hours: float | None = Query(default=24.0)):
    """Lightweight counter from decisions table for UI."""
    try:
        since = time.time() - (hours or 24.0) * 3600.0
        con = _db()
        cur = con.execute("SELECT COUNT(*) FROM decisions WHERE ts >= ?", (since,))
        n = int(cur.fetchone()[0])
        return {
            "count": n,
            "explore_rate": 0.0,
            "p95_latency_miss_rate": 0.0,
            "bitrate_mape": 0.0,
            "post_bler_mean": None,
            "post_bler_p95": None,
        }
    except Exception:
        return {
            "count": 0,
            "explore_rate": 0.0,
            "p95_latency_miss_rate": None,
            "bitrate_mape": None,
            "post_bler_mean": None,
            "post_bler_p95": None,
        }
