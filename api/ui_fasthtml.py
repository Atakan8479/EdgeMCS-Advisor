# api/ui_fasthtml.py
# Pretty ML UI for your FastAPI /predict (UI-only; no calibration or KPI blocks)

from fasthtml.common import *   # components + fast_app + serve()
import os, json, math, requests, csv, re, random
from pathlib import Path
from starlette.requests import Request
from urllib.parse import parse_qs

# If UI_DATASET_FILE isn't set, set it to your file if present
if not os.getenv("UI_DATASET_FILE"):
    _candidate = Path("/Users/atakanozcan/FinalCompiledData_02022023.csv")
    if _candidate.exists():
        os.environ["UI_DATASET_FILE"] = str(_candidate)

# ---------------- Config ----------------
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY  = os.getenv("API_KEY")
HEADERS  = {"ngrok-skip-browser-warning": "true"} if "ngrok-free.dev" in BASE_URL else {}
if API_KEY:
    HEADERS["x-api-key"] = API_KEY

MODELS_DIR  = Path("models")
SCHEMA_PATH = MODELS_DIR / "schema.json"
TITLE = "IoT Uplink MCS Predictor — Machine Learning Interface"

ADVISOR_STATE = {}

schema = {"features":{"numeric":[],"categorical":[]}, "target":"MCS_CWD0", "classes":[]}
try:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
except Exception:
    pass

NUMS    = schema.get("features", {}).get("numeric", [])
CATS    = schema.get("features", {}).get("categorical", [])
CLASSES = schema.get("classes", [])
TARGET  = schema.get("target", "MCS_CWD0")


def _read_json(path: Path):
    try:
        if path.exists():
            return json.loads(path.read_text("utf-8"))
    except Exception:
        pass
    return None


def _model_info_from_anywhere():
    ok_m, meta = api_get("/model_metadata", timeout=5)
    if ok_m and isinstance(meta, dict):
        info = {
            "model_name": meta.get("model_name"),
            "model_path": meta.get("model_path"),
            "trained_on": meta.get("trained_on"),
        }
        m = meta.get("metrics") or {}
        info["accuracy"] = m.get("accuracy")
        info["macro_f1"] = m.get("macro_f1")
        return info

    metadata = _read_json(MODELS_DIR / "metadata.json") or {}
    metrics  = _read_json(MODELS_DIR / "metrics.json")  or {}

    if metadata or metrics:
        info = {
            "model_name": metadata.get("model_name"),
            "model_path": metadata.get("model_path") or str((MODELS_DIR / "best_model.joblib").resolve()),
            "trained_on": metadata.get("trained_on"),
            "accuracy":  (metadata.get("metrics") or {}).get("accuracy"),
            "macro_f1":  (metadata.get("metrics") or {}).get("macro_f1"),
        }
        if isinstance(metrics, dict):
            info["accuracy"] = metrics.get("accuracy", info.get("accuracy"))
            info["macro_f1"] = metrics.get("macro_f1", info.get("macro_f1"))
        return info

    info = {
        "model_name": os.getenv("MODEL_NAME"),
        "model_path": os.getenv("MODEL_PATH"),
        "trained_on": os.getenv("TRAINED_ON") or os.getenv("UI_DATASET_FILE"),
        "accuracy": os.getenv("TEST_ACC"),
        "macro_f1": os.getenv("TEST_MACRO_F1"),
    }
    if any(info.values()):
        return info

    return {
        "model_name": None,
        "model_path": None,
        "trained_on": None,
        "accuracy": None,
        "macro_f1": None,
    }


def _short_path(p: str | None):
    if not p:
        return "—"
    try:
        name = Path(p).name
        return name or p
    except Exception:
        return p


def feature_form():
    return render_core_fields(_example_defaults())


def _rand_hosts(n: int | None = None) -> list[dict]:
    if n is None:
        n = random.randint(1, 5)
    base_names = ["mec-a","mec-b","mec-c","mec-d","edge-1","edge-2","edge-3","fog-a","fog-b","fog-c"]
    random.shuffle(base_names)
    hosts = []
    for i in range(n):
        hosts.append({
            "name": base_names[i] if i < len(base_names) else f"mec-{i+1}",
            "rtt_ms": round(random.uniform(2.0, 40.0), 1),
            "cpu_util": round(random.uniform(0.05, 0.95), 2),
            "queue_len": random.randint(0, 10)
        })
    return hosts


_CAT_CHOICES = {
    "Network Type": ["NR", "LTE"],
    "Band": ["n78", "n41", "n258", "B7", "B3"],
    "ANT": ["2x2", "4x4", "8x8"],
    "MCS_CWD1": ["QPSK", "16QAM", "64QAM", "256QAM"],
}


def _rand_numeric(name: str) -> float:
    if name == "SINR":          return round(random.uniform(-5, 30), 1)
    if name == "RB":            return int(random.uniform(1, 100))
    if name == "BLER":          return round(random.uniform(0.0, 0.2), 3)
    if name in ("CQI_CWD0","CQI_CWD1","CQI_MEAN"): return int(random.uniform(1, 15))
    if name == "TH_SUM":        return round(random.uniform(0.5, 50.0), 2)
    if name in ("TH_CWD0","TH_CWD1"): return round(random.uniform(0.1, 25.0), 2)
    if name == "Num_Carriers":  return random.choice([1, 2, 3])
    if name == "TRANS":         return random.choice([0, 1])
    if name == "RANK":          return random.choice([1, 2])
    return round(random.uniform(0.0, 10.0), 2)


def _rand_categorical(name: str) -> str:
    opts = _CAT_CHOICES.get(name)
    return random.choice(opts) if opts else ""


def get_live_keys():
    ok, sch = api_get("/schema", timeout=5)
    if ok and isinstance(sch, dict):
        feats = sch.get("features", {}) or {}
        nums = list(feats.get("numeric", []) or [])
        cats = list(feats.get("categorical", []) or [])
        return nums, cats
    return NUMS, CATS


def _example_defaults() -> dict:
    ok, ex = api_get("/example_payload", timeout=5)
    if ok and isinstance(ex, dict):
        feats = ex.get("features") or {}
        if isinstance(feats, dict) and feats:
            return feats
    ok_s, sch = api_get("/schema", timeout=5)
    nums = (sch.get("features", {}).get("numeric", []) if ok_s else NUMS)
    cats = (sch.get("features", {}).get("categorical", []) if ok_s else CATS)
    return {**{k: 0 for k in nums}, **{k: "" for k in cats}}


def _coerce_val(k: str, v: str):
    if v == "" or v is None:
        return None
    if k in NUMS:
        try:
            x = float(v)
            if math.isnan(x) or math.isinf(x):
                return None
            return x
        except Exception:
            return None
    return v


def api_get(path: str, timeout=5):
    try:
        r = requests.get(f"{BASE_URL}{path}", headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, {"error": str(e)}


def api_post(path: str, body: dict, timeout=20):
    try:
        r = requests.post(f"{BASE_URL}{path}", json=body, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, {"error": str(e)}


def predict(features: dict):
    return api_post("/predict", {"features": features}, timeout=60)


def find_dataset() -> Path | None:
    root = Path.cwd()
    env_path = os.getenv("UI_DATASET_FILE")
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = (root / p).resolve()
        if p.exists() and p.is_file():
            return p
    bases = [
        root, root / "data", root / "datasets",
        root.parent, root.parent / "data", root.parent / "datasets",
        Path.home(), Path.home() / "Downloads", Path.home() / "Documents",
    ]
    candidates: list[Path] = []
    for base in bases:
        if not base.exists():
            continue
        candidates += list(base.glob("FinalCompiledData_*.csv"))
        candidates += list(base.rglob("FinalCompiledData_*.csv"))
        candidates += [p for p in base.rglob("*.csv") if "finalcompileddata" in p.name.lower()]
    home_exact = Path.home() / "FinalCompiledData_02022023.csv"
    if home_exact.exists() and home_exact.is_file():
        candidates.append(home_exact)
    uniq, seen = [], set()
    for p in candidates:
        if p.is_file():
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                uniq.append(p)
    if not uniq:
        return None
    uniq.sort(key=lambda p: (p.stat().st_mtime, p.stat().st_size), reverse=True)
    return uniq[0]


def dataset_head_info(path: Path, max_cols=10):
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(8192)
            try:
                dialect = csv.Sniffer().sniff(sample)
                f.seek(0)
                reader = csv.reader(f, dialect)
            except Exception:
                f.seek(0)
                reader = csv.reader(f)
            header = next(reader, [])
        cols = header[:max_cols]
        more = max(0, len(header) - len(cols))
        return cols, more
    except Exception:
        return [], 0


def hero():
    return Div(
        P("This interface sends feature values to a FastAPI ",
          Code("/predict"),
          " endpoint and returns a predicted Modulation and Coding Scheme (MCS).",
          cls="lead"),
        P("It uses a trained machine learning pipeline and an IoT dataset (uplink KPIs)."),
        cls="hero"
    )


def status_card():
    ok_h, health = api_get("/health", timeout=5)
    ok_s, sch    = api_get("/schema", timeout=5)

    n_num = (sch.get("features", {}).get("numeric", []) if ok_s else NUMS)
    n_cat = (sch.get("features", {}).get("categorical", []) if ok_s else CATS)
    classes = (sch.get("classes", []) if ok_s else CLASSES)
    target  = (sch.get("target") if ok_s else TARGET)

    rows = [
        Tr(Td("API Base URL"), Td(Code(BASE_URL))),
        Tr(Td("Model loaded"), Td(str(health.get("ok") if ok_h else False))),
        Tr(Td("Numeric features"), Td(str(len(n_num)))),
        Tr(Td("Categorical features"), Td(str(len(n_cat)))),
        Tr(Td("Target variable"), Td(str(target))),
        Tr(Td("Classes"), Td(", ".join(map(str, classes)) if classes else "—")),
    ]
    if not ok_h:
        rows.append(Tr(Td("Health error"), Td(str(health.get("error")))))
    if not ok_s:
        rows.append(Tr(Td("Schema error"), Td(str(sch.get("error")))))

    mi = _model_info_from_anywhere()
    rows += [
        Tr(Td("Model"), Td(str(mi.get("model_name") or "—"))),
        Tr(Td("Model file"), Td(Code(_short_path(mi.get("model_path"))))),
        Tr(Td("Trained on"), Td(Code(_short_path(mi.get("trained_on"))))),
        Tr(Td("Test Accuracy / Macro-F1"),
           Td(f"{mi.get('accuracy','—')} / {mi.get('macro_f1','—')}")),
    ]
    return Card(H3("Backend Status"), Table(*rows, cls="striped"))


def dataset_card():
    ds = find_dataset()
    if not ds:
        return Card(
            H3("Dataset"),
            P("No file matching ", Code("FinalCompiledData_*.csv"),
              " found in project root, ", Code("data/"), " or ", Code("datasets/"), "."),
            P("Set ", Code("UI_DATASET_FILE"),
              " to the exact path on your machine.")
        )
    cols, more = dataset_head_info(ds)
    more_txt = (f" (+{more} more)" if more else "")
    return Card(
        H3("Dataset"),
        P("Using file: ", Code(ds.name)),
        P("Columns (first few): ", Code(", ".join(cols) + more_txt) if cols else "—"),
        P("Shown for provenance; predictions come from the trained model."),
    )


def proba_table(proba: dict, ordered_classes):
    rows = [Tr(Th("Class"), Th("Probability"), Th(""))]
    for cls in ordered_classes:
        p = float(proba.get(cls, 0.0))
        bar = Div(style=f"height:8px;width:{int(p*100)}%;background:#ddd;")
        rows.append(Tr(Td(str(cls)), Td(f"{p:.4f}"), Td(bar)))
    return Table(*rows, cls="striped")


def render_core_fields(values: dict | None = None):
    values = values or {}
    live_nums, live_cats = get_live_keys()
    numeric = (live_nums or ["SINR","RB","BLER","CQI_CWD0","CQI_CWD1","TH_SUM","TH_CWD0","TH_CWD1",
                             "Num_Carriers","TRANS","RANK","CQI_MEAN"])
    categorical = (live_cats or ["Network Type","Band","ANT","MCS_CWD1"])

    core_order = [
        "SINR","RB","BLER","CQI_CWD0","CQI_CWD1","TH_SUM","TH_CWD0","TH_CWD1",
        "Num_Carriers","TRANS","RANK","CQI_MEAN","Network Type","Band","ANT","MCS_CWD1"
    ]
    all_keys = numeric + categorical
    core = [k for k in core_order if k in all_keys]

    def pretty(k: str) -> str:
        if k.replace("_","").isupper():
            return k.replace("_"," ")
        return " ".join(
            w if w.isupper() else w.capitalize()
            for w in k.replace("_"," ").split()
        )

    items = []
    for name in core:
        is_num = name in numeric
        val = values.get(name, "")
        if isinstance(val, (int, float)):
            val = str(val)
        items.append(
            Div(
                Label(pretty(name), cls="block text-sm mb-1"),
                Input(
                    name=name,
                    type=("number" if is_num else "text"),
                    step=("any" if is_num else None),
                    placeholder=pretty(name),
                    value=val,
                    cls="w-full"
                ),
                cls="flex flex-col w-full"
            )
        )
    return Div(*items, id="core_fields", cls="flex flex-col gap-3 w-full")


def render_qos_block(values: dict | None = None):
    values = values or {}

    def _val(k):
        v = values.get(k, "")
        return "" if v is None else v

    t_val = str(values.get("traffic_type") or "")

    return Div(
        H4("QoS"),
        Div(
            Div(
                Label("Target P95 Latency (ms)", cls="block text-sm mb-1"),
                Input(
                    name="target_p95_latency_ms",
                    type="number",
                    step="any",
                    placeholder="e.g., 40",
                    value=_val("target_p95_latency_ms"),
                    cls="w-full"
                ),
                cls="flex flex-col w-full"
            ),
            Div(
                Label("Min Bitrate (Mbps)", cls="block text-sm mb-1"),
                Input(
                    name="min_bitrate",
                    type="number",
                    step="any",
                    placeholder="e.g., 5.0",
                    value=_val("min_bitrate"),
                    cls="w-full"
                ),
                cls="flex flex-col w-full"
            ),
            Div(
                Label("Max Bitrate (Mbps)", cls="block text-sm mb-1"),
                Input(
                    name="max_bitrate",
                    type="number",
                    step="any",
                    placeholder="leave blank for no cap",
                    value=_val("max_bitrate"),
                    cls="w-full"
                ),
                cls="flex flex-col w-full"
            ),
            Div(
                Label("Traffic Type", cls="block text-sm mb-1"),
                Select(
                    Option("—", value="",        selected=(t_val == "")),
                    Option("telemetry", value="telemetry", selected=(t_val == "telemetry")),
                    Option("video",     value="video",     selected=(t_val == "video")),
                    Option("control",   value="control",   selected=(t_val == "control")),
                    name="traffic_type", cls="w-full"
                ),
                cls="flex flex-col w-full"
            ),
        ),
        Div(
            Button(
                "QoS: Low-latency (≤40ms, ≥5 Mbps)",
                hx_post="/qos_preset_low",
                hx_target="#qos_wrap",
                hx_swap="outerHTML",
                cls="btn"
            ),
            Button(
                "Clear QoS",
                hx_post="/qos_clear",
                hx_target="#qos_wrap",
                hx_swap="outerHTML",
                cls="btn"
            ),
            style="display:flex; gap:12px; flex-wrap:wrap; margin-top:8px; margin-bottom:15px;"
        ),
        id="qos_wrap",
        cls="card"
    )


def render_hosts_block(text: str | None = None):
    placeholder = (
        '[{"name":"mec-a","rtt_ms":4.2,"cpu_util":0.32,"queue_len":1}, '
        '{"name":"mec-b","rtt_ms":3.7,"cpu_util":0.61,"queue_len":2}]'
    )
    content = text or ""
    return Div(
        H4("MEC Hosts (JSON Array)"),
        Textarea(
            content,
            name="hosts_json",
            placeholder=placeholder,
            rows=4,
            cls="w-full"
        ),
        Div(
            Button(
                "Insert Hosts",
                hx_post="/hosts_randomize",
                hx_target="#hosts_wrap",
                hx_swap="outerHTML",
                type="button",
                cls="btn"
            ),
            Button(
                "Clear",
                hx_post="/hosts_clear",
                hx_target="#hosts_wrap",
                hx_swap="outerHTML",
                type="button",
                cls="btn"
            ),
            style="display:flex; gap:12px; flex-wrap:wrap; margin-top:8px;"
        ),
        id="hosts_wrap",
        cls="card"
    )


def advisor_result_view(res: dict):
    if not res or not isinstance(res, dict):
        return Div("No recommendation yet.")

    dec_id  = res.get("decision_id", "—")
    mcs     = res.get("mcs", "—")
    conf    = res.get("confidence", None)
    risk    = res.get("risk", "—")
    br      = res.get("bitrate_hint", None)
    psize   = res.get("packet_size", None)
    host    = res.get("mec_host") or {}
    probs   = res.get("probabilities") or {}

    conf_txt  = f"{conf:.4f}" if isinstance(conf, (int, float)) else "—"
    br_txt    = f"{br:.2f}" if isinstance(br, (int, float)) else "—"
    psize_txt = str(psize) if psize is not None else "—"

    ordered = (schema.get("classes") or sorted(probs.keys()))
    proba_tbl = proba_table(probs, ordered)

    expl = (res.get("rationale") or {}).get("explored", False)
    t    = (
        (res.get("rationale") or {}).get("traffic_type")
        or ((res.get("inputs") or {}).get("qos") or {}).get("traffic_type")
    )

    host_rows = [
        Tr(Td("Name"), Td(str(host.get("name", "—")))),
        Tr(Td("RTT (ms)"), Td(str(host.get("rtt_ms", "—")))),
        Tr(Td("CPU util"), Td(str(host.get("cpu_util", "—")))),
        Tr(Td("Queue len"), Td(str(host.get("queue_len", "—")))),
    ] if host else [Tr(Td("—"), Td("No host provided"))]

    return Div(
        H4("Decision"),
        Table(
            Tr(Td("Decision ID"), Td(Code(dec_id))),
            Tr(Td("MCS"), Td(Code(str(mcs)))),
            Tr(Td("Confidence / Risk"), Td(f"{conf_txt} / {risk}")),
            Tr(Td("Bitrate hint"), Td(br_txt)),
            Tr(Td("Packet size (bytes)"), Td(psize_txt)),
            Tr(Td("Exploration"), Td("yes" if expl else "no")),
            Tr(Td("Traffic Type"), Td(str(t or "—"))),
            cls="striped"
        ),
        H4("Chosen MEC Host"),
        Table(*host_rows, cls="striped"),
        H4("Class Probabilities"),
        proba_tbl,
        Details(Summary("Raw JSON"), Pre(json.dumps(res, indent=2))),
        cls="card"
    )


def advisor_card(state):
    qos_block   = render_qos_block()
    hosts_block = render_hosts_block()
    result_shell = Div("No recommendation yet.", id="advisor_result", cls="card")

    return Card(
        H3("Advisor — Bitrate & MEC Offload"),
        Form(
            H4("Core Features"),
            render_core_fields(_example_defaults()),
            Div(
                Button(
                    "Insert Random Core Features",
                    hx_post="/advisor_randomize",
                    hx_target="#core_fields",
                    hx_swap="outerHTML",
                    type="button",
                    cls="btn"
                ),
                style="margin: 4px 0 12px 0;"
            ),
            qos_block,
            hosts_block,
            Div(
                Button("Get Recommendation", type="submit", cls="btn"),
                style="display:flex; gap:12px; flex-wrap:wrap; margin-top:8px;"
            ),
            hx_post="/advisor_recommend",
            hx_target="#advisor_result",
            hx_swap="outerHTML",
            method="post"
        ),
        result_shell
    )


app, rt = fast_app()


@rt("/")
def get():
    left  = Div(
        status_card(),
        dataset_card(),
        style=(
            "flex:1;min-width:340px;padding-right:16px;"
            "display:flex;flex-direction:column;gap:12px;"
        )
    )
    right = Div(
        advisor_card(ADVISOR_STATE),
        style=(
            "flex:1;min-width:340px;padding-left:16px;"
            "display:flex;flex-direction:column;gap:16px;"
        )
    )

    return Titled(
        TITLE,
        hero(),
        Div(
            left,
            right,
            style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap;"
        ),
        Footer(
            P(
                "Tip: set ", Code("API_BASE_URL"),
                " if your API is on ngrok. Optional auth header: ", Code("API_KEY")
            )
        )
    )


@rt("/advisor_randomize")
def post():
    live_nums, live_cats = get_live_keys()
    numeric = (live_nums or ["SINR","RB","BLER","CQI_CWD0","CQI_CWD1","TH_SUM","TH_CWD0","TH_CWD1",
                             "Num_Carriers","TRANS","RANK","CQI_MEAN"])
    categorical = (live_cats or ["Network Type","Band","ANT","MCS_CWD1"])

    core_keys = [
        "SINR","RB","BLER","CQI_CWD0","CQI_CWD1","TH_SUM","TH_CWD0","TH_CWD1",
        "Num_Carriers","TRANS","RANK","CQI_MEAN","Network Type","Band","ANT","MCS_CWD1"
    ]
    vals = {}
    for k in core_keys:
        if k in numeric:
            vals[k] = _rand_numeric(k)
        elif k in categorical:
            vals[k] = _rand_categorical(k)
    return render_core_fields(vals)


@rt("/load_defaults")
def post():
    return Div(feature_form(), id="formwrap")


@rt("/predict")
async def post(request: Request):
    raw = {}
    try:
        if "application/json" in (request.headers.get("content-type") or ""):
            j = await request.json()
            if isinstance(j, dict):
                raw = j
    except Exception:
        pass

    if not raw:
        try:
            form = await request.form()
            if form:
                raw = dict(form)
        except Exception:
            pass

    if not raw:
        try:
            body = (await request.body() or b"").decode("utf-8", "ignore")
            if body:
                raw = {k: v[0] for k, v in parse_qs(body).items()}
        except Exception:
            pass

    def _maybe_json(x):
        if isinstance(x, str):
            try:
                y = json.loads(x)
                return y if isinstance(y, dict) else {}
            except Exception:
                return {}
        return x if isinstance(x, dict) else {}

    if "payload" in raw:
        raw = _maybe_json(raw["payload"])
    elif "form" in raw:
        raw = _maybe_json(raw["form"])

    if not raw or (len(raw) == 1 and raw.get("__sent") == "1"):
        return Card(
            H3("Please enter features"),
            P("All fields are empty. Fill at least one field, then click Predict."),
            Details(Summary("Debug (raw)"), Pre(json.dumps(raw, indent=2))),
        )

    live_nums, _ = get_live_keys()
    numeric_keys = set((live_nums or NUMS) or [])

    feats, nonempty = {}, False
    for k, v in raw.items():
        if k == "__sent":
            continue
        if isinstance(v, str):
            v = v.strip()
        if v not in ("", None, "null"):
            nonempty = True

        if v in ("", None, "null"):
            feats[k] = None
        elif k in numeric_keys:
            try:
                x = float(v)
                feats[k] = None if (math.isnan(x) or math.isinf(x)) else x
            except Exception:
                feats[k] = None
        else:
            feats[k] = v

    if not nonempty:
        return Card(
            H3("Please enter features"),
            P("All fields are empty. Fill at least one field, then click Predict."),
            Details(Summary("Debug (raw)"), Pre(json.dumps(raw, indent=2))),
        )

    ok, out = predict(feats)
    if not ok:
        return Card(
            H3("Prediction Error"),
            P("Could not reach the FastAPI backend at ", Code(f"{BASE_URL}/predict")),
            Pre(str(out.get("error"))),
            P("Start API: ", Code("uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload")),
        )

    ok_m, meta = api_get("/model_metadata", timeout=5)
    model_name = (meta or {}).get("model_name") if ok_m else None
    model_path = (meta or {}).get("model_path") if ok_m else None
    trained_on = (meta or {}).get("trained_on") if ok_m else None
    mm = (meta or {}).get("metrics") if ok_m else {}
    acc = mm.get("accuracy")
    f1m = mm.get("macro_f1")

    if not ok_m:
        mi = _model_info_from_anywhere()
        model_name = model_name or mi.get("model_name")
        model_path = model_path or mi.get("model_path")
        trained_on = trained_on or mi.get("trained_on")
        acc = acc if acc is not None else mi.get("accuracy")
        f1m = f1m if f1m is not None else mi.get("macro_f1")

    pred = out.get("prediction", "?")
    proba = out.get("proba", {})
    cls_order = (schema.get("classes") or sorted(proba.keys()))
    return Card(
        H3("Prediction: ", Span(str(pred), style="font-family:monospace;")),
        proba_table(proba, cls_order),
        H4("Model info (live)"),
        Table(
            Tr(Td("Model"), Td(str(model_name or "—"))),
            Tr(Td("Model file"), Td(Code(_short_path(model_path)))),
            Tr(Td("Trained on"), Td(Code(_short_path(trained_on)))),
            Tr(Td("Test Accuracy / Macro-F1"),
               Td(f"{acc if acc is not None else '—'} / {f1m if f1m is not None else '—'}")),
            cls="striped"
        ),
        Details(Summary("Payload sent"), Pre(json.dumps({"features": feats}, indent=2))),
        Details(Summary("Debug (raw)"), Pre(json.dumps(raw, indent=2))),
    )


@rt("/hosts_randomize")
def post():
    hosts = _rand_hosts()
    sample = json.dumps(hosts)
    return render_hosts_block(sample)


@rt("/whatif")
def get():
    if "SINR" not in NUMS:
        return Titled("What-if", P("SINR not in numeric schema."))
    slider = Input(
        type="range",
        name="SINR",
        value="10",
        min="-5",
        max="30",
        step="0.5",
        hx_post="/whatif_update",
        hx_target="#whatif_out",
        hx_trigger="input"
    )
    return Titled("What-if: vary SINR", slider, Div(id="whatif_out"))


@rt("/whatif_update")
def post(SINR: str, **rest):
    feats = {k: (0.0 if k in NUMS else None) for k in (NUMS + CATS)}
    feats["SINR"] = _coerce_val("SINR", SINR)
    ok, out = predict(feats)
    if not ok:
        return Card(H4("Backend unavailable"), Pre(str(out.get("error"))))
    pred = out.get("prediction", "?")
    proba = out.get("proba", {})
    cls_order = CLASSES or sorted(proba.keys())
    return Card(H4(f"SINR={feats['SINR']} → {pred}"), proba_table(proba, cls_order))


@rt("/qos_preset_low")
def post():
    return render_qos_block({"target_p95_latency_ms": "40", "min_bitrate": "5", "max_bitrate": ""})


@rt("/qos_clear")
def post():
    return render_qos_block({})


@rt("/hosts_preset_example")
def post():
    sample = (
        '[{"name":"mec-a","rtt_ms":4.2,"cpu_util":0.32,"queue_len":1}, '
        '{"name":"mec-b","rtt_ms":3.7,"cpu_util":0.61,"queue_len":2}]'
    )
    return render_hosts_block(sample)


@rt("/hosts_clear")
def post():
    return render_hosts_block("")


@rt("/advisor_recommend")
async def post(request: Request):
    try:
        form = await request.form()
    except Exception:
        form = {}

    live_nums, live_cats = get_live_keys()
    numeric = set(live_nums or [])
    categorical = set(live_cats or [])

    feats = {}
    for k, v in form.items():
        if k in ("target_p95_latency_ms","min_bitrate","max_bitrate","hosts_json"):
            continue
        if v in ("", None):
            continue
        if k in numeric:
            try:
                feats[k] = float(v)
            except Exception:
                pass
        elif k in categorical or k not in ("__sent",):
            feats[k] = v

    def _fget(name):
        v = form.get(name)
        if v in (None, ""):
            return None
        try:
            return float(v)
        except Exception:
            return None

    qos = {}
    p95 = _fget("target_p95_latency_ms")
    minbr = _fget("min_bitrate")
    maxbr = _fget("max_bitrate")
    t = form.get("traffic_type")
    if p95 is not None:
        qos["target_p95_latency_ms"] = p95
    if minbr is not None:
        qos["min_bitrate"] = minbr
    if maxbr is not None:
        qos["max_bitrate"] = maxbr
    if t in ("telemetry","video","control"):
        qos["traffic_type"] = t
    if not qos:
        qos = None

    hosts = None
    hj = form.get("hosts_json")
    if hj and str(hj).strip():
        try:
            parsed = json.loads(hj)
            if isinstance(parsed, list):
                hosts = parsed
        except Exception as ex:
            return Div(
                H4("Advisor Result"),
                Div(f"Invalid hosts_json: {ex}", cls="error"),
                id="advisor_result",
                cls="card"
            )

    payload = {"features": feats, "qos": qos, "hosts": hosts}
    ok, out = api_post("/recommend", payload, timeout=30)
    if not ok:
        return Div(
            H4("Advisor Result"),
            P("Recommendation failed."),
            Pre(json.dumps(out, indent=2)),
            id="advisor_result",
            cls="card"
        )

    return Div(
        advisor_result_view(out),
        id="advisor_result"
    )


serve()
