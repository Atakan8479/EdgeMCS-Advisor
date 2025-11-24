# IoT Uplink MCS Predictor & Advisor

This project provides a **FastAPI backend** and a **FastHTML-based UI** for predicting the Modulation and Coding Scheme (MCS\_CWD0) of an IoT/5G uplink and giving simple **bitrate + MEC offload recommendations**.

The system is built around a real-world dataset (`FinalCompiledData_02022023.csv`) with KPIs such as SINR, RB, BLER, CQI, throughput per codeword, number of carriers, and antenna/rank information. A trained machine learning pipeline is loaded from `models/best_model.joblib`, and a schema file `models/schema.json` describes numeric and categorical features and target classes.

---

## Features

### Backend (`api/app.py`)

- FastAPI service exposing:
  - `GET /health` – basic health check with model and schema info.
  - `GET /schema` – full schema (`numeric`, `categorical`, `target`, `classes`).
  - `GET /example_payload` – zero/empty example payload for UI.
  - `POST /predict` – raw MCS\_CWD0 prediction with class probabilities.
  - `POST /recommend` – advisor endpoint for:
    - MCS selection with guardrails (confidence, BLER threshold).
    - Bitrate hint computation.
    - Packet size selection.
    - Optional MEC host selection based on RTT, CPU, and queue length.
  - `POST /feedback` – log realized bitrate/latency/BLER for later analysis.
  - `GET /metrics_window` – simple counter based on SQLite decisions table.

- Policy logic:
  - BLER normalization (0–1, supports percentage inputs).
  - Derived features: `TH_SUM` and `CQI_MEAN` if not provided.
  - Confidence-based risk levels: `low`, `medium`, `high`.
  - High-BLER safety rule (`EXTREME_BLER > 0.50` ⇒ force QPSK).
  - Optional ε-greedy exploration controlled via environment variables.

- Persistence:
  - Lightweight logging to `advisor_logs.jsonl`.
  - SQLite database (`advisor.db`) with three tables:
    - `decisions` – one row per recommendation.
    - `mec_hosts` – chosen MEC host per decision.
    - `class_probabilities` – QPSK / 16QAM / 64QAM / 256QAM probabilities.

### UI (`api/ui_fasthtml.py`)

- FastHTML app that:
  - Connects to the FastAPI backend (`API_BASE_URL`).
  - Uses `/schema` and `/example_payload` to render dynamic forms.
  - Sends requests to `/predict` and `/recommend`.
  - Shows:
    - Backend status (features, target, classes, accuracy + macro-F1).
    - Dataset card (using `FinalCompiledData_*.csv` if found locally).
    - Class probability table with bars.
    - Advisor card:
      - Core features form (numeric + categorical).
      - QoS block (latency target, min/max bitrate, traffic type).
      - MEC hosts JSON block with random host generator.
      - Result card with decision ID, MCS, confidence, risk, bitrate hint, packet size, chosen MEC host, and full JSON.

- **Privacy-friendly UI:**
  - Only shows **filenames** for the model file and trained-on dataset.
  - Full filesystem paths are hidden from the browser.

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
