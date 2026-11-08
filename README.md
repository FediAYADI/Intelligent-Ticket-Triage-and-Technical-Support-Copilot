# Intelligent Ticket Triage and Technical Support Copilot

Academic MLOps project: a three-component system for technical-support ticket
processing — intelligent classification & routing, a RAG-based response
copilot, and anomaly/incident detection — with Airflow orchestration,
EvidentlyAI drift monitoring, a Prometheus + Grafana dashboard, and a
Streamlit human-in-the-loop interface.

## Architecture

```
raw CSV ──► PySpark pipeline ──► processed splits (CSV + SQLite)
                                    │
        ┌───────────────────────────┼────────────────────────────┐
        ▼                           ▼                            ▼
 DistilBERT multi-task      sentence-transformers + FAISS   KMeans + z-score
 classifier (type, urgency,   RAG copilot (cited answers,    volume anomaly
 sentiment, team) + confidence  FAQ-gap detection)            detection
        │                           │                            │
        └──────────► routing (priority score, review queue) ◄─────┘
                                    │
              SQLite store ──► Streamlit UI (5 pages)
                                    │
              Prometheus metrics ◄──┘──► Grafana dashboard
Airflow DAG: ingest → train → reindex KB → anomalies → Evidently report
```

See `docs/architecture.md` for details.

## Dataset

**Primary source:** `Tobi-Bueck/customer-support-tickets`
(Hugging Face, CC-BY-NC-4.0) — ~20k multilingual customer-support tickets
with `subject`, `body`, `answer`, `type`, `queue`, `priority`, `language`,
tags. We keep the English subset (~12k) and map:

| Project label | Source column | Notes |
|---|---|---|
| `ticket_type` (category) | `type` | Incident / Request / Problem / Change |
| `urgency` | `priority` | low / medium / high — **the source has no "critical" class** (see Limitations) |
| `team` (routing target) | `queue` | 10 support queues |
| `sentiment` | *derived* | lexicon heuristic (`src/data_processing/sentiment.py`) — the source has **no sentiment labels** |
| `created_at` | *simulated* | the source has **no timestamps**; spread over 90 days + 3 injected bursts for anomaly demos |
| RAG answers | `answer` | real agent resolutions used as KB documents |

Raw files live in `data/raw/tobi_bueck/` (downloaded from the HF hub).
Processed outputs: `data/processed/tickets_{train,val,test}.csv`,
`tickets_all.csv`, `dataset_stats.json`.

## Setup

```bash
pip install -r requirements.txt
# Optional heavy services:
#   pip install apache-airflow      # or use docker-compose.yml
#   docker compose up -d            # Airflow + Prometheus + Grafana
```

Requires Java 17+ for PySpark (local mode). Python 3.12 tested.

## Pipeline

```bash
# 1. Data engineering (PySpark)
python -m src.data_processing.pipeline

# 2. Train multi-task DistilBERT classifier
python -m src.classification.train        # writes models/classification/... + reports/classification_metrics.json

# 3. Build RAG knowledge base (embeddings + FAISS)
python -m src.rag.indexer

# 4. Detect anomalies/incidents
python -m src.anomaly_detection.detector  # writes data/processed/incidents.csv

# 5. Evidently drift + data-quality report
python -m src.monitoring.evidently_report # writes reports/evidently_*

# 6. Seed the app DB and simulate incoming tickets
python scripts/seed_db.py --predict-sample 300
python scripts/simulate_incoming.py --n 25 --burst 30   # --burst injects a payment-outage spike

# 7. Run the UI + metrics endpoint
streamlit run dashboard/app.py            # http://localhost:8501
# metrics are served on :8000/metrics (started automatically by the app/scripts)
```

## Components

- **Classification** — `src/classification/`: shared DistilBERT encoder + 4
  heads (category, urgency, sentiment, team). Confidence = min head prob;
  below `confidence_threshold` (config) → human-review queue.
- **Routing** — `src/routing/router.py`: priority = w·urgency +
  w·frustration + w·team-load; configurable weights/thresholds.
- **RAG copilot** — `src/rag/`: sentence-transformers (`all-MiniLM-L6-v2`) +
  FAISS index over resolved tickets + FAQ articles. Two generation backends:
  `extractive` (default, CPU) or `hf` generative (Phi-3.5-mini/Gemma-2-2B via
  config, needs GPU). Always returns cited sources; flags insufficient
  information below `min_similarity`. `faq_gaps.py` clusters uncovered
  questions → FAQ suggestions for human validation.
- **Anomaly detection** — `src/anomaly_detection/detector.py`: hourly
  per-team volume z-scores + KMeans topical description of anomalous windows
  → incident alerts persisted to `data/processed/incidents.csv`.
- **MLOps** — `airflow/dags/` (ingest→train→reindex→anomalies→evidently),
  `src/monitoring/evidently_report.py` (drift + data quality),
  `src/monitoring/metrics.py` (Prometheus), `grafana/` + `prometheus/` +
  `docker-compose.yml`.
- **UI** — `dashboard/app.py`: Dashboard, Ticket Management (corrections),
  AI Copilot (accept/modify/reject feedback), Incident Monitoring,
  FAQ Management.

## Tests

```bash
pytest -q    # 8 tests: sentiment, routing, storage, anomaly burst detection
```

## Limitations (documented deviations)

- **CPU-only torch** (`torch 2.14.0+cpu`): DistilBERT trains fine on CPU with
  ~12k short tickets; generative SLMs (Phi-3.5/Gemma-2) are impractical, so the
  default RAG backend is extractive. The `hf` backend is implemented and
  config-switchable.
- **Urgency has 3 classes** (low/medium/high): the source dataset provides no
  "critical" label; mapping is documented rather than fabricated.
- **Sentiment and timestamps are simulated/heuristic** (see Dataset table).
- **PySpark on Windows** has no binary wheels and its Python workers fail to
  launch in `local` mode on some setups. The pipeline therefore runs Spark for
  ingestion/schema-alignment/filtering/cleaning/deduplication, then applies
  the sentiment heuristic and timestamp simulation in pandas after
  `toPandas()` (documented in `pipeline.py`). `--engine pandas` runs the same
  logic without Spark at all; on Linux/macOS or a cluster the same code runs
  unchanged, and Python UDFs can be re-enabled there.

## Project structure

```
config/config.yaml      all tunables
src/data_processing/    PySpark pipeline + sentiment heuristic
src/classification/     multi-task model, train, predict
src/routing/            priority score + team routing
src/rag/                indexer, copilot, faq gap detection
src/anomaly_detection/  volume z-score + clustering
src/monitoring/         Prometheus metrics + Evidently report
src/evaluation/         (metrics helpers used by train.py)
src/storage.py          SQLite runtime store
airflow/dags/           orchestration DAG
dashboard/app.py        Streamlit UI
grafana/ prometheus/    monitoring assets
docker-compose.yml      Airflow + Prometheus + Grafana
reports/                generated metrics & drift reports
tests/                  pytest suite
```
