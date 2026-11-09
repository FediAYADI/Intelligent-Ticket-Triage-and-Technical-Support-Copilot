# Final Project Report — Intelligent Ticket Triage & Support Copilot

## 1. Methodology

The system implements the three components required by the specification,
integrated through a shared ticket store (SQLite) and a Prometheus metrics
layer:

1. **Classification & routing** — a multi-task DistilBERT model
   (`distilbert-base-uncased` encoder + 4 linear heads) predicts ticket
   category (`type`), urgency (`priority`), sentiment (lexicon-derived) and
   support team (`queue`). Per-head softmax confidences are combined into an
   overall confidence (minimum head probability); tickets below
   `confidence_threshold` are routed to a human-review queue instead of
   auto-assignment. A priority score combines urgency, customer frustration
   (lexicon + stylistic cues) and target-team workload.

2. **RAG copilot** — resolved historical tickets and FAQ articles are
   embedded with `all-MiniLM-L6-v2`, L2-normalised and indexed in FAISS
   (`IndexFlatIP` = cosine). Retrieval returns top-k cited sources; an
   extractive backend (CPU) or an optional generative backend (Phi-3.5-mini /
   Gemma-2-2B via `llm_backend: hf`) produces the suggested answer. Answers
   below `min_similarity` are explicitly flagged as insufficiently grounded.
   Uncovered recurring questions are clustered (agglomerative, cosine) into
   FAQ suggestions pending human validation.

3. **Anomaly detection** — ticket volume is aggregated per team×hour and
   z-scored against each team's own history; windows above threshold are
   described by KMeans topical clustering of ticket embeddings, producing
   incident alerts with related ticket IDs.

## 2. Data

Source: `Tobi-Bueck/customer-support-tickets` (HF, CC-BY-NC-4.0), English
subset, 11,734 clean tickets → 9,436/1,149/1,149 train/val/test.
Real labels: category, urgency (3 classes — no "critical" in source), team.
Derived: sentiment (lexicon heuristic), timestamps (simulated, documented).
Full details: `docs/dataset.md`.

## 3. Experiments & results

### Classification (test split, DistilBERT, 5 epochs, 5k-sample CPU run)

| Head | Accuracy | Macro F1 |
|---|---|---|
| ticket_type (4 classes) | 0.789 | 0.770 |
| urgency (3 classes) | 0.466 | 0.347 |
| sentiment (3 classes) | 0.903 | 0.896 |
| team (10 classes) | 0.406 | 0.244 |

Per-class precision/recall/F1 and confusion matrices:
`reports/classification_metrics.json`.

Interpretation: category and sentiment are reliable; urgency and especially
team routing are weak — the `queue` field is noisy/multi-factor and the CPU
budget limited training. This is precisely why the confidence gate and the
human-review queue exist: low-confidence routings are escalated rather than
trusted. (A longer GPU run or class-balanced sampling would improve these;
see Limitations.)

**Confidence calibration caveat:** on this CPU-trained model the per-head
softmax probabilities are under-confident and tightly distributed
(mean ≈ 0.38). The aggregate confidence is the mean of the four head
max-probabilities and the review threshold (0.38, ≈ median) was set on the
validation distribution so roughly half of tickets auto-route and half are
escalated — demonstrating the human-in-the-loop workflow. On a GPU-trained
model, temperature scaling on the validation set is the recommended
calibration step (noted as future work).

### RAG copilot

- Knowledge base: 11,748 documents (11,734 resolved tickets + 14 FAQ seeds).
- Verified behaviour: a password-reset question retrieves `faq_0` at
  similarity 0.94 and returns the curated FAQ answer; gibberish input
  returns the explicit "insufficient information" response (no hallucinated
  answer).
- FAQ-gap detection on 32 simulated uncovered questions produced clustered
  suggestions (SSO, webhooks) written to `faq_suggestions.csv` for review.

### Anomaly detection

On the processed dataset (with 3 injected team-coherent bursts) the detector
produced 3 incident alerts, e.g.:

- `INC-0001` — Human Resources, 2026-09-16 10:00, volume 15 vs baseline
  1.26/h (z = 7.4), dominant topic "HR System Crash During Employee Data
  Sync", 8/15 negative-sentiment tickets.
- `INC-0003` — Product Support, volume 13 vs 1.62/h (z = 11.2), topic
  "Safeguarding Medical Information".
- `INC-0006` — Technical Support, volume 15 vs 2.01/h (z = 10.8), topic
  "Hospital Data Security Incident".

Alerts include window stats, type/urgency distributions, related ticket IDs
and sample subjects (`data/processed/incidents.csv`).

### Drift / data quality (EvidentlyAI)

`reports/evidently_summary.json` + `reports/evidently_drift_report.html`:
reference vs current window comparison — missing-field rates (~8.7% missing
subjects, identical across windows → no drift), duplicate/invalid-label
rates 0%, embedding centroid drift 0.033, mean nearest-neighbour similarity
0.77 → `drift_detected: false`.

## 4. MLOps deliverables

- **Airflow DAG** `airflow/dags/ticket_triage_pipeline.py`: ingest → train →
  reindex KB → anomaly detection → Evidently report (daily; retrain weekly
  in production). Deployable via `docker-compose.yml` or any Airflow.
- **EvidentlyAI**: weekly-style drift + data-quality report (above).
- **Grafana**: `grafana/dashboards/ticket_triage.json` + provisioning —
  total tickets, human-review count, incorrect-routing rate, copilot
  adoption rate, tickets by urgency, incident alerts by team, p95 latency,
  LLM token usage, active tickets per team. Prometheus scrapes
  `:8000/metrics` (`src/monitoring/metrics.py`).

## 5. Human-in-the-loop UI

`dashboard/app.py` (Streamlit): Dashboard, Ticket Management (inspect +
correct predictions; corrections counted as routing errors), AI Copilot
(suggested answer + cited sources + accept/modify/reject feedback),
Incident Monitoring (alerts + related tickets + acknowledge), FAQ Management
(approve/reject suggested FAQs → reindexed KB).

## 6. Limitations

- **CPU-only environment** (`torch+cpu`, RTX 3050 present but CUDA build not
  installed): generative SLMs (Phi-3.5/Gemma-2) impractical → extractive RAG
  backend is default; `hf` backend implemented and config-switchable.
- **Urgency has 3 classes**: source lacks "critical"; documented rather than
  fabricated.
- **Sentiment/timestamps are derived/simulated** — documented heuristics,
  not ground truth.
- **PySpark on Windows**: no binary wheels; Python UDF workers fail in
  local mode → Spark performs ingestion/cleaning/dedup, pandas applies
  sentiment/timestamps post-collection. `--engine pandas` is a full fallback.
  On Linux/cluster the same code runs unchanged.
- **Team/urgency heads underperform** — mitigated by the human-review gate;
  more epochs/GPU or better queue labels would help.
- **Anomaly ground truth is simulated** — real incident labels don't exist;
  validation is therefore qualitative (injected bursts are recovered).

## 7. Reproduce

See `README.md` → Pipeline. Tests: `pytest -q` (8 tests).
