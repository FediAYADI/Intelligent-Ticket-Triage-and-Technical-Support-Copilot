# Architecture

## Data flow

1. **Ingestion/cleaning (PySpark)** — `src/data_processing/pipeline.py`
   reads all CSVs in `data/raw/tobi_bueck`, filters `language=en`, drops
   empty/duplicate tickets, normalises whitespace, derives `ticket_text =
   subject + body`, maps source columns to project labels, adds simulated
   `created_at` timestamps and lexicon `sentiment`, then writes stratified
   train/val/test splits.

2. **Classification** — `src/classification/model.py` defines
   `MultiTaskTicketClassifier`: a DistilBERT encoder with mean pooling and
   four linear heads. `train.py` fine-tunes all heads jointly (sum of
   cross-entropy losses), checkpoints on validation macro-F1 of `ticket_type`,
   and evaluates on the test split → `reports/classification_metrics.json`
   (accuracy, precision, recall, F1, confusion matrices per head).

3. **Routing** — `predict.py` returns per-head labels + probabilities;
   `overall_confidence = min(head confidences)`. Below
   `classification.confidence_threshold` → `queue=human_review`.
   `router.py` computes `priority_score = 0.5·urgency + 0.3·frustration +
   0.2·(1 - team_load)` and assigns the predicted `team`.

4. **RAG copilot** — `indexer.py` embeds `question` fields (resolved-ticket
   subject+body and FAQ entries) with `all-MiniLM-L6-v2`, L2-normalises and
   stores a FAISS `IndexFlatIP` (cosine) + parquet doc store under
   `models/embeddings/rag_index/`. `copilot.py` retrieves top-k, applies
   `min_similarity`, and generates via the configured backend
   (`extractive` default; `hf` for Phi-3.5/Gemma-2). Responses always include
   sources; low similarity → explicit "insufficient information".
   `faq_gaps.py` agglomerative-clusters uncovered questions → suggested FAQ
   entries pending human approval in the UI.

5. **Anomaly detection** — `detector.py` aggregates volume per team×hour,
   z-scores against each team's own history, flags z > 2.5 with volume ≥ 5,
   then describes each window via KMeans topical clustering → alerts in
   `data/processed/incidents.csv`.

6. **MLOps**
   - *Airflow*: `airflow/dags/ticket_triage_pipeline.py` chains
     ingest → train → reindex → anomalies → evidently (daily; retrain weekly
     in production).
   - *EvidentlyAI*: `evidently_report.py` compares reference vs current
     windows — data-quality metrics (missing fields, invalid labels,
     duplicates, short texts) + embedding drift (centroid distance,
     nearest-neighbour similarity) + optional Evidently HTML report.
   - *Prometheus/Grafana*: `metrics.py` exposes counters/histograms/gauges
     (tickets, review queue, routing errors, copilot actions, anomalies,
     latency, tokens, active tickets) on `:8000/metrics`;
     `grafana/dashboards/ticket_triage.json` + provisioning +
     `prometheus/prometheus.yml` + `docker-compose.yml` wire the dashboard.

7. **UI** — `dashboard/app.py` (Streamlit, sidebar navigation):
   Dashboard stats/charts, ticket list + manual corrections (logged as
   routing errors), copilot answer + sources + accept/modify/reject feedback
   (persisted to SQLite + Prometheus), incident alerts with related tickets,
   FAQ suggestion approval workflow.

## Storage

- **CSV/parquet** — immutable pipeline artifacts (raw → processed → models).
- **SQLite** (`app_data.db` at project root) — runtime state: tickets,
  copilot feedback, team workloads.
- **FAISS index + parquet** — RAG knowledge base.

## Feedback loops (per spec)

- Low-confidence tickets → human review queue; corrections recorded as
  `routing_errors` metrics and ticket `status='correctled'`.
- Copilot accept/modify/reject → `copilot_feedback` table +
  `copilot_suggestions_total` metric → adoption rate on Grafana.
- Approved FAQ suggestions → `faq_approved.csv` → reindexed by the Airflow
  reindex task.
- Evidently drift report → trigger for re-labelling/retraining (documented;
  the retrain task runs on schedule regardless).
