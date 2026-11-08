# Dataset Documentation

## Source

- **Name:** Tobi-Bueck/customer-support-tickets
- **URL:** https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets
- **License:** CC-BY-NC-4.0 (attribution required; non-commercial use)
- **DOI:** 10.57967/hf/6184
- **Raw files used:**
  - `dataset-tickets-multi-lang-4-20k.csv` (20,000 rows, en+de)
  - `dataset-tickets-german_normalized_50_5_2.csv` (8,077 rows, de — filtered out)

## Raw schema

`subject, body, answer, type, queue, priority, language, tag_1..tag_8`

## Label mapping (documented)

| Project field | Source | Real or derived |
|---|---|---|
| `ticket_text` | `subject + " " + body` | real |
| `answer` (RAG payload) | `answer` | real agent resolution |
| `ticket_type` | `type` → Incident/Request/Problem/Change | **real** |
| `urgency` | `priority` → low/medium/high | **real** (no "critical" class in source) |
| `team` | `queue` → 10 support queues | **real** |
| `sentiment` | lexicon heuristic (`src/data_processing/sentiment.py`) | **derived** — source has no sentiment labels |
| `created_at` | simulated uniform-over-90-days + 3 team-coherent bursts | **simulated** — source has no timestamps |

## Cleaning rules

- keep `language == "en"`
- require `body` length > 40 chars and `answer` length > 20 chars
- whitespace normalisation, dedupe on `ticket_text`

## Splits

Stratified-random by seeded RNG: train 80% / val 10% / test 10%
(≈ 9,436 / 1,149 / 1,149 after Spark run).

## Reproduce

```bash
python -m src.data_processing.pipeline --engine spark   # or pandas
```

Outputs: `data/processed/tickets_{train,val,test,all}.csv`,
`data/processed/dataset_stats.json`.
