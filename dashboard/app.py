"""Streamlit human-in-the-loop interface.

Pages: Dashboard | Ticket Management | AI Copilot | Incident Monitoring | FAQ Management

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring import metrics  # noqa: E402
from src.storage import connect, fetch_all, record_feedback  # noqa: E402
from src.utils import load_config, project_path  # noqa: E402

st.set_page_config(page_title="Ticket Triage Copilot", layout="wide")
metrics.start_metrics_server()
CFG = load_config()

PAGES = ["Dashboard", "Ticket Management", "AI Copilot",
         "Incident Monitoring", "FAQ Management"]


# ---------------------------------------------------------------- helpers
def load_tickets(where: str = "", params: tuple = ()) -> pd.DataFrame:
    with connect(CFG) as conn:
        rows = fetch_all(conn, "tickets", where, params)
    return pd.DataFrame(rows)


def load_incidents() -> pd.DataFrame:
    p = Path(CFG["paths"]["processed_dir"]) / "incidents.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_faq_suggestions() -> pd.DataFrame:
    p = Path(CFG["paths"]["kb_dir"]) / "faq_suggestions.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_resource
def get_copilot():
    from src.rag.copilot import Copilot
    return Copilot()


# ---------------------------------------------------------------- pages
def page_dashboard():
    st.title("Triage Dashboard")
    df = load_tickets()
    if df.empty:
        st.info("No tickets yet — run `python scripts/seed_db.py` then "
                "`python scripts/simulate_incoming.py`.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total tickets", len(df))
    c2.metric("Urgent (high)", int((df["urgency"] == "high").sum()))
    c3.metric("Awaiting review", int(df["needs_review"].sum()))
    c4.metric("Open incidents", len(load_incidents()))

    st.subheader("Tickets by category")
    st.bar_chart(df["ticket_type"].value_counts())
    st.subheader("Tickets by urgency")
    st.bar_chart(df["urgency"].value_counts())
    st.subheader("Tickets by team")
    st.bar_chart(df["team"].value_counts())


def page_tickets():
    st.title("Ticket Management")
    df = load_tickets("ORDER BY priority_score DESC")
    if df.empty:
        st.info("No tickets in the store.")
        return

    review_only = st.checkbox("Show only human-review queue")
    view = df[df["needs_review"] == 1] if review_only else df
    st.dataframe(
        view[["ticket_id", "subject", "team", "queue", "ticket_type",
              "urgency", "sentiment", "confidence", "priority_score", "status"]],
        use_container_width=True,
    )

    st.subheader("Inspect & correct a ticket")
    ids = view["ticket_id"].tolist()
    tid = st.selectbox("Ticket", ids)
    row = view[view["ticket_id"] == tid].iloc[0]
    st.write(f"**Subject:** {row['subject']}")
    st.write(f"**Body:** {row['body']}")
    pred = json.loads(row["predictions_json"]) if row.get("predictions_json") else {}
    cols = st.columns(4)
    new_team = cols[0].selectbox("Team", sorted(df["team"].dropna().unique()),
                                 index=sorted(df["team"].dropna().unique()).index(row["team"])
                                 if row["team"] in df["team"].values else 0)
    new_type = cols[1].selectbox("Category", sorted(df["ticket_type"].dropna().unique()),
                                 index=sorted(df["ticket_type"].dropna().unique()).index(row["ticket_type"])
                                 if row["ticket_type"] in df["ticket_type"].values else 0)
    new_urg = cols[2].selectbox("Urgency", ["low", "medium", "high"],
                                index=["low", "medium", "high"].index(row["urgency"])
                                if row["urgency"] in ["low", "medium", "high"] else 1)
    new_sent = cols[3].selectbox("Sentiment", ["negative", "neutral", "positive"],
                                 index=["negative", "neutral", "positive"].index(row["sentiment"])
                                 if row["sentiment"] in ["negative", "neutral", "positive"] else 1)

    if st.button("Save corrections"):
        with connect(CFG) as conn:
            conn.execute(
                """UPDATE tickets SET team=?, ticket_type=?, urgency=?,
                sentiment=?, status='corrected', needs_review=0 WHERE ticket_id=?""",
                (new_team, new_type, new_urg, new_sent, tid),
            )
            # count as routing correction if auto-routed team changed
            if pred.get("team") and pred["team"] != new_team:
                metrics.record_routing_error(new_team)
        st.success("Corrections saved.")
        st.rerun()


def page_copilot():
    st.title("AI Support Copilot")
    copilot = get_copilot()
    df = load_tickets("WHERE queue='auto' OR needs_review=1 ORDER BY priority_score DESC")
    if df.empty:
        st.info("No tickets available.")
        return
    ids = df["ticket_id"].tolist()
    tid = st.selectbox("Select a ticket", ids)
    row = df[df["ticket_id"] == tid].iloc[0]
    query = f"{row['subject']} — {row['body']}"

    if st.button("Generate suggested response"):
        with st.spinner("Retrieving from knowledge base..."):
            result = copilot.answer(query)
        st.session_state[f"copilot_{tid}"] = result

    result = st.session_state.get(f"copilot_{tid}")
    if not result:
        return

    if not result["grounded"]:
        st.warning("Insufficient information in the knowledge base — "
                   "human expertise required.")
    st.subheader("Suggested response")
    st.write(result["answer"])
    st.caption(f"Backend: {result['backend']}")

    st.subheader("Sources")
    for s in result["sources"]:
        st.markdown(f"- **{s['doc_id']}** ({s['kind']}, sim={s['similarity']:.2f}) "
                    f"— {s['question'][:120]}")

    st.subheader("Feedback")
    action = st.radio("Action", ["accepted", "modified", "rejected"], horizontal=True)
    final = st.text_area("Final answer (if modified)", value=result["answer"])
    if st.button("Submit feedback"):
        with connect(CFG) as conn:
            record_feedback(conn, tid, result["answer"], final, action)
        metrics.record_copilot_action(action)
        st.success(f"Feedback recorded: {action}")


def page_incidents():
    st.title("Incident Monitoring")
    inc = load_incidents()
    if inc.empty:
        st.info("No incidents detected yet — run "
                "`python -m src.anomaly_detection.detector`.")
        return
    st.dataframe(inc[["alert_id", "detected_at", "team", "hour", "volume",
                      "zscore", "topic", "status"]], use_container_width=True)
    aid = st.selectbox("Alert", inc["alert_id"].tolist())
    a = inc[inc["alert_id"] == aid].iloc[0]
    st.write(f"**Team:** {a['team']}  |  **Window:** {a['hour']}  |  "
             f"**Volume:** {a['volume']} (baseline {a['baseline_mean']})  |  "
             f"**z-score:** {a['zscore']}")
    st.write(f"**Topic:** {a['topic']}")
    st.write(f"**Type distribution:** {a['ticket_type_dist']}")
    st.write(f"**Urgency distribution:** {a['urgency_dist']}")
    ids = json.loads(a["ticket_ids"]) if isinstance(a["ticket_ids"], str) else a["ticket_ids"]
    if ids:
        st.subheader("Related tickets")
        tdf = load_tickets()
        st.dataframe(tdf[tdf["ticket_id"].astype(str).isin(map(str, ids))]
                     [["ticket_id", "subject", "team", "urgency"]],
                     use_container_width=True)
    if st.button("Acknowledge alert"):
        path = Path(CFG["paths"]["processed_dir"]) / "incidents.csv"
        inc2 = inc.copy()
        inc2.loc[inc2["alert_id"] == aid, "status"] = "acknowledged"
        inc2.to_csv(path, index=False)
        st.success("Alert acknowledged.")
        st.rerun()


def page_faq():
    st.title("FAQ Management")
    sug = load_faq_suggestions()
    if sug.empty:
        st.info("No FAQ suggestions yet — uncovered-question clusters appear "
                "after copilot usage / `src.rag.faq_gaps` runs.")
        return
    pending = sug[sug["status"] == "pending"]
    st.subheader(f"Pending suggestions ({len(pending)})")
    for _, r in pending.iterrows():
        with st.expander(f"{r['question'][:100]}  (cluster={r['cluster_size']})"):
            st.write(r["question"])
            st.text_area("Draft answer", value=r["draft_answer"], key=f"ans_{r.name}",
                         height=100)
            c1, c2 = st.columns(2)
            if c1.button("Approve", key=f"ok_{r.name}"):
                approved = Path(CFG["paths"]["kb_dir"]) / "faq_approved.csv"
                new = pd.DataFrame([{
                    "question": r["question"],
                    "answer": r["draft_answer"],
                    "category": r["category"],
                    "team": "General Inquiry",
                }])
                new.to_csv(approved, mode="a", header=not approved.exists(), index=False)
                sug.loc[sug.index == r.name, "status"] = "approved"
                sug.to_csv(Path(CFG["paths"]["kb_dir"]) / "faq_suggestions.csv", index=False)
                st.success("Approved — re-run `python -m src.rag.indexer` to index it.")
            if c2.button("Reject", key=f"no_{r.name}"):
                sug.loc[sug.index == r.name, "status"] = "rejected"
                sug.to_csv(Path(CFG["paths"]["kb_dir"]) / "faq_suggestions.csv", index=False)
                st.rerun()


PAGES_FUNCS = {
    "Dashboard": page_dashboard,
    "Ticket Management": page_tickets,
    "AI Copilot": page_copilot,
    "Incident Monitoring": page_incidents,
    "FAQ Management": page_faq,
}

st.sidebar.title("Ticket Triage Copilot")
page = st.sidebar.radio("Navigate", PAGES)
PAGES_FUNCS[page]()
