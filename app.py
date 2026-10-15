"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


async def recent_sessions(limit: int = 10) -> list[dict[str, Any]]:
    """Return recent audit sessions for the sidebar."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MAX(timestamp) AS last_event,
                   CASE MAX(
                       CASE risk_level
                           WHEN 'high' THEN 3
                           WHEN 'med'  THEN 2
                           ELSE 1
                       END
                   )
                       WHEN 3 THEN 'high'
                       WHEN 2 THEN 'med'
                       ELSE 'low'
                   END AS worst_risk,
                   COUNT(*) AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def load_thread(thread_id: str) -> tuple[str, dict | None, dict | None]:
    """Restore the latest values and interrupt payload for a saved graph thread."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as checkpointer:
        await checkpointer.setup()
        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(config)

        values = dict(snapshot.values or {})
        interrupts = list(snapshot.interrupts)
        if not interrupts:
            for task in snapshot.tasks:
                interrupts.extend(task.interrupts)

        payload = interrupts[0].value if interrupts else None
        final = values if values and not snapshot.next else None
        return values.get("pr_url", ""), payload, final


async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Start or resume one graph thread."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as checkpointer:
        await checkpointer.setup()
        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            return await graph.ainvoke(
                {"pr_url": pr_url, "thread_id": thread_id}, config
            )
        return await graph.ainvoke(Command(resume=resume_value), config)


def save_result(result: dict) -> None:
    """Store a graph result in Streamlit session state."""
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.final = None
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result


def render_approval_card(payload: dict) -> dict | None:
    """Render the 58–72% approval card and return a resume value on click."""
    confidence = payload["confidence"]
    st.subheader(f"Approval requested — confidence {confidence:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for comment in payload.get("comments", []):
        line = comment.get("line") or "?"
        st.markdown(
            f"- **[{comment['severity']}]** `{comment['file']}:{line}` — "
            f"{comment['body']}"
        )

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    thread_id = st.session_state.thread_id
    feedback = st.text_input(
        "Feedback (optional)", key=f"approval_feedback_{thread_id}"
    )
    approve_col, reject_col, edit_col = st.columns(3)
    if approve_col.button("Approve", type="primary", key=f"approve_{thread_id}"):
        return {"choice": "approve", "feedback": feedback}
    if reject_col.button("Reject", key=f"reject_{thread_id}"):
        return {"choice": "reject", "feedback": feedback}
    if edit_col.button("Edit", key=f"edit_{thread_id}"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """Render the <58% escalation form and return the reviewer's answers."""
    confidence = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {confidence:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    thread_id = st.session_state.thread_id
    questions = payload.get("questions", [])
    with st.form(f"escalation_{thread_id}"):
        answers = {
            question: st.text_area(
                question,
                key=f"escalation_answer_{thread_id}_{index}",
            ).strip()
            for index, question in enumerate(questions)
        }
        submitted = st.form_submit_button("Submit answers", type="primary")

    if submitted:
        unanswered = [question for question, answer in answers.items() if not answer]
        if unanswered:
            st.warning("Please answer every escalation question before continuing.")
            return None
        return answers
    return None


# Streamlit requires page configuration before other UI calls.
st.set_page_config(page_title="HITL PR Review", page_icon="🔎", layout="wide")
st.title("HITL PR Review Agent")
st.caption("Confidence-based review with human approval, escalation, and audit replay")


for key, default in {
    "thread_id": None,
    "pr_url": "",
    "interrupt_payload": None,
    "final": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


with st.sidebar:
    st.header("Recent sessions")
    try:
        sessions = asyncio.run(recent_sessions())
    except Exception as exc:
        sessions = []
        st.error(f"Could not load audit sessions: {exc}")

    if not sessions:
        st.caption("No audited sessions yet.")
    for session in sessions:
        label = (
            f"{session['worst_risk'].upper()} · {session['events']} events\n"
            f"{session['pr_url']}"
        )
        if st.button(label, key=f"session_{session['thread_id']}", use_container_width=True):
            try:
                restored_pr, restored_interrupt, restored_final = asyncio.run(
                    load_thread(session["thread_id"])
                )
            except Exception as exc:
                st.error(f"Could not restore session: {exc}")
            else:
                st.session_state.thread_id = session["thread_id"]
                st.session_state.pr_url = restored_pr or session["pr_url"]
                st.session_state.interrupt_payload = restored_interrupt
                st.session_state.final = restored_final
                st.rerun()


with st.form("start"):
    pr_url = st.text_input(
        "PR URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review", type="primary")


if submitted:
    if not pr_url.strip():
        st.warning("Enter a GitHub pull-request URL.")
    else:
        st.session_state.pr_url = pr_url.strip()
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.interrupt_payload = None
        st.session_state.final = None
        try:
            with st.spinner("Fetching PR and asking the LLM..."):
                result = asyncio.run(
                    run_graph(st.session_state.pr_url, st.session_state.thread_id)
                )
            save_result(result)
        except Exception as exc:
            st.error(f"Review failed: {exc}")


payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload.get("kind")
    if kind == "approval_request":
        answer = render_approval_card(payload)
    elif kind == "escalation":
        answer = render_escalation_card(payload)
    else:
        st.error(f"Unknown interrupt kind: {kind}")
        answer = None

    if answer is not None:
        try:
            with st.spinner("Resuming review..."):
                result = asyncio.run(
                    run_graph(
                        st.session_state.pr_url,
                        st.session_state.thread_id,
                        resume_value=answer,
                    )
                )
            save_result(result)
            st.rerun()
        except Exception as exc:
            st.error(f"Could not resume review: {exc}")


if st.session_state.final is not None:
    final = st.session_state.final
    action = str(final.get("final_action") or "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"✓ {action} — comment posted")
        st.link_button("View pull request on GitHub", st.session_state.pr_url)
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    else:
        st.info(f"final_action = {action}")

    analysis = final.get("analysis")
    if analysis is not None:
        if isinstance(analysis, dict):
            confidence = analysis["confidence"]
            summary = analysis["summary"]
            comments = analysis.get("comments", [])
        else:
            confidence = analysis.confidence
            summary = analysis.summary
            comments = analysis.comments

        st.subheader(f"Final review — confidence {confidence:.0%}")
        st.markdown(summary)
        for comment in comments:
            if isinstance(comment, dict):
                severity = comment["severity"]
                file = comment["file"]
                line = comment.get("line") or "?"
                body = comment["body"]
            else:
                severity = comment.severity
                file = comment.file
                line = comment.line or "?"
                body = comment.body
            st.markdown(
                f"- **[{severity}]** `{file}:{line}` — {body}"
            )

    st.caption(
        f"thread_id = {st.session_state.thread_id} · replay: "
        f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )
