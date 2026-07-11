from __future__ import annotations

import re

import streamlit as st

from app import generate_sql_pipeline, invalidate_cache_singleton
from database.excel_importer import import_excel_dataset
from database.query_results import execute_select_preview, compute_full_stats
from embeddings.faiss_store import FaissSchemaStore
from retrieval.few_shot_retriever import FaissFewShotStore
from ui.interpretation import interpret, generate_llm_insight, generate_followup_answer


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _md_to_html(text: str) -> str:
    """Convert basic markdown to HTML for safe embedding inside a <div>."""
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    # Horizontal rule
    text = re.sub(r"^---$", "<hr style='border-color:#1E4D6B; margin:12px 0;'>",
                  text, flags=re.MULTILINE)
    # Blockquotes (policy context lines)
    def _bq(m: re.Match) -> str:
        return (
            f"<blockquote style='border-left:3px solid #38BDF8; "
            f"padding-left:12px; color:#94A3B8; margin:8px 0 8px 0;'>"
            f"{m.group(1)}</blockquote>"
        )
    text = re.sub(r"^> (.+)$", _bq, text, flags=re.MULTILINE)
    # Paragraph breaks
    paragraphs = text.split("\n\n")
    html_parts = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if para.startswith("<hr") or para.startswith("<blockquote"):
            html_parts.append(para)
        else:
            para = para.replace("\n", "<br>")
            html_parts.append(f"<p style='margin:6px 0;'>{para}</p>")
    return "\n".join(html_parts)


def _render_direct_answer(direct_answer: str) -> None:
    """Renders the direct answer box as a single self-contained HTML block."""
    html_content = _md_to_html(direct_answer)
    st.markdown(
        f"""
        <div style="
            background-color:#0F2027;
            border:1px solid #1E4D6B;
            border-left:5px solid #38BDF8;
            border-radius:10px;
            padding:18px 22px;
            margin-bottom:12px;
        ">
            <p style="
                color:#38BDF8; font-size:13px; font-weight:600;
                letter-spacing:0.08em; text-transform:uppercase;
                margin:0 0 12px 0; font-family:'Inter',sans-serif;
            ">✅ Direct Answer to Your Question</p>
            <div style="
                color:#E2E8F0; font-size:15px;
                line-height:1.75; font-family:'Inter',sans-serif;
            ">
                {html_content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_llm_insight(text: str) -> None:
    st.markdown(
        f"""
        <div style="
            background-color:#1A1A2E;
            border:1px solid #6D28D9;
            border-left:5px solid #A78BFA;
            border-radius:10px;
            padding:18px 22px;
            margin:8px 0 16px 0;
        ">
            <p style="
                color:#A78BFA; font-size:13px; font-weight:600;
                letter-spacing:0.08em; text-transform:uppercase;
                margin:0 0 10px 0; font-family:'Inter',sans-serif;
            ">🤖 AI-Generated Specific Insight</p>
            <p style="
                color:#E2E8F0; font-size:15px;
                line-height:1.75; margin:0;
                font-family:'Inter',sans-serif;
            ">{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _show_execution_plan(output) -> None:
    st.subheader("Execution Plan")
    st.code("\n".join(output.optimization.execution_plan))
    st.metric("Planning / execution time",
              f"{output.optimization.execution_time_ms} ms")
    if output.optimization.index_recommendations:
        st.subheader("Index Recommendations")
        st.write(output.optimization.index_recommendations)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────
def render() -> None:
    st.set_page_config(page_title="Jan Aadhaar NL2SQL", layout="wide")
    st.title("Jan Aadhaar NL2SQL")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Local Setup")
        auto_pull    = st.checkbox("Pull missing Ollama models", value=False)
        run_profile  = st.checkbox("Execute generated query for timing", value=False)
        bypass_cache = st.checkbox("Bypass semantic query cache", value=False)
        show_results = st.checkbox("Show matching entries", value=True)
        result_limit = st.number_input(
            "Maximum displayed rows",
            min_value=10, max_value=1000, value=200, step=10,
        )

        if st.button("Clear semantic query cache"):
            import os
            from config.settings import settings
            for p in [settings.data_dir / "cache.faiss",
                      settings.data_dir / "cache_metadata.json"]:
                if p.exists():
                    os.remove(p)
            invalidate_cache_singleton()
            st.success("Semantic query cache cleared.")

        if st.button("Load default primary dataset (500K)"):
            with st.spinner("Loading records..."):
                try:
                    import_excel_dataset("new_dataset/Jan_Aadhaar_500K_FINAL.xlsx")
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.success("Primary database is ready.")

        uploaded_data = st.file_uploader("Import custom Excel dataset", type=["xlsx"])
        if uploaded_data is not None and st.button("Load uploaded dataset"):
            with st.spinner("Loading records into SQLite..."):
                try:
                    report = import_excel_dataset(uploaded_data, uploaded_data.name)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.success(f"Loaded {report.rows_loaded} citizen records.")

        if st.button("Rebuild schema & few-shot indices"):
            with st.spinner("Rebuilding FAISS indices..."):
                FaissSchemaStore().build(force=True)
                FaissFewShotStore().build(force=True)
            st.success("Indices rebuilt.")

    # ── Question + Generate button ─────────────────────────────────────────────
    question = st.text_area(
        "Natural language question",
        value="Show all boys above 21 in Jaipur.",
        height=100,
    )

    generate_clicked = st.button("Generate SQL", type="primary")

    # ── Run pipeline when Generate is clicked ─────────────────────────────────
    if generate_clicked:
        # Clear previous results and AI insight
        st.session_state.pop("pipeline_result", None)
        st.session_state.pop("llm_insight", None)
        st.session_state.pop("followup_chat", None)

        with st.spinner("Retrieving schema context and generating SQL locally..."):
            try:
                output = generate_sql_pipeline(
                    question,
                    ask_model_pull=auto_pull,
                    include_optimization=True,
                    run_query_for_profile=run_profile,
                    bypass_cache=bypass_cache,
                )
            except Exception as exc:
                st.error(str(exc))
                return

        # Fetch results immediately and store everything in session_state
        preview = None
        interp  = None
        if show_results and output.sql:
            try:
                preview = execute_select_preview(
                    output.sql,
                    max_rows=int(result_limit),
                    fuzzy_target=output.fuzzy_target,
                    is_fuzzy=output.is_fuzzy,
                )
                if not preview.rows.empty:
                    # Compute stats on FULL result set, not just 200 preview rows
                    full_stats = compute_full_stats(output.sql)
                    interp = interpret(
                        question=question,
                        sql=output.sql,
                        df=preview.rows,
                        total_rows=full_stats.get("total", preview.displayed_rows),
                        truncated=preview.truncated,
                        precomputed_stats=full_stats if full_stats else None,
                    )
            except Exception as exc:
                st.session_state["preview_error"] = str(exc)

        st.session_state["pipeline_result"] = {
            "question": question,
            "output":   output,
            "preview":  preview,
            "interp":   interp,
        }

    # ── Render from session_state (survives any button click) ─────────────────
    if "pipeline_result" not in st.session_state:
        return

    data     = st.session_state["pipeline_result"]
    question = data["question"]
    output   = data["output"]
    preview  = data["preview"]
    interp   = data["interp"]

    # Tier badge
    tier_info = {
        "fast_path": ("⚡ Tier 0: Fast Path Engine",
                      "Deterministic rule-based SQL generated instantly (< 5 ms) "
                      "without LLM calls.", "#F59E0B"),
        "cache":     ("🟢 Tier 1: Exact Cache Hit",
                      "Retrieved matching SQL from semantic query cache "
                      "(similarity >= 0.98).", "#10B981"),
        "cache_swapped": ("🟢 Tier 1.5: Smart Cache (AST Swapped)",
                          "Retrieved structurally similar query and swapped "
                          "parameters.", "#06B6D4"),
        "llm":       ("🤖 Tier 2: LLM Fallback",
                      "Generated SQL using local LLM with dynamic schema context "
                      "and semantic few-shots.", "#8B5CF6"),
    }
    info = tier_info.get(output.source)
    if info:
        title, desc, bc = info
        st.markdown(
            f"""<div style="padding:15px;border-radius:10px;
                background-color:#1E293B;border-left:5px solid {bc};
                margin-bottom:20px;">
                <h4 style="margin:0;color:#F8FAFC;font-weight:600;
                    font-family:'Inter',sans-serif;">{title}</h4>
                <p style="margin:5px 0 0 0;color:#94A3B8;font-size:14px;
                    font-family:'Inter',sans-serif;">{desc}</p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.subheader("Generated SQL")
    st.code(output.sql, language="sql")

    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence",        output.confidence)
    c2.metric("Retrieved tables",  len(output.retrieved_tables))
    c3.metric("Retrieved columns", len(output.retrieved_columns))

    if output.query_corrections:
        st.subheader("Query Corrections")
        st.write(output.query_corrections)
        st.caption(f"Normalized question: {output.normalized_question}")

    left, right = st.columns(2)
    with left:
        st.subheader("Retrieved Tables")
        st.write(output.retrieved_tables)
    with right:
        st.subheader("Retrieved Columns")
        st.write(output.retrieved_columns)

    if output.validation_errors:
        st.subheader("Validation Errors")
        st.error("; ".join(output.validation_errors))

    # ── Results + Interpretation ───────────────────────────────────────────────
    if show_results and output.sql:
        if "preview_error" in st.session_state:
            st.error(f"Results could not be displayed: "
                     f"{st.session_state['preview_error']}")
        elif preview is None:
            pass
        elif preview.rows.empty:
            st.info("The query returned no matching entries in the currently loaded dataset.")
        else:
            if output.is_fuzzy:
                st.subheader(f"Similarity Matches for '{output.fuzzy_target}'")
                st.info("Filtered by Jaro-Winkler similarity >= 0.80, sorted descending.")
            else:
                st.subheader("Matching Entries")

            st.dataframe(preview.rows, use_container_width=True, hide_index=True)

            if output.is_fuzzy:
                st.caption(
                    f"Showing {preview.displayed_rows} similarity match(es)"
                    + ("; more exist." if preview.truncated else ".")
                )
            else:
                st.caption(
                    f"Showing {preview.displayed_rows} matching row(s)"
                    + ("; more rows exist — download CSV for full set."
                       if preview.truncated else ".")
                )

            st.download_button(
                "⬇️ Download displayed results as CSV",
                data=preview.rows.to_csv(index=False).encode("utf-8"),
                file_name="query_results_preview.csv",
                mime="text/csv",
            )

            if interp:
                st.markdown("---")
                st.markdown("## 📊 Result Interpretation")
                st.caption(
                    "Auto-generated analysis of the results above. "
                    "Choose the view that suits your background."
                )

                # Direct answer box — single HTML block, content inside the box
                _render_direct_answer(interp.direct_answer)

                # AI Insight button — persists because output is in session_state
                col_btn, col_note = st.columns([1, 3])
                with col_btn:
                    ai_clicked = st.button(
                        "🤖 Generate AI Insight",
                        help="Uses local Ollama to write a specific 3-sentence "
                             "interpretation. Takes 5–15 seconds.",
                    )
                with col_note:
                    st.caption(
                        "Calls your local Ollama model — takes 5–15 seconds. "
                        "No data leaves your machine."
                    )

                if ai_clicked:
                    with st.spinner("Generating specific AI insight..."):
                        insight = generate_llm_insight(
                            question=question,
                            sql=output.sql,
                            stats=interp.stats,
                            total=preview.displayed_rows,
                            truncated=preview.truncated,
                            intents=interp.intents,
                        )
                    st.session_state["llm_insight"] = insight

                if "llm_insight" in st.session_state:
                    _render_llm_insight(st.session_state["llm_insight"])

                # Three audience tabs
                st.markdown("<br>", unsafe_allow_html=True)
                tab_s, tab_t, tab_e = st.tabs([
                    "🟢  Simple — For Everyone",
                    "🔵  Technical — For Analysts",
                    "🟣  Executive — For Decision Makers",
                ])
                with tab_s:
                    st.markdown(interp.simple)
                with tab_t:
                    st.markdown(interp.technical)
                with tab_e:
                    st.markdown(interp.executive)
                # ── FOLLOW-UP Q&A CHAT ────────────────────────────────────
                st.markdown("---")
                st.markdown("### 💬 Ask a Follow-up Question")
                st.caption(
                    "Ask anything about these results. "
                    "The AI answers based on the data above and remembers "
                    "your conversation."
                )

                # Initialise chat history for this result set
                if "followup_chat" not in st.session_state:
                    st.session_state["followup_chat"] = []

                # Render existing chat messages
                for msg in st.session_state["followup_chat"]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                # Chat input — does not navigate away on submit
                followup_q = st.chat_input(
                    "e.g. How many of them are likely eligible for widow pension?"
                )

                if followup_q:
                    # Show user message immediately
                    st.session_state["followup_chat"].append(
                        {"role": "user", "content": followup_q}
                    )
                    with st.chat_message("user"):
                        st.markdown(followup_q)

                    # Generate and show assistant answer
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            answer = generate_followup_answer(
                                original_question=question,
                                sql=output.sql,
                                stats=interp.stats,
                                total=preview.displayed_rows,
                                truncated=preview.truncated,
                                intents=interp.intents,
                                chat_history=st.session_state["followup_chat"][:-1],
                                followup_question=followup_q,
                            )
                        st.markdown(answer)

                    st.session_state["followup_chat"].append(
                        {"role": "assistant", "content": answer}
                    )

    if output.optimization:
        _show_execution_plan(output)


if __name__ == "__main__":
    render()