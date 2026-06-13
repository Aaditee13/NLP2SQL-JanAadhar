from __future__ import annotations

import streamlit as st

from app import generate_sql_pipeline
from database.excel_importer import import_excel_dataset
from database.query_results import execute_select_preview
from embeddings.faiss_store import FaissSchemaStore
from retrieval.few_shot_retriever import FaissFewShotStore


def render() -> None:
    st.set_page_config(page_title="Jan Aadhaar NL2SQL", layout="wide")
    st.title("Jan Aadhaar NL2SQL")

    with st.sidebar:
        st.header("Local Setup")
        auto_pull = st.checkbox("Pull missing Ollama models", value=False)
        run_profile = st.checkbox("Execute generated query for timing", value=False)
        bypass_cache = st.checkbox("Bypass semantic query cache", value=False)
        show_results = st.checkbox("Show matching entries", value=True)
        result_limit = st.number_input("Maximum displayed rows", min_value=10, max_value=1000, value=200, step=10)
        
        if st.button("Clear semantic query cache"):
            import os
            from config.settings import settings
            cache_path = settings.data_dir / "cache.faiss"
            metadata_path = settings.data_dir / "cache_metadata.json"
            if cache_path.exists():
                os.remove(cache_path)
            if metadata_path.exists():
                os.remove(metadata_path)
            st.success("Semantic query cache cleared.")

        if st.button("Load default dummy dataset"):
            with st.spinner("Loading records from Dummy_Data_Set.xlsx..."):
                try:
                    import_excel_dataset("dummy_dataset/Dummy_Data_Set.xlsx")
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.success("Demo database is ready using Dummy_Data_Set.xlsx.")
        uploaded_data = st.file_uploader("Import dummy Excel dataset", type=["xlsx"])
        if uploaded_data is not None and st.button("Load uploaded dataset"):
            with st.spinner("Loading records into the local SQLite database..."):
                try:
                    report = import_excel_dataset(uploaded_data, uploaded_data.name)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.success(f"Loaded {report.rows_loaded} citizen records.")
        if st.button("Rebuild schema & few-shot indices"):
            with st.spinner("Embedding metadata with Ollama and rebuilding FAISS..."):
                FaissSchemaStore().build(force=True)
                FaissFewShotStore().build(force=True)
            st.success("Indices rebuilt.")

    question = st.text_area(
        "Natural language question",
        value="Show all boys above 21 in Jaipur.",
        height=100,
    )
    if st.button("Generate SQL", type="primary"):
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

        tier_info = {
            "fast_path": (
                "⚡ Tier 0: Fast Path Engine",
                "Deterministic rule-based SQL generated instantly (< 5ms) without LLM calls.",
                "#F59E0B"
            ),
            "cache": (
                "🟢 Tier 1: Exact Cache Hit",
                "Retrieved matching SQL from semantic query cache (similarity >= 0.98).",
                "#10B981"
            ),
            "cache_swapped": (
                "🟢 Tier 1.5: Smart Cache (AST Swapped)",
                "Retrieved structurally similar query from cache (similarity >= 0.85) and swapped parameters (gender/district/age).",
                "#06B6D4"
            ),
            "llm": (
                "🤖 Tier 2: LLM Fallback",
                "Generated SQL using local LLM with dynamic schema context and semantic few-shots.",
                "#8B5CF6"
            )
        }
        
        info = tier_info.get(output.source)
        if info:
            title, desc, border_color = info
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:10px; background-color:#1E293B; border-left:5px solid {border_color}; margin-bottom:20px;">
                    <h4 style="margin:0; color:#F8FAFC; font-weight:600; font-family:'Inter', sans-serif;">{title}</h4>
                    <p style="margin:5px 0 0 0; color:#94A3B8; font-size:14px; font-family:'Inter', sans-serif;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("Generated SQL")
        st.code(output.sql, language="sql")

        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", output.confidence)
        c2.metric("Retrieved tables", len(output.retrieved_tables))
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

        if show_results and output.sql:
            if output.is_fuzzy:
                st.subheader(f"Similarity Matches for '{output.fuzzy_target}'")
                st.info(f"Showing results filtered by Jaro-Winkler similarity >= 0.80, sorted descending.")
            else:
                st.subheader("Matching Entries")
            try:
                preview = execute_select_preview(
                    output.sql,
                    max_rows=int(result_limit),
                    fuzzy_target=output.fuzzy_target,
                    is_fuzzy=output.is_fuzzy,
                )
            except Exception as exc:
                st.error(f"Results could not be displayed: {exc}")
            else:
                if preview.rows.empty:
                    st.info("The query returned no matching entries in the currently loaded dataset.")
                else:
                    st.dataframe(preview.rows, width="stretch", hide_index=True)
                    if output.is_fuzzy:
                        st.caption(
                            f"Showing {preview.displayed_rows} similarity match(es)"
                            + ("; more candidate rows exist in the database." if preview.truncated else ".")
                        )
                    else:
                        st.caption(
                            f"Showing {preview.displayed_rows} matching row(s)"
                            + ("; more rows exist." if preview.truncated else ".")
                        )
                    st.download_button(
                        "Download displayed results as CSV",
                        data=preview.rows.to_csv(index=False).encode("utf-8"),
                        file_name="query_results_preview.csv",
                        mime="text/csv",
                    )


        if output.optimization:
            st.subheader("Execution Plan")
            st.code("\n".join(output.optimization.execution_plan))
            st.metric("Planning / execution time", f"{output.optimization.execution_time_ms} ms")
            if output.optimization.index_recommendations:
                st.subheader("Index Recommendations")
                st.write(output.optimization.index_recommendations)


if __name__ == "__main__":
    render()
