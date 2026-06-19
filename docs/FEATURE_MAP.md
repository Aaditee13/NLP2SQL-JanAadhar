# Feature Map: Jan Aadhaar NL2SQL

This document maps every feature of the system to its implementation status, key files, entry points, and any known gaps.

---

## Feature Status Legend

- **Complete** — fully implemented and tested
- **Partial** — implemented but with known gaps, bugs, or missing edge cases
- **Stub** — file/module exists but content is minimal or placeholder
- **Planned** — mentioned in docs/README but not yet implemented

---

## Core Pipeline Features

### 1. Natural Language to SQL (End-to-End Pipeline)

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `app.py:generate_sql_pipeline()`, `app.py:PipelineOutput` |
| **Entry point** | `generate_sql_pipeline(question, ...)` at `app.py:732` |
| **Notes** | Orchestrates all sub-modules: normalize → embed → FAISS → retrieve → prompt → generate → post-process → validate → optimize. Max 3 retry attempts with error feedback. Returns empty SQL if all attempts fail. |

---

### 2. Query Typo Normalization

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `normalization/query_normalizer.py:QueryNormalizer`, `normalization/query_normalizer.py:normalize_query()` |
| **Entry point** | `normalize_query(question)` at `normalization/query_normalizer.py:183` |
| **Notes** | RapidFuzz WRatio scoring at threshold 88. Context-aware: location-protected tokens (after "in/from/at") stay at 88; person-protected tokens (after "named/called" or in quotes) require 95. Bank abbreviations (`sbi`, `pnb`, `bob`, etc.) expanded to full stored DB names. Direct corrections dictionary for known typos. |

---

### 3. Ollama Embedding with nomic-embed-text

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `embeddings/ollama_embeddings.py:OllamaEmbedder` |
| **Entry point** | `OllamaEmbedder.embed(text)` — called by `FaissSchemaStore` |
| **Notes** | Calls `ollama.Client.embeddings()`. Normalizes vector to unit length (so IndexFlatIP = cosine similarity). Single-text and batch versions available. |

---

### 4. FAISS Schema Index (Build, Load, Search)

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `embeddings/faiss_store.py:FaissSchemaStore`, `data/schema.faiss`, `data/schema_metadata.json` |
| **Entry point** | `FaissSchemaStore().build()` or `FaissSchemaStore().search(question)` |
| **Notes** | Uses `IndexFlatIP`. Persisted to disk; loaded on each app start unless `force=True`. Index contains schema documents (tables + columns), not citizen records. Rebuild via `--build-index` CLI flag or "Rebuild schema index" button in Streamlit sidebar. |

---

### 5. Hybrid Schema Retrieval with Domain Gating

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `retrieval/schema_retriever.py:SchemaRetriever` |
| **Entry point** | `SchemaRetriever(store).retrieve(question)` at `retrieval/schema_retriever.py:90` |
| **Notes** | Combines vector recall, lexical alias matching, domain term gates (BANK_TERMS, CASTE_TERMS, EDUCATION_TERMS, MINORITY_TERMS, RURAL_TERMS), geography injection, join key enrichment, and column pruning. Returns `RetrievalResult` with tables, columns, relationships, confidence score (avg top-5 FAISS scores). |

---

### 6. Dynamic RAG Prompt Builder

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `prompting/prompt_builder.py:PromptBuilder` |
| **Entry point** | `PromptBuilder().build(retrieval_result, previous_error, dialect)` at `prompting/prompt_builder.py:77` |
| **Notes** | Builds ~500-token prompt with SQL safety constraints, location pre-classification (district vs block/village), dynamic per-column rules (education casing, caste, bank names, is_rural, family member count, minority, unbanked). Supports sqlite and postgresql dialect hints. |

---

### 7. LLM SQL Generation via Ollama Qwen

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `llm/ollama_client.py:OllamaSqlGenerator`, `llm/ollama_client.py:OllamaModelManager` |
| **Entry point** | `OllamaSqlGenerator().generate(prompt)` at `llm/ollama_client.py:78` |
| **Notes** | temperature=0, top_p=0.1, num_ctx=2048, num_predict=256, keep_alive=30m. `_clean_sql()` strips markdown code fences. `OllamaModelManager` handles model checking and pulling with CLI input prompt or Streamlit checkbox. |

---

### 8. SQL Post-Processing (19-Step Repair Pipeline)

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `app.py:_post_process_sql()` (lines 76-661), `app.py:_fix_no_bank_sql()` (lines 664-729) |
| **Entry point** | Called in the generation loop at `app.py:768-769` |
| **Notes** | Addresses systematic LLM output errors: free-text LIKE rewrites, bank name UPPER(), categorical casing normalization, education LOWER(), is_rural integer mapping, district casing/redirect, IN clause handling (district, bank, caste, categorical), alias mismatch correction, spurious JOIN pruning, transitive JOIN injection, family member count fixes. Step 1.0 now emits a dual-arm OR clause for fuzzy queries: `(col LIKE '%prefix%' OR col_phonetic LIKE 'phonetic_key%')`. |

---

### 9. SQL Validation (Read-Only + Schema + Join Checks)

| Field | Detail |
|---|---|
| **Status** | Partial |
| **Key files** | `validation/sql_validator.py:SQLValidator` |
| **Entry point** | `SQLValidator().validate(sql, allowed_tables, allowed_columns)` |
| **Notes** | Validates SELECT-only, no DDL/DML keywords, known tables/columns, valid join relationships, columns within retrieved context. Columns ending in `_phonetic` are whitelisted as internal infrastructure (injected by post-processor, never shown to LLM). **Known issues**: (1) Active debug `print()` statements (lines 108-118) pollute stdout on every call. (2) Column context validation exempts `member` and `family` tables (`col.split(".")[0].lower() not in {"member", "family"}`), weakening the context boundary check. |

---

### 10. Retry Loop with Error Feedback

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `app.py:generate_sql_pipeline()` lines 761-779 |
| **Entry point** | Part of `generate_sql_pipeline()` |
| **Notes** | Runs up to `settings.max_retries` (default 3) times. On failure, `previous_error` is passed to `PromptBuilder.build()` which injects it as "Previous SQL was invalid: {error}. Fix it." Returns empty string if all attempts fail. |

---

### 11. SQL EXPLAIN Plan and Index Recommendations

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `optimization/query_optimizer.py:QueryOptimizer` |
| **Entry point** | `QueryOptimizer().profile(sql, run_query=False)` |
| **Notes** | Runs `EXPLAIN QUERY PLAN` on SQLite. `recommend_indexes()` scans all `COLUMNS` entries for non-indexed columns appearing in the SQL. Optional actual query execution for timing (only when `run_query=True`). |

---

## Data Pipeline Features

### 12. Excel Dataset Import

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `database/excel_importer.py:import_excel_dataset()`, `database/excel_importer.py:DatasetImportReport` |
| **Entry point** | `import_excel_dataset(source, source_name, database_url)` |
| **Notes** | Reads `.xlsx` (or `.csv` — partial, not exposed in UI/CLI). Groups by `ENROLLMENT_ID`. Identifies HOF by `MEM_TYPE='HOF'` or `RELATION_WITH_HOF='self'`. Cleans caste values (strips leading digits, title-cases). Computes `phonetic_key()` for all name columns and stores results in `*_phonetic` columns. Runs `ALTER TABLE ADD COLUMN` migrations (try/except guarded) so existing databases gain phonetic columns automatically on next import. Truncates tables before insert (SQLite: DELETE; PostgreSQL: TRUNCATE RESTART IDENTITY). Returns `DatasetImportReport` with counts. |

---

### 13. Demo Database Seeding

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `app.py:run_cli()` lines 814-816, `ui/streamlit_app.py` lines 21-28 |
| **Entry point** | CLI: `python app.py --seed-demo-db` / Streamlit: "Load default dummy dataset" button |
| **Notes** | Calls `import_excel_dataset("dummy_dataset/Dummy_Data_Set.xlsx")`. The actual dummy dataset is `dummy_dataset/Dummy_Data_Set.xlsx` (not committed to git in readable form, but present as binary). |

---

### 14. Query Result Preview and Execution

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `database/query_results.py:execute_select_preview()`, `database/query_results.py:QueryResultPreview` |
| **Entry point** | `execute_select_preview(sql, max_rows, database_url, fuzzy_target, is_fuzzy, threshold)` |
| **Notes** | Always re-validates SQL before execution. Wraps query in `SELECT * FROM (...) LIMIT N+1` subquery. For fuzzy queries: fetches 1000 rows and applies Jaro-Winkler reranking. Returns `QueryResultPreview { rows: DataFrame, truncated, displayed_rows }`. |

---

## UI Features

### 15. Streamlit Web Interface

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `ui/streamlit_app.py:render()` |
| **Entry point** | `streamlit run app.py` → `render()` via `_is_streamlit()` detection |
| **Notes** | Full-featured UI: question input, SQL display, confidence/table/column metrics, query corrections, retrieved schema lists, validation error display, results dataframe, CSV download, EXPLAIN plan. Sidebar: auto-pull models, timing toggle, show-results toggle, row limit selector, dataset load buttons, schema index rebuild. |

---

### 16. CSV Download of Results

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `ui/streamlit_app.py` lines 116-121 |
| **Entry point** | "Download displayed results as CSV" button in Streamlit |
| **Notes** | Uses `st.download_button` with `preview.rows.to_csv(index=False).encode('utf-8')`. Filename: `query_results_preview.csv`. |

---

### 17. CLI Interface

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `app.py:run_cli()` lines 803-879 |
| **Entry point** | `python app.py [question] [flags]` |
| **Notes** | Flags: `--seed-demo-db`, `--build-index`, `--import-excel <path>`, `--show-results`, `--no-explain`, `--run-profile-query`. Interactive mode (no question arg) prompts via `input()`. |

---

## Intelligence Features

### 18. Fuzzy Name Search (Phonetic + Jaro-Winkler)

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `normalization/fuzzy_match.py:phonetic_key()`, `normalization/fuzzy_match.py:is_fuzzy_intent()`, `normalization/fuzzy_match.py:extract_fuzzy_target()`, `normalization/fuzzy_match.py:fuzzy_rerank()` |
| **Entry point** | Intent detected in `app.py:generate_sql_pipeline()` lines 751-759; applied in `database/query_results.py:execute_select_preview()` |
| **Notes** | Detects triggers: "similar to", "name like", "sounds like", "spelled like", "fuzzy search for", "approximate matches for", "resembling". Extracts target name. SQL filter uses a dual-arm OR: `(col LIKE '%prefix%' OR col_phonetic LIKE 'phonetic_key%')` — the orthographic arm catches close spellings (Poonam/Poonem), the phonetic arm catches systematic romanization variants where the first 3 chars differ (Poonam→Poo vs Punam→Pun, Shweta→Shw vs Sweta→Swe). Post-execution reranking in `fuzzy_rerank()` uses three strategies (best wins): full-string JW, per-word JW for single-word targets, and phonetic key match (floors score at 0.90). JW threshold: 0.80 default. |

---

### 19. Bilingual Caste Expansion

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `app.py:_CASTE_GROUPS` (lines 36-56), `app.py:_post_process_sql()` steps 1.1 and 1.2 |
| **Entry point** | Part of `_post_process_sql()` |
| **Notes** | 20 caste groups with English and Hindi variants. A LIKE for 'Rajput' expands to `(caste LIKE '%Rajput%' OR caste LIKE '%Rajpoot%' OR caste LIKE '%राजपूत%')`. IN clauses also expanded. Triggered for both `=` and `LIKE` and `IN (...)` on the `caste` column. |

---

### 20. Location Classification and District Redirect

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `prompting/prompt_builder.py:_extract_location_hints()`, `prompting/prompt_builder.py:_classify_location()`, `app.py:_post_process_sql()` steps 6/8/8.5/9 |
| **Entry point** | Called during `PromptBuilder.build()` and `_post_process_sql()` |
| **Notes** | Extracts location tokens from "in/from/at" prepositions including conjunctions ("Jaipur and Ajmer"). Classifies each as known Rajasthan district or unknown. Injects prompt rules pre-emptively. Post-process also catches district-formatted SQL with non-district place names and redirects to `block LIKE '%val%' OR village LIKE '%val%'`. |

---

### 21. Unbanked / No-Bank-Account Query Handling

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `app.py:_NO_BANK_WORDS`, `app.py:_fix_no_bank_sql()` lines 664-729 |
| **Entry point** | Called after `_post_process_sql()` in the generation loop at `app.py:769` |
| **Notes** | Detects phrases: "no bank", "without bank", "don't have", "do not have", "no account", "unbanked", "without account". Injects `LEFT JOIN bank_details ... WHERE bank_details.bank_id IS NULL`. Promotes INNER JOIN to LEFT JOIN. Strips spurious `member_type = 'MEM'` the LLM often adds. |

---

## Evaluation and Testing Features

### 22. Benchmark Runner

| Field | Detail |
|---|---|
| **Status** | Partial |
| **Key files** | `evaluation/benchmark.py:run_benchmark()`, `evaluation/benchmark_cases.json` |
| **Entry point** | `python -m evaluation.benchmark` |
| **Notes** | Only 3 benchmark cases. Measures exact_match, schema_accuracy, retrieval_accuracy, latency_ms. **Bug**: `retrieval_accuracy` computation compares `output.retrieved_columns` (list of `table.column`) against `expected_terms` extracted as tokens containing "." from the normalized expected SQL. This is meaningful only if expected SQL contains qualified column names — inconsistent across the 3 cases. |

---

### 23. Unit Tests

| Field | Detail |
|---|---|
| **Status** | Partial |
| **Key files** | `tests/test_sql_validator.py`, `tests/test_prompt_builder.py`, `tests/test_schema_retriever.py`, `tests/test_schema_metadata.py`, `tests/test_query_normalizer.py`, `tests/test_fuzzy_match.py`, `tests/test_ollama_client.py`, `tests/test_excel_results.py` |
| **Entry point** | `pytest` from project root |
| **Notes** | All tests are unit tests. No integration tests invoking the full pipeline (Ollama is not mocked in any test except `StubModelManager` in `test_ollama_client.py`). `test_excel_results.py` creates an in-memory SQLite test database. Missing: tests for `_fix_no_bank_sql()`, `generate_sql_pipeline()` end-to-end, optimization module, evaluation module. |

---

## Configuration and Infrastructure Features

### 24. Environment Variable Configuration

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `config/settings.py:Settings` |
| **Entry point** | `from config.settings import settings` |
| **Notes** | Frozen dataclass, all settings readable at import. Supports `OLLAMA_BASE_URL`, `SQL_MODEL`, `EMBEDDING_MODEL`, `OLLAMA_KEEP_ALIVE`, `DATABASE_URL`, `MAX_SQL_RETRIES`, `RETRIEVAL_TOP_K`. |

---

### 25. Environment Verification Script

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `scripts/verify_environment.py` |
| **Entry point** | `python scripts/verify_environment.py` |
| **Notes** | Checks all required Python packages, ollama executable presence, Ollama server reachability, and required model availability. |

---

### 26. Windows PowerShell Setup Script

| Field | Detail |
|---|---|
| **Status** | Complete (Windows only) |
| **Key files** | `scripts/setup.ps1` |
| **Entry point** | `.\scripts\setup.ps1` in PowerShell |
| **Notes** | Runs pip install, verify_environment. Does NOT pull Ollama models automatically — prints the pull commands for the user to run. No equivalent `setup.sh` for macOS/Linux. |

---

### 27. PDF Project Guide Generation

| Field | Detail |
|---|---|
| **Status** | Complete |
| **Key files** | `scripts/generate_project_guide_pdf.py`, `docs/Jan_Aadhaar_NL2SQL_Project_Guide.pdf` |
| **Entry point** | `python scripts/generate_project_guide_pdf.py` |
| **Notes** | Generates a 14-section PDF using ReportLab. Includes architecture flowchart and schema relationship diagram drawn programmatically as custom `Flowable` objects. Output: `docs/Jan_Aadhaar_NL2SQL_Project_Guide.pdf`. |

---

## Planned / Missing Features (Per README / Docs)

### 28. Pension / NFSA / eKYC Query Support

| Field | Detail |
|---|---|
| **Status** | Planned (not implemented) |
| **Key files** | None — columns not in `database/models.py` or `database/schema_metadata.py` |
| **Entry point** | N/A |
| **Notes** | README example questions mention pension, NFSA status, eKYC pending. No such columns exist in the current schema. These are likely production Jan Aadhaar columns not yet added to the demo. |

---

### 29. Production Authorization / Row-Level Security

| Field | Detail |
|---|---|
| **Status** | Planned |
| **Key files** | `docs/production_deployment.md` (describes requirements) |
| **Entry point** | N/A |
| **Notes** | District/department-level authorization predicates, role-based column masking, query allow-list for sensitive columns. Not implemented. |

---

### 30. Mandatory LIMIT on List Queries

| Field | Detail |
|---|---|
| **Status** | Planned |
| **Key files** | `docs/production_deployment.md` |
| **Entry point** | N/A |
| **Notes** | No enforcement today. For production, queries returning potentially millions of rows need a mandatory LIMIT injected by the validator or post-processor. |

---

### 31. CSV Import via UI

| Field | Detail |
|---|---|
| **Status** | Partial (backend exists, not exposed) |
| **Key files** | `database/excel_importer.py:import_excel_dataset()` line 53 |
| **Entry point** | N/A in UI |
| **Notes** | `import_excel_dataset()` has a CSV branch triggered by filename ending in `.csv`. The Streamlit file uploader only accepts `["xlsx"]` (`ui/streamlit_app.py:29`). CLI `--import-excel` flag is not documented to support CSV. The CSV path is not tested. |
