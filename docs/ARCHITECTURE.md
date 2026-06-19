# Architecture: Jan Aadhaar NL2SQL

## Overview

The system is a **fully local Retrieval-Augmented Generation (RAG) pipeline** for natural language to SQL translation. No external APIs are called at any point. All computation runs on the local machine using Ollama for LLM inference and FAISS for vector search.

---

## End-to-End Data Flow

```
User (Streamlit UI or CLI)
│
│  Natural language question
│  e.g. "Show all female beneficiaries receiving pension in Jaipur district."
│
▼
[1] QUERY NORMALIZATION                  normalization/query_normalizer.py
│   RapidFuzz corrects typos in:
│   - Rajasthan district names (jaipor → Jaipur)
│   - Known terms (femail → female, benificiaries → beneficiaries)
│   - Proper-noun protection (person names and quoted tokens use 95 threshold)
│   - Bank abbreviations expanded (sbi → STATE BANK OF INDIA)
│   Returns: QueryNormalizationResult { original, normalized, corrections }
│
▼
[2] QUERY EMBEDDING                      embeddings/ollama_embeddings.py
│   OllamaEmbedder.embed(normalized_question)
│   → POST http://localhost:11434  (nomic-embed-text model)
│   → 768-dim float32 vector, L2-normalized
│
▼
[3] FAISS SCHEMA SEARCH                  embeddings/faiss_store.py
│   FaissSchemaStore.search(question, top_k=16)
│   → IndexFlatIP.search() on pre-built schema.faiss
│   → Returns top-16 scored schema documents (tables + columns)
│   Index location: data/schema.faiss
│   Metadata: data/schema_metadata.json
│
▼
[4] HYBRID SCHEMA RETRIEVAL              retrieval/schema_retriever.py
│   SchemaRetriever.retrieve(question)
│   Combines:
│     a) Vector results (top 6 semantic columns not already lexically matched)
│     b) Lexical matching (column aliases, district names, sample values)
│     c) Domain gating (bank columns only if banking terms present; caste_category
│        only if SC/ST/OBC/GEN terms present; minority only if minority terms present)
│     d) Geography injection (family table if location preposition found)
│     e) Join key enrichment (add foreign keys if both sides of join are relevant)
│     f) Display enrichment (add member_name if "show/list/find" detected)
│     g) Column pruning (_prune_columns removes columns that fail domain checks)
│   Returns: RetrievalResult { tables, columns, relationships, confidence }
│
▼
[5] FUZZY INTENT DETECTION               normalization/fuzzy_match.py
│   is_fuzzy_intent(question): detects "similar to", "name like", "sounds like", etc.
│   extract_fuzzy_target(question): extracts the target name from fuzzy queries
│
▼
[6] PROMPT CONSTRUCTION                  prompting/prompt_builder.py
│   PromptBuilder.build(retrieval_result)
│   Builds a ~500-line structured prompt containing:
│   - System role and SQL constraints (no DDL, no SELECT *, read-only)
│   - Location pre-classification (district vs block/village rules injected)
│   - Dynamic column-specific rules (education casing, minority NULL handling,
│     caste IN expansion guidance, bank UPPER() rules, unbanked LEFT JOIN,
│     is_rural integer encoding, family member count rules)
│   - Available tables list
│   - Relevant columns with descriptions, types, sample values
│   - Allowed relationships (JOIN keys only)
│   - Error feedback (if retry: includes previous validation error)
│   - User question
│
▼
[7] SQL GENERATION (LLM)                 llm/ollama_client.py
│   OllamaSqlGenerator.generate(prompt)
│   → POST http://localhost:11434/api/generate  (qwen2.5-coder:3b model)
│   Settings: temperature=0, top_p=0.1, num_ctx=2048, num_predict=256
│   _clean_sql() strips markdown code fences, ensures trailing semicolon
│   Returns: raw SQL string
│
▼
[8] SQL POST-PROCESSING                  app.py: _post_process_sql(), _fix_no_bank_sql()
│   19-step regex pipeline (in _post_process_sql):
│     Step 0:  Fix mismatched table alias prefixes
│     Step 1:  Convert = to LIKE for free-text columns (names, villages, etc.)
│     Step 1.0: Fuzzy name broadening — dual-arm OR: orthographic prefix LIKE '%XXX%' + phonetic LIKE 'key%'
│     Step 1.1: Caste IN clause → bilingual LIKE expansion
│     Step 1.2: Caste LIKE → bilingual expansion (English + Hindi variants)
│     Step 1.5: Strip spurious IS NOT NULL on nullable location sub-columns
│     Step 1.6: Prune unused family JOIN
│     Step 2:  bank_name = → UPPER(bank_name) LIKE
│     Step 2.5: bank_name IN → UPPER(bank_name) IN (uppercased values)
│     Step 3:  Categorical value normalization (gender, caste_category, marital_status)
│     Step 3.5: Categorical IN clause normalization
│     Step 4:  Categorical LIKE → = for exact-match categoricals
│     Step 4:  education 'illiterate' → LOWER() handling
│     Step 5:  is_rural text/boolean → INTEGER 0 or 1
│     Step 6:  District casing → canonical Rajasthan district name
│     Step 8:  District LIKE → = for known districts
│     Step 8.5: district IN with non-district values → block/village redirect
│     Step 9:  district = non-district-value → block/village LIKE redirect
│     Steps 10-15: Family/member count query corrections
│     Step 16: bank_account_number → bank_account alias normalization
│     Step 17: Strip trailing LLM commentary after semicolon
│     Step 18: Inject missing JOIN if column prefix references unjoined table
│     Step 19: Re-run alias prefix correction after any injected tables
│   _fix_no_bank_sql(): LEFT JOIN injection and IS NULL for unbanked queries
│
▼
[9] SQL VALIDATION                       validation/sql_validator.py
│   SQLValidator.validate(sql, allowed_tables, allowed_columns)
│   Checks:
│   - Parseable (sqlparse)
│   - Single statement
│   - SELECT only (not INSERT/UPDATE/DELETE/DROP/etc.)
│   - No semicolons within statement
│   - No disallowed keywords
│   - All tables are in known schema
│   - All tables are within retrieved context
│   - All qualified columns exist (table.column); `*_phonetic` columns exempt as internal infrastructure
│   - All qualified column table references are in FROM/JOIN
│   - Columns outside retrieved context rejected (with exception for member/family)
│   - All JOINs use declared relationships
│   Returns: ValidationResult { valid, errors }
│
▼
[10] RETRY LOOP                          app.py: generate_sql_pipeline()
│   If validation.valid == False and attempts < settings.max_retries (3):
│     → previous_error injected into next prompt → back to step 6
│   If max retries exhausted: sql = "" (no output)
│
▼
[11] OPTIMIZATION                        optimization/query_optimizer.py
│   QueryOptimizer.profile(sql, run_query=False)
│   → SQLite EXPLAIN QUERY PLAN (always)
│   → Optional actual execution (only if run_query_for_profile=True)
│   → recommend_indexes(): scans COLUMNS metadata for non-indexed columns in SQL
│   Returns: OptimizationReport { execution_plan, execution_time_ms, index_recommendations }
│
▼
[12] RESULT DISPLAY                      ui/streamlit_app.py OR app.py run_cli()
│   - Generated SQL (code block in UI)
│   - Confidence score (average of top-5 FAISS scores)
│   - Retrieved tables and columns
│   - Spelling corrections applied
│   - Validation errors (if any)
│   - Fuzzy match results (if fuzzy query)
│   - Database result rows (execute_select_preview, capped at result_limit)
│   - EXPLAIN plan and timing
│   - Index recommendations
│   - CSV download button (Streamlit only)
```

---

## Component Descriptions

### `config/settings.py`
A frozen `@dataclass` providing all runtime configuration. Reads from environment variables with sensible defaults. The `database_url` property dynamically selects between the `DATABASE_URL` env var and the local SQLite path.

Key settings:
- `sql_model` = `"qwen2.5-coder:3b"` (env: `SQL_MODEL`)
- `embedding_model` = `"nomic-embed-text"` (env: `EMBEDDING_MODEL`)
- `ollama_keep_alive` = `"30m"` (env: `OLLAMA_KEEP_ALIVE`)
- `max_retries` = `3` (env: `MAX_SQL_RETRIES`)
- `retrieval_top_k` = `16` (env: `RETRIEVAL_TOP_K`)
- `faiss_index_path` = `data/schema.faiss`
- `sqlite_path` = `data/jan_aadhaar_demo.sqlite`

### `database/models.py`
SQLAlchemy 2.0 declarative models. Three tables:
- `Family` — household geography (district, block, village, ward, is_rural). Includes `family_head_name_phonetic` (indexed) for phonetic name search.
- `Member` — citizen demographics (age, gender, caste, income, occupation, education, etc.). Includes `member_name_phonetic` (indexed), `father_name_phonetic`, `mother_name_phonetic`, `spouse_name_phonetic` for phonetic name search.
- `BankDetails` — bank account and DBT status

Two composite indexes defined at module level: `ix_member_gender_caste_age`, `ix_family_geo`.

### `database/schema_metadata.py`
Authoritative schema description layer. Contains:
- `ColumnMeta` dataclass: physical name, description, data_type, aliases, sample_values, indexed flag, semantic_name
- `TableMeta` dataclass: table name, description, aliases
- `RAJASTHAN_DISTRICTS_41`: canonical list of all 41 Rajasthan districts
- `TABLES`: 3 table entries
- `COLUMNS`: ~30 column entries with rich alias and sample value lists
- `RELATIONSHIPS`: 2 declared foreign key join paths
- Helper functions: `all_table_names()`, `columns_by_table()`

This file is the single source of truth for what the LLM is allowed to know about the schema.

### `database/excel_importer.py`
Reads `.xlsx` (or `.csv`) files in the Jan Aadhaar export column format. Groups rows by `ENROLLMENT_ID` to identify families. Identifies HOF (Head of Family) by `MEM_TYPE='HOF'` or `RELATION_WITH_HOF='self'`. Computes `phonetic_key()` for all name columns (`member_name`, `father_name`, `mother_name`, `spouse_name`, `family_head_name`) at import time and stores them in the corresponding `*_phonetic` columns. Runs `ALTER TABLE ADD COLUMN` migrations (guarded by try/except) so existing databases gain the phonetic columns automatically on next import. Bulk-inserts `Family`, `Member`, `BankDetails` records after truncating the target tables. Returns `DatasetImportReport`.

Required columns: `DISTRICT_NAME_ENG`, `ENROLLMENT_ID`, `MEMBER_ID`, `NAME_EN`, `AGE`, `GENDER`.

### `database/query_results.py`
`execute_select_preview()` — validates SQL, wraps it in a `SELECT * FROM (...) LIMIT N+1` subquery, executes it, and optionally applies fuzzy reranking. Returns `QueryResultPreview { rows: DataFrame, truncated: bool, displayed_rows: int }`.

### `embeddings/ollama_embeddings.py`
`OllamaEmbedder` wraps the `ollama.Client` for embedding calls. Single-text and batch embedding. Normalizes every vector to unit length so inner product == cosine similarity.

### `embeddings/faiss_store.py`
`FaissSchemaStore` builds, saves, loads, and searches the FAISS index.
- `schema_documents()`: generates text representations of all tables and columns from `schema_metadata.py`
- `build()`: embeds all documents and creates `IndexFlatIP`; persists to `data/schema.faiss` + `data/schema_metadata.json`
- `search()`: embeds query, calls `index.search()`, returns top-k scored documents

### `retrieval/schema_retriever.py`
The most complex module. `SchemaRetriever.retrieve()` implements a hybrid strategy:
1. Lexical matching against column aliases, sample values, and known terms
2. Vector recall from FAISS with domain gating
3. Geography-aware injection (family columns for location queries)
4. Column pruning via `_prune_columns()` — removes domains not requested
5. Join key enrichment for multi-table queries
6. Display column injection for "show/list" queries

Domain term sets (`BANK_TERMS`, `CASTE_TERMS`, `EDUCATION_TERMS`, etc.) gate which columns are permitted to enter the prompt context.

### `prompting/prompt_builder.py`
`PromptBuilder.build()` assembles the LLM prompt from a `RetrievalResult`. Includes:
- Hard SQL safety rules (no DDL, read-only, no SELECT *)
- Semantic wording rules (boy=Male, widow=Widow+Female, senior citizen=age>=60, etc.)
- Pre-classified location rules (district vs block/village)
- Dynamic column-specific rules based on what was retrieved
- MULTIPLE VALUE FILTERING RULE (use OR not AND for multi-value same-column filters)
- The actual schema context (tables, columns with descriptions, relationships)

### `llm/ollama_client.py`
`OllamaModelManager`: checks/pulls Ollama models with user permission (CLI) or auto-pull checkbox (Streamlit).
`OllamaSqlGenerator`: calls `ollama.Client.generate()` with temperature=0, num_ctx=2048, num_predict=256. `_clean_sql()` strips markdown fences from model output.

### `validation/sql_validator.py`
`SQLValidator.validate()` uses `sqlparse` to parse the SQL, then runs regex-based checks for table names, column references, join relationships, and write keyword blocking. String literals are masked before checking to prevent false positives on keyword-shaped values. **Note**: contains active debug `print()` statements on every call (lines 108-118).

### `optimization/query_optimizer.py`
`QueryOptimizer.profile()` runs `EXPLAIN QUERY PLAN` on validated SQL, optionally executes it, and calls `recommend_indexes()` which scans all `COLUMNS` metadata to flag unindexed columns that appear in the SQL.

### `normalization/query_normalizer.py`
`QueryNormalizer.normalize()` uses RapidFuzz `process.extractOne()` with `fuzz.WRatio` to fuzzy-match tokens against a vocabulary of district names, column aliases, and common terms. Context-aware: location-protected tokens (after "in/from/at") use the base threshold (88); person-protected tokens (after "named/called" or in quotes) require 95 confidence.

### `normalization/fuzzy_match.py`
Contains three components:
- `phonetic_key(name)` — reduces an Indian name to a canonical phonetic key by normalizing common romanization variants: vowel doubling (`oo/ou→u`, `aa→a`, `ee/ii→i`), consonant aspiration (`sh→s`, `bh→b`, `kh→k`, `th→t`, `gh→g`, `ph→p`, `ch→c`, `jh→j`), North Indian v/b interchange (`v→b`), and gemination (double consonants → single). Maps Poonam↔Punam, Shweta↔Sweta, Vijay↔Bijay, Geeta↔Gita, etc.
- Fuzzy intent detection via regex patterns (`is_fuzzy_intent()`, `extract_fuzzy_target()`).
- `fuzzy_rerank()` — three-strategy scoring (best wins): (1) full-string Jaro-Winkler, (2) per-word JW for single-word targets against multi-word DB names, (3) phonetic key match floors score at 0.90, catching romanization variants that JW misses or scores below threshold.

### `ui/streamlit_app.py`
Single `render()` function. Sidebar: model pull, timing toggle, show-results toggle, row limit, default/uploaded dataset load, schema index rebuild. Main area: question text input, Generate SQL button, SQL display, metrics, corrections, retrieved schema, validation errors, results dataframe, CSV download, EXPLAIN plan. Calls `generate_sql_pipeline()` from `app.py`.

### `evaluation/benchmark.py`
`run_benchmark()` reads `evaluation/benchmark_cases.json`, runs each case through `generate_sql_pipeline()`, computes `exact_match`, `schema_accuracy` (validator result), `retrieval_accuracy` (partial; see PROJECT_OVERVIEW known gaps), `latency_ms`. Run with `python -m evaluation.benchmark`.

### `app.py`
Top-level orchestrator. Contains:
- `PipelineOutput` dataclass
- `_post_process_sql()` — 19-step SQL repair pipeline
- `_fix_no_bank_sql()` — unbanked query handler
- `generate_sql_pipeline()` — the main pipeline function
- `run_cli()` — argparse CLI entry point
- `_is_streamlit()` — detects Streamlit context
- `__main__` guard — dispatches to `render()` or `run_cli()`

---

## Module Dependency Map

```
app.py
├── config.settings
├── database.excel_importer
├── database.query_results
├── database.schema_metadata  (RAJASTHAN_DISTRICTS_41)
├── embeddings.faiss_store
├── llm.ollama_client
├── normalization.fuzzy_match
├── normalization.query_normalizer
├── optimization.query_optimizer
├── prompting.prompt_builder
├── retrieval.schema_retriever
└── validation.sql_validator

ui/streamlit_app.py
└── app.generate_sql_pipeline
└── database.excel_importer
└── database.query_results
└── embeddings.faiss_store

embeddings/faiss_store.py
├── config.settings
├── database.schema_metadata  (COLUMNS, TABLES)
└── embeddings.ollama_embeddings

retrieval/schema_retriever.py
├── config.settings
├── database.schema_metadata  (COLUMNS, TABLES, RELATIONSHIPS, RAJASTHAN_DISTRICTS_41)
└── embeddings.faiss_store

prompting/prompt_builder.py
├── database.schema_metadata  (COLUMNS, RAJASTHAN_DISTRICTS_41)
└── retrieval.schema_retriever  (RetrievalResult, LOCATION_PREPOSITIONS, LOCATION_STOPWORDS)

validation/sql_validator.py
└── database.schema_metadata  (RELATIONSHIPS, all_table_names, columns_by_table)

optimization/query_optimizer.py
├── database.connection
├── database.schema_metadata  (COLUMNS)
└── validation.sql_validator

database/excel_importer.py
├── database.connection
└── database.models

database/query_results.py
├── database.connection
├── normalization.fuzzy_match
└── validation.sql_validator

normalization/query_normalizer.py
└── database.schema_metadata  (COLUMNS, RAJASTHAN_DISTRICTS_41)

normalization/fuzzy_match.py
└── (no internal imports)

evaluation/benchmark.py
├── app.generate_sql_pipeline
└── validation.sql_validator
```

---

## Database Design

### Tables

#### `family`
Represents one household / Jan Aadhaar enrollment unit.

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `family_id` | INTEGER | PK | Auto surrogate key |
| `jan_aadhaar_number` | VARCHAR(20) | NOT NULL, UNIQUE | Family enrollment ID |
| `family_head_name` | VARCHAR(120) | NOT NULL | Name of HOF |
| `family_head_name_phonetic` | VARCHAR(120) | NULL | Phonetic key of family_head_name; indexed |
| `district` | VARCHAR(80) | NOT NULL | One of 41 Rajasthan districts |
| `city` | VARCHAR(80) | NULL | Populated only for urban families |
| `block` | VARCHAR(80) | NULL | Populated only for rural families |
| `gram_panchayat` | VARCHAR(100) | NULL | Rural only |
| `village` | VARCHAR(100) | NULL | Rural only |
| `ward` | VARCHAR(40) | NULL | Urban only |
| `is_rural` | BOOLEAN (INTEGER) | NULL | 1=rural, 0=urban |

Index: `ix_family_geo (district, block, gram_panchayat, village)`

#### `member`
Represents one individual citizen member of a family.

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `member_id` | INTEGER | PK | |
| `family_id` | INTEGER | FK → family | |
| `jan_aadhaar_member_id` | VARCHAR(40) | NOT NULL, UNIQUE | Format: `{enrollment_id}-{member_id}` |
| `member_name` | VARCHAR(120) | NOT NULL | |
| `member_name_phonetic` | VARCHAR(120) | NULL | Phonetic key of member_name; indexed |
| `father_name` | VARCHAR(120) | NULL | |
| `father_name_phonetic` | VARCHAR(120) | NULL | |
| `mother_name` | VARCHAR(120) | NULL | |
| `mother_name_phonetic` | VARCHAR(120) | NULL | |
| `spouse_name` | VARCHAR(120) | NULL | |
| `spouse_name_phonetic` | VARCHAR(120) | NULL | |
| `date_of_birth` | DATE | NULL | |
| `age` | INTEGER | NULL | Range: 8–103 in dummy data |
| `gender` | VARCHAR(16) | NOT NULL | Stored as 'Male' or 'Female' |
| `mobile_number` | VARCHAR(16) | NULL | |
| `caste_category` | VARCHAR(32) | NULL | Stored as 'SC', 'ST', 'OBC', 'GEN' |
| `marital_status` | VARCHAR(32) | NULL | 'Married', 'Unmarried', 'Widow' |
| `member_type` | VARCHAR(20) | NULL | 'HOF' or 'MEM' |
| `relation_with_hof` | VARCHAR(40) | NULL | 'Self', 'Son', 'Daughter', 'Husband', etc. |
| `caste` | VARCHAR(180) | NULL | Detailed caste name, mixed case, bilingual |
| `income` | INTEGER | NULL | Annual income in rupees |
| `occupation` | VARCHAR(80) | NULL | |
| `minority` | VARCHAR(40) | NULL | 'Muslim' or 'Jain'; 96% NULL |
| `education` | VARCHAR(80) | NULL | 'illiterate' (lowercase), 'Graduate', etc. |

Index: `ix_member_gender_caste_age (gender, caste_category, age)`

#### `bank_details`
One record per bank account per member (a member may have multiple accounts).

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `bank_id` | INTEGER | PK | |
| `member_id` | INTEGER | FK → member | |
| `bank_account` | VARCHAR(32) | NOT NULL | |
| `bank_name` | VARCHAR(120) | NOT NULL | Inconsistent casing in real data |
| `ifsc_code` | VARCHAR(16) | NOT NULL | |
| `dbt_status` | VARCHAR(24) | NOT NULL | Default 'Active' |

### Declared Relationships (for JOIN validation)
```
member.family_id    → family.family_id
bank_details.member_id → member.member_id
```

Transitive join to link `family` and `bank_details` must pass through `member`.

### Schema Divergence Warning
`database/ddl/schema.sql` contains extra columns (`permanent_address`, `current_address`, `ration_card_number`, `email`, `photo_path`, `aadhaar_masked`, `voter_id`, `pan_number`, `religion`, `disability_status`) not present in `database/models.py`. The Python models are the runtime-authoritative schema; the DDL file is outdated.

---

## Deployment

### Local Development / Demo

```powershell
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Pull Ollama models
ollama pull qwen2.5-coder:3b
ollama pull nomic-embed-text

# 3. Verify environment
python scripts/verify_environment.py

# 4. Load demo data + build FAISS index
python app.py --seed-demo-db --build-index

# 5. Run Streamlit UI
streamlit run app.py

# OR run CLI
python app.py "Show all female beneficiaries in Jaipur"
```

### Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `SQL_MODEL` | `qwen2.5-coder:3b` | SQL generation model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_KEEP_ALIVE` | `30m` | How long model stays resident in Ollama |
| `DATABASE_URL` | `sqlite:///data/jan_aadhaar_demo.sqlite` | Database connection string |
| `MAX_SQL_RETRIES` | `3` | Max LLM retry attempts |
| `RETRIEVAL_TOP_K` | `16` | Number of FAISS results per query |

### Streamlit Detection
`app.py` uses `_is_streamlit()` to check `streamlit.runtime.scriptrunner.get_script_run_ctx()`. When run via `streamlit run app.py`, it calls `ui.streamlit_app.render()`. When run as `python app.py`, it calls `run_cli()`.

### Production Notes
Per `docs/production_deployment.md`: point `DATABASE_URL` at PostgreSQL or another production RDBMS. Rebuild the FAISS index after any `database/schema_metadata.py` changes. Add authorization, row limits, query timeouts, and audit logging before production.
