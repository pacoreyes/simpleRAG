# Development Log

<!--
  MAINTENANCE RULES
  - This log is APPEND-ONLY. Never delete or modify past entries.
  - Each calendar day gets one dated section (## YYYY-MM-DD).
  - Within a day, add sub-sections for distinct work blocks; do not merge them.
  - Record: what changed, why it changed, decisions made, and open items.
  - At the end of each session update the "Open items" list under that day:
    cross off resolved items with ~~strikethrough~~, add new ones.
    Do not delete resolved items — the history of what was pending matters.
-->

## 2026-06-30

### Session summary

First full working session on the project. Started from a blank `models.py`, a flat `src/` layout, and a `DATA_PIPELINE.md` design document. By end of session the project has a clean package structure, working models, a configured test suite, and a documented pipeline entry point.

---

### Analysis phase

- Evaluated `DATA_PIPELINE.md` against what is actually needed to build the pipeline. Identified critical gaps: unknown source schema, no chunking strategy, no embedding model named, ambiguous `generated_questions`/`subtopic` fields, no deduplication logic for variant rows, and missing Pinecone index configuration.
- Inspected the RagQuAS Parquet file (`test-00000-of-00001.parquet`): confirmed 201 rows, 19 columns, Spanish text, fields named `topic`, `answer`, `question`, `variant`, `context_1..5`, `link_1..5`, `text_1..5`.
- Audited all existing utils: `data_transformation_helpers.py`, `io_helpers.py`, `llm_helpers.py`, `pinecone_helpers.py`. Found that `network_helpers.py` had a hard `dagster` dependency incompatible with this project and was dropped.
- Resolved all gaps: dropped `generated_questions` and `subtopic`; confirmed `deduplicate_by_priority()` solves the variant-row problem; chose `generate_cache_key()` (SHA256) as the `doc_id` strategy.

### Settings changes (`src/simple_rag/settings.py`)

- **Embedding model**: replaced `Snowflake/snowflake-arctic-embed-s` (local BYOV model, English only) with `multilingual-e5-large` (Pinecone Inference API, multilingual — required for Spanish dataset).
- **Embedding dimensions**: `384 → 1024` to match `multilingual-e5-large` output size.
- **Chunking**: `CHUNK_TARGET_TOKENS 500 → 300` (web snippets, not long documents); `CHUNK_OVERLAP_SENTENCES 4 → 1` (overlap designed for long prose); added `CHUNK_MIN_SENTENCES = 2` to replace hardcoded `5` in `build_chunks()`.
- **Paths**: added `ASSETS_DIRPATH = DATA_DIR / "assets"`.
- Identified vestigial settings (`VECTOR_DB_DIRPATH`, `VECTOR_DB_BATCH_SIZE`, `LLM_MAX_ENTITIES_PER_CHUNK`, `LLM_MAX_RELATIONS_PER_CHUNK`) — left in place pending cleanup decision.

### Models (`src/simple_rag/models.py`)

Wrote three Pydantic models defining the pipeline's data contracts:
- `SourceRow` — 19-column mirror of the Parquet schema.
- `DocumentRecord` — post-explode/dedup intermediate with `doc_id`, `is_gold_for`.
- `ChunkRecord` — final Pinecone upsert payload with `chunk_id`, `chunk_index`, `char_length`, `token_count`.

### Data transformation helpers

- Made `build_chunks()` and `chunk_text()` dynamic: defaults now read from `settings.CHUNK_TARGET_TOKENS`, `settings.CHUNK_OVERLAP_SENTENCES`, `settings.CHUNK_MIN_SENTENCES` instead of hardcoded values.
- Added `from simple_rag.settings import settings` import.

### Package restructure

Moved from a flat `src/` layout to a proper named package:

```
Before:
  src/
    settings.py
    models.py
    data_pipeline/utils/...

After:
  src/
    simple_rag/
      settings.py
      models.py
      utils/           ← shared (io, llm, pinecone helpers)
      data_pipeline/
        run.py
        utils/         ← pipeline-specific (chunking, dedup)
      rag/             ← empty, ready to build
```

- All imports updated to `simple_rag.*` namespace.
- `data_pipeline/utils/__init__.py` cleared (had imports for non-existent modules and `dagster`).
- `network_helpers.py` not migrated (dagster dependency, unused in this project).

### Test suite (`tests/`)

Built 57 unit tests across two files:
- `test_models.py` — 18 tests covering all three Pydantic models, required/optional fields, `is_gold_for` isolation between instances, and validation errors.
- `test_data_transformation.py` — 39 tests covering `split_sentences`, `build_chunks` (8 cases including edge cases and settings-default wiring), `chunk_text`, `deduplicate_by_priority` (6 cases), `normalize_and_clean_text`, `strip_json_fences`, and `generate_cache_key`.
- Added `conftest.py` to wire `src/` into sys.path for test discovery.
- Added `pytest`, `pytest-asyncio`, `pytest-mock`, `bandit`, `ruff`, `ty` to `[dependency-groups] dev`.
- Added `[tool.ruff]`, `[tool.bandit]`, `asyncio_mode = "auto"` to `pyproject.toml`.
- Fixed missing `click` dependency (required by spaCy CLI).
- All 57 tests pass.

### Pipeline entry point

Created `src/simple_rag/data_pipeline/run.py` — the single file to invoke to run the complete pipeline:

```
python -m simple_rag.data_pipeline.run
```

The file documents all 5 stages with TODO markers pending implementation:
1. Load dataset → `list[SourceRow]`
2. Explode & deduplicate → `list[DocumentRecord]`
3. Chunk → `list[ChunkRecord]`, save `chunks-processed.parquet`
4. Extract keywords (Gemini)
5. Embed & upsert to Pinecone

### Documentation

- `README.md`: project overview, architecture diagram, data flow, Pinecone vector schema, evaluation capability description, prerequisites, installation, configuration table, run instructions, test commands, and model descriptions.
- `LOG.md`: this file.

---

### `pyproject.toml` — full audit and rewrite

Complete audit of `pyproject.toml` against the actual codebase and package architecture. Rewrote the file to fix all identified issues.

**Build system added (`[build-system]` + `[tool.hatch.build.targets.wheel]`)**
Previously absent — without it the package could not be formally installed. Added `hatchling` as the build backend (uv-native, zero extra config) and pointed it at `src/simple_rag`. Confirmed working: `uv sync` now builds and installs `simple-rag-demo==0.1.0`.

**CLI entry point registered (`[project.scripts]`)**
Added `run-pipeline = "simple_rag.data_pipeline.run:run"`. After `uv sync`, the pipeline can be invoked simply as `run-pipeline` from anywhere, in addition to `python -m simple_rag.data_pipeline.run`.

**Project metadata completed**
Added `description`, `readme`, `license`, `authors` to `[project]`.

**Dependency cleanup — 13 packages removed from environment**

| Removed | Reason |
|---|---|
| `langchain-text-splitters` | Not imported anywhere; pulled in `langchain-core`, `langsmith`, and 10 transitive packages |
| `curl-cffi` | Only used in `network_helpers.py`, which was dropped (dagster dependency) |
| `ftfy` | Not imported anywhere in `src/` |
| `torch` (direct dep) | Still present transitively via `spacy-transformers`; listing it directly was redundant |
| `pinecone[asyncio]` extra | The extra does not exist in pinecone 9.1.0; changed to `pinecone` |
| `google.genai` | Wrong package name (uses dot); corrected to `google-genai` |

**Duplicate spaCy entries consolidated**
Three lines (`spacy[transformers]`, `spacy>=3.7.0`, `spacy-transformers>=1.3.0`) collapsed to two clean lines with explicit version pins.

**Tool configurations completed**
- `[tool.ruff.format]` added (`quote-style = "double"`, `indent-style = "space"`)
- `[tool.ty]` added (`src = { root = "src" }`)
- `ruff format` `line-length` field moved to `[tool.ruff]` only (the `[tool.ruff.format]` section does not accept it — caught and fixed during sync)

**Security findings — `transformers` CVEs (open)**
The `transformers==4.49.0` package (transitive via `spacy-transformers`) carries multiple CVEs rated 7.5–7.8 (ReDoS and insufficient information in huggingface/transformers). The current codebase does not load any transformer model at runtime — `data_transformation_helpers.py` uses only `spacy.blank("en")` with the rule-based `sentencizer`. Attack surface is zero today.

Decision pending: if `spacy[transformers]` and `spacy-transformers` are removed and replaced with plain `spacy>=3.7.0`, the CVEs and all CUDA/NVIDIA packages disappear. Trade-off: `rag/` would need to re-add them when a transformer-based reranker is introduced. See open items.

**Outdated dependency noted**
`spacy-transformers` 1.3.9 is installed; 1.4.0 is available. Not upgraded yet — pending the decision on whether to keep `spacy-transformers` at all.

---

### Open items (after pyproject.toml audit)

- ~~Document README and LOG~~ ✓
- Implement Steps 1–5 in `run.py` (pipeline logic).
- Decide whether to remove vestigial settings (`VECTOR_DB_*`, `LLM_MAX_ENTITIES_PER_CHUNK`, `LLM_MAX_RELATIONS_PER_CHUNK`).
- Begin `simple_rag/rag/` — query router, retrieval, answer generation, evaluation harness.
- Add `.env.example` file.
- **Security decision**: drop `spacy[transformers]` + `spacy-transformers` in favour of plain `spacy>=3.7.0` to eliminate `transformers` CVEs? Or keep and upgrade to `spacy-transformers==1.4.0`?

---

### Separation-of-concerns (SoC) audit and refactor

Full audit of all modules for import-direction violations and misplaced responsibilities. Three violations found in shared `utils/`, three misplaced functions in `data_pipeline/utils/`.

#### Violations found and fixed

**`pinecone_helpers.py` (shared utils) — imported project settings**

`from simple_rag.settings import settings` was present, making the module project-specific. Retry configuration was also bound to project constants.

Fix: removed `settings` import entirely. `retry_count` (default 5) and `backoff_factor` (default 2.0) are now explicit function parameters on both `generate_embeddings_pinecone` and `generate_embeddings_pinecone_async`. Replaced Dagster context logging with `logging.getLogger(__name__)`.

**`llm_helpers.py` (shared utils) — `import torch` at module level**

`torch` was imported at the top of the file, causing startup overhead on every import of `llm_helpers` even when GPU detection is never needed.

Fix: moved `import torch` inside `get_device()` (lazy import).

**`data_transformation_helpers.py` — two generic functions with no pipeline-specific knowledge**

- `format_list_natural_language()` — pure string formatting, no domain knowledge.
- `strip_json_fences()` — generic LLM output parser for JSON cleanup, no domain knowledge.

Fix: moved both to their correct locations (see table below).

**`prepare_chunks_for_extraction()` — dead parameter**

The function accepted `overlap_sentences` but never used it.

Fix: removed the dead parameter; simplified function to a single list comprehension. Updated the call site in `chunk_text()`.

#### Functions moved

| Function | From | To | Reason |
|---|---|---|---|
| `format_list_natural_language` | `data_pipeline/utils/data_transformation_helpers.py` | `utils/text_helpers.py` (new file) | Generic string formatting; no pipeline knowledge |
| `strip_json_fences` | `data_pipeline/utils/data_transformation_helpers.py` | `utils/llm_helpers.py` | Generic LLM output parser; belongs with other LLM utilities |

#### Files created

- `src/simple_rag/utils/text_helpers.py` — new shared utils module for generic string formatting; contains `format_list_natural_language()`.

#### Files modified

| File | Change |
|---|---|
| `src/simple_rag/utils/pinecone_helpers.py` | Removed `settings` import; retry params made explicit; added `logging`; removed Dagster context param |
| `src/simple_rag/utils/llm_helpers.py` | Lazy `torch` import; added `strip_json_fences`; fixed header |
| `src/simple_rag/data_pipeline/utils/data_transformation_helpers.py` | Removed `format_list_natural_language`, `strip_json_fences`; removed `import re`, `Sequence`; fixed dead param in `prepare_chunks_for_extraction` |
| `tests/test_data_transformation.py` | Updated `strip_json_fences` import to `simple_rag.utils.llm_helpers` |

#### Tests added

- `tests/test_text_helpers.py` — 9 tests for `format_list_natural_language` (empty, None, single, two items, Oxford comma, duplicates, None values in list, non-string coercion).

Test count: **57 → 66** (all pass).

#### Architecture documentation

Added "Architecture decisions" section to `README.md` with 6 rules and dependency direction table. Updated architecture tree and module table to include `text_helpers.py` and `test_text_helpers.py`.

#### Borderline case (not moved)

`normalize_and_clean_text()` in `data_transformation_helpers.py` strips whitespace and returns `None` for empty strings. Technically generic, but trivial enough that moving it would add indirection without benefit. Left in place.

---

### Open items for next session

- ~~Document README and LOG~~ ✓
- ~~SoC audit and refactor~~ ✓
- ~~**Security decision**: drop `spacy[transformers]` + `spacy-transformers`~~ ✓ — resolved; see below
- Implement Steps 1–5 in `run.py` (pipeline logic).
- Decide whether to remove vestigial settings (`VECTOR_DB_*`, `LLM_MAX_ENTITIES_PER_CHUNK`, `LLM_MAX_RELATIONS_PER_CHUNK`).
- Begin `simple_rag/rag/` — query router, retrieval, answer generation, evaluation harness.
- Add `.env.example` file.

---

### `transformers` CVE resolution

Removed `spacy[transformers]>=3.7.0` and `spacy-transformers>=1.3.0` from `pyproject.toml`. Replaced with plain `spacy>=3.7.0`.

`uv sync` removed 14 packages:

```
- transformers==4.49.0      ← CVEs 7.5–7.8 eliminated
- torch==2.12.1
- huggingface-hub==0.36.2
- tokenizers==0.21.4
- safetensors==0.8.0
- spacy-transformers==1.3.9
- spacy-alignments==0.9.2
- filelock, fsspec, hf-xet, mpmath, networkx, sympy (transitive)
```

All 66 tests pass. README "Known issues" section updated to "None at this time" with a preserved comment for future reference if `spacy-transformers` is re-added for a transformer-based reranker in `rag/`.

---

### Spanish language fix (`spacy.blank`)

`data_transformation_helpers.py` was using `spacy.blank("en")` for sentence segmentation despite the dataset being entirely in Spanish. The English tokenizer mis-handles Spanish abbreviations (`Sr.`, `Dr.`, `p.ej.`, `etc.`) and does not recognise `¿`/`¡` as sentence-start markers.

Fix:
- Added `SPACY_LANGUAGE: str = "es"` to `settings.py`.
- `_get_nlp_model()` now calls `spacy.blank(settings.SPACY_LANGUAGE)` instead of the hardcoded `"en"`.

No other components needed changes: tiktoken `cl100k_base` is language-agnostic; Pinecone `multilingual-e5-large` already covers Spanish; Gemini is multilingual (keyword extraction prompts, not yet written, will ask for Spanish output explicitly).

All 66 tests pass.

---

### Click CLI for pipeline phase control

Replaced the single `run()` function in `run.py` with a Click CLI group (`cli`) exposing three subcommands. Motivation: preprocessing (Steps 1–3) and Pinecone loading (Steps 4–5) have different re-run patterns and different API key requirements; running them as separate commands avoids unnecessary re-work and makes iteration faster during development.

#### New CLI surface

```
run-pipeline preprocess [--limit N]   # Steps 1–3: load → dedup → chunk → save Parquet
run-pipeline load                     # Steps 4–5: keywords → embed → upsert
run-pipeline all          [--limit N]  # Steps 1–5 in sequence
run-pipeline --help
run-pipeline <subcommand> --help
```

`--limit N` caps source rows for test runs without changing source data.

`load` guards against missing Parquet with a clear `FileNotFoundError` pointing to `preprocess`.

#### Files changed

| File | Change |
|---|---|
| `src/simple_rag/data_pipeline/run.py` | Replaced `run()` with `_run_preprocess()`, `_run_load()`, and Click group `cli` with three subcommands |
| `pyproject.toml` | `run-pipeline` entry point changed from `run:run` to `run:cli` |
| `README.md` | "Running the pipeline" section rewritten with subcommand table and usage examples |

---

### Open items for next session

- ~~Document README and LOG~~ ✓
- ~~SoC audit and refactor~~ ✓
- ~~Security decision: `transformers` CVEs~~ ✓
- ~~Spanish language fix (spaCy)~~ ✓
- ~~Click CLI for pipeline phase control~~ ✓
- ~~Preprocess pipeline implementation (Steps 1–3) + 40 tests~~ ✓
- ~~`is_gold_for` removed (evaluation field, not needed for simple RAG)~~ ✓
- Implement Steps 4–5 in `run.py` (keyword extraction + embed/upsert).
- Decide whether to remove vestigial settings (`VECTOR_DB_*`, `LLM_MAX_ENTITIES_PER_CHUNK`, `LLM_MAX_RELATIONS_PER_CHUNK`).
- Begin `simple_rag/rag/` — query router, retrieval, answer generation.
- Add `.env.example` file.

---

### `is_gold_for` removed — design decision

**Context**: The RagQuAS dataset was built for retrieval evaluation. The pipeline was accumulating SHA256 hashes of question texts into an `is_gold_for` field on every `DocumentRecord` and `ChunkRecord`, and propagating it into Pinecone metadata. This required a Polars `group_by + agg` step in `explode_and_deduplicate` to collect question IDs per document.

**Decision**: The project goal is a simple RAG that answers questions — not an evaluation harness. The `is_gold_for` field serves no purpose in retrieval or answer generation. It was dead weight in the data model, the pipeline, and Pinecone metadata.

**Changes made**:
- `models.py`: `is_gold_for: list[str] = []` removed from `DocumentRecord` and `ChunkRecord`
- `data_transformation_helpers.py`: `explode_and_deduplicate()` simplified — removed `question_id` computation, `group_by + agg`, and join; now a single `.sort().unique().drop()` chain
- `data_transformation_helpers.py`: `chunk_documents()` — removed `is_gold_for` propagation
- `tests/test_models.py`: removed 5 `is_gold_for` tests
- `tests/test_preprocess.py`: removed 3 `is_gold_for` tests, updated fixtures
- `README.md`: Pinecone vector schema updated; "Evaluation capability" section replaced with "Retrieval strategy" describing topic-filtered RAG flow and keyword hybrid search

**Test count**: 106 → 97 (9 tests removed, all were `is_gold_for` assertions)

**Pipeline re-run**: `run-pipeline preprocess` still produces 201 rows → 132 documents → 1063 chunks correctly.

---

### Keyword extraction reverted — design decision

**Context**: Step 4 was implemented as a Gemini batch call to extract 5–10 Spanish keywords per chunk (`extract_keywords_async`). This added a `keywords: list[str] = []` field to `ChunkRecord`, a `_KEYWORD_PROMPT` template, async concurrency logic, and a Gemini client dependency to the pipeline.

**Decision**: The project goal is a simple RAG. The dataset already provides `topic` (32 known labels), `doc_id`, `source_domain`, and `link` — all the metadata needed to filter and cite without keyword extraction. Adding Gemini-generated keywords introduces operational complexity (rate-limiting, partial failure, re-run logic) without a material quality gain over topic-based filtering. The `topic` filter narrows the search space from 1063 chunks to ~30–50 before semantic ranking runs, which is a stronger signal than sparse keyword matching would add.

**Changes made**:
- `models.py`: removed `keywords: list[str] = []` from `ChunkRecord`
- `data_transformation_helpers.py`: removed `asyncio`, `tqdm`, `generate_text_gemini_async`, `strip_json_fences` imports; removed `_KEYWORD_PROMPT` constant and `extract_keywords_async()` function; `load_chunks_parquet()` docstring cleaned
- `data_pipeline/run.py`: removed `asyncio`, `get_gemini_client`, `extract_keywords_async` imports; `_run_load()` now goes directly to embedding — no keyword step
- `settings.py`: removed `GEMINI_CONCURRENT_REQUESTS` (only existed for keyword batching)
- `tests/test_preprocess.py`: removed `"keywords"` from `test_all_columns_present` expected set; fixed `DocumentRecord(is_gold_for=[])` stale call
- All 97 tests pass.

---

### Vestigial settings cleanup

Removed four settings that were never used by any live code path:

| Setting removed | Was used for |
|---|---|
| `VECTOR_DB_BATCH_SIZE` | Local ChromaDB/FAISS (pre-Pinecone) |
| `VECTOR_DB_DIRPATH` | Local vector DB directory (pre-Pinecone) |
| `LLM_MAX_ENTITIES_PER_CHUNK` | Knowledge-graph extraction (dropped earlier) |
| `LLM_MAX_RELATIONS_PER_CHUNK` | Knowledge-graph extraction (dropped earlier) |

`VECTOR_DB_DIRPATH` was also being auto-created in `_compute_and_create_paths()`; that creation was removed from the `model_validator`.

---

### Steps 4–5 implemented — embed + upsert to Pinecone

**`pinecone_helpers.py`** — two new functions added:

- `get_pinecone_client(api_key) -> Pinecone` — thin factory, encapsulates client init.
- `ensure_index(pc, index_name, dimension, metric, cloud, region)` — creates the Pinecone serverless index if it does not already exist (AWS us-east-1, cosine, 1024 dims by default). No-op if the index is already present.
- `upsert_chunks(pc, index_name, chunks, embeddings, batch_size=100)` — zips `ChunkRecord` list with embeddings; builds vector dicts with metadata fields `topic`, `doc_id`, `link`, `source_domain`, `chunk_index`, `text`; upserts in batches with a tqdm progress bar.

`text` is stored in Pinecone metadata so the RAG router can retrieve chunk content without a separate Parquet lookup.

**`_run_load()` in `data_pipeline/run.py`** — rewritten as two genuine steps:

```
Step 1/2 — Load chunks, generate embeddings via multilingual-e5-large
Step 2/2 — ensure_index, upsert_chunks to "ragchunks"
```

Guards:
- `FileNotFoundError` if `chunks-processed.parquet` does not exist (run `preprocess` first)
- `RuntimeError` if `PINECONE_API_KEY` is empty

**Live run result** (`run-pipeline all`):

```
Preprocess : 201 source rows → 132 unique documents → 1063 chunks
Load       : 1063 embeddings (multilingual-e5-large, ~1m 45s)
             1063 vectors upserted to ragchunks (~1m 36s)
```

Pinecone index `ragchunks` is now populated and queryable.

---

### Second SoC audit

Full re-audit after Steps 4–5 implementation introduced new code.

#### Violation found and fixed

**`pinecone_helpers.py` (shared utils) — imported `ChunkRecord` from `simple_rag.models`**

`upsert_chunks()` accepted `List[ChunkRecord]` and directly read domain field names (`chunk.chunk_id`, `chunk.topic`, `chunk.doc_id`, `chunk.link`, `chunk.source_domain`, `chunk.chunk_index`, `chunk.text`). This made a generic utility aware of this project's data model, breaking the rule that `utils/` must not import from `simple_rag.models` or know any domain concepts.

Fix:
- Renamed `upsert_chunks` → `upsert_vectors(pc, index_name, vectors: List[dict], batch_size)` — accepts pre-serialised Pinecone vector dicts, no model knowledge.
- Removed `TYPE_CHECKING` guard and `ChunkRecord` import entirely.
- Vector dict construction (with domain field names) moved to `data_pipeline/run.py` where it belongs.
- `clear_index` signature updated: now accepts `pc: Pinecone` instead of creating its own client (consistency with other helpers).

#### Stale docstring fixed

`explode_and_deduplicate()` docstring still described `is_gold_for` accumulation (steps 3–4 in the original description), which was removed in the `is_gold_for` cleanup. Updated to accurately describe the current three-step flow.

#### Code preserved intentionally

- `save_chunks` / `load_chunks` (JSONL I/O in `data_transformation_helpers.py`) — kept; reusable for caching intermediate results in the router.
- `generate_embeddings_pinecone_async` — kept; the router will use it to embed user queries concurrently.
- `clear_index` — kept; useful for full re-indexing operations.
- `get_device()` in `llm_helpers.py` — kept; may be needed if a local reranker is added to the router.

#### Files changed

| File | Change |
|---|---|
| `src/simple_rag/utils/pinecone_helpers.py` | `upsert_chunks` → `upsert_vectors(vectors: List[dict])`; removed `ChunkRecord` import; `clear_index` takes `pc` not `api_key`; `generate_embeddings_pinecone_async` restored |
| `src/simple_rag/data_pipeline/run.py` | Vector dict construction moved here; import updated to `upsert_vectors` |
| `src/simple_rag/data_pipeline/utils/data_transformation_helpers.py` | `explode_and_deduplicate` docstring corrected |

All 97 tests pass.

---

### Open items

- ~~Document README and LOG~~ ✓
- ~~SoC audit and refactor~~ ✓
- ~~Security decision: `transformers` CVEs~~ ✓
- ~~Spanish language fix (spaCy)~~ ✓
- ~~Click CLI for pipeline phase control~~ ✓
- ~~Preprocess pipeline implementation (Steps 1–3) + 40 tests~~ ✓
- ~~`is_gold_for` removed (evaluation field, not needed for simple RAG)~~ ✓
- ~~Keyword extraction reverted (over-engineering; existing metadata is sufficient)~~ ✓
- ~~Vestigial settings removed~~ ✓
- ~~Steps 4–5 implemented and verified live (embed + upsert)~~ ✓
- ~~Second SoC audit — `upsert_chunks` violation fixed~~ ✓
- Begin `simple_rag/rag/` — topic classifier, Pinecone query with topic filter, source-diversity reranking, Gemini answer generation with citations.
- Add `.env.example` file.

---

### Orphan directory cleanup

Two empty directories were found under `data_volume/` with no live code paths referencing them:

- `data_volume/datasets/` — originally intended for downloaded datasets; the actual source file (`test-00000-of-00001.parquet`) lives in `data_volume/assets/` instead. `DATASETS_DIRPATH` was being auto-created by `settings._compute_and_create_paths()` on every process start despite having no purpose.
- `data_volume/vector_db/` — leftover from an earlier local file-based vector DB design (pre-Pinecone). The setting `VECTOR_DB_DIRPATH` was removed in a previous session but the directory persisted on disk from earlier runs.

**Changes:**
- Both directories deleted from disk.
- `DATASETS_DIRPATH` field and its assignment removed from `settings.py`.
- `DATASETS_DIRPATH` removed from the `dirs_to_create` auto-creation list.
- `data_volume/` now contains only `assets/` (source parquet + pipeline output parquet).

All 97 tests pass.
