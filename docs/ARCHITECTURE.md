# Arquitectura

Estructura completa del paquete, reglas de capas, y límites de imports para `simple_rag`. Ver el [README.md](../README.md) principal para una vista rápida, instalación, y uso.

## Estructura completa del paquete

```
simple_rag_demo/
├── data_volume/
│   └── assets/                           ← toda la data vive aquí
│       ├── test-00000-of-00001.parquet   ← dataset de origen (RagQuAS, 201 filas)
│       └── chunks-processed.parquet      ← salida del pipeline (generada por el paso de preprocess)
│
├── src/
│   └── simple_rag/                       ← paquete de nivel superior
│       ├── settings.py                   ← toda la configuración (rutas, modelos, parámetros de chunking)
│       ├── models.py                     ← modelos Pydantic: SourceRow, DocumentRecord, ChunkRecord
│       │
│       ├── utils/                        ← helpers compartidos (usados por ambos subsistemas)
│       │   ├── io_helpers.py             ← I/O asíncrono de archivos, generación de cache key SHA256
│       │   ├── llm_helpers.py            ← cliente de Gemini + tokenizer (tiktoken) + parsing de salida del LLM
│       │   ├── pinecone_helpers.py       ← embed + upsert de Pinecone (sync y async)
│       │   └── text_helpers.py           ← utilidades genéricas de formateo de strings
│       │
│       ├── data_pipeline/                ← pipeline batch — se ejecuta una vez para construir el índice
│       │   ├── run.py                    ← punto de entrada (python -m simple_rag.data_pipeline.run)
│       │   └── utils/
│       │       └── data_transformation_helpers.py  ← chunking, deduplicación, limpieza de texto
│       │
│       └── rag/                          ← sistema de consulta del RAG
               ├── router.py            ← Clasificador de tema (Gemini 2.5 Flash + thinking)
               ├── retrieval.py         ← Búsqueda semántica en Pinecone con filtro de tema
               ├── generator.py         ← Generador de respuestas (Gemini 2.5 Flash Lite)
               ├── evaluate.py          ← Judges de faithfulness + corrección de respuesta
               ├── pipeline.py          ← Orquestador: router → retrieval → generación → eval
               └── utils/
                   ├── routing_helpers.py     ← RouterDecision → filtro de Pinecone
                   ├── retrieval_helpers.py   ← límite por diversidad de fuentes, normalización de chunks
                   ├── generation_helpers.py  ← armado del prompt de respuesta
                   ├── evaluation_helpers.py  ← Recall@k, MRR, similitud coseno, parsing JSON del judge
                   └── gold_data_helpers.py   ← lookup de doc_ids/answer/topic gold desde el parquet de RagQuAS
│
├── app/
│   └── app.py                            ← UI de chat en Chainlit (chainlit run app/app.py -w)
│
└── tests/
    ├── conftest.py                   ← configuración de pytest; flag --integration; marks
    ├── test_models.py                ← tests de contrato de los modelos Pydantic
    ├── test_data_transformation.py   ← tests de chunking, dedup, limpieza de texto
    ├── test_text_helpers.py          ← tests de formateo de strings
    ├── test_llm_helpers.py           ← tests de strip_json_fences
    ├── test_preprocess.py            ← tests del pipeline de preprocess completo
    ├── test_router.py                ← RouterDecision, build_pinecone_filter, KNOWN_TOPICS
    ├── test_retrieval.py             ← tests de cap_chunks_per_doc
    ├── test_evaluate.py              ← recall@k, MRR, similitud coseno
    └── test_rag_integration.py       ← 15 preguntas gold + métricas agregadas (--integration)
```

## Reglas de arquitectura

Estas reglas se aplican por convención y deben preservarse a medida que el código crece.

### 1. `simple_rag/utils/` — solo helpers genéricos, a nivel de tecnología

Las funciones aquí deben ser reusables en cualquier proyecto de Python sin modificación. No tienen ningún conocimiento de RagQuAS, texto en español, estrategia de chunking, nombres de índices de Pinecone, o cualquier concepto de dominio.

**Imports permitidos:** solo librería estándar y paquetes de terceros.
**Imports prohibidos:** `simple_rag.models`, `simple_rag.settings`, `simple_rag.data_pipeline`, `simple_rag.rag`.

Si una función referencia el nombre de un campo, un dataset, una plantilla de prompt, o un setting del proyecto, no pertenece a `utils/`.

| Módulo | Responsabilidad |
|---|---|
| `io_helpers.py` | I/O asíncrono de archivos, serialización JSON, cache keys SHA256 |
| `llm_helpers.py` | Configuración del cliente de Gemini, generación de texto (sync + async), parsing de salida del LLM, tokenización |
| `pinecone_helpers.py` | Cliente de Pinecone, gestión de índices, embedding (sync + async), upsert genérico de vectores |
| `text_helpers.py` | Utilidades genéricas de formateo de strings |

### 2. `simple_rag/data_pipeline/` — solo el lado de ingesta

Es dueño de todo el camino desde el dataset crudo hasta los vectores indexados: carga, explode, deduplicación, chunking, embedding, upsert. Su `utils/data_transformation_helpers.py` contiene lógica pura específica de dominio (chunking, deduplicación, limpieza de texto) separada de la orquestación de `run.py` — helpers locales al subsistema, distintos del `simple_rag/utils/` genérico de arriba.

**Imports permitidos:** `simple_rag.utils`, `simple_rag.models`, `simple_rag.settings`.
**Imports prohibidos:** cualquier cosa de `simple_rag.rag`.

### 3. `simple_rag/rag/` — solo el lado de consulta

Sigue el mismo patrón de `utils/` local para lógica de dominio que `data_pipeline/`: la lógica pura (armado de filtros, reranking de chunks, armado de prompts, métricas de IR/eval, carga de data gold) vive en `rag/utils/*_helpers.py`, mientras que `router.py`, `retrieval.py`, `generator.py`, `evaluate.py`, y `pipeline.py` contienen solo orquestación (llamadas a LLMs, llamadas a Pinecone, secuenciación).

Es dueño de todo el camino desde la query del usuario hasta la respuesta generada: routing, retrieval, reranking, generación, evaluación.

**Imports permitidos:** `simple_rag.utils`, `simple_rag.models`, `simple_rag.settings`.
**Imports prohibidos:** cualquier cosa de `simple_rag.data_pipeline`.

### 4. Sin imports entre subsistemas

`data_pipeline` y `rag` son subsistemas independientes que comparten infraestructura (`utils`, `models`, `settings`) pero nunca deben importarse entre sí. Una función que ambos necesiten pertenece a `utils/`, no a ninguno de los dos subsistemas.

### 5. `models.py` — única fuente de verdad para los contratos de datos

Todos los modelos Pydantic viven aquí. Ni `data_pipeline` ni `rag` definen sus propios modelos. El archivo de modelos no tiene imports de ningún otro lugar de `simple_rag`, salvo librería estándar y Pydantic.

### 6. `settings.py` — única fuente de verdad para la configuración

Todos los valores configurables viven aquí. Los subsistemas y sus utils leen desde `settings`; no definen sus propias constantes para cosas que podrían cambiar entre ejecuciones. Los utils compartidos (`simple_rag/utils/`) no deben importar `settings` — reciben la configuración como parámetros de función, para que quien llama (pipeline o rag) decida qué valores pasar.

### 7. Comentarios y docstrings en código están en español; los docs del proyecto están en español

Dado que los usuarios finales de la app son hispanohablantes, todos los docstrings, comentarios inline, mensajes de log, texto de ayuda del CLI, y demás texto explicativo en el código están escritos en español. Los términos específicos de IT/AI/dev se mantienen en inglés cuando traducirlos sería antinatural u oscurecería la API real (`chunk`, `embedding`, `token`, `pipeline`, `router`, `batch`, `cache`, `backoff`, `thread`, `hash`, siglas JSON/API/CLI/SDK, y las palabras clave `Args:`/`Returns:` de los docstrings). El README y este documento de arquitectura también están en español; los demás docs internos de planificación (`LOG.md`, `ROUTER.md`, `DATA_PIPELINE.md`, `PRESENTATION.md`) se mantienen en inglés.

### Dirección de dependencias (resumen)

```
simple_rag.data_pipeline  →  simple_rag.utils      ✓
simple_rag.data_pipeline  →  simple_rag.models     ✓
simple_rag.data_pipeline  →  simple_rag.settings   ✓
simple_rag.rag            →  simple_rag.utils      ✓
simple_rag.rag            →  simple_rag.models     ✓
simple_rag.rag            →  simple_rag.settings   ✓
simple_rag.data_pipeline  →  simple_rag.rag        ✗  NUNCA
simple_rag.rag            →  simple_rag.data_pipeline  ✗  NUNCA
simple_rag.utils          →  simple_rag.models     ✗  NUNCA
simple_rag.utils          →  simple_rag.settings   ✗  NUNCA
```

## Cómo funciona el router (`QueryRouter`)

```
                  CÓMO FUNCIONA EL ROUTER (QueryRouter) — PROCESO COMPLETO
═══════════════════════════════════════════════════════════════════════════════

  Query del usuario (string, en español)
  "¿Cuál es la forma más fácil de reclamar cuando un vuelo se ha retrasado?"
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. ARMADO DEL PROMPT                                                     │
│    _SYSTEM_PROMPT + KNOWN_TOPICS (~30 temas, texto exacto) + query       │
│    Incluye guía de desambiguación para pares de temas confusos           │
│    (p. ej. "lenguaje" vs "Lingüística", "turismo" vs "viajes")           │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. LLAMADA AL LLM — Gemini 2.5 Flash + thinking                          │
│    generate_text_gemini_async(                                          │
│        model="models/gemini-2.5-flash",                                  │
│        temperature=0.0,                                                  │
│        mime_type="application/json",                                     │
│        thinking_budget=1024,                                             │
│    )                                                                      │
│    → retry con backoff exponencial ante 503 / 429 / UNAVAILABLE          │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼  respuesta cruda del LLM (JSON, puede venir con fences de markdown)
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. PARSING → RouterDecision                                              │
│    strip_json_fences(raw) → json.loads() → RouterDecision(              │
│        topics: list[str],                                                │
│        filter_mode: "exact" | "multi" | "none",                          │
│        query_rewrite: str,                                               │
│        confidence: float,                                                │
│        reasoning: str,                                                   │
│    )                                                                      │
│    Si falla el parsing → fallback: topics=[], filter_mode="none",        │
│                           confidence=0.0                                  │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. CHEQUEO DE CONFIDENCE                                                  │
│    confidence < ROUTER_CONFIDENCE_THRESHOLD (0.6)?                       │
│      SÍ → filter_mode se fuerza a "none" (no hay confianza, no filtra)   │
│      NO → se mantiene el filter_mode decidido por el LLM                 │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼  RouterDecision final — produce DOS salidas que pipeline.py usa juntas
        │
        ├──► 5a. build_pinecone_filter(decision)  — FILTRO DE METADATA
        │        filter_mode="exact" + topics no vacío → {"topic":{"$eq":topics[0]}}
        │        filter_mode="multi" + len(topics) > 1 → {"topic":{"$in":topics}}
        │        cualquier otro caso                    → None (sin filtro)
        │
        └──► 5b. decision.query_rewrite            — TEXTO A EMBEBER
                 "Cómo hacer un reclamo por retraso de vuelo"
                 (versión limpia de la query, sin saludos/relleno conversacional,
                  usada para la búsqueda semántica — NO la pregunta original)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. pipeline.py → retrieve(                                               │
│        query            = decision.query_rewrite,   ← de 5b              │
│        pinecone_filter  = build_pinecone_filter(decision),  ← de 5a      │
│        pc               = pinecone_client,                               │
│        top_k            = RAG_TOP_K (5),                                 │
│    )                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. retrieval.py — dentro de retrieve()                                   │
│                                                                            │
│    a) Embed:                                                             │
│       generate_embeddings_pinecone_async(                                │
│           texts=[query_rewrite],                                         │
│           model="multilingual-e5-large",                                 │
│           input_type="query",                                            │
│       ) → query_vector                                                   │
│                                                                            │
│    b) Query explícita a Pinecone:                                        │
│       index.query(                                                       │
│           vector = query_vector,                                         │
│           top_k  = max(RAG_CANDIDATE_POOL_SIZE, top_k) = max(15, 5) = 15,│
│           filter = pinecone_filter,        ← {"topic":{"$eq":            │
│                                                "reclamaciones"}}          │
│           include_metadata = True,                                       │
│       ) → hasta 15 candidatos (más de los 5 finales, para tener margen   │
│           antes de aplicar diversidad de fuentes)                        │
│                                                                            │
│    c) match_to_chunk() — normaliza cada match a un dict plano            │
│                                                                            │
│    d) cap_chunks_per_doc(chunks, max_per_doc=2, limit=5)                 │
│       reordena por diversidad de fuentes (máx. 2 chunks por doc_id)      │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  chunks finales (≤5), rankeados por similitud semántica contra
  query_rewrite, dentro del topic filtrado por el router


──────────────────────────────── EJEMPLO CONCRETO ────────────────────────────────

  query = "¿Cuál es la forma más fácil de reclamar cuando un vuelo
            se ha retrasado?"
                    │
                    ▼
  RouterDecision(
      topics       = ["reclamaciones"],
      filter_mode  = "exact",
      query_rewrite= "Cómo hacer un reclamo por retraso de vuelo",
      confidence   = 1.00,
      reasoning    = "La consulta pregunta directamente sobre cómo
                       reclamar un retraso de vuelo, cubierto por
                       'reclamaciones'."
  )
                    │  confidence 1.00 ≥ 0.6 → no se fuerza a "none"
                    ▼
        ┌───────────┴────────────┐
        ▼                        ▼
  filtro de metadata       texto a embeber
  {"topic":{"$eq":         "Cómo hacer un reclamo
    "reclamaciones"}}       por retraso de vuelo"
        └───────────┬────────────┘
                     ▼
     retrieve(query=texto_a_embeber, pinecone_filter=filtro)
                     │
                     ▼
        chunks de Pinecone (filtrados por topic, rankeados
        por similitud semántica contra la query reescrita)
```

## Internals de localización de la UI (Chainlit)

`.chainlit/config.toml` configura `language = "es"` (forzado para todos los usuarios, no detectado por navegador) y `name = "Asistente"`. Existen dos archivos de pantalla de bienvenida y ambos importan:

- `chainlit_es.md` — lo que realmente se renderiza, ya que Chainlit busca primero `chainlit_{language}.md` y `language` está fijado a `"es"`.
- `chainlit.md` — se mantiene sincronizado con el mismo contenido en español. `init_markdown()` de Chainlit corre en cada `chainlit run` y regenera silenciosamente un archivo placeholder genérico en inglés si `chainlit.md` no existe, así que este archivo debe existir con contenido real en lugar de eliminarse.

`.chainlit/config.toml` está trackeado en git (no está en gitignore) porque contiene esta configuración específica del proyecto; `.chainlit/translations/` (los paquetes de idioma propios de Chainlit, regenerados automáticamente) está en gitignore.

Los strings de la UI en tiempo de ejecución en `app/app.py` (nombres de steps, etiquetas de métricas, resumen de sesión) están en español, manteniendo `"Router"` como nombre del step — un préstamo del inglés ya establecido y usado de forma consistente en todo el contenido en español de este proyecto.
