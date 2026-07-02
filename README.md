# Simple RAG Demo

Un demo de Retrieval-Augmented Generation (RAG) construido sobre **[RagQuAS](https://huggingface.co/datasets/IIC/RagQuAS)** — 201 pares de pregunta-respuesta escritos a mano en español, en ~30 dominios (seguros, primeros auxilios, astronomía, recetas, reclamaciones de viaje, veterinaria, yoga, etc.), creado por el Instituto de Ingeniería del Conocimiento específicamente para evaluar un sistema RAG completo.

## Características principales

- **Router con conciencia de tema** (Gemini 2.5 Flash + thinking) pre-filtra el espacio de búsqueda vectorial por dominio antes de correr la búsqueda semántica, en lugar de depender solo del retrieval denso.
- **Reordenamiento por diversidad de fuentes** limita los chunks por documento de origen para que ningún documento individual domine los resultados del top-k.
- **Generación fundamentada con citas** — las respuestas se construyen únicamente a partir de los chunks recuperados, citan sus fuentes, y señalan explícitamente cualquier parte de la pregunta que el contexto no cubra.
- **Harness de evaluación integrado** — Recall@k, MRR, similitud de respuesta basada en embeddings, y LLM-judge de faithfulness/corrección, conectados a un test suite de `pytest --integration` con umbrales de calidad estrictos.
- **Arquitectura en capas limpia** — ingesta (`data_pipeline/`) y consulta (`rag/`) son subsistemas independientes que solo comparten infraestructura genérica. Ver [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **UI de chat en Chainlit**, alineada con el idioma del dataset, español.

---

## Arquitectura

```
simple_rag_demo/
├── data_volume/assets/         ← dataset de origen + salida del pipeline (Parquet)
├── src/simple_rag/
│   ├── settings.py             ← toda la configuración
│   ├── models.py                ← modelos Pydantic compartidos
│   ├── utils/                   ← helpers genéricos, a nivel de tecnología
│   ├── data_pipeline/           ← pipeline batch: dataset → índice de Pinecone
│   └── rag/                     ← sistema de consulta: router → retrieval → generación → eval
├── app/app.py                   ← UI de chat en Chainlit
└── tests/                       ← tests unitarios + test suite de integración
```

`data_pipeline` y `rag` son subsistemas independientes — comparten `utils/`, `models.py`, y `settings.py`, pero nunca se importan entre sí. Tanto `data_pipeline/` como `rag/` mantienen sus propios helpers puros específicos de dominio en un subpaquete `utils/` local, separado de su código de orquestación. Reglas completas de capas, límites de imports, y el árbol de archivos anotado: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Flujo de datos (data pipeline de ingesta de datos)

```
Parquet de RagQuAS (201 filas, 19 columnas)
        │
        ▼
 Cargar → aprovechamos text_1..5 por fila → DocumentRecord por documento único
          Deduplicar por doc_id (SHA256 del texto)      → 201 filas → 132 documentos únicos
        │
        ▼
 Chunking de cada documento (≤300 tokens, 1 oración de overlap)  → 132 documentos → 1063 chunks
        │
        ▼
 Embed (Pinecone Inference: multilingual-e5-large, 1024 dims)
        │
        ▼
 Upsert al índice de Pinecone "ragchunks"
   metadata: topic, doc_id, link, source_domain, chunk_index, text
```

### Estrategia de retrieval (consulta)

El dataset abarca más de 30 dominios sin solapamiento semántico. El router aprovecha la metadata `topic` para pre-filtrar el índice antes de la búsqueda semántica, acotando el pool de candidatos de 1063 a ~30–50 chunks relevantes al tema:

```
Pregunta del usuario → El router clasifica la query (tema + filter mode, vía Gemini)
                     → Query a Pinecone con filter={"topic": "<tema_detectado>"}
                     → Top-k chunks semánticamente similares dentro de ese tema
                     → Rerank: máximo 2 chunks por doc_id (diversidad de fuentes)
                     → Gemini genera la respuesta a partir de los chunks recuperados + cita fuentes
```

Este enfoque de tres niveles — filtro de tema → búsqueda semántica → reordenamiento por diversidad de fuentes — mejora la precisión sin requerir metadata adicional más allá de la que el dataset ya provee.

---

## Primeros pasos

### Prerrequisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes)
- Una cuenta de [Pinecone](https://www.pinecone.io/) (el pipeline crea automáticamente el índice `ragchunks` en la primera ejecución)
- Una clave de API de [Google Gemini](https://aistudio.google.com/) (usada por el router de consulta del RAG y el generador de respuestas)

### Instalación

```bash
git clone <repo-url>
cd simple_rag_demo
uv sync --group dev
```

### Configuración

Copia `.env.example` a `.env` y completa tus claves:

```
GEMINI_API_KEY=...
PINECONE_API_KEY=...
```

El resto de los parámetros configurables viven en `src/simple_rag/settings.py` con sus valores por defecto — se pueden sobrescribir vía variables de entorno.

### Ejecutar el pipeline

Después de `uv sync`, el comando CLI `run-pipeline` queda registrado en tu shell, con tres subcomandos para que cada fase se pueda ejecutar y volver a ejecutar de forma independiente.

| Comando | Pasos | Qué hace |
|---|---|---|
| `run-pipeline preprocess` | 1–3 | Carga el dataset → explode/deduplica → chunking → guarda `chunks-processed.parquet` |
| `run-pipeline load` | 4–5 | Lee el Parquet → embed (multilingual-e5-large) → upsert a Pinecone con metadata |
| `run-pipeline all` | 1–5 | Ambas fases en secuencia |

```bash
# Primera ejecución típica
run-pipeline preprocess
run-pipeline load

# O todo junto
run-pipeline all

# Probar con un subconjunto pequeño antes de comprometerse con el dataset completo
run-pipeline preprocess --limit 20
run-pipeline load
```

`preprocess` y `all` aceptan `--limit N` para acotar la cantidad de filas de origen — útil para iterar durante el desarrollo. `load` requiere que `chunks-processed.parquet` exista (escrito por `preprocess`) y lanza un error claro si falta. Ejecutar desde la raíz del proyecto — `.env` se lee automáticamente.

---

## Usando el sistema RAG

### UI de chat (Chainlit)

La forma principal de usar el sistema RAG es la UI de chat en Chainlit, en `app/app.py`. Conecta `build_pipeline()` y, para cada query, muestra la decisión del router, los chunks recuperados, y las métricas de evaluación como steps colapsables antes de la respuesta final. La UI está totalmente localizada al español — ver [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#internals-de-localización-de-la-ui-chainlit) para los detalles internos de la localización.

```bash
uv run chainlit run app/app.py -w
```

Después abre la URL que imprime Chainlit (por defecto `http://localhost:8000`).

- Haz cualquier pregunta en español dentro de los ~30 dominios del dataset (seguros, yoga, veterinaria, astronomía, gastronomía, reclamaciones, primeros auxilios, viajes, etc.) — vas a ver la decisión de tema del router, los chunks recuperados, y un score de faithfulness.
- Haz una pregunta **exactamente** como aparece en el dataset de RagQuAS (p. ej. una de las 15 preguntas en `tests/test_rag_integration.py`) para además ver Recall@3/5, MRR, precisión del router, y similitud de respuesta — la app compara tu pregunta contra la data gold cargada desde el parquet de origen.
- El pipeline ya debe estar cargado en Pinecone (`run-pipeline all`) y `.env` debe tener `GEMINI_API_KEY` / `PINECONE_API_KEY` válidas antes de iniciar la UI de chat.

### Uso como librería (script / REPL)

El sistema RAG también se puede usar directamente como librería. El archivo `.env` se lee automáticamente vía pydantic-settings.

```python
import asyncio
from simple_rag.rag.pipeline import build_pipeline

async def main():
    pipeline = build_pipeline()
    result = await pipeline.run("¿Cuáles son los beneficios del yoga?")

    print(result.answer)
    print("Decisión del router:", result.router_decision.topics, result.router_decision.filter_mode)
    for chunk in result.chunks:
        print(f"  [{chunk['topic']}] {chunk['source_domain']} — score {chunk['score']:.3f}")

asyncio.run(main())
```

```bash
uv run python my_script.py
```

Pasa `gold_topic`, `gold_doc_ids`, y `gold_answer` a `pipeline.run(...)` para además obtener scores de Recall@k, precisión del router, y similitud de respuesta en `result.eval` — así es como el test suite de integración mide la calidad contra el dataset de RagQuAS.

### Qué conecta `build_pipeline()`

| Componente | Modelo / Servicio | Rol |
|---|---|---|
| `QueryRouter` | Gemini 2.5 Flash + thinking | Clasifica la query → tema + filter mode |
| `retrieve()` | Índice `ragchunks` de Pinecone | Búsqueda semántica filtrada por tema (top-5) |
| `generate_answer()` | Gemini 2.5 Flash Lite | Respuesta en español fundamentada en los chunks recuperados |
| Judge de `faithfulness()` | Gemini 2.5 Flash Lite | El LLM puntúa la fundamentación de la respuesta de 1 a 5 |

Todos los componentes se crean a partir de variables de entorno (`GEMINI_API_KEY`, `PINECONE_API_KEY`) leídas desde `.env`.

---

## Testing

```bash
# Tests unitarios (no requieren claves de API)
uv run pytest

# Lint / formato / seguridad / tipos
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run bandit -r src/
uv run ty check src/
```

El test suite de integración ejercita el pipeline completo contra 15 preguntas gold curadas (una por tema), midiendo recall de retrieval, calidad del ranking, precisión del router, y faithfulness de las respuestas. Requiere un índice de Pinecone ya cargado (`run-pipeline all`) y claves de API en vivo, y se omite por defecto:

```bash
# Suite de integración completa: 15 tests de pregunta individual + métricas agregadas
uv run pytest tests/test_rag_integration.py --integration -v -s

# Solo métricas agregadas (más rápido)
uv run pytest tests/test_rag_integration.py::test_aggregate_metrics --integration -v -s
```

**Últimos resultados medidos** (router Gemini 2.5 Flash + generador/judge Gemini 2.5 Flash Lite):

| Métrica | Valor | Umbral |
|---|---|---|
| Precisión del router | 100% | ≥ 70% |
| Recall@3 | 76.4% | ≥ 50% |
| Recall@5 | 88.2% | ≥ 55% |
| MRR | 0.967 | ≥ 0.45 |
| Similitud de respuesta | 0.933 | — |
| Faithfulness | 5.00/5 | ≥ 3.5 |

Historial completo de benchmarks y análisis: ver `LOG.md`.

---

## Modelos de datos

- **`SourceRow`** — mapeo directo del schema Parquet de 19 columnas de RagQuAS. Cuatro campos requeridos (`topic`, `answer`, `question`, `variant`); todos los campos `text_i`, `context_i`, y `link_i` son opcionales.
- **`DocumentRecord`** — un documento de origen único después de aplicar explode y deduplicar `SourceRow`s. `doc_id` es un SHA256 del contenido del texto, estable entre ejecuciones.
- **`ChunkRecord`** — un chunk de un `DocumentRecord`, listo para el upsert en Pinecone (el `id` del vector es `{doc_id}_chunk_{index}`). Hereda `doc_id`, `topic`, `link`, y `source_domain` de su padre, y agrega `chunk_index`, `char_length`, y `token_count`. Estos mismos campos son los que se guardan como metadata en Pinecone al momento de la consulta — `text` arma la ventana de contexto del LLM, `topic` es el campo de filtro del router, y `link`/`source_domain` respaldan las citas de las respuestas.

---

## Problemas conocidos

Ninguno por el momento.

---

## Licencia

MIT — ver `pyproject.toml`.

## Para saber más

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — reglas completas de capas, límites de imports, diagrama de dependencias, árbol de archivos anotado, e internals de la localización de Chainlit.
- `LOG.md` — historial de desarrollo, decisiones, y tendencias de benchmarks a lo largo del tiempo.
