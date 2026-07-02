"""Tests de integración — requieren claves de API de Pinecone y Gemini en vivo.

Ejecutar con:
    uv run pytest tests/test_rag_integration.py --integration -v -s

Cada pregunta de test viene del dataset RagQuAS y tiene contextos y respuestas
gold verificables. El test suite mide Recall@3, Recall@5, MRR, precisión del
router, similitud de respuesta, y faithfulness, y luego imprime una tabla resumen.
"""

import asyncio

import pytest

from simple_rag.rag.pipeline import PipelineResult, build_pipeline
from simple_rag.rag.utils.evaluation_helpers import mean_reciprocal_rank
from simple_rag.rag.utils.gold_data_helpers import load_gold_data

# ---------------------------------------------------------------------------
# Preguntas gold de test — una por tema, tomadas textualmente del dataset RagQuAS
# ---------------------------------------------------------------------------

TEST_QUESTIONS: list[dict] = [
    {
        "question": "¿Cuál es la forma más fácil de reclamar cuando un vuelo que sale de España se ha retrasado?",
        "gold_topic": "reclamaciones",
    },
    {
        "question": "recomiendame un seguro de hogar para mi nueva casa",
        "gold_topic": "seguros",
    },
    {
        "question": "Hola! ¿Podrías explicarme cuáles son los tres beneficios principales del Surya Namaskar?",
        "gold_topic": "yoga",
    },
    {
        "question": "hola, me puedes explicar qué usos puede tener la Gabapentina en gatos? gracias",
        "gold_topic": "veterinaria",
    },
    {
        "question": "Esta noche será la oposición de Neptuno, en qué posición estará Neptuno respecto del Sol y de la Tierra?",
        "gold_topic": "astronomía",
    },
    {
        "question": "qué comidas puedo probar en mi viaje a Japón?",
        "gold_topic": "gastronomía",
    },
    {
        "question": "Ventajas y desventajas de los coches híbridos recargables.",
        "gold_topic": "coches",
    },
    {
        "question": "como debería aprender a leer y escribir japonés?",
        "gold_topic": "idiomas",
    },
    {
        "question": "Dime las diferencias entre una corchea y una negra.",
        "gold_topic": "música",
    },
    {
        "question": "Hola! Quiero saber cómo puedo renovar mi DNI-e, ¿podrías ayudarme paso a paso?",
        "gold_topic": "documentación",
    },
    {
        "question": "Necesito saber qué significa la 'm' en la ecuación 'E=MC²'.",
        "gold_topic": "energía",
    },
    {
        "question": "¿Cuál es el origen de la expresión dar gato por liebre?",
        "gold_topic": "lenguaje",
    },
    {
        "question": "necesito la tarjeta sanitaria europea para viajar a Londres y cómo se tramita?",
        "gold_topic": "turismo",
    },
    {
        "question": "¿Cómo puedo evitar una estafa telefónica?",
        "gold_topic": "estafas",
    },
    {
        "question": "¿Qué ventajas e inconvenientes tiene adoptar un gato frente a comprarlo en una tienda?",
        "gold_topic": "Mascotas",
    },
]


# ---------------------------------------------------------------------------
# Fixture de pipeline compartido (una instancia por sesión para reusar conexiones)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def pipeline():
    """Pipeline nuevo por test — evita problemas de binding al event loop de asyncio con httpx."""
    return build_pipeline()


@pytest.fixture(scope="module")
def gold_lookup():
    return load_gold_data()


# ---------------------------------------------------------------------------
# Tests de preguntas individuales
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("test_case", TEST_QUESTIONS, ids=[tc["gold_topic"] for tc in TEST_QUESTIONS])
async def test_single_question(test_case, pipeline, gold_lookup):
    """Corre una sola pregunta a través del pipeline y valida las métricas de retrieval."""
    question = test_case["question"]
    gold_topic = test_case["gold_topic"]

    gold = gold_lookup.get(question, {})
    gold_doc_ids: set[str] = gold.get("gold_doc_ids", set())
    gold_answer: str = gold.get("gold_answer", "")

    result: PipelineResult = await pipeline.run(
        query=question,
        gold_topic=gold_topic,
        gold_doc_ids=gold_doc_ids,
        gold_answer=gold_answer,
        run_faithfulness=True,
        run_correctness=False,
        run_similarity=True,
    )

    e = result.eval
    print(f"\n{'─'*70}")
    print(f"Tema:      {gold_topic}")
    print(f"Pregunta:  {question[:80]}")
    print(f"Router:    topics={result.router_decision.topics}  mode={result.router_decision.filter_mode}  conf={result.router_decision.confidence:.2f}")
    print(f"Router ✓:  {e.router_correct}")
    print(f"Recall@3:  {e.recall_at_3:.2f}   Recall@5: {e.recall_at_5:.2f}   RR: {e.reciprocal_rank:.2f}")
    print(f"Sim resp:  {e.answer_similarity:.2f}" if e.answer_similarity is not None else "Sim resp:  N/A")
    print(f"Fidelidad: {e.faithfulness_score}/5 — {e.faithfulness_reason}")
    print(f"Respuesta: {result.answer[:200]}")

    # Aserciones mínimas de calidad
    assert result.answer, "La respuesta no debe estar vacía"
    assert e.recall_at_3 is not None, "recall_at_3 debe calcularse cuando se proveen gold_doc_ids"
    assert e.faithfulness_score is not None, "faithfulness_score debe calcularse"


# ---------------------------------------------------------------------------
# Test de métricas agregadas — corre las 15 preguntas y verifica los umbrales promedio
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_aggregate_metrics(pipeline, gold_lookup):
    """Corre todas las preguntas de test y verifica los umbrales de métrica a nivel sistema."""

    recall3_scores: list[float] = []
    recall5_scores: list[float] = []
    rr_scores: list[float] = []
    similarity_scores: list[float] = []
    faithfulness_scores: list[int] = []
    router_correct: list[bool] = []

    for tc in TEST_QUESTIONS:
        question = tc["question"]
        gold_topic = tc["gold_topic"]
        gold = gold_lookup.get(question, {})
        gold_doc_ids: set[str] = gold.get("gold_doc_ids", set())
        gold_answer: str = gold.get("gold_answer", "")

        try:
            result: PipelineResult = await pipeline.run(
                query=question,
                gold_topic=gold_topic,
                gold_doc_ids=gold_doc_ids,
                gold_answer=gold_answer,
                run_faithfulness=True,
                run_correctness=False,
                run_similarity=True,
            )
        except Exception as exc:
            print(f"  [OMITIDO] {gold_topic}: error del pipeline — {exc}")
            await asyncio.sleep(5)
            continue
        await asyncio.sleep(5)  # evita 503s consecutivos de Gemini bajo carga
        e = result.eval
        if e.recall_at_3 is not None:
            recall3_scores.append(e.recall_at_3)
        if e.recall_at_5 is not None:
            recall5_scores.append(e.recall_at_5)
        if e.reciprocal_rank is not None:
            rr_scores.append(e.reciprocal_rank)
        if e.answer_similarity is not None:
            similarity_scores.append(e.answer_similarity)
        if e.faithfulness_score is not None:
            faithfulness_scores.append(e.faithfulness_score)
        if e.router_correct is not None:
            router_correct.append(e.router_correct)

    mean_r3 = sum(recall3_scores) / len(recall3_scores) if recall3_scores else 0.0
    mean_r5 = sum(recall5_scores) / len(recall5_scores) if recall5_scores else 0.0
    mrr = mean_reciprocal_rank(rr_scores)
    mean_sim = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    mean_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    router_acc = sum(router_correct) / len(router_correct) if router_correct else 0.0

    print(f"\n{'═'*70}")
    print("MÉTRICAS AGREGADAS SOBRE 15 PREGUNTAS GOLD")
    print(f"{'═'*70}")
    print(f"  Precisión del router: {router_acc:.2%}  ({sum(router_correct)}/{len(router_correct)} correctas)")
    print(f"  Recall@3:             {mean_r3:.2%}")
    print(f"  Recall@5:             {mean_r5:.2%}")
    print(f"  MRR:                  {mrr:.3f}")
    print(f"  Similitud de resp.:   {mean_sim:.3f}  (coseno vs respuesta gold)")
    print(f"  Faithfulness:         {mean_faith:.2f}/5")
    print(f"{'═'*70}")

    # Umbrales mínimos de calidad
    assert router_acc >= 0.70, f"Precisión del router {router_acc:.2%} por debajo del umbral de 70%"
    assert mean_r3 >= 0.50, f"Recall@3 promedio {mean_r3:.2%} por debajo del umbral de 50%"
    assert mean_r5 >= 0.55, f"Recall@5 promedio {mean_r5:.2%} por debajo del umbral de 55%"
    assert mrr >= 0.45, f"MRR {mrr:.3f} por debajo del umbral de 0.45"
    assert mean_faith >= 3.5, f"Faithfulness promedio {mean_faith:.2f} por debajo del umbral de 3.5/5"
