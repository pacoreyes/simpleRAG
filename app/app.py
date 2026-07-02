# -----------------------------------------------------------
# Simple RAG Demo - Chainlit UI
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Punto de entrada de Chainlit: `chainlit run app/app.py -w`.

Conecta el pipeline existente de simple_rag.rag a una UI de chat. Cada turno
muestra la decisión del router, los chunks recuperados, y las métricas de
evaluación como steps colapsables, seguidos de la respuesta generada con
los links de las fuentes.
"""

import logging

import chainlit as cl

from simple_rag.rag.pipeline import PipelineResult, build_pipeline
from simple_rag.rag.utils.gold_data_helpers import load_gold_data

log = logging.getLogger(__name__)

# Datos de referencia de solo lectura — seguro cargarlos una vez al importar
# (acá no hay clientes atados al event loop, solo una lectura de parquet).
GOLD_LOOKUP: dict[str, dict] = load_gold_data()


@cl.on_chat_start
async def on_chat_start() -> None:
    # Se construye por sesión: los clientes de Gemini/Pinecone subyacentes
    # atan su transporte httpx async al event loop activo en el momento de
    # construcción, así que un singleton a nivel de módulo arriesga problemas
    # de binding entre sesiones.
    cl.user_session.set("pipeline", build_pipeline())
    cl.user_session.set("metric_history", [])

    await cl.Message(
        content=(
            "**Simple RAG Demo** — pregúntame algo en español (ver preguntas de ejemplo "
            "arriba, en la pantalla de bienvenida)."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    pipeline = cl.user_session.get("pipeline")
    query = message.content.strip()

    gold = GOLD_LOOKUP.get(query)

    async with cl.Step(name="Router", type="tool") as router_step:
        router_step.input = query

        result: PipelineResult = await pipeline.run(
            query=query,
            gold_topic=gold["gold_topic"] if gold else None,
            gold_doc_ids=gold["gold_doc_ids"] if gold else None,
            gold_answer=gold["gold_answer"] if gold else None,
            run_faithfulness=True,
            run_correctness=False,
            run_similarity=True,
        )

        decision = result.router_decision
        router_step.output = (
            f"topics={decision.topics}  filter_mode={decision.filter_mode}  "
            f"confidence={decision.confidence:.2f}\n\n"
            f"query_rewrite: {decision.query_rewrite}\n\n"
            f"reasoning: {decision.reasoning}"
        )

    async with cl.Step(name="Recuperación", type="retrieval") as retrieval_step:
        retrieval_step.input = {
            "filter_mode": decision.filter_mode,
            "topics": decision.topics,
            "top_k": len(result.chunks),
        }
        if result.chunks:
            lines = [
                f"[{c['topic']}] {c['source_domain']} — score {c['score']:.3f}"
                for c in result.chunks
            ]
            retrieval_step.output = "\n".join(lines)
        else:
            retrieval_step.output = "No se recuperaron fragmentos."

    e = result.eval
    eval_lines = [f"Fidelidad: {e.faithfulness_score}/5 — {e.faithfulness_reason}"]
    if gold:
        eval_lines.append(f"Router correcto: {e.router_correct}")
        eval_lines.append(f"Recall@3: {e.recall_at_3:.2f}   Recall@5: {e.recall_at_5:.2f}")
        eval_lines.append(f"MRR (esta consulta): {e.reciprocal_rank:.2f}")
        if e.answer_similarity is not None:
            eval_lines.append(f"Similitud de respuesta: {e.answer_similarity:.3f}")

    async with cl.Step(name="Evaluación", type="tool") as eval_step:
        eval_step.output = "\n".join(eval_lines)

    history: list[dict] = cl.user_session.get("metric_history")
    history.append({
        "faithfulness": e.faithfulness_score,
        "recall_at_3": e.recall_at_3,
    })
    cl.user_session.set("metric_history", history)

    faith_scores = [h["faithfulness"] for h in history if h["faithfulness"] is not None]
    if faith_scores:
        avg_faith = sum(faith_scores) / len(faith_scores)
        session_summary = f"Fidelidad promedio de la sesión ({len(faith_scores)} consultas): {avg_faith:.2f}/5"
    else:
        session_summary = None

    seen_links: set[str] = set()
    unique_sources: list[dict] = []
    for c in result.chunks:
        link = c["link"]
        if link and link in seen_links:
            continue
        seen_links.add(link)
        unique_sources.append(c)

    sources = [
        cl.Text(
            name=c["source_domain"] or c["link"] or f"chunk_{i}",
            content=c["link"] or "(sin enlace)",
            display="inline",
        )
        for i, c in enumerate(unique_sources)
    ]

    await cl.Message(content=result.answer, elements=sources).send()

    if session_summary:
        await cl.Message(content=session_summary, author="Métricas de sesión").send()
