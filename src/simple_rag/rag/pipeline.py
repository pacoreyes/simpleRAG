# -----------------------------------------------------------
# Simple RAG Demo - Pipeline Orchestrator
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------


"""Pipeline del RAG — orquesta router → recuperación → generación → evaluación."""

import logging
from typing import Optional

from google import genai
from pinecone import Pinecone
from pydantic import BaseModel

from simple_rag.rag.evaluate import answer_correctness, answer_similarity, faithfulness
from simple_rag.rag.generator import generate_answer
from simple_rag.rag.retrieval import retrieve
from simple_rag.rag.router import QueryRouter, RouterDecision
from simple_rag.rag.utils.evaluation_helpers import recall_at_k, reciprocal_rank
from simple_rag.rag.utils.routing_helpers import build_pinecone_filter
from simple_rag.settings import settings
from simple_rag.utils.llm_helpers import get_gemini_client
from simple_rag.utils.pinecone_helpers import get_pinecone_client

log = logging.getLogger(__name__)


class EvalScores(BaseModel):
    router_correct: Optional[bool] = None
    recall_at_3: Optional[float] = None
    recall_at_5: Optional[float] = None
    reciprocal_rank: Optional[float] = None
    answer_similarity: Optional[float] = None
    faithfulness_score: Optional[int] = None
    faithfulness_reason: Optional[str] = None
    correctness_score: Optional[int] = None
    correctness_reason: Optional[str] = None


class PipelineResult(BaseModel):
    query: str
    router_decision: RouterDecision
    chunks: list[dict]
    answer: str
    eval: EvalScores = EvalScores()


class RAGPipeline:
    """Pipeline de RAG de punta a punta: router → recuperación → generación → (opcionalmente) evaluación."""

    def __init__(
        self,
        gemini_client: genai.Client,
        pinecone_client: Pinecone,
        top_k: int = settings.RAG_TOP_K,
    ) -> None:
        self._gemini = gemini_client
        self._pc = pinecone_client
        self._top_k = top_k
        self._router = QueryRouter(client=gemini_client)

    async def run(
        self,
        query: str,
        gold_topic: Optional[str] = None,
        gold_doc_ids: Optional[set[str]] = None,
        gold_answer: Optional[str] = None,
        run_faithfulness: bool = True,
        run_correctness: bool = False,
        run_similarity: bool = True,
    ) -> PipelineResult:
        """Ejecuta el pipeline completo para una sola query.

        Args:
            query: Pregunta del usuario en español.
            gold_topic: Etiqueta de tema gold del dataset (para evaluar la precisión del router).
            gold_doc_ids: Set de doc_ids de contextos gold (para evaluar Recall@k).
            gold_answer: String de respuesta gold (para evaluar similitud y corrección).
            run_faithfulness: Si se debe ejecutar el LLM-judge de faithfulness (cuesta una llamada a Gemini).
            run_correctness: Si se debe ejecutar el LLM-judge de corrección de respuesta (cuesta una llamada a Gemini).
            run_similarity: Si se debe calcular la similitud de respuesta basada en embeddings.

        Returns:
            PipelineResult con la respuesta y todos los scores de evaluación disponibles.
        """
        # 1 — Enrutar
        decision = await self._router.route_async(query)
        log.info(
            "Router: filter_mode=%s topics=%s confidence=%.2f",
            decision.filter_mode,
            decision.topics,
            decision.confidence,
        )

        # 2 — Recuperar
        pinecone_filter = build_pinecone_filter(decision)
        chunks = await retrieve(
            query=decision.query_rewrite,
            pc=self._pc,
            pinecone_filter=pinecone_filter,
            top_k=self._top_k,
        )
        log.info("Se recuperaron %d chunks (filter=%s)", len(chunks), pinecone_filter)

        # 3 — Generar
        answer_text = await generate_answer(
            query=query,
            chunks=chunks,
            client=self._gemini,
        )

        # 4 — Evaluar (solo cuando se provee data gold)
        scores = EvalScores()
        retrieved_doc_ids = [c["doc_id"] for c in chunks]

        if gold_topic is not None:
            scores.router_correct = (
                gold_topic in decision.topics if decision.topics else False
            )

        if gold_doc_ids is not None:
            scores.recall_at_3 = recall_at_k(retrieved_doc_ids, gold_doc_ids, k=3)
            scores.recall_at_5 = recall_at_k(retrieved_doc_ids, gold_doc_ids, k=5)
            scores.reciprocal_rank = reciprocal_rank(retrieved_doc_ids, gold_doc_ids)

        if gold_answer is not None and run_similarity:
            scores.answer_similarity = await answer_similarity(
                generated=answer_text,
                gold=gold_answer,
                pc=self._pc,
            )

        if run_faithfulness and chunks:
            faith = await faithfulness(
                generated_answer=answer_text,
                retrieved_chunks=chunks,
                client=self._gemini,
            )
            scores.faithfulness_score = faith.get("score")
            scores.faithfulness_reason = faith.get("reason")

        if gold_answer is not None and run_correctness:
            correctness = await answer_correctness(
                generated_answer=answer_text,
                gold_answer=gold_answer,
                client=self._gemini,
            )
            scores.correctness_score = correctness.get("score")
            scores.correctness_reason = correctness.get("reason")

        return PipelineResult(
            query=query,
            router_decision=decision,
            chunks=chunks,
            answer=answer_text,
            eval=scores,
        )


def build_pipeline() -> RAGPipeline:
    """Construye un RAGPipeline a partir de la configuración del entorno."""
    gemini = get_gemini_client(settings.GEMINI_API_KEY)
    pc = get_pinecone_client(settings.PINECONE_API_KEY)
    return RAGPipeline(gemini_client=gemini, pinecone_client=pc)
