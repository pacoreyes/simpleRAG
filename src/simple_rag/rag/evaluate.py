# -----------------------------------------------------------
# Simple RAG Demo - Evaluate
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

import logging

from google import genai
from pinecone import Pinecone

from simple_rag.rag.utils.evaluation_helpers import cosine_similarity, parse_judge_response
from simple_rag.settings import settings
from simple_rag.utils.llm_helpers import generate_text_gemini_async
from simple_rag.utils.pinecone_helpers import generate_embeddings_pinecone_async

log = logging.getLogger(__name__)

_FAITHFULNESS_PROMPT = """Evalúa si la respuesta generada está completamente respaldada por el contexto proporcionado.

Contexto recuperado:
{context}

Respuesta generada:
{answer}

¿Contiene la respuesta únicamente información presente en el contexto? Penaliza afirmaciones sin soporte.

Devuelve JSON: {{"score": <entero 1-5>, "reason": "<una frase>"}}
Donde: 1=sin soporte, 3=parcialmente, 5=completamente respaldada."""

_CORRECTNESS_PROMPT = """Compara la respuesta generada con la respuesta de referencia (gold).

Respuesta de referencia:
{gold_answer}

Respuesta generada:
{generated_answer}

¿Son semánticamente equivalentes? Ignora diferencias de estilo o formato.

Devuelve JSON: {{"score": <entero 1-5>, "reason": "<una frase>"}}
Donde: 1=incorrecta, 3=parcialmente correcta, 5=completamente correcta."""


async def answer_similarity(
    generated: str,
    gold: str,
    pc: Pinecone,
    model: str = settings.DEFAULT_EMBEDDINGS_MODEL_NAME,
) -> float:
    """Similitud coseno entre los embeddings de la respuesta generada y la gold.

    Ambos textos se embeben con input_type='passage' para que estén en el
    mismo espacio de embedding sin importar su longitud o redacción.

    Args:
        generated: String de la respuesta generada.
        gold: String de la respuesta gold del dataset.
        pc: Cliente de Pinecone inicializado.
        model: Nombre del modelo de embedding de Pinecone Inference.

    Returns:
        Similitud coseno en [-1.0, 1.0], típicamente [0.5, 1.0] para textos similares.
    """
    embeddings = await generate_embeddings_pinecone_async(
        pc=pc,
        texts=[generated, gold],
        model=model,
        input_type="passage",
        batch_size=2,
        description="Embebiendo respuestas",
    )
    return cosine_similarity(embeddings[0], embeddings[1])


async def faithfulness(
    generated_answer: str,
    retrieved_chunks: list[dict],
    client: genai.Client,
    model: str = settings.GEMINI_MODEL,
) -> dict:
    """Score de LLM-judge sobre si la respuesta está fundamentada en el contexto recuperado.

    Args:
        generated_answer: La respuesta producida por el generador.
        retrieved_chunks: Dicts de chunks de retrieval (cada uno debe tener una clave 'text').
        client: Cliente de Gemini configurado.
        model: Nombre del modelo de Gemini para la llamada al judge.

    Returns:
        Dict con 'score' (int 1-5) y 'reason' (str).
    """
    context = "\n\n".join(c["text"] for c in retrieved_chunks if c.get("text"))
    prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=generated_answer)
    try:
        raw = await generate_text_gemini_async(
            client=client,
            prompt=prompt,
            model_name=model,
            max_tokens=512,
            temperature=0.0,
            mime_type="application/json",
        )
    except RuntimeError as e:
        log.warning("Error de API en faithfulness: %s", str(e)[:200])
        return {"score": 0, "reason": "error de API"}
    return parse_judge_response(raw, error_prefix="Faithfulness")


async def answer_correctness(
    generated_answer: str,
    gold_answer: str,
    client: genai.Client,
    model: str = settings.GEMINI_MODEL,
) -> dict:
    """Score de LLM-judge sobre la equivalencia semántica entre la respuesta generada y la gold.

    Args:
        generated_answer: La respuesta producida por el generador.
        gold_answer: String de la respuesta gold del dataset.
        client: Cliente de Gemini configurado.
        model: Nombre del modelo de Gemini para la llamada al judge.

    Returns:
        Dict con 'score' (int 1-5) y 'reason' (str).
    """
    prompt = _CORRECTNESS_PROMPT.format(
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )
    try:
        raw = await generate_text_gemini_async(
            client=client,
            prompt=prompt,
            model_name=model,
            max_tokens=256,
            temperature=0.0,
            mime_type="application/json",
        )
    except RuntimeError as e:
        log.warning("Error de API en correctness: %s", str(e)[:200])
        return {"score": 0, "reason": "error de API"}
    return parse_judge_response(raw, error_prefix="Correctness")
