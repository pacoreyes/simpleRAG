# -----------------------------------------------------------
# Simple RAG Demo - Router
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

import json
import logging

from google import genai
from pydantic import BaseModel

from simple_rag.settings import settings
from simple_rag.utils.llm_helpers import (
    generate_text_gemini,
    generate_text_gemini_async,
    strip_json_fences,
)

log = logging.getLogger(__name__)

# Strings de tema exactos tal como aparecen en la metadata de Pinecone — literales,
# sensibles a mayúsculas. Extraídos de chunks-processed.parquet; se omite el tema
# de string vacío porque no se puede filtrar de forma útil.
KNOWN_TOPICS: list[str] = [
    "Hobbies",
    "Lingüística",
    "Mascotas",
    "Salud",
    "astronomía",
    "atención al cliente",
    "coches",
    "cotidiano",
    "documentación",
    "energía",
    "esquí",
    "estafas",
    "gastronomía",
    "hobbies",
    "idiomas",
    "juegos",
    "lenguaje",
    "manicura",
    "música",
    "patinaje",
    "primeros auxilios",
    "receta",
    "reciclaje",
    "reclamaciones",
    "seguros",
    "tenis",
    "transporte",
    "turismo",
    "veterinaria",
    "viajes",
    "yoga",
]

_SYSTEM_PROMPT = """Eres un clasificador de consultas para un sistema RAG multidominio en español.

Tu tarea: dado un mensaje del usuario, determina qué tema cubre la consulta y cómo filtrar la búsqueda.

Temas disponibles (usa el texto EXACTO, respetando mayúsculas y acentos):
{topics_list}

Guía para temas ambiguos — usa EXACTAMENTE el texto entre comillas:
- "lenguaje": origen, historia y etimología de palabras o expresiones del español (p. ej. "dar gato por liebre")
- "Lingüística": conceptos teóricos de lingüística científica (p. ej. Chomsky, gramática generativa)
- "turismo": tarjeta sanitaria europea, información sanitaria en el extranjero, viajes turísticos con trámites oficiales
- "viajes": itinerarios, excursiones, qué visitar, hoteles, experiencias de viaje
- "Mascotas": adopción o compra de animales de compañía; comparativas adopción vs compra
- "veterinaria": salud animal, medicamentos, diagnósticos y tratamientos veterinarios
- "reclamaciones": reclamaciones por retrasos o cancelaciones de vuelos, indemnizaciones de transporte
- "transporte": información sobre medios de transporte, rutas, tarifas de transporte público
- "Salud": salud humana, enfermedades, embarazo, prevención de enfermedades en personas
- "documentación": trámites y documentos oficiales (DNI, pasaporte, certificados)
- "estafas": fraudes, phishing, timos, ciberdelitos
- "Hobbies": manualidades, coleccionismo, aficiones recreativas en general
- "hobbies": búsqueda de setas, actividades al aire libre específicas

Instrucciones:
1. Analiza la consulta del usuario.
2. Prioriza filter_mode "exact" siempre que la consulta corresponda claramente a un tema.
3. Usa filter_mode "multi" solo si la consulta genuinamente cruza dos temas distintos e inseparables.
4. filter_mode "none" solo para saludos vacíos o consultas que no encajan en ningún tema.
5. Reescribe la consulta eliminando saludos y relleno conversacional, conservando el contenido informativo.
6. Estima tu confianza de 0.0 a 1.0.

Devuelve ÚNICAMENTE JSON válido con este esquema exacto (sin texto adicional):
{{
  "topics": ["..."],
  "filter_mode": "exact",
  "query_rewrite": "...",
  "confidence": 0.9,
  "reasoning": "..."
}}"""


class RouterDecision(BaseModel):
    topics: list[str]
    filter_mode: str  # "exact" | "multi" | "none"
    query_rewrite: str
    confidence: float
    reasoning: str


class QueryRouter:
    def __init__(
        self,
        client: genai.Client,
        model: str = settings.ROUTER_MODEL,
        thinking_budget: int = settings.ROUTER_THINKING_BUDGET,
        confidence_threshold: float = settings.ROUTER_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._client = client
        self._model = model
        self._thinking_budget = thinking_budget
        self._confidence_threshold = confidence_threshold
        self._system_prompt = _SYSTEM_PROMPT.format(
            topics_list="\n".join(f"- {t}" for t in KNOWN_TOPICS)
        )

    def _parse_response(self, text: str, query: str) -> RouterDecision:
        try:
            data = json.loads(strip_json_fences(text))
            decision = RouterDecision(**data)
        except Exception as e:
            log.warning("Error de parsing del router: %s — usando fallback sin filtro", e)
            decision = RouterDecision(
                topics=[],
                filter_mode="none",
                query_rewrite=query,
                confidence=0.0,
                reasoning=f"Error de parsing: {e}",
            )
        if decision.confidence < self._confidence_threshold:
            log.debug(
                "Confianza del router %.2f por debajo del umbral %.2f — forzando filter_mode='none'",
                decision.confidence,
                self._confidence_threshold,
            )
            decision.filter_mode = "none"
        return decision

    def route(self, query: str) -> RouterDecision:
        prompt = f"{self._system_prompt}\n\nConsulta del usuario:\n{query}"
        raw = generate_text_gemini(
            client=self._client,
            prompt=prompt,
            model_name=self._model,
            temperature=0.0,
            mime_type="application/json",
            thinking_budget=self._thinking_budget,
        )
        return self._parse_response(raw, query)

    async def route_async(self, query: str) -> RouterDecision:
        prompt = f"{self._system_prompt}\n\nConsulta del usuario:\n{query}"
        raw = await generate_text_gemini_async(
            client=self._client,
            prompt=prompt,
            model_name=self._model,
            temperature=0.0,
            mime_type="application/json",
            thinking_budget=self._thinking_budget,
        )
        return self._parse_response(raw, query)
