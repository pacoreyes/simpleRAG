# Simple RAG Demo

Un demo de Retrieval-Augmented Generation sobre el dataset RagQuAS — 201 pares
de pregunta-respuesta en español, en ~30 dominios (seguros, yoga, veterinaria,
astronomía, gastronomía, reclamaciones, primeros auxilios, viajes, y más).

Haz una pregunta en español. Cada respuesta muestra la decisión de tema del
router, los fragmentos recuperados, y las métricas de evaluación (faithfulness
siempre; Recall@k, MRR, precisión del router, y similitud de respuesta cuando
tu pregunta coincide textualmente con una de las preguntas gold del dataset).

## Prueba una de estas preguntas gold

Pega cualquiera de estas **exactamente como están escritas** para desbloquear
el set completo de métricas (Recall@3/5, MRR, precisión del router, similitud
de respuesta) junto con la respuesta:

| Tema | Pregunta |
|---|---|
| reclamaciones | ¿Cuál es la forma más fácil de reclamar cuando un vuelo que sale de España se ha retrasado? |
| seguros | recomiendame un seguro de hogar para mi nueva casa |
| yoga | Hola! ¿Podrías explicarme cuáles son los tres beneficios principales del Surya Namaskar? |
| veterinaria | hola, me puedes explicar qué usos puede tener la Gabapentina en gatos? gracias |
| astronomía | Esta noche será la oposición de Neptuno, en qué posición estará Neptuno respecto del Sol y de la Tierra? |
| gastronomía | qué comidas puedo probar en mi viaje a Japón? |
| coches | Ventajas y desventajas de los coches híbridos recargables. |
| idiomas | como debería aprender a leer y escribir japonés? |
| música | Dime las diferencias entre una corchea y una negra. |
| documentación | Hola! Quiero saber cómo puedo renovar mi DNI-e, ¿podrías ayudarme paso a paso? |
| energía | Necesito saber qué significa la 'm' en la ecuación 'E=MC²'. |
| lenguaje | ¿Cuál es el origen de la expresión dar gato por liebre? |
| turismo | necesito la tarjeta sanitaria europea para viajar a Londres y cómo se tramita? |
| estafas | ¿Cómo puedo evitar una estafa telefónica? |
| Mascotas | ¿Qué ventajas e inconvenientes tiene adoptar un gato frente a comprarlo en una tienda? |

Cualquier otra pregunta dentro de estos ~30 dominios también recibe una
respuesta con decisión del router, fragmentos recuperados, y una puntuación
de faithfulness — solo que sin las métricas de comparación con el gold.
