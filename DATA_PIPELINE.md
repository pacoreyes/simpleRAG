dataset: RagQuAS (Retrieval-Augmented Generation and Question-Answering in Spanish) was built by the Instituto de Ingeniería del Conocimiento specifically to evaluate a complete RAG system, not to train one. It's small and dense: 201 rows, all in Spanish, hand-written and reviewed by two computational linguists, spanning ~30 domains (insurance, first aid, recipes, astronomy, customer service, travel claims, veterinary, yoga, etc.).

the dataset is located in data_volume/assets/test-00000-of-00001.parquet

Goal: building a simple RAG system
Multi-domain open QA assistant (my recommendation). Index all text_i documents into one vector store and let the user ask anything. This shows off open-domain retrieval and is the most impressive in a demo because questions jump across unrelated topics. The differentiator: since you have context_i and gold answer, you can put real numbers on screen — retrieval recall@k (did the retrieved chunk overlap the gold context?) and answer faithfulness against the gold answer. Most RAG demos can't show that.

Metadata use:
The one genuinely strong native field is topic. It's low-cardinality (~30 values), clean, and human-assigned — exactly the shape Pinecone filtering likes (filter={"topic": {"$eq": "seguros"}} or {"$in": [...]}). Filtering to the right domain prevents the classic failure where a query about insurance retrieves a lexically similar chunk from "reclamaciones."
The link_i fields are weak as filter metadata but worth carrying as display/provenance metadata, and you can cheaply derive a source_domain from them (parse the URL host) if you ever want to filter or boost by source.
Everything else — question, variant, answer, context_i — are query/eval fields, not document metadata. You don't filter on them, but context_i becomes useful for evaluation (more below).

The catch: filling the filter at query time
An LLM/classifier router that maps the query to one of the ~30 topics before retrieval (a "self-query" retriever does exactly this — the LLM extracts a filter from natural language). More impressive, slightly more brittle.

Deterministic, cheap, do these regardless:

doc_id / parent_doc_id / chunk_id and chunk_index — stable IDs linking each chunk back to its source text_i. This is what enables small-to-big / parent-document retrieval (retrieve a precise chunk, then feed the LLM the fuller parent text), deduplication across variant rows, and citation. This single addition usually does more for answer quality than any fancy field.
source_domain, char_length / token_count — for source boosting and for filtering out junk-sized chunks.

Metadata that makes evaluation work
Because this dataset ships gold contexts and answers, store a stable doc_id and a flag for which chunks correspond to each question's gold context_i. Then after a Pinecone query you can check whether the retrieved chunk IDs include the gold one and compute recall@k directly — turning your demo into a measurable RAG, which is the dataset's original purpose.

Pinecone practicalities
Keep metadata flat — Pinecone accepts only strings, numbers, booleans, and lists of strings, no nested objects. Only put low-cardinality fields in your filters (topic, subtopic, source_domain); store the chunk text and links as metadata for retrieval/display but don't treat them as filters. Respect the ~40 KB-per-vector metadata limit — fine for chunk text, but don't stuff an entire long document into every chunk's metadata.

A concrete minimal schema for your demo

{
  "text": "the chunk content (for display + reranking)",
  "topic": "seguros",
  "subtopic": "indemnizacion-vuelos",
  "doc_id": "doc_0042",
  "parent_doc_id": "doc_0042",
  "chunk_index": 1,
  "source_domain": "airhelp.com",
  "link": "https://www.airhelp.com/...",
  "generated_questions": ["¿Cómo reclamo un vuelo retrasado?", "..."],
  "keywords": ["vuelo", "retraso", "indemnización", "AirHelp"],
  "is_gold_for": ["q_0042"]
}
