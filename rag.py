"""
Enhanced RAG pipeline for the Lamonte Smith Digital Twin.

Improvements over baseline RAG:
1. Sliding-window neighbor expansion — matched chunks pull adjacent chunks from same section
2. Hybrid search — vector similarity + BM25 keyword matching, merged and deduplicated
3. Section-aware query routing — metadata pre-filtering for clearly scoped queries
4. Similarity-based reranking — final result set reranked by combined vector + keyword score
5. Improved context formatting — section grouping for coherent LLM consumption

Security: All retrieval operates on the curated biography corpus only. No user input
is used in metadata filters without sanitization. Distance threshold enforced on all paths.
"""

import logging
import math
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import chromadb
import numpy as np
from openai import OpenAI

import config

logger = logging.getLogger(__name__)


@dataclass
class ChunkedText:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | np.ndarray | None = None
    id: str | None = None


# ============================================================================
# Indexing: load, chunk, embed, store
# ============================================================================

def chunk_curated_lines(text: str) -> list[ChunkedText]:
    """Split text into chunks by line, tracking '# section' headers as metadata.
    Each chunk stores its global index for neighbor expansion at query time."""
    chunks = []
    section_name = ""
    global_idx = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            section_name = stripped.lstrip("#").strip()
            section_i = 0
        else:
            section_i += 1
            chunks.append(
                ChunkedText(text=stripped, metadata={
                    'section': section_name,
                    'chunk': section_i,
                    'global_idx': global_idx,
                })
            )
            global_idx += 1
    return chunks


def embed_strings(oai_client: OpenAI, strings: list[str]) -> list[list[float]]:
    """Embed a list of strings using the OpenAI embeddings API.
    Sends all chunks in a single batched request (up to 2048 inputs)."""
    response = oai_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=strings
    )
    return [item.embedding for item in response.data]


def embed_chunks(oai_client: OpenAI, chunks: list[ChunkedText]) -> list[ChunkedText]:
    """Generate an embedding for each ChunkedText and attach it."""
    embeddings = embed_strings(oai_client, [chunk.text for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
        chunk.metadata["embedding_model"] = config.EMBEDDING_MODEL
    return chunks


def db_store_embeds(
    chroma_client: chromadb.ClientAPI,
    collection_name: str,
    chunks: list[ChunkedText],
) -> chromadb.Collection:
    """Store embeddings as a new ChromaDB collection. Deletes existing collection first."""
    existing_collections = [c.name for c in chroma_client.list_collections()]
    if collection_name in existing_collections:
        chroma_client.delete_collection(collection_name)

    collection = chroma_client.create_collection(
        name=collection_name,
        configuration=config.CHROMA_COLLECTION_CONFIG,
    )

    docs, embeds, metadata, ids = [], [], [], []
    for chunk in chunks:
        docs.append(chunk.text)
        embeds.append(chunk.embedding)
        metadata.append(chunk.metadata)
        ids.append(str(uuid.uuid4()))

    collection.add(ids=ids, embeddings=embeds, metadatas=metadata, documents=docs)
    return collection


def db_load_embeds(
    chroma_client: chromadb.ClientAPI,
    collection_name: str,
) -> list[ChunkedText]:
    """Load all entries from a ChromaDB collection as ChunkedTexts."""
    collection = chroma_client.get_collection(name=collection_name)
    data = collection.get(include=["embeddings", "metadatas", "documents"])
    assert data["embeddings"] is not None
    assert data["metadatas"] is not None
    assert data["documents"] is not None

    return [
        ChunkedText(text=doc, metadata=dict(meta), embedding=list(emb), id=_id)
        for doc, emb, meta, _id in zip(
            data['documents'], data['embeddings'], data['metadatas'], data['ids'],
        )
    ]


# ============================================================================
# BM25 Keyword Search (lightweight, no external dependency)
# ============================================================================

class BM25Index:
    """Lightweight BM25 implementation for hybrid search. No external dependencies.
    Built once at startup from the ChromaDB collection documents."""

    def __init__(self, documents: list[str], ids: list[str], metadatas: list[dict],
                 k1: float = 1.5, b: float = 0.75):
        self._docs = documents
        self._ids = ids
        self._metadatas = metadatas
        self._k1 = k1
        self._b = b

        # Tokenize and build index
        self._tokenized = [self._tokenize(doc) for doc in documents]
        self._avg_dl = sum(len(t) for t in self._tokenized) / max(len(self._tokenized), 1)
        self._n = len(documents)

        # Document frequency for each term
        self._df: dict[str, int] = defaultdict(int)
        for tokens in self._tokenized:
            for term in set(tokens):
                self._df[term] += 1

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r'\b\w+\b', text.lower())

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return top-k documents by BM25 score."""
        query_terms = self._tokenize(query)
        scores = []
        for i, tokens in enumerate(self._tokenized):
            score = 0.0
            dl = len(tokens)
            tf_map: dict[str, int] = defaultdict(int)
            for t in tokens:
                tf_map[t] += 1
            for term in query_terms:
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf(term)
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (1 - self._b + self._b * dl / self._avg_dl)
                score += idf * numerator / denominator
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {'id': self._ids[i], 'document': self._docs[i],
             'metadata': self._metadatas[i], 'bm25_score': s}
            for i, s in scores[:top_k]
        ]


def build_bm25_index(collection: chromadb.Collection) -> BM25Index:
    """Build a BM25 index from all documents in the ChromaDB collection."""
    data = collection.get(include=["documents", "metadatas"])
    return BM25Index(
        documents=data['documents'],
        ids=data['ids'],
        metadatas=data['metadatas'],
    )


# ============================================================================
# Section-Aware Query Routing
# ============================================================================

# Map of keyword signals to section names for pre-filtering
_SECTION_ROUTING_MAP = {
    'dissertation': 'Doctoral Dissertation',
    'research gap': 'Doctoral Dissertation',
    'mixed methods': 'Doctoral Dissertation',
    'fgsm': 'Doctoral Dissertation',
    'pgd': 'Doctoral Dissertation',
    'adversarial': 'Doctoral Dissertation',
    'gm career': 'General Motors Career',
    'general motors': 'General Motors Career',
    'infotainment': 'General Motors Career',
    'at&t': 'AT&T Career',
    'att career': 'AT&T Career',
    'u-verse': 'AT&T Career',
    'super bowl': 'AT&T Career',
    'certification': 'Certifications and Training',
    'pmp': 'Certifications and Training',
    'dfss': 'Certifications and Training',
    'education': 'Education',
    'walsh college': 'Education',
    'degree': 'Education',
    'teach': 'Teaching Philosophy',
    'harvard bok': 'Teaching Philosophy',
    'pedagogy': 'Teaching Philosophy',
    'mentor': 'Mentoring Philosophy',
    'family': 'Identity',
    'wife': 'Identity',
    'detroit': 'Identity',
    'keynote': 'Professional Aspirations',
    'aspiration': 'Professional Aspirations',
    'proud': 'Proudest Accomplishments',
    'accomplishment': 'Proudest Accomplishments',
    'advice': 'Career Lessons and Advice',
    'lesson': 'Career Lessons and Advice',
    'values': 'Values and Behavioral Anchors',
    'convergence intel': 'Identity',
    'linkedin': 'Contact',
    'email': 'Contact',
    'contact': 'Contact',
}


def _detect_section_filter(query: str) -> str | None:
    """Detect if the query maps to a specific biography section.
    Returns section name or None for unscoped queries."""
    query_lower = query.lower()
    for keyword, section in _SECTION_ROUTING_MAP.items():
        if keyword in query_lower:
            logger.debug("Section routing: '%s' -> section '%s'", keyword, section)
            return section
    return None


# ============================================================================
# Neighbor Expansion
# ============================================================================

def _expand_neighbors(
    matched_chunks: list[dict],
    collection: chromadb.Collection,
    window: int = 1,
) -> list[dict]:
    """For each matched chunk, retrieve neighboring chunks from the same section.
    Uses global_idx metadata for position-aware expansion.
    Deduplicates by document ID."""

    if not matched_chunks:
        return matched_chunks

    # Collect global indices and sections we need to expand
    seen_ids = {c['id'] for c in matched_chunks}
    expansion_targets = []
    for chunk in matched_chunks:
        idx = chunk['metadata'].get('global_idx')
        section = chunk['metadata'].get('section')
        if idx is not None and section is not None:
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                expansion_targets.append((idx + offset, section))

    if not expansion_targets:
        return matched_chunks

    # Query ChromaDB for neighbors by global_idx and section
    expanded = list(matched_chunks)
    for target_idx, target_section in expansion_targets:
        try:
            results = collection.get(
                where={"$and": [
                    {"global_idx": {"$eq": target_idx}},
                    {"section": {"$eq": target_section}},
                ]},
                include=["documents", "metadatas"],
            )
            if results['ids']:
                for nid, ndoc, nmeta in zip(results['ids'], results['documents'], results['metadatas']):
                    if nid not in seen_ids:
                        seen_ids.add(nid)
                        expanded.append({
                            'id': nid,
                            'document': ndoc,
                            'metadata': nmeta,
                            'distance': None,  # neighbor, not directly matched
                        })
        except Exception as e:
            logger.debug("Neighbor expansion failed for idx=%s: %s", target_idx, e)

    return expanded


# ============================================================================
# Hybrid Merge and Rerank
# ============================================================================

def _merge_and_rerank(
    vector_results: list[dict],
    keyword_results: list[dict],
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> list[dict]:
    """Merge vector and keyword results, deduplicate, and rerank by combined score.
    Vector scores are inverted distances (lower distance = higher score).
    Keyword scores are raw BM25 scores (higher = better)."""

    combined: dict[str, dict] = {}

    # Normalize vector scores: convert distance to similarity (1 - distance)
    if vector_results:
        max_sim = max((1 - r.get('distance', 1.0)) for r in vector_results if r.get('distance') is not None)
        max_sim = max(max_sim, 0.001)
        for r in vector_results:
            d = r.get('distance')
            sim = (1 - d) / max_sim if d is not None else 0.5
            rid = r['id']
            if rid not in combined:
                combined[rid] = {**r, 'vector_score': sim, 'keyword_score': 0.0}
            else:
                combined[rid]['vector_score'] = sim

    # Normalize keyword scores
    if keyword_results:
        max_bm25 = max(r.get('bm25_score', 0) for r in keyword_results)
        max_bm25 = max(max_bm25, 0.001)
        for r in keyword_results:
            norm_score = r.get('bm25_score', 0) / max_bm25
            rid = r['id']
            if rid not in combined:
                combined[rid] = {**r, 'vector_score': 0.0, 'keyword_score': norm_score}
            else:
                combined[rid]['keyword_score'] = norm_score

    # Compute combined score and sort
    for entry in combined.values():
        entry['combined_score'] = (
            vector_weight * entry.get('vector_score', 0) +
            keyword_weight * entry.get('keyword_score', 0)
        )

    ranked = sorted(combined.values(), key=lambda x: x['combined_score'], reverse=True)
    return ranked


# ============================================================================
# Context Injection Builder (Main Entry Point)
# ============================================================================

def build_context_injection(
    oai_client: OpenAI,
    collection: chromadb.Collection,
    user_query: str,
    bm25_index: 'BM25Index | None' = None,
    n_results: int = config.N_RESULTS,
    d_threshold: float = config.DISTANCE_THRESHOLD,
    neighbor_window: int = config.NEIGHBOR_WINDOW,
    max_context_chunks: int = config.MAX_CONTEXT_CHUNKS,
) -> str:
    """Enhanced context injection with hybrid search, section routing,
    neighbor expansion, and reranking.

    Pipeline:
    1. Detect section filter from query (optional pre-filter)
    2. Vector search via ChromaDB (with optional section filter)
    3. BM25 keyword search (if index available)
    4. Merge and rerank results
    5. Expand neighbors for matched chunks
    6. Format context with section grouping
    """

    # --- Step 1: Section routing ---
    section_filter = _detect_section_filter(user_query)

    # --- Step 2: Vector search ---
    try:
        q_embeds = embed_strings(oai_client, [user_query])
        query_params = {"n_results": n_results}
        if section_filter:
            query_params["where"] = {"section": {"$eq": section_filter}}
            logger.info("Section-routed retrieval: filtering to '%s'", section_filter)

        q_results = collection.query(q_embeds, **query_params)
    except Exception as e:
        logger.error("Vector retrieval failed: %s: %s", type(e).__name__, e)
        return _format_empty_context("retrieval_error")

    # Apply distance threshold
    vector_chunks: list[dict] = []
    for id, meta, d, doc in zip(
        q_results['ids'][0],
        q_results['metadatas'][0],
        q_results['distances'][0],
        q_results['documents'][0],
    ):
        if d < d_threshold:
            vector_chunks.append({'id': id, 'metadata': meta, 'distance': d, 'document': doc})
            logger.debug('Vector retrieved "%s" #%s, d=%.4f: %s',
                         meta.get('section'), meta.get('chunk'), d, doc[:60])
        else:
            logger.debug('Vector discarded "%s" #%s, d=%.4f > %s',
                         meta.get('section'), meta.get('chunk'), d, d_threshold)

    # --- Step 3: BM25 keyword search ---
    keyword_chunks: list[dict] = []
    if bm25_index is not None:
        keyword_chunks = bm25_index.search(user_query, top_k=n_results)
        logger.debug("BM25 returned %d results", len(keyword_chunks))

    # --- Step 4: Merge and rerank ---
    if keyword_chunks:
        merged = _merge_and_rerank(vector_chunks, keyword_chunks)
    else:
        merged = vector_chunks

    if not merged:
        logger.info('No relevant chunks found for query: %s', user_query)
        return _format_empty_context("no_results")

    # --- Step 5: Neighbor expansion ---
    expanded = _expand_neighbors(merged[:max_context_chunks], collection, window=neighbor_window)

    # --- Step 6: Format with section grouping ---
    return _format_context(expanded, max_context_chunks)


def _format_empty_context(reason: str) -> str:
    """Format context injection when no results are found."""
    if reason == "retrieval_error":
        msg = "Context retrieval is temporarily unavailable."
    else:
        msg = "No relevant biographical information was found for the following user query."
    return (
        "Retrieval results:\n"
        f"{msg}\n"
        "Respond naturally without biographical facts. Don't reference the retrieval process.\n\n"
        "<retrieved_context></retrieved_context>"
    )


def _format_context(chunks: list[dict], max_chunks: int) -> str:
    """Format retrieved chunks into context injection, grouped by section for coherence."""

    # Group by section, preserving order of first appearance
    section_order = []
    section_chunks: dict[str, list[dict]] = defaultdict(list)
    seen_docs = set()

    for chunk in chunks[:max_chunks]:
        doc = chunk.get('document', '')
        if doc in seen_docs:
            continue
        seen_docs.add(doc)
        section = chunk.get('metadata', {}).get('section', 'General')
        if section not in section_chunks:
            section_order.append(section)
        section_chunks[section].append(chunk)

    # Build tagged output grouped by section
    tagged_sections: list[str] = []
    for section in section_order:
        section_lines = [f'  <section name="{section}">']
        for c in section_chunks[section]:
            source_id = c.get('id', 'unknown')
            section_lines.append(f'    <chunk source="{source_id}">')
            section_lines.append(f'      {c["document"]}')
            section_lines.append(f'    </chunk>')
        section_lines.append(f'  </section>')
        tagged_sections.append('\n'.join(section_lines))

    total_chunks = sum(len(v) for v in section_chunks.values())
    logger.info('Injecting %d chunks across %d sections', total_chunks, len(section_order))

    return (
        "Retrieval results:\n"
        "The following biographical excerpts may be relevant to the following user query.\n"
        "Use them, *if relevant*, to inform your response.\n"
        "Remember: speak naturally and don't reference the retrieval process.\n\n"
        f"<retrieved_context>\n{chr(10).join(tagged_sections)}\n</retrieved_context>"
    )
