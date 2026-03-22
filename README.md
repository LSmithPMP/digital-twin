# Virtual Lamonte — AI Digital Twin

A security-hardened, RAG-powered conversational AI that responds as a digital representation of **Lamonte Smith** — doctoral candidate, Senior Software Design Release Engineer at General Motors, and AI/ML & cybersecurity researcher.

**Live Demo:** [lsmithpmp-digital-twin.hf.space](https://lsmithpmp-digital-twin.hf.space)

## How It Works

Biographical facts in `data/biography.txt` (231 lines, 23 sections, 188 retrievable chunks) are chunked, embedded using OpenAI `text-embedding-3-large`, and stored in a ChromaDB vector index. At runtime, user queries pass through an enhanced retrieval pipeline:

1. **Section-aware routing** — keyword detection pre-filters retrieval to the most relevant biography section
2. **Hybrid search** — vector similarity (ChromaDB) + BM25 keyword matching, merged and deduplicated
3. **Similarity-based reranking** — combined vector + keyword scores rank results by relevance
4. **Neighbor expansion** — matched chunks pull adjacent chunks from the same section for coherence
5. **Context injection** — retrieved chunks formatted with section grouping and injected as developer-role context

The LLM (OpenAI `gpt-4o` via the Responses API) generates responses grounded in retrieved facts through a Gradio `ChatInterface`.

## Security-by-Design Architecture

Security is embedded at every layer, not bolted on. A dedicated `security.py` module enforces defense-in-depth:

- **Input Validation** — 14 compiled regex patterns detect prompt injection; 2,000-char length limit
- **Input Sanitization** — Control characters stripped; whitespace normalized; null bytes removed
- **Rate Limiting** — 10 queries/minute per session; 5 notifications/hour per session
- **Conversation Depth** — 50-turn limit prevents progressive context extraction
- **Output Filtering** — 40+ architecture-disclosure terms scanned; matches trigger safe redirect
- **Context Pruning** — Stale RAG injections removed after 5 turns to prevent context bloat
- **Session Isolation** — Per-session state; no shared state; no disk persistence
- **Tool Governance** — Strict JSON schema validation; recursion bounded to 10 calls
- **Startup Audit** — Verifies secrets, .gitignore coverage, and telemetry settings at boot

All 15 identified cybersecurity risks resolve to **Low** residual severity through design controls.

## Tech Stack

- **Frontend:** Gradio ChatInterface
- **LLM:** OpenAI GPT-4o (Responses API)
- **Embeddings:** OpenAI text-embedding-3-large (3072 dimensions)
- **Vector Store:** ChromaDB (HNSW, cosine distance)
- **Keyword Search:** Native BM25 implementation (zero external dependencies)
- **Deployment:** Hugging Face Spaces
- **Security:** Custom security.py module

## Project Structure
```
app.py              — Gradio entry point with 4 security gates
config.py           — Configuration (models, thresholds, security params)
inference.py        — LLM streaming with tool execution and output filtering
rag.py              — Enhanced RAG: hybrid search, reranking, neighbor expansion
security.py         — Input validation, output filtering, rate limiting, audit
tools.py            — Tool registry and implementations
prompts.py          — System message with anti-fabrication rules
data/               — biography.txt (231 lines, 23 sections)
assets/             — Avatar image
scripts/            — build_vectors.py for offline vector store generation
```

## Running Locally
```bash
git clone https://github.com/LSmithPMP/digital-twin.git
cd digital-twin
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
python scripts/build_vectors.py
python app.py
```

## About

**Lamonte Smith** — Doctoral candidate (AI/ML & Cybersecurity) at Walsh College | Senior Software Design Release Engineer at General Motors | 20+ years across automotive, telecom, and IT.

*AI/ML-powered intelligence where the machines that run the world meet the threats that break them.*

- **Live App:** [lsmithpmp-digital-twin.hf.space](https://lsmithpmp-digital-twin.hf.space)
- **GitHub:** [github.com/LSmithPMP/digital-twin](https://github.com/LSmithPMP/digital-twin)
- **HF Space:** [huggingface.co/spaces/LSmithPMP/digital-twin](https://huggingface.co/spaces/LSmithPMP/digital-twin)
- **LinkedIn:** [linkedin.com/in/lamonte-smith-7518b4248](https://www.linkedin.com/in/lamonte-smith-7518b4248/)
- **X:** [x.com/LSmithPMP](https://x.com/LSmithPMP)

Built for the **SuperDataScience AI Engineer Lab 5-Week Sprint**.

## License

MIT
