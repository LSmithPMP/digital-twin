# Virtual Lamonte — AI Digital Twin

> **AI Engineer Lab 6-Week Sprint · SuperDataScience**  
> Lamonte Smith

A security-hardened, RAG-powered conversational AI that responds as a digital representation of **Lamonte Smith** — dual doctoral candidate at Walsh College, Senior Software Design Release Engineer at General Motors, and AI/ML & cybersecurity researcher.

**Live Demo:** [lsmithpmp-digital-twin.hf.space](https://lsmithpmp-digital-twin.hf.space)

---

## How It Works

Biographical facts in `data/biography.txt` (231 lines, 23 sections, 188 retrievable chunks) are chunked, embedded using OpenAI `text-embedding-3-large`, and stored in a ChromaDB vector index. At runtime, user queries pass through an enhanced retrieval pipeline:

1. **Section-aware routing** — keyword detection pre-filters retrieval to the most relevant biography section
2. **Hybrid search** — vector similarity (ChromaDB) + BM25 keyword matching, merged and deduplicated
3. **Similarity-based reranking** — combined vector + keyword scores rank results by relevance
4. **Neighbor expansion** — matched chunks pull adjacent chunks from the same section for coherence
5. **Context injection** — retrieved chunks formatted with section grouping and injected as developer-role context

The LLM (OpenAI `gpt-4o` via the Responses API) generates responses grounded in retrieved facts through a Gradio `ChatInterface`.

---

## Security-by-Design Architecture

Security is embedded at every layer, not bolted on. A dedicated `security.py` module enforces defense-in-depth across all 15 identified cybersecurity risks — all resolved to **Low** residual severity through design controls:

| Control | Implementation |
|---------|---------------|
| Input Validation | 14 compiled regex patterns detect prompt injection; 2,000-char length limit |
| Input Sanitization | Control characters stripped; whitespace normalized; null bytes removed |
| Rate Limiting | 10 queries/minute per session; 5 notifications/hour per session |
| Conversation Depth | 50-turn limit prevents progressive context extraction |
| Output Filtering | 40+ architecture-disclosure terms scanned; matches trigger safe redirect |
| Context Pruning | Stale RAG injections removed after 5 turns to prevent context bloat |
| Session Isolation | Per-session state; no shared state; no disk persistence |
| Tool Governance | Strict JSON schema validation; recursion bounded to 10 calls |
| Startup Audit | Verifies secrets, .gitignore coverage, and telemetry settings at boot |

See [SECURITY.md](SECURITY.md) for full security policy and responsible disclosure.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI_GPT--4o-412991?style=flat&logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1B1B2F?style=flat&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-F97316?style=flat&logo=gradio&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat&logo=huggingface&logoColor=black)

- **Frontend:** Gradio ChatInterface
- **LLM:** OpenAI GPT-4o (Responses API)
- **Embeddings:** OpenAI text-embedding-3-large (3072 dimensions)
- **Vector Store:** ChromaDB (HNSW, cosine distance)
- **Keyword Search:** Native BM25 implementation (zero external dependencies)
- **Deployment:** Hugging Face Spaces
- **Security:** Custom `security.py` module

---

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

---

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

---

## Academic Context

| Field | Detail |
|-------|--------|
| Program | AI Engineer Lab 6-Week Sprint |
| Institution | SuperDataScience |
| Focus | RAG systems, security-by-design, conversational AI |

---

## About

**Lamonte Smith** — Dual doctoral candidate (DBA — AIML Leadership · PhD — Cybersecurity) at Walsh College · Senior Software Design Release Engineer at General Motors · 20+ years across automotive, telecom, and IT.

- **Live App:** [lsmithpmp-digital-twin.hf.space](https://lsmithpmp-digital-twin.hf.space)
- **HF Space:** [huggingface.co/spaces/LSmithPMP/digital-twin](https://huggingface.co/spaces/LSmithPMP/digital-twin)
- **GitHub:** [github.com/LSmithPMP](https://github.com/LSmithPMP)
- **LinkedIn:** [linkedin.com/in/lamonte-smith-7518b4248](https://www.linkedin.com/in/lamonte-smith-7518b4248/)

---

## License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">
<sub>Built by <a href="https://github.com/LSmithPMP">Lamonte Smith</a> · SuperDataScience AI Engineer 6-Week Sprint · 2026</sub>
</div>
