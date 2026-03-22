"""
Lamonte Smith Digital Twin — Application Entry Point

Security-first design: All user input passes through validation, sanitization,
rate limiting, and conversation depth checks before reaching the LLM. All model
output passes through disclosure filtering before rendering to the user.
"""

import logging
import os

import chromadb
import gradio as gr
from openai import OpenAI
from openai.types.responses import ResponseInputItemParam

import config
import inference
import prompts
import rag
import security
import tools

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)
for name in (__name__, 'inference', 'rag', 'tools', 'security'):
    logging.getLogger(name).setLevel(config.LOG_LEVEL)


### Environment Setup

on_hf_spaces = os.environ.get("SPACE_ID") is not None

if on_hf_spaces:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=config.HUGGINGFACE_DATASET_REPO,
        repo_type='dataset',
        local_dir=config.CHROMA_PATH.name,
        token=os.environ.get('HF_TOKEN'),
    )
else:
    from dotenv import load_dotenv
    load_dotenv()


### Startup Security Audit
startup_warnings = security.audit_startup_security()
if startup_warnings:
    for w in startup_warnings:
        logger.critical(w)

oai_client = OpenAI()
chroma_client = chromadb.PersistentClient(config.CHROMA_PATH, config.CHROMA_CLIENT_SETTINGS)
collection = chroma_client.get_collection(config.CHROMA_COLLECTION_NAME)
tool_registry = tools.build_all_tools()

# Build BM25 keyword index for hybrid search
bm25_index = rag.build_bm25_index(collection)
logger.info("BM25 index built with %d documents", len(bm25_index._docs))


### Session Callback with Security Controls

_RATE_LIMITED_MSG = ("I need a moment to catch up — you're sending messages faster than I can "
                     "process them. Give me a few seconds and try again.")
_INPUT_TOO_LONG_MSG = ("That message is a bit long for me to process effectively. "
                       "Could you shorten it and try again?")
_INJECTION_MSG = ("I appreciate the creativity, but I'm designed to answer questions about "
                  "Lamonte's background, research, and career. What can I help you with?")
_DEPTH_LIMIT_MSG = ("We've had a great conversation! For best results, I'd recommend "
                    "starting a fresh chat. I perform best in shorter sessions.")


def _prune_stale_injections(api_messages: list, max_retained: int):
    """Remove old RAG context injections from conversation history to prevent
    context window bloat. Keeps the system prompt (first developer message)
    and only the most recent N developer-role context injections.
    
    Security benefit: Limits the accumulated context an attacker could
    probe through progressive extraction over many turns."""
    # Find all developer-role messages that contain retrieval results (not the system prompt)
    injection_indices = [
        i for i, m in enumerate(api_messages)
        if isinstance(m, dict) and m.get('role') == 'developer'
        and 'retrieved_context' in m.get('content', '')
    ]
    # Remove all but the most recent max_retained injections
    if len(injection_indices) > max_retained:
        to_remove = injection_indices[:-max_retained]
        for idx in reversed(to_remove):  # reverse to preserve indices
            api_messages.pop(idx)
        logger.debug("Pruned %d stale RAG injections, kept %d",
                     len(to_remove), max_retained)


def gradio_input_callback(user_input: str,
                          gradio_history: list[gr.ChatMessage],
                          api_messages: list[ResponseInputItemParam]):
    """
    Security-hardened callback. All input passes through validation, sanitization,
    rate limiting, and depth checks before reaching the LLM pipeline.
    """

    # --- SECURITY GATE 1: Rate limiting ---
    if not security.rate_limiter.check_query_rate():
        yield [gr.ChatMessage(role="assistant", content=_RATE_LIMITED_MSG)], api_messages
        return

    # --- SECURITY GATE 2: Input validation ---
    is_valid, reason = security.validate_input(user_input)
    if not is_valid:
        if reason == "input_too_long":
            msg = _INPUT_TOO_LONG_MSG
        elif reason == "injection_detected":
            msg = _INJECTION_MSG
        else:
            msg = _INJECTION_MSG
        yield [gr.ChatMessage(role="assistant", content=msg)], api_messages
        return

    # --- SECURITY GATE 3: Input sanitization ---
    user_input = security.sanitize_input(user_input)

    # --- SECURITY GATE 4: Conversation depth check ---
    if not security.check_conversation_depth(api_messages):
        yield [gr.ChatMessage(role="assistant", content=_DEPTH_LIMIT_MSG)], api_messages
        return

    # --- SECURE PIPELINE: Build context and stream response ---
    if not api_messages:
        api_messages.append({"role": "developer", "content": prompts.SYSTEM_MESSAGE})

    # --- CONTEXT WINDOW MANAGEMENT: Prune stale RAG injections ---
    # Keep only the N most recent developer-role context injections to prevent
    # context bloat, token waste, and conflicting information across turns.
    _prune_stale_injections(api_messages, config.MAX_RETAINED_INJECTIONS)

    rag_context = rag.build_context_injection(
        oai_client, collection, user_input, bm25_index=bm25_index
    )
    api_messages.append({"role": "developer", "content": rag_context})
    api_messages.append({"role": "user", "content": user_input})

    # Secure debug logging — never log full prompts or secrets
    logger.debug("Processing query (%d chars, %d messages in history)",
                 len(user_input), len(api_messages))

    yield from inference.stream_turn(oai_client, api_messages, tool_registry)


### Gradio UI

greeting: gr.MessageDict = {
    "role": "assistant", "content": "Hey there! \U0001f44b I'm Virtual Lamonte. "
    "Ask me about my AI/ML research, my work at AT&T or GM, my doctoral journey at Walsh College, "
    "or anything about Agentic AI, cybersecurity, wireless infrastructure, or autonomous vehicles. "
    "How can I help?"
}

_avatar_path = config.BASE_DIR / 'assets' / 'avatar.png'
_favicon_path = config.BASE_DIR / 'assets' / 'favicon.ico'

chatbot = gr.Chatbot(
    [greeting],
    type='messages',
    show_label=False,
    avatar_images=(None, str(_avatar_path) if _avatar_path.exists() else None),
    scale=1,
)

api_messages = gr.State([])

demo = gr.ChatInterface(
    fn=gradio_input_callback,
    chatbot=chatbot,
    additional_inputs=[api_messages],
    additional_outputs=[api_messages],
    additional_inputs_accordion=gr.Accordion(visible=False),
    title='Virtual Lamonte',
    fill_height=True,
    fill_width=False,
)

custom_css = (
    ".main { max-width: 800px !important; margin: auto !important; }\n"
    "h1 { text-align: left !important; }\n"
    ".avatar-container { width: 50px !important; height: 50px !important; }\n"
    ".avatar-container img { padding: 0 !important; }\n"
    ".role { align-self: center !important; }\n"
    ".message-buttons-left { display: none !important; }\n"
    ".thought-group { width: fit-content !important; padding-right: var(--spacing-xxl) !important}\n"
    "footer { height: 5px !important; visibility: hidden !important; }\n"
)

if __name__ == "__main__":
    demo.launch()
