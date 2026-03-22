"""
Security module for the Digital Twin application.
Implements defense-in-depth controls: input validation, output filtering,
rate limiting, and content safety checks.

Security design principle: All controls are enforced at the application layer,
independent of LLM behavior or platform-level protections.
"""

import logging
import re
import time
from collections import defaultdict

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

# Maximum characters allowed per user message
MAX_INPUT_LENGTH = 2000

# Maximum conversation turns before session reset is recommended
MAX_CONVERSATION_TURNS = 50

# Patterns commonly used in prompt injection attempts
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"you\s+are\s+now\s+(a|an|the)\s+",
    r"system\s*prompt",
    r"reveal\s+(your|the)\s+(instructions|prompt|rules|system)",
    r"act\s+as\s+(a|an|if)\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"forget\s+(everything|all|your\s+instructions)",
    r"override\s+(your|the|all)\s+",
    r"<\s*/?\s*system\s*>",
    r"<\s*/?\s*developer\s*>",
    r"\[\s*INST\s*\]",
    r"\[\s*/\s*INST\s*\]",
]
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def validate_input(user_input: str) -> tuple[bool, str]:
    """
    Validate and sanitize user input before processing.

    Returns:
        (is_valid, message): If invalid, message contains the reason.
    """
    # Length check
    if not user_input or not user_input.strip():
        return False, "empty_input"

    if len(user_input) > MAX_INPUT_LENGTH:
        return False, "input_too_long"

    # Prompt injection pattern detection
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(user_input):
            logger.warning("Prompt injection pattern detected in user input")
            return False, "injection_detected"

    return True, "valid"


def sanitize_input(user_input: str) -> str:
    """
    Sanitize user input by stripping control characters and normalizing whitespace.
    Applied after validation passes.
    """
    # Remove null bytes and control characters (keep newlines and tabs)
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', user_input)
    # Normalize excessive whitespace
    sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)
    return sanitized.strip()


# ---------------------------------------------------------------------------
# Output Filtering
# ---------------------------------------------------------------------------

# Phrases that should never appear in model output (architecture disclosure)
_OUTPUT_BLOCKLIST = [
    "system prompt",
    "system message",
    "developer role",
    "retrieved_context",
    "chromadb",
    "chroma_path",
    "biography.txt",
    "build_vectors",
    "rag.py",
    "inference.py",
    "prompts.py",
    "config.py",
    "tools.py",
    "app.py",
    "security.py",
    "openai_api_key",
    "pushover_user",
    "pushover_token",
    "hf_token",
    ".env",
    "gr.state",
    "api_messages",
    "tool_registry",
    "distance_threshold",
    "n_results",
    "bm25_index",
    "bm25index",
    "bm25 index",
    "section_filter",
    "neighbor_window",
    "neighbor expansion",
    "global_idx",
    "context_injection",
    "chunk source=",
    "section name=",
    "_prune_stale",
    "max_context_chunks",
    "max_retained_injections",
    "query routing",
    "hybrid search",
    "rerank",
]


def filter_output(text: str) -> str:
    """
    Scan model output for inadvertent disclosure of system architecture,
    internal file names, or secret references. Replace flagged content
    with a safe in-character response.
    """
    text_lower = text.lower()
    for phrase in _OUTPUT_BLOCKLIST:
        if phrase in text_lower:
            logger.warning("Output filter triggered on phrase: %s", phrase)
            return ("I appreciate the question, but I'm not able to share details about "
                    "my internal architecture. Feel free to ask me about my research, "
                    "career, or professional interests instead.")
    return text


# ---------------------------------------------------------------------------
# Rate Limiting (Per-Session)
# ---------------------------------------------------------------------------

class SessionRateLimiter:
    """
    Token-bucket rate limiter scoped to individual sessions.
    Prevents query flooding and tool abuse at the application layer.
    """

    def __init__(self, max_queries_per_minute: int = 10, max_notifications_per_hour: int = 5):
        self._query_timestamps: defaultdict[str, list[float]] = defaultdict(list)
        self._notification_timestamps: defaultdict[str, list[float]] = defaultdict(list)
        self._max_qpm = max_queries_per_minute
        self._max_nph = max_notifications_per_hour

    def check_query_rate(self, session_id: str = "default") -> bool:
        """Return True if query is within rate limit."""
        now = time.time()
        window = [t for t in self._query_timestamps[session_id] if now - t < 60]
        self._query_timestamps[session_id] = window
        if len(window) >= self._max_qpm:
            logger.warning("Query rate limit exceeded for session %s", session_id[:8])
            return False
        self._query_timestamps[session_id].append(now)
        return True

    def check_notification_rate(self, session_id: str = "default") -> bool:
        """Return True if notification is within rate limit."""
        now = time.time()
        window = [t for t in self._notification_timestamps[session_id] if now - t < 3600]
        self._notification_timestamps[session_id] = window
        if len(window) >= self._max_nph:
            logger.warning("Notification rate limit exceeded for session %s", session_id[:8])
            return False
        self._notification_timestamps[session_id].append(now)
        return True


# Singleton rate limiter instance
rate_limiter = SessionRateLimiter()


# ---------------------------------------------------------------------------
# Conversation Depth Guard
# ---------------------------------------------------------------------------

def check_conversation_depth(api_messages: list) -> bool:
    """
    Check if conversation has exceeded the maximum safe depth.
    Deep conversations increase context window exposure and extraction risk.
    Returns True if within safe limits.
    """
    user_msg_count = sum(
        1 for m in api_messages
        if isinstance(m, dict) and m.get('role') == 'user'
    )
    if user_msg_count >= MAX_CONVERSATION_TURNS:
        logger.info("Conversation depth limit reached (%d turns)", user_msg_count)
        return False
    return True


# ---------------------------------------------------------------------------
# Startup Security Audit
# ---------------------------------------------------------------------------

def audit_startup_security() -> list[str]:
    """
    Run security checks at application startup.
    Returns a list of warnings (empty list = all clear).
    """
    import os
    warnings = []

    # Check that required secrets are present
    if not os.environ.get("OPENAI_API_KEY"):
        warnings.append("CRITICAL: OPENAI_API_KEY not set in environment")

    # Check that .env is not accidentally committed
    env_path = config.BASE_DIR / '.env'
    gitignore_path = config.BASE_DIR / '.gitignore'
    if env_path.exists() and gitignore_path.exists():
        gitignore_content = gitignore_path.read_text()
        if '.env' not in gitignore_content:
            warnings.append("WARNING: .env exists but is not in .gitignore")

    # Check ChromaDB telemetry is disabled
    if config.CHROMA_CLIENT_SETTINGS.anonymized_telemetry is not False:
        warnings.append("WARNING: ChromaDB telemetry is not disabled")

    for w in warnings:
        logger.warning("Startup security audit: %s", w)

    return warnings
