import re
import unicodedata
from typing import Iterable, Tuple

PROMPT_INJECTION_PATTERNS = [
    re.compile(
        r"ignore\s+(all\s+)?(the\s+)?previous\s+(prompt|prompts|instruction|instructions|rules|context)",
        re.IGNORECASE,
    ),
    re.compile(
        r"forget\s+(all\s+)?(the\s+)?previous\s+(prompt|prompts|instruction|instructions|rules)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(disregard|override)\s+(all\s+)?(the\s+)?previous\s+(prompt|prompts|instruction|instructions|rules)",
        re.IGNORECASE,
    ),
    re.compile(r"only\s+use\s+(this|the)\s+(new\s+)?prompt", re.IGNORECASE),
    re.compile(r"reset\s+(all\s+)?(your\s+)?instructions", re.IGNORECASE),
    re.compile(
        r"ignore\s+(all\s+)?system\s+(prompt|prompts|instruction|instructions)",
        re.IGNORECASE,
    ),
    re.compile(r"abaikan\s+(seluruh\s+)?instruksi\s+sebelumnya", re.IGNORECASE),
    re.compile(r"lupakan\s+(seluruh\s+)?instruksi\s+sebelumnya", re.IGNORECASE),
    re.compile(r"gunakan\s+(hanya|cuma)\s+(prompt|perintah)\s+baru", re.IGNORECASE),
]

PROMPT_INJECTION_KEYWORD_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("ignore", "previous", "instructions"),
    ("ignore", "system", "prompt"),
    ("only", "use", "prompt"),
    ("reset", "your", "instructions"),
    ("forget", "all", "instructions"),
    ("abaikan", "instruksi", "sebelumnya"),
    ("lupakan", "instruksi", "sebelumnya"),
    ("hanya", "gunakan", "perintah"),
    ("gunakan", "perintah", "baru"),
)


def normalize_guardrail_text(text: str) -> str:
    """Lowercase text, strip diacritics, and keep only alphanumerics for guardrail checks."""
    normalized = unicodedata.normalize("NFKD", text or "")
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii", "ignore")
    lowered = ascii_only.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _contains_keywords(normalized_text: str, keywords: Iterable[str]) -> bool:
    return all(keyword in normalized_text for keyword in keywords)


def contains_prompt_injection_attempt(text: str) -> bool:
    """Detect prompt-injection attempts, even if the attacker uses another language."""
    if not text:
        return False
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    normalized = normalize_guardrail_text(text)
    if not normalized:
        return False
    return any(
        _contains_keywords(normalized, keywords)
        for keywords in PROMPT_INJECTION_KEYWORD_GROUPS
    )


def prompt_injection_response() -> str:
    return (
        "Maaf, saya tetap mengikuti pedoman resmi Optimaxx dan tidak dapat mengabaikan instruksi yang berlaku. "
        "Silakan ajukan pertanyaan terkait layanan kami agar saya bisa membantu. \n"
        "📧 Email: info@optimaxx.id\n🌐 Website: https://optimaxx.id"
    )
