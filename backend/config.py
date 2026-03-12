"""
config.py — Environment variables, constants, domain system prompts,
and the provider/model catalogue.

All other modules import from here; nothing imports from them in return,
so there are no circular dependencies.
"""

import os
import logging
from typing import Dict, Any

from dotenv import load_dotenv
import urllib3

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Environment / API keys ────────────────────────────────────────────────────

load_dotenv()

PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_ENV        = os.getenv("PINECONE_ENV", "us-east-1")
AWS_ACCESS_KEY      = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY      = os.getenv("AWS_SECRET_ACCESS_KEY")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

USER_DB_PATH = os.getenv("USER_DB_PATH", "users.db")
UPLOAD_DIR   = "temp_uploads"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is required")

# ── Pinecone index names ──────────────────────────────────────────────────────

PINECONE_INDEX_TEXT  = "pdf-query-index"
PINECONE_INDEX_IMAGE = "pdf-query-index-image"
PINECONE_INDEX_AUDIO = "pdf-query-index-audio"
PINECONE_INDEX_VIDEO = "pdf-query-index-video"

INDEX_CT_MAP: Dict[str, str] = {
    PINECONE_INDEX_IMAGE: "image",
    PINECONE_INDEX_AUDIO: "audio",
    PINECONE_INDEX_VIDEO: "video",
    PINECONE_INDEX_TEXT:  "text",
}

# ── Domain system prompts ─────────────────────────────────────────────────────
# Each domain injects expert context into the LLM's system role.
# The same gpt-4o / claude / gemini becomes a domain specialist without
# loading a separate model.

DOMAIN_SYSTEM_PROMPTS: Dict[str, str] = {
    "General": (
        "You are a knowledgeable and helpful AI assistant. "
        "Answer questions clearly, accurately and concisely based on the provided context. "
        "If the context does not contain the answer, say so honestly."
    ),
    "Legal": (
        "You are an expert legal analyst with deep knowledge of contract law, "
        "regulatory compliance, case law, and legal documentation. "
        "When analysing legal text, highlight obligations, rights, risks, and "
        "potential issues. Cite relevant clauses or precedents where applicable. "
        "Always remind the user to consult a qualified attorney for formal advice. "
        "Answer based strictly on the provided document context."
    ),
    "Medical": (
        "You are a clinical knowledge assistant with expertise in medicine, "
        "pharmacology, diagnostics, and patient care documentation. "
        "Interpret medical terminology, lab values, and clinical notes accurately. "
        "Always note that this is informational and not a substitute for professional "
        "medical advice. Answer based on the provided document context."
    ),
    "Finance": (
        "You are an expert financial analyst with deep knowledge of accounting, "
        "investment analysis, financial modelling, SEC filings, and market dynamics. "
        "Interpret financial statements, ratios, and market data precisely. "
        "Highlight key financial risks and opportunities. "
        "Remind users to seek professional financial advice. "
        "Answer based on the provided document context."
    ),
    "Code": (
        "You are a senior software engineer and code reviewer with expertise in "
        "multiple programming languages, design patterns, algorithms, and best "
        "practices. When analysing code or technical documents, identify bugs, "
        "suggest improvements, explain logic clearly, and provide working examples. "
        "Answer based on the provided document context."
    ),
    "Science": (
        "You are a scientific research assistant with expertise in reading and "
        "interpreting academic papers, experimental data, and scientific literature. "
        "Explain findings accurately, contextualise methodologies, and identify "
        "the significance of results. Answer based on the provided document context."
    ),
    "Creative": (
        "You are a creative writing assistant with deep knowledge of narrative "
        "structure, character development, style, and genre conventions. "
        "Help analyse, improve, or generate creative content thoughtfully. "
        "Answer based on the provided document context."
    ),
    "Customer Service": (
        "You are a customer service specialist focused on resolving customer issues "
        "clearly, empathetically and efficiently. Reference product or service "
        "documentation to provide accurate answers. Keep responses professional "
        "and solution-oriented. Answer based on the provided document context."
    ),
    "Vision": (
        "You are a multimodal AI assistant specialising in visual content analysis. "
        "Interpret image descriptions, visual data, and related documents accurately. "
        "Answer based on the provided document context."
    ),
}

AVAILABLE_DOMAINS = list(DOMAIN_SYSTEM_PROMPTS.keys())


def get_domain_system_prompt(domain: str) -> str:
    """Return domain-aware system prompt, defaulting to General."""
    return DOMAIN_SYSTEM_PROMPTS.get(domain, DOMAIN_SYSTEM_PROMPTS["General"])


# ── Provider / model catalogue ────────────────────────────────────────────────

PROVIDER_MODELS: Dict[str, Any] = {
    "openai": {
        "gpt-4o":        {"label": "GPT-4o",        "cost": "$0.005/1K tokens"},
        "gpt-4-turbo":   {"label": "GPT-4 Turbo",   "cost": "$0.01/1K tokens"},
        "gpt-4":         {"label": "GPT-4",          "cost": "$0.03/1K tokens"},
        "gpt-3.5-turbo": {"label": "GPT-3.5 Turbo", "cost": "$0.001/1K tokens"},
    },
    "anthropic": {
        "claude-opus-4-5":           {"label": "Claude Opus 4.5",   "cost": "$0.015/1K tokens"},
        "claude-sonnet-4-5":         {"label": "Claude Sonnet 4.5", "cost": "$0.003/1K tokens"},
        "claude-3-5-haiku-20241022": {"label": "Claude 3.5 Haiku",  "cost": "$0.0008/1K tokens"},
    },
    "gemini": {
        "gemini-1.5-pro":   {"label": "Gemini 1.5 Pro",   "cost": "$0.007/1K tokens"},
        "gemini-1.5-flash": {"label": "Gemini 1.5 Flash", "cost": "$0.0007/1K tokens"},
        "gemini-pro":       {"label": "Gemini Pro",        "cost": "$0.0005/1K tokens"},
    },
}
