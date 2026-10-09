"""Runtime configuration. Everything tunable lives here and is overridable via env / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- LLM -------------------------------------------------------------
    # provider: groq | openrouter | openai | ollama | anthropic | gemini | none
    # "none" runs the deterministic fallback planner (no network) - used by tests/CI.
    # Primary provider, then ordered fallbacks (comma-separated). Models per provider come from
    # <PROVIDER>_MODEL / <PROVIDER>_VISION_MODEL env vars, else the defaults in llm.PROVIDERS.
    provider: str = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    fallback_providers: tuple[str, ...] = tuple(
        p.strip() for p in os.getenv("LLM_FALLBACK", "groq").split(",") if p.strip())
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "24"))
    turn_deadline_s: float = float(os.getenv("TURN_DEADLINE_S", "240"))  # LLM wall-clock budget per turn

    # --- Paths -----------------------------------------------------------
    claims_dir: Path = Path(os.getenv("CLAIMS_DIR", ROOT / "claims"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", ROOT / "outputs"))
    cache_dir: Path = Path(os.getenv("CACHE_DIR", ROOT / ".cache"))
    state_dir: Path = Path(os.getenv("STATE_DIR", ROOT / ".state"))

    # --- Business policy knobs (see README "Policy decisions") ------------
    # Every settlement in the sample data shows Net Payout == ACV (deductible not
    # subtracted). We flag it as a warning; set true to make it block finalization.
    strict_settlement_math: bool = _bool("STRICT_SETTLEMENT_MATH", False)
    # Warnings (adjuster concerns, math) block finalization when true.
    warnings_block_finalize: bool = _bool("WARNINGS_BLOCK_FINALIZE", False)
    # OCR line confidence thresholds.
    ocr_good: float = float(os.getenv("OCR_GOOD", "0.90"))
    ocr_fair: float = float(os.getenv("OCR_FAIR", "0.75"))

    required_documents: tuple[str, ...] = field(
        default=("police_report", "finance_agreement", "settlement_breakdown")
    )


settings = Settings()
for _d in (settings.output_dir, settings.cache_dir, settings.state_dir):
    _d.mkdir(parents=True, exist_ok=True)
