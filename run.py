"""Run the MoneyGuard local sandbox (Python 3.11+, standard library only)."""
import argparse
import errno
import os
import sys
from pathlib import Path

from moneyguard.server import serve

ROOT = Path(__file__).resolve().parent


def load_env():
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if key.strip() in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL", "GROQ_API_KEY", "GROQ_MODEL",
                                   "CUSTOM_AI_BASE_URL", "CUSTOM_AI_API_KEY", "CUSTOM_AI_MODEL", "CUSTOM_AI_NAME", "MONEYGUARD_AI_ORDER", "MONEYGUARD_HOST", "MONEYGUARD_PORT", "MONEYGUARD_PUBLIC_HOSTS",
                                   "MONEYGUARD_LOCAL_MODEL_URL", "MONEYGUARD_LOCAL_TEXT_MODEL", "MONEYGUARD_LOCAL_VISION_MODEL",
                                   "MONEYGUARD_SPEECH_MODEL_DIR", "MONEYGUARD_SPEECH_GROQ_API_KEY", "MONEYGUARD_SPEECH_GEMINI_API_KEY", "MONEYGUARD_SPEECH_GEMINI_MODEL",
                                   "MONEYGUARD_LOCAL_STRUCTURED_PLANS", "MONEYGUARD_SEMANTIC_SCREENING", "MONEYGUARD_DECISION_LAB"):
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


if __name__ == "__main__":
    if sys.version_info < (3, 11):
        sys.exit("MoneyGuard needs Python 3.11 or newer.")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_env()
    parser = argparse.ArgumentParser(description="MoneyGuard: approval-gated cash-flow rescue (synthetic sandbox)")
    parser.add_argument("--host", default=os.getenv("MONEYGUARD_HOST", "127.0.0.1"),
                        help="Bind address (loopback by default; Docker uses 0.0.0.0 behind a local port mapping)")
    parser.add_argument("--port", type=int, default=int(os.getenv("MONEYGUARD_PORT", "8787")))
    parser.add_argument("--db", default=str(ROOT / "data" / "moneyguard.sqlite3"))
    parser.add_argument("--no-demo", action="store_true", help="Start without the synthetic demo businesses")
    args = parser.parse_args()
    # Optional cloud AI. Several providers (Gemini, Groq, OpenRouter, a custom OpenAI-compatible one) are tried in
    # order, so a busy one never blocks an answer. None is needed for the app to work. See moneyguard/ai_providers.py.
    from moneyguard.ai_providers import label, make_transport, providers_from_env
    try:
        providers = providers_from_env(os.environ)
    except ValueError as exc:
        sys.exit(f"AI provider settings in .env: {exc}")
    if providers:
        key, model, transport = "configured", label(providers), make_transport(providers)
    else:
        key, model, transport = os.getenv("OPENAI_API_KEY", "").strip(), os.getenv("OPENAI_MODEL", "gpt-5-mini"), None
    try:
        serve(args.db, args.port, key, model, not args.no_demo, args.host, transport)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        sys.exit(f"Port {args.port} is already in use, probably by another MoneyGuard still running.\n"
                 f"Stop it with Ctrl+C in its terminal, or start this one on another port:\n"
                 f"  python3 run.py --port {args.port + 1}")
