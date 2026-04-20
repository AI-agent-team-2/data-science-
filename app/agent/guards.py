from __future__ import annotations

# Backward-compatible shim.
# Prefer importing from `app.guards.prompt_injection`.

from app.guards.prompt_injection import (  # noqa: F401
    apply_guard,
    detect_prompt_injection,
    known_domain_constraint_response,
    rewrite_suspicious_query,
)
