"""Ticker consistency guards for the multi-subgraph pipeline.

Every stage subgraph in the parent graph is preceded by a TickerGuard node
that verifies the instrument under analysis before the stage runs:

1. **Format check (fail-fast)**: the ticker must be a well-formed symbol
   (e.g. 600519.SH, 300394.SZ, NVDA, 7203.T). Anything else aborts the run
   with a descriptive error.
2. **Consistency check (fail-fast)**: the ticker in the shared state must
   match the anchor value (input_ticker) captured at run start. If any
   agent, tool result, or future stage rewrites company_of_interest to a
   different symbol, the pipeline stops before the next stage can analyse
   the wrong instrument.
3. **Normalisation (write-back)**: bare A-share codes are completed to the
   canonical CODE.EXCHANGE form (600519 -> 600519.SH, 000001 -> 000001.SZ)
   following the same rule the akshare data layer uses, and the suffix case
   is fixed (600519.sh -> 600519.SH). The canonical value is written back
   into company_of_interest so every downstream agent and tool call sees
   one unambiguous instrument.

The guards are pure functions over the parent AgentState, so they also work
as standalone utilities and are trivially unit-testable.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

# 600519.SH / 300394.SZ / 430047.BJ (case-insensitive; suffix is uppercased)
_A_SHARE_WITH_SUFFIX_RE = re.compile(r"^\d{6}\.(SH|SZ|BJ)$", re.IGNORECASE)
# Bare 6-digit A-share code, e.g. "600519"
_A_SHARE_BARE_RE = re.compile(r"^\d{6}$")
# Generic CODE.EXCHANGE (NVDA, CNC.TO, 7203.T, BRK.A) with optional index caret
_GENERIC_SYMBOL_RE = re.compile(r"^\^?[A-Za-z0-9]+(\.[A-Za-z0-9]+)?$")


def normalize_ticker(ticker: str) -> str:
    """Normalise a ticker to its canonical form.

    - strips surrounding whitespace,
    - uppercases the exchange suffix of A-share codes (600519.sh -> 600519.SH),
    - completes bare A-share codes (600519 -> 600519.SH, 000001 -> 000001.SZ)
      using the same SH/SZ rule as the akshare data layer,
    - uppercases generic symbols (cnc.to -> CNC.TO).

    Raises:
        ValueError: if the value is not a well-formed ticker.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError(f"ticker must be a non-empty string, got {ticker!r}")

    raw = ticker.strip()
    if len(raw) > 20:
        raise ValueError(f"ticker too long ({len(raw)} chars): {ticker!r}")

    if _A_SHARE_WITH_SUFFIX_RE.fullmatch(raw):
        # Explicit suffix wins; just fix the case: 600519.sh -> 600519.SH
        return f"{raw[:6]}.{raw[7:].upper()}"
    if _A_SHARE_BARE_RE.fullmatch(raw):
        # Follow the data layer: 6/5-prefixed codes are Shanghai, else Shenzhen.
        # (Beijing Stock Exchange codes are not auto-completed here; extend
        # the data layer first if you need BJ support.)
        return f"{raw}.SH" if raw[0] in ("6", "5") else f"{raw}.SZ"
    if _GENERIC_SYMBOL_RE.fullmatch(raw):
        return raw.upper()

    raise ValueError(
        f"invalid ticker {ticker!r}: expected CODE.EXCHANGE like 600519.SH / "
        "300394.SZ / 7203.T, a bare 6-digit A-share code, or an alphabetic "
        "symbol like NVDA"
    )


def is_valid_ticker(ticker: str) -> bool:
    """True if ticker normalises without error."""
    try:
        normalize_ticker(ticker)
        return True
    except ValueError:
        return False


def create_ticker_guard(stage_name: str) -> Any:
    """Create a ticker-consistency guard node for the stage that follows it.

    Args:
        stage_name: Display name of the upcoming stage, used in error messages
            so failures point at exactly where the pipeline stopped.

    Returns:
        A LangGraph node (state: AgentState) -> dict. Raises ValueError
        when the ticker is malformed or diverges from the run's anchor
        (input_ticker); otherwise returns the normalised ticker write-back
        (or an empty dict when nothing changed).
    """

    def ticker_guard(state: Dict[str, Any]) -> Dict[str, Any]:
        current = state.get("company_of_interest", "")
        anchor = state.get("input_ticker", "")

        # 1) Format check — fail fast on anything that is not a ticker
        try:
            normalized = normalize_ticker(current)
        except ValueError as exc:
            raise ValueError(
                f"[TickerGuard:{stage_name}] refusing to enter stage: "
                f"company_of_interest is not a valid ticker. {exc}"
            ) from exc

        # 2) Consistency check — must still match the anchor captured at run start
        if anchor:
            try:
                anchor_norm = normalize_ticker(anchor)
            except ValueError:
                anchor_norm = anchor.strip().upper()
            if normalized != anchor_norm:
                raise ValueError(
                    f"[TickerGuard:{stage_name}] ticker inconsistency detected: "
                    f"company_of_interest is {current!r} but this run is anchored "
                    f"to {anchor!r}. A previous stage changed the instrument; "
                    "the pipeline stopped to avoid analysing the wrong symbol."
                )

        # 3) Normalisation write-back (e.g. 600519 -> 600519.SH, 600519.sh -> 600519.SH)
        if normalized != current:
            logger.info(
                "[TickerGuard:%s] normalising ticker %r -> %r before entering stage",
                stage_name, current, normalized,
            )
            return {"company_of_interest": normalized}
        return {}

    return ticker_guard
