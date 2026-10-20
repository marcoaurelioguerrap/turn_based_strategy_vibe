"""Deterministic, stateless randomness.

Every random value in the game is derived from the game seed plus the context of
the decision (turn number, action sequence number, unit ids).  Nothing random is
stored in the game state, which keeps serialisation small and makes replays
exact: the same seed and the same actions always produce the same events.
"""

from __future__ import annotations

import hashlib

#: The Advance Wars style damage jitter, as a percentage of base damage.
LUCK_RANGE = 12


def _digest_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def roll_percent(seed: int, *parts: object) -> int:
    """A stable value in ``0..99`` for the given context."""
    return _digest_int(seed, *parts) % 100


def luck_percent(seed: int, *parts: object) -> int:
    """A stable damage jitter in ``-LUCK_RANGE..+LUCK_RANGE`` percent."""
    return _digest_int(seed, *parts) % (2 * LUCK_RANGE + 1) - LUCK_RANGE
