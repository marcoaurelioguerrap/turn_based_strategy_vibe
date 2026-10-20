"""Errors raised while validating an action.

Messages are written for whoever is driving the game (a human UI or an LLM
agent), so they name the unit, state the constraint that was violated and say
what to do instead.  An agent can feed the message straight back into its next
attempt.
"""

from __future__ import annotations


class ActionError(ValueError):
    """An action that the rules do not allow."""
