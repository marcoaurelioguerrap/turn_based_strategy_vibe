"""The pygame hot-seat UI for the tbs engine (milestone M3).

This package is a *client* of the pure engine: it reads a
:class:`~tbs.engine.state.GameState`, enumerates the same legal options the
agent API serves, and mutates the game only through ``rules.perform``.  The
engine never imports anything from here, so agents, the server, text views and
the window all describe the same game.

Modules are split so that all decision logic lives in pygame-free modules and
can be tested headlessly:

``palette.py``
    pure colour choices (terrain, players, highlights).
``layout.py``
    pure hex/pixel geometry, hit-testing and the viewport camera.
``interaction.py``
    pure ``(state, selection, click) -> Intent`` mode machine.
``render.py``
    pygame drawing only.
``app.py``
    pygame window and event loop; ``py -m tbs.ui.app``.
"""
