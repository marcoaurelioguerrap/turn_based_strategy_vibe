# tbs — a turn-based strategy engine built for LLM agents

A modern-setting hex wargame in the spirit of Advance Wars / Panzer Corps 2 /
Daisenryaku, built as a **headless, deterministic, JSON-serializable Python
engine** with zero dependencies. Agents — LLM, swarm, or scripted — receive a
prompt-sized view of the game, enumerate everything each unit may legally do,
and answer with plain JSON orders; every order is validated on its own and a
rejection names the violated rule in one sentence, so an agent can fix its
answer and retry instead of guessing.

A UI, an LLM driver, and a replay tool are interchangeable clients of the same
pure core: `state + action → new_state + events`.

## Status

Milestones from the agreed plan (kept in `~/.opencode/plan/turn-based-strategy-plan.md`):

| | Milestone | State |
|---|---|---|
| M1 | Core: hex grid, state, scenario loading, movement, serialisation | done |
| M2 | Combat: damage matrix, counterattacks, ranges, ammo | done |
| M4 | Economy: capture, income, production | done (tuning is ongoing by design) |
| M5 | Logistics: supply domains, fuel, resupply, transports | done |
| M6 | Victory conditions, save/load, replay log | done |
| M7 | LLM agent API | done — views, options, previews, order docs, and the STDIO JSON-lines server (`tbs/api/server.py`) |
| M8 | An agent plays a full game | scripted-agent play is tested (`tests/test_agent_play.py`, `tests/test_agent_server.py`); a real LLM game is the next step |
| M3 | pygame UI | done — a hot-seat window (`tbs/ui/`, `py -m tbs.ui.app`) over the same action API |
| M9+ | Fog of war, swarm orchestrator, scenario editor | planned |

Everything that follows is enforced by **240 passing tests** (`py -m pytest`,
well under a second) — each claim in this README is pinned by one of them.

## Run it

Python 3.11 or newer. The engine is stdlib-only; `pytest` is the sole dev
dependency, and the optional window needs `pygame-ce`.

```
py -m pip install -e ".[dev]"       # or just: py -m pip install pytest
py -m pytest                        # 240 tests in ~1 s
py scripts/demo.py                  # a two-minute tour: view → bad order rejected → two AI half-turns → save/load/resume
py -m pip install -e ".[ui]"        # the hot-seat window (pygame-ce; import name is `pygame`)
py -m tbs.ui.app                    # opens the main menu: New Game / Load Game / Quit
py -m tbs.ui.app --no-menu          # skip the title screen and play the bundled scenario
py -m tbs.api.server                # the agent server: JSON objects in, JSON objects out, one per line
```

The window is a *client* of the same pure core: clicking only ever produces an
order the engine already lists as legal, so a game played by hand saves,
hashes and replays exactly like an agent's.

- **Menus:** a main menu (New Game → scenario picker, Load Game, Quit), an
  in-game pause menu (`Esc` with nothing selected, or `M`), and a per-unit
  order menu (Move / Attack / Capture / Load / Unload / Status / Hold / Cancel).
  Entries that are illegal right now are shown greyed out — "Attack (0)" when
  nothing is in range.
- **Movement range:** selecting a unit strokes the *boundary* of every hex it
  can reach, and lights its attackable enemies; the same `rules.legal_moves` /
  `rules.attack_options` the agent API serves. Choosing Attack strokes the
  weapons' range boundary instead.
- **Hover:** resting on a hex shows the terrain, cover, owner and coordinates,
  plus the standing unit's HP, ammo (current/max), fuel, rearm and this-turn status.
- **Toolbar:** vector-drawn icon buttons down the left border for Menu, End
  Turn, Units (the current player's roster), Build, Save, Load, Log,
  Coordinates and zoom.
- **Keyboard:** arrows move a white cursor hex; on a legal destination it draws
  the engine's exact path as a white trail with an arrowhead, and `Enter`
  confirms. `Shift`+arrows pan, `Home` recentres. In the order menu `W`/`S`
  moves the highlight and `F` activates it; `I` opens status, `U` the roster.
  Moving keeps the unit selected so a direct attack can follow.
- **Mouse:** left click selects and commits, right click or `Esc` backs out,
  order / weapon-picker / build-menu rows are clickable, mouse wheel zooms.
  Other keys: `Tab` cycles ready units, `Space`/`E` end turn, `H` hold, `B`
  build menu (then a number), `C` coordinates, `L` log, `F5`/`F9` quick
  save/load.




## Quickstart

```python
from tbs.engine import rules, view
from tbs.engine.data import available_scenarios, load_scenario, load_specs
from tbs.engine.state import new_game

scenario = load_scenario(available_scenarios()[0])   # data/maps/border_skirmish.json
state = new_game(scenario, load_specs())
rules.start_game(state)                               # scenario-authored hp/fuel/ammo survive turn 1

print(view.render_map(state))                         # terrain, ownership, units as aligned text

options = rules.unit_options(state, "A5")             # everything A5 may legally do, right now
order = {"action": "attack", "unit": "A5", "target": options["attacks"][0]["target"]}
events = rules.perform(state, order)                  # validated; returns the events it caused
rules.perform(state, {"action": "end_turn"})
```

## The agent loop

1. **Ask for a view.** `view.view_json(state, "A")` is the whole game one agent
   needs as one JSON object: map layers, sites and owners, unit table, turn and
   status, a rules summary, coordinate explanations — *and* for each of the
   asking player's units a `unit_actions` entry listing its legal attacks (with
   luck-free damage previews), moves, capture hints, load/unload options, and
   `"hold": true` (a unit is never left without a legal order). For the bundled
   14×8 skirmish the entire view is ~10 KB of JSON.
2. **Choose an order.** Attack previews, move endpoints and capture hints all
   arrive in the view itself; `rules.unit_options()` is the same data as typed
   Python objects, including `preview_damage` per target and human-readable
   blockers. Enumerating options consumes and changes nothing.
3. **Submit it.** `rules.perform(state, order)` validates one order at a time.
   Bad orders raise `ActionError` phrased for correction, e.g.
   `A1 cannot attack B1 at range 11: mg reaches 1-1, B1 is 11 away`. Orders may
   be submitted in any order and by different agents; a coordinator ends the turn.

## Action protocol

All orders are JSON objects; coordinates are axial `[q, r]` (see *Coordinates*).

| Order | Shape | Notes |
|---|---|---|
| move | `{"action":"move","unit":"A2","path":[[q,r],…]}` | contiguous path from the unit's hex, within move + fuel |
| attack | `{"action":"attack","unit":"A5","target":"B5"}` | optional `"weapon"` id; omitted = engine picks the best legal weapon |
| capture | `{"action":"capture","unit":"A3"}` | unit must stand on a capturable site not already its own |
| load | `{"action":"load","unit":"A4","cargo":"A3"}` | cargo must be adjacent; transport classes gate what fits |
| unload | `{"action":"unload","unit":"A4","cargo":"A3","target":[q,r]}` | drop onto a hex adjacent to the transport |
| build | `{"action":"build","site":[q,r],"unit":"infantry"}` | own factory (land), port (naval), airport (air); new units act next turn |
| hold | `{"action":"hold","unit":"A1"}` | always legal for a unit that may still act |
| end_turn | `{"action":"end_turn"}` | the coordinator's order; runs resupply, fuel attrition and victory checks |

`tests/test_agent_play.py` drives the bundled scenario with a purely
options-derived greedy policy, replays the whole game from its recorded orders,
and proves a reader who sees only the view JSON can pick valid attacks from it.

## The agent server

`py -m tbs.api.server` wraps that exact protocol as a process: one JSON object
per line in, one per line out, starting with an unprompted greeting that names
the scenario and protocol. Commands are `hello`, `help` (serves the full
protocol as data at runtime), `state`, `status`, `options`, `order` (fields
inlined next to the `cmd`), `save` and `load` — every answer is `{"ok":true,…}`
or `{"ok":false,"error":"one-sentence reason to retry from"}`. Rejected orders
never touch the game, broken lines never crash the server, and the served
`state_hash` after each order lets an agent verify what really changed.
Options: `--scenario <path>`, `--seed <n>`, `--session <save file>`, `--root <dir>`.

```
> {"cmd":"state"}
< {"ok":true,"view":{...whole prompt view...}}
> {"cmd":"order","action":"attack","unit":"A5","target":"B5","weapon":"howitzer"}
< {"ok":true,"events":[...],"state_hash":"...","status":"ongoing","turn":1,...}
> {"cmd":"order","action":"attack","unit":"A1","target":"B1"}
< {"ok":false,"error":"A1 cannot attack B1 at range 11: mg reaches 1-1, B1 is 11 away"}
```

## Coordinates

Internally everything is axial `[q, r]`. Maps are authored as text rows in the
odd-r offset layout (`rows` in scenario JSON), where row/column counting reads
naturally; the engine converts once at load. Scenario unit placements use
`offset` `[column, row]` coordinates for the same reason. The `view` module
explains both conventions inside every view.

## Combat

Integer HP 0–100. A hit is computed in this exact order, all integer math:

1. `power × multiplier% ÷ 100` — the weapon × damage-class matrix
   (`infantry`, `light`, `heavy`, `air`, `naval`, `sub`) from `data/units.json`;
2. `× attacker_hp ÷ 100` — damage scales with condition;
3. counterattacks are computed at 50 % power, and only at distance 1 with a
   direct weapon (`min_range = 1`); indirect attacks are never counterattacked;
4. terrain cover reduces the result (`data/terrain.json` cover percentages);
5. luck, if the scenario enables `damage_luck` (it is on by default): a roll in
   a −12 %…+12 % band;
6. minimum 1 damage.

A unit may move and then attack in the same turn, with one restriction: **indirect
weapons (`min_range > 1`) cannot fire after the unit has moved**. Direct weapons
still can, so a moved Destroyer keeps its Deck Gun but loses Shore Bombardment.
`rules.attack_options` filters those weapons out, the same way the UI and the
agent server see it.

Luck is **stateless**: every roll is a pure function of the game seed plus the
exact context of the act that rolls it (turn number, action sequence number,
attacker, defender, weapon, phase). Nothing random is ever stored in a state, so
re-runs are identical even with dice on.

## Supply, fuel, capture

- Every weapon has limited **ammunition**, spent one point per shot; a weapon at
  zero cannot fire. Ammunition is the only firing limit: a unit can attack
  anywhere as long as it has ammo and a target in range. The Tank carries 3 Main
  Gun shells and 4 Machine Gun bursts.
- Aircraft fly above the surface: one air unit may share a hex with one land or
  naval unit. Two aircraft or two surface units still block each other, and a
  stacked aircraft and surface unit may move through the shared hex freely.
- Supply lines govern **rearming**. A unit is in supply when it stands on or
  next to one of its own bases whose domain covers it, or next to (distance ≤ 1)
  a friendly supply vehicle — Truck and APC for land, Transport Ship for naval.
  Cargo riding inside a supply vehicle counts as supplied.
- At the start of its turn every supplied unit is fully replenished (fuel and
  ammo). `start_game` deliberately does *not* replenish, so a scenario's
  authored hp/fuel/ammo survive turn 1; placements that omit ammo start full.
- Air and naval units that end their turn at 0 fuel lose HP, unless a base or
  supply vehicle is near enough to top them up.
- Capture is progress accumulation: a unit standing on a capturable site adds
  its current HP to that site's progress each capture action; one capturer at a
  time per site. Cities, factories, ports and airports are capturable; capturing
  an HQ ends the game immediately. Elimination or the scenario's turn limit ends
  it as elimination or draw.
- Income is paid at each turn start from owned sites; production happens on the
  matching build sites; a unit built this turn cannot act until the next.

## Determinism, saves, replay

- `state_hash(state)` hashes the full serialized state *minus the log* — it is
  the position; `log_hash(state)` hashes the recorded history. They track
  different things on purpose.
- `serialize.replay(scenario, orders)` rebuilds a game from its orders alone —
  identical events, positions and hashes — and a seed override gives exact
  what-if branches.
- Save files are plain JSON (`save_game`/`load_game`); a loaded game continues
  byte-identically from where the original stopped.

## Layout

```
tbs/engine/     pure logic, no I/O, no pygame, stdlib only
  hexgrid.py    axial coords, distances, ranges, Dijkstra pathfinding
  data.py       GameSpecs: units/terrain/scenarios + the data validator
  state.py      GameState, Unit, Tile, Player, new_game()
  actions.py    action types + JSON (de)serialisation
  rules.py      turn flow, order validation, legal-action queries, economy, supply
  combat.py     damage maths (integer-only)
  rng.py        stateless dice: seed + acting context → 0..99 roll
  view.py       text and JSON views — the same views agents receive
  serialize.py  save/load, replay, state/log hashes
  errors.py     ActionError with correction-shaped messages
tbs/api/
  server.py     the STDIO JSON-lines agent server over the engine API
tbs/ui/
  palette.py    UI colours (pure)
  layout.py     hex/pixel geometry, camera, hit-testing, reachable-area outline (pure)
  menu.py       menu model: items, enabled state, navigation (pure)
  info.py       hover text: tile and unit facts (pure)
  buttons.py    icon-toolbar layout and hit-testing (pure)
  interaction.py click resolver: selection + mode + action menu + cursor -> order dict (pure)
  render.py     pygame drawing only
  app.py        the pygame window, menus and event loop (`py -m tbs.ui.app`)
data/
  units.json    14 unit types + the weapon × damage-class matrix
  terrain.json  11 tile types: cover, move costs, build/capture/supply rules
  maps/         scenarios as text-row maps + regions + starting armies
scripts/demo.py headless end-to-end tour of the API
tests/          240 tests pinning every rule above
```

## Tests

| File | Tests | What it pins |
|---|---|---|
| `test_hexgrid.py` | 17 | coordinates, distances, diagonals, pathfinding |
| `test_movement.py` | 22 | move costs, fuel, blocking, occupied hexes, no second move after moving |
| `test_combat.py` | 30 | the full damage chain, counters, cover, weapon choice, ammo as the firing limit, direct fire after moving and indirect fire only when stationary |
| `test_logistics.py` | 23 | supply domains, resupply, load/unload, fuel attrition |
| `test_capture_economy.py` | 14 | capture progress, income, production, victory by capture |
| `test_scenarios.py` | 11 | scenario validation, fairness (material symmetry), error strings |
| `test_stacking.py` | 11 | aircraft sharing a hex with one land/naval unit, two-layer movement, targeting, unloading, production and views |
| `test_determinism.py` | 11 | replays, seeded dice, hashes, save/load continuation |
| `test_data_integrity.py` | 26 | cross-file data rules + every validator message |
| `test_agent_play.py` | 4 | greedy-agent play, order-log replay, view/API agreement, view-only orders |
| `test_agent_server.py` | 7 | JSON-lines protocol over the wire, structured rejections, save/load continuation, the real `py -m tbs.api.server` process |
| `test_ui_layout.py` | 11 | hex/pixel round trip, centring, hit-testing, zoom, the reachable-area outline, and a guard that `engine/` never imports pygame |
| `test_ui_menu.py` | 5 | menu navigation skips disabled entries and wraps around |
| `test_ui_info.py` | 5 | hover text for terrain, owners, units, ammo, rearm and status |
| `test_ui_buttons.py` | 3 | toolbar layout, overflow and hit-testing |
| `test_ui_interaction.py` | 22 | select/move/attack/build/action-menu/weapon-picker held to the engine's own options; cursor and attack region; move-then-attack for direct weapons; blocked-attack reasons; a UI-played turn replays from its orders |
| `test_ui_app.py` | 18 | the real pygame loop on SDL's dummy driver: draw, menus, hover, toolbar, click-to-move, cursor, action menu, status, pause, end turn, build menu, zoom |

## Next

- **M8:** a real LLM agent playing a full game through the server; the scripted-agent scaffolding and the wire protocol are both in place and tested.
- **M3 polish:** combat animations, snapshot undo and a minimap.
