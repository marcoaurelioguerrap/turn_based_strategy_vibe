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

## Download and play

- **Play the game on Windows (no Python):** download `tbs-windows-x64.zip` from
  the [latest release](https://github.com/marcoaurelioguerrap/turn_based_strategy_vibe/releases/latest),
  extract it and run `tbs.exe`.
- **Game source:** this repository. `py -m pip install -e ".[ui]"` then
  `py -m tbs.ui.app`.
- **Play against NCM:** clone the
  [NCM fork](https://github.com/marcoaurelioguerrap/NCM_fork), download the base
  checkpoint from its [releases](https://github.com/marcoaurelioguerrap/NCM_fork/releases),
  and follow `tbs_ncm/README.md`. NCM needs Python, PyTorch and the checkpoint;
  the standalone game build does not include it.
- **Experiment report:** [`ncm_training.pdf`](https://github.com/marcoaurelioguerrap/turn_based_strategy_vibe/releases/latest/download/ncm_training.pdf)
  covers the data, tokenizer, generator and league results.

The game is a hot-seat window out of the box. The **New Game vs LLM** menu needs
an OpenAI-compatible endpoint; the NCM opponent is launched from the NCM checkout
with `py -m tbs_ncm.play_ui`.

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
| M8 | An agent plays a full game | single local LLM plays complete games (headless verified to a turn-60 draw; same loop in pygame, `tbs/agents/`) |
| M10 | Council of generals | done — a commander plus role generals and a deterministic merger (`tbs/agents/council.py`) |
| M11 | Map pack and scenario picker | done — four new maps, picker metadata/thumbnails, scrollable menus, fit-to-view camera |
| M3 | pygame UI | done — a hot-seat window (`tbs/ui/`, `py -m tbs.ui.app`) over the same action API |
| M9+ | Fog of war, scenario editor | planned |

Everything that follows is enforced by **407 passing tests** (`py -m pytest`,
a few seconds) — each claim in this README is pinned by one of them.

## Run it

Python 3.11 or newer. The engine is stdlib-only; `pytest` is the sole dev
dependency, and the optional window needs `pygame-ce`.

```
py -m pip install -e ".[dev]"       # or just: py -m pip install pytest
py -m pytest                        # 407 tests in ~10 s
py scripts/demo.py                  # a two-minute tour: view → bad order rejected → two AI half-turns → save/load/resume
py scripts/landing_benchmark.py     # scripted self-play landing/load/unload metrics on the island maps
py scripts/generate_ferry_data.py   # NCM-format rows from ferry-capable scripted games
py -m pip install -e ".[ui]"        # the hot-seat window (pygame-ce; import name is `pygame`)
py -m tbs.ui.app                    # opens the main menu: New Game / New Game vs LLM / Load Game / Quit
py -m tbs.ui.app --no-menu          # skip the title screen and play Border Skirmish
py -m tbs.api.server                # the agent server: JSON objects in, JSON objects out, one per line
py -m tbs.agents.match --a greedy --b llm --llm-model "<model>"   # headless LLM match with a replayable log
py -m tbs.agents.match --a greedy --b council --llm-model "<model>"  # the M10 council of generals
```

The window is a *client* of the same pure core: clicking only ever produces an
order the engine already lists as legal, so a game played by hand saves,
hashes and replays exactly like an agent's.

- **Menus:** a main menu (New Game → scenario picker, Load Game, Quit), an
  in-game pause menu (`Esc` with nothing selected, or `M`), and a per-unit
  order menu (Move / Attack / Capture / Load / Unload / Status / Hold / Cancel).
  Entries that are illegal right now are shown greyed out — "Attack (0)" when
  nothing is in range.
- **Scenario picker:** every map in `data/maps/` is listed by title with a
  size/turns/difficulty hint, a terrain-coloured thumbnail and a one-line
  description. Long lists scroll (`Up`/`Down`, `PageUp`/`PageDown`, mouse) and
  the picker opens on the explicit default map, `Border Skirmish`.
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

## Maps

Ten scenarios ship in `data/maps/`: the five original maps and five `_m`
variants at about ten times the tile area. The picker reads each file's `title`,
`description`, `difficulty` and `fair` fields; `name` stays the stable id used
by saves. A bare launch, the agent server and `tbs.agents.match` all default to
`border_skirmish`, resolved by name rather than by filename order.

| Map | Size | Theme | Fair |
|---|---|---|---|
| `border_skirmish` | 14×8 | Intro: ridge + bay, all three domains | yes |
| `mountain_pass` | 18×10 | Two foot-and-vehicle passes, no navy | yes |
| `archipelago` | 20×12 | Island war: ports, transports, destroyers | yes |
| `urban_delta` | 16×10 | Dense city blocks, factories, income race | yes |
| `last_stand` | 16×10 | Asymmetric attack/defence, larger Red army | no |
| `border_skirmish_m` | 44×25 | Large ridge + bay, 25 units a side | yes |
| `mountain_pass_m` | 56×32 | Large pass war, 23 units a side | yes |
| `archipelago_m` | 64×38 | Large island war, 28 units a side | yes |
| `urban_delta_m` | 50×32 | Large city and river, 26 units a side | yes |
| `last_stand_m` | 50×32 | Large attack/defence, 16 Blue vs 29 Red | no |

Maps are plain JSON: text rows in odd-r offset layout, `regions` for ownership,
`units` placed with `offset` `[column, row]`. The bundled fair maps are
hex-true mirrored (a plain string reverse is **not** a mirror on odd rows) and
are held to the same material checks as the intro map by
`tests/test_scenarios.py`: equal functional sites, army, income, objectives and
every starting unit in supply.

Every team is capped at **50 live units** by `max_units_per_player` (a scenario
field, default 50). Production is refused with a correction-shaped message at
the cap, and `build_options` and the agent view expose the limit.

The `_m` maps also set `view_move_limit: 12`. The hot-seat UI still uses the
full `rules.legal_moves`/`legal_actions` API; the agent view keeps a strategic
sample of destinations (sites, contact, frontier, nearest/farthest). That keeps
a legal-order reranker such as the NCM league tractable on a 2,400-tile board.

Run any map directly:

```
py -m tbs.ui.app --scenario data/maps/archipelago_m.json
py -m tbs.api.server --scenario data/maps/urban_delta_m.json
```

The large maps open on the current player's HQ at a playable zoom; `Home`
switches to the full-map overview, and the camera pans with `Shift`+arrows,
middle-mouse drag or the arrow keys.

### The large-map league

`scripts/league_m.py` runs the five scripted persona teachers from the NCM
checkout in a round-robin over the `_m` maps (no model weights required), with
40 turns and up to 12 orders per side per turn:

```
py scripts/league_m.py --max-turns 40 --max-steps 12
```

It writes `docs/league_m/report.json` and `report.md`. Raise `--max-turns` and
`--max-steps` for a deeper league. `--ncm-root` points at the NCM checkout when
it is not the sibling `..\NCM_fork`. The trained NCM model league command is in
*Playing against NCM* below.

### The World Cup

`scripts/world_cup.py` runs a 10-team World Cup over the five `_m` maps: the five
trained NCM personas, four randomized conventional AIs and the greedy baseline.
Two groups of five play a five-round round-robin (one bye per team); the top two
in each group meet in cross-group semifinals, then the third-place match and the
final. The group stage rotates through all five new maps, while the four fair
mirrored maps host the knockout. It is 24 games, 30 turns and up to 10 orders
per side per turn by default.

Run it with the NCM virtualenv so torch and the checkpoints are available:

```powershell
..\NCM_fork\.venv\Scripts\python.exe scripts\world_cup.py `
  --ncm-root ..\NCM_fork `
  --resume
```

The defaults are `--generator mixed_generator.pt --decode legal --prefix-len
4096 --max-turns 30 --max-steps 10`. A full run is resumable; `--resume`
continues an interrupted tournament and `--report-only` rebuilds the reports
without loading models. Use `--dry-run` to print the draw, and `--limit N` for a
short smoke run. If legal-order decoding is too slow for a deeper run, the NCM
large-map fast mode is `--decode sample --retries 1`.

The human report and resume state are written to `docs/world_cup/report.md`,
`report.json` and `tournament.json`. Match saves, order logs and meta sidecars
are written directly to the NCM dashboard directory,
`..\NCM_fork\artifacts\tbs\matches`, and the league is written as a batch under
`..\NCM_fork\artifacts\tbs\personas\world_cup_<seed>`. The NCM dashboard therefore
shows every game in **Games** and the whole tournament as one batch in
**Overview**:

```powershell
cd ..\NCM_fork
.\.venv\Scripts\python.exe -m tbs_ncm.dashboard --open
```

`--allow-asymmetric-knockout` opts `last_stand_m` into the knockout rotation;
the default keeps it in the group stage and uses the four fair maps for
elimination games.


## Quickstart

```python
from tbs.engine import rules, view
from tbs.engine.data import default_scenario, load_scenario, load_specs
from tbs.engine.state import new_game

scenario = load_scenario(default_scenario())          # data/maps/border_skirmish.json
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
   14×8 skirmish the entire view is ~10 KB of JSON. On the large `_m` maps the
   local-LLM prompt keeps every attack/capture/load and a bounded strategic
   sample of legal move destinations, so an 8K-context model still fits.
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

## Amphibious operations

Land units cannot enter the sea; a Transport Ship must lift them. The rules
already allow a transport to **move and then load or unload one cargo in the
same turn**, and an unloaded unit may act immediately. What a one-step view
could not express was the multi-turn operation, so the view now carries an
advisory `ferry` block on maps with sea transports:

```json
"ferry": {
  "stranded": ["A3", "A4"],
  "transports": ["A10"],
  "operations": [
    {"transport": "A10", "cargo": "A3", "action": "pickup",
     "cargo_to": [3, 5], "target": [15, 3]}
  ]
}
```

`island` ids (sparse, only on maps with a ferry layer) and `stranded: true`
make the geography explicit; `operations` names a transport, cargo, pickup or
landing hex and the strategic target. The operations are advisory, not new
legal actions: every coordinate comes from `rules.legal_moves`,
`load_options` or `unload_options`, and the engine still validates each order.

The scripted `GreedyAgent` now executes these hints before its normal
attack/capture/approach policy: it moves a cargo to the pickup, loads, sails to
the beach, unloads, and then hands the landed unit back to the ordinary capture
logic. It also repositions an empty transport toward the next stranded unit
instead of sailing it at the enemy. On the stock `archipelago` opening this
lands the first infantry on the enemy home island on the player's second turn.

The same hint is available to the LLM prompt and to the compact NCM codec. The
codec adds one line per operation (`transport cargo action cargo_to
transport_to at target`, `-` for absent fields), and the LLM prompt explains
how to execute it. The NCM player uses a small **learned action selector**
trained on these states to choose the action character (`l`, `u`, `p`, `s`,
`r`); the full NCM still supplies the arguments through legal candidates. No
scripted ferry override is called by the deployed NCM. The coordinator now
allows one move followed by one non-move action for the same unit, matching the
engine's `move then load / unload` rule; duplicate moves are still ignored.

Two model-free tools measure and collect the behavior:

```
py scripts/landing_benchmark.py
py scripts/generate_ferry_data.py --amphibious-repeat 4
```

`landing_benchmark.py` runs scripted self-play on `archipelago` and
`archipelago_m` and reports first landing turn, units that ever landed, units
that reached the enemy home island, and load/unload counts. The default seed
set lands on turn 2 on the small map and turn 7 on the large map.
`generate_ferry_data.py` writes the same `prefix`/`target` JSONL rows the NCM
trainer consumes, tags load/unload rows with `"amphibious": true`, and gives a
new NCM fine-tune the demonstrations the original border-skirmish dataset never
contained.

## Compact codecs and stepwise turns

`tbs/agents/codec.py` is a stdlib-only transport for models with a token budget.
`encode_state` turns a `state_view` into deterministic line-oriented text:
terrain rows, every site with owner and capture progress, every unit with
position/hp/fuel/ammo and its acted/moved/new/loaded/cargo flags, the legal
build list, and (on maps with transports) the `ferry` recommendations. For the
bundled scenario, the first five player-turn views encode to
1,061–1,125 characters — **511–541 GPT-2 tokens** measured with the released NCM
prefix tokenizer, against 3.8–4.7k tokens for the full JSON view. Static spec
data and `unit_actions` are deliberately absent: legality stays the
coordinator's job, and a rejection comes back as feedback.

Orders have a one-line form (`m A2 3,4`, `a A5 B5 howitzer`, `c A4`,
`b 1,1 infantry`, `e`). `decode_order(encode_order(order))` returns the
canonical agent order; an engine move path collapses to its destination and
`normalize_order` expands it back to the identical path.
`tests/test_agent_codec.py` round-trips every legal order the engine offers from
a real mid-game state, plus unloads created by a real load and turn flip.

For models that decide one order at a time,
`tbs.agents.turn.run_turn_stepwise` applies a batch and asks again with a fresh
view until no unit can act, the agent sends `end_turn`, it returns nothing
actionable, the game ends, or `max_steps` is reached. Format failures are
retried with feedback; transport failures propagate. Whole-turn agents work
unchanged: their plan is applied, and another call happens only if a unit can
still act. A one-order-per-call agent plays and replays a full game exactly.
`StepwiseSession` exposes the same loop as `next_request` / `record_plan` /
`ready` / `finish`, for a UI that runs each call in a worker thread.

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

## Playing against a local LLM

M8 adds a local, OpenAI-compatible opponent. Every LLM turn is one
`/chat/completions` request: the model receives the same JSON view an external
agent gets and answers with a whole-turn plan, then the coordinator validates
each order through the engine. Rejections are sent back once with the engine's
correction message and a fresh view; after that the turn ends anyway, so a weak
model can never stall the game. The HTTP call runs in a worker thread, so the
window keeps drawing while the model thinks.

Reference setup used for the NCM league (Unsloth Desktop):

```
http://127.0.0.1:8888/v1
ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF   (quant IQ3_S)
```

The server lists the loaded weights under the id without the `:quant` suffix;
check `/v1/models` for the loaded context. The client compacts the view and can
disable the model's reasoning mode; use `--llm-thinking` only when the server's
request window is large enough. The earlier 74B Nemotron match is preserved in
the NCM report as a historical run.

The server above requires a bearer token (an unauthenticated probe returns HTTP
401), so pass the key it prints, either as `--llm-api-key` or through
`TBS_LLM_API_KEY`. There is no default key in the repository.

Play in the window:

```
$env:TBS_LLM_BASE_URL = "http://127.0.0.1:8888/v1"
$env:TBS_LLM_MODEL    = "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF"
$env:TBS_LLM_API_KEY  = "<the key Unsloth Desktop shows>"
py -m tbs.ui.app --no-menu --agent-side B
```

Without `--no-menu`, start the window with the env vars set and choose
**New Game vs LLM** in the main menu, then pick your side. Other useful flags:
`--llm-temperature`, `--llm-max-tokens`, `--llm-timeout`, `--llm-retries`,
`--llm-attempts`, `--llm-json-mode`, and `--llm-log logs/llm/play.jsonl`.
During the LLM's turn, `Esc`/`M` opens the pause menu; **Take over for B**
disables the agent, **Resume LLM** turns it back on.

## The council of generals (M10)

`--agent-kind council` replaces the single opponent with a staff of agents on
the same local model. One council turn is:

1. **Commander** — one short plain-text call sets the intent: main effort,
   1–3 priority targets/sites, and whether to build. It never issues unit
   orders, so it cannot make an illegal move.
2. **Generals** — each general gets the board plus only its own `unit_actions`,
   its doctrine, and the commander's brief. Generals with no units and no
   affordable builds are skipped.
3. **Merge** — a deterministic, priority-ordered merger keeps one order per
   unit, enforces each general's slice, allows builds only from the
   Quartermaster, and keeps duplicate attacks on the same target (focus fire).
4. **Coordinator** — the existing `turn.py` loop validates every order through
   the engine, retries once with feedback, and always ends the turn.

| General | Doctrine | Commands | Builds |
|---|---|---|---|
| Vanguard | Armour/artillery offensive | tank, artillery, AA missile | no |
| Anvil | Infantry line, capture, HQ screen | infantry, HQ | no |
| Talon | Air strikes and fuel discipline | helicopter, attack plane, bomber | no |
| Fleet | Sea control and escort | destroyer, cruiser, submarine | no |
| Quartermaster | Economy, logistics, production | APC, Truck, Transport Ship + unassigned types | yes |

Talon and Fleet are skipped automatically when the map has no air or naval
units. The bundled skirmish has both, so a full council makes six calls per
turn (commander + five generals). For a faster game, pass a subset:

```powershell
$env:TBS_LLM_BASE_URL = "http://127.0.0.1:8888/v1"
$env:TBS_LLM_MODEL    = "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF"
$env:TBS_LLM_API_KEY  = "<the key Unsloth Desktop shows>"

py -m tbs.agents.match --a greedy --b council `
  --council-members vanguard,anvil,quartermaster `
  --max-turns 20 --out saves/council_match.json --llm-log logs/llm/council.jsonl

py -m tbs.ui.app --no-menu --agent-side B --agent-kind council
```

Useful flags: `--council-members` (comma-separated ids), `--council-config`
(a JSON roster override), `--council-no-commander` (one fewer call per turn),
and `--council-parallel` (ask generals concurrently; the local server may still
serialize them). All accepted orders replay exactly, even though the generals'
choices are stochastic. Per-general prompts, briefs, accepted orders and merge
drops are written to `--llm-log`.

A custom roster is a JSON file:

```json
{"generals":[
  {"id":"spear","name":"Spear","title":"the point","doctrine":"Attack.",
   "unit_types":["tank","artillery"],"priority":5,"can_build":false},
  {"id":"clerk","name":"Clerk","title":"the ledger","doctrine":"Build and capture.",
   "unit_types":[],"priority":90,"can_build":true,"reserve":true}
]}
```

## Playing against NCM

The NCM player lives in the separate
[NCM fork](https://github.com/marcoaurelioguerrap/NCM_fork) (`tbs_ncm/`) because
it needs torch. It trains its own character-level tokenizer on order lines, then
an NCM generator that reads the compact state text and writes one order per step.

Clone both repositories side by side, download the base checkpoint from the
[fork's releases](https://github.com/marcoaurelioguerrap/NCM_fork/releases) into
`NCM_fork/artifacts/tbs`, and launch the window from the NCM environment:

```powershell
git clone https://github.com/marcoaurelioguerrap/turn_based_strategy_vibe.git
git clone https://github.com/marcoaurelioguerrap/NCM_fork.git
cd NCM_fork
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt pygame-ce
$env:TBS_GAME_REPO = "..\turn_based_strategy_vibe"
.venv\Scripts\python.exe -m tbs_ncm.play_ui --no-menu --agent-side B --ncm-decode legal
```

On an NVIDIA Windows machine, install the CUDA torch wheel before the
requirements; `SETUP-LOCAL.md` in the fork has the exact command. The release
contains the tokenizer, the base generator and `config.yaml`. The five persona
checkpoints used in the 60-game league are not part of the base release: the
league table is recorded evidence from those checkpoints, not a claim the base
download can rerun. The reports and match files are in
[`docs/all_ai_league`](docs/all_ai_league).

The UI uses `StepwiseSession`: one order at a time with a fresh view. The
default `--ncm-decode legal` is a constrained decoder: it samples once for the
model's intended action/unit, ranks the view's legal orders for that choice with
the model, and returns the best legal order. The older free-generation path
(`--ncm-decode sample`) repairs an unparseable or illegal sample by snapping the
model's chosen unit/action to the nearest legal option, resamples, and only then
falls back to the greedy agent. `--ncm-max-steps` bounds the batches per turn;
`--ncm-checkpoint` selects the artifact directory (default `artifacts/tbs`).

The NCM bridge can also play the council headlessly and write the result where
its dashboard tracks it (`artifacts/tbs/matches`):

```powershell
cd NCM_fork
.\.venv\Scripts\python.exe -m tbs_ncm.play --opponent council --ncm-side B `
  --seed 60001 --council-members vanguard,anvil,quartermaster --council-parallel `
  --max-turns 40
```

Dashboards read the match save, its `.orders.json` replay and the `.meta.json`
sidecar written next to it.

The large `_m` maps form their own persona league. The league resolves maps by
the JSON `name`, so no NCM code changes are needed; point `--maps` at the new
stems and raise the turn cap:

```powershell
cd NCM_fork
.\.venv\Scripts\python.exe -m tbs_ncm.persona_league `
  --maps border_skirmish_m,mountain_pass_m,archipelago_m,urban_delta_m,last_stand_m `
  --games-per-map 1 --max-turns 20 --max-steps 6 `
  --decode sample --retries 1 --prefix-len 4096 `
  --out-dir artifacts/tbs/personas/league_m_20261025 `
  --match-dir artifacts/tbs/matches/m_league
```

The report is written as `report.json` / `report.md` in `--out-dir`, and every
game keeps its replayable save, order log and meta sidecar under `--match-dir`.
`--decode sample` keeps the model league practical on a 2,400-tile board; drop
it to use the slower legal-order reranker. The scripted league above is the
fast balance run; use `--pairs aggressive:defensive` for a quick model smoke
instead of the full round-robin. The compact codec is ~3.4-5.0k characters on
these maps, so the league uses a 4096-token prefix instead of the old 2048
default.

The 60-game all-AI league is a **system** result, not a raw-generation result.
With `--decode legal`, the NCM samples an intended action/unit and then ranks
the legal candidate lines. In the released 60-game run only 0.13% of NCM
Vanguard's orders (27 of 20,103) and 1.05% of NCM Siege's (213 of 20,338) were
exact raw generations; the rest were chosen by the constrained decoder, with
zero fallback. The 25.2% exact-match figure in the NCM report is raw generation
on held-out states, a different measurement. The quality table is in
[`docs/all_ai_league/quality_20261030_v2.json`](docs/all_ai_league/quality_20261030_v2.json).

Headless smoke test with a replayable order log and a final save:

```powershell
cd NCM_fork
.\.venv\Scripts\python.exe -m tbs_ncm.play --opponent greedy --ncm-side B --seed 7
```

The command writes a replayable save and order log; the NCM dashboard reads the
same sidecars. An LLM's choices are not deterministic, but its accepted orders
are a deterministic game.

The prompt lists move **destinations** (`unit_actions[id].move_to`), not paths.
The model answers with `{"action":"move","unit":"A2","to":[q,r]}` and the
coordinator expands that into the engine's current cheapest path, so a model
never has to invent a route. Any other order follows the normal action schema.

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
  amphibious.py landmass ids, stranded units and advisory ferry hints
  serialize.py  save/load, replay, state/log hashes
  errors.py     ActionError with correction-shaped messages
tbs/api/
  server.py     the STDIO JSON-lines agent server over the engine API
tbs/agents/
  protocol.py   Agent/TurnRequest/AgentPlan + tolerant JSON plan parser
  llm.py        stdlib OpenAI-compatible client, prompt builder, single LLM agent
  generals.py   council roster: doctrines, unit slices, filtered views
  council.py    M10 commander + generals + deterministic merge
  turn.py       validate whole-turn plans, retry rejections, always end the turn;
                run_turn_stepwise re-asks with a fresh view after each batch
  codec.py      compact deterministic state text + one-line order codec (stdlib)
  ferry.py      executes the view's ferry hints as legal orders
  greedy.py     deterministic scripted baseline agent (now ferry-aware)
  league.py     World Cup draw, schedule, standings, bracket and report (pure)
  match.py      headless LLM-vs-agent CLI with save + replayable order log
tbs/ui/
  palette.py    UI colours (pure)
  layout.py     hex/pixel geometry, camera (including fit-to-view), hit-testing, reachable-area outline (pure)
  menu.py       menu model: items, enabled state, scrolling window, navigation (pure)
  picker.py     scenario metadata + thumbnail sampling for the picker (pure)
  info.py       hover text: tile and unit facts (pure)
  buttons.py    icon-toolbar layout and hit-testing (pure)
  interaction.py click resolver: selection + mode + action menu + cursor -> order dict (pure)
  render.py     pygame drawing only
  app.py        the pygame window, menus and event loop (`py -m tbs.ui.app`)
data/
  units.json    14 unit types + the weapon × damage-class matrix
  terrain.json  11 tile types: cover, move costs, build/capture/supply rules
  maps/         ten scenarios: five originals and five ~10x `_m` maps
scripts/demo.py headless end-to-end tour of the API
scripts/landing_benchmark.py first-landing, enemy-home and load/unload metrics
scripts/generate_ferry_data.py NCM-format rows from ferry-capable scripted games
scripts/league_m.py fast scripted-persona round-robin on the `_m` maps
scripts/world_cup.py 10-team NCM + conventional-AI World Cup with dashboard feeds
tests/          407 tests pinning every rule above
```

## Tests

| File | Tests | What it pins |
|---|---|---|
| `test_hexgrid.py` | 17 | coordinates, distances, diagonals, pathfinding |
| `test_movement.py` | 22 | move costs, fuel, blocking, occupied hexes, no second move after moving |
| `test_combat.py` | 30 | the full damage chain, counters, cover, weapon choice, ammo as the firing limit, direct fire after moving and indirect fire only when stationary |
| `test_logistics.py` | 23 | supply domains, resupply, load/unload, fuel attrition |
| `test_capture_economy.py` | 15 | capture progress, income, production, victory by capture, the 50-unit team cap |
| `test_scenarios.py` | 28 | scenario validation, per-map fairness (sites, army, income, supply), hex-true mirroring, the asymmetric map |
| `test_stacking.py` | 11 | aircraft sharing a hex with one land/naval unit, two-layer movement, targeting, unloading, production and views |
| `test_determinism.py` | 11 | replays, seeded dice, hashes, save/load continuation |
| `test_data_integrity.py` | 27 | cross-file data rules + every validator message |
| `test_agent_play.py` | 4 | greedy-agent play, order-log replay, view/API agreement, view-only orders |
| `test_agent_server.py` | 7 | JSON-lines protocol over the wire, structured rejections, save/load continuation, the real `py -m tbs.api.server` process |
| `test_agent_protocol.py` | 8 | tolerant plan parsing: fenced/embedded/bare JSON, end_turn absorption, retryable format errors |
| `test_agent_llm.py` | 12 | OpenAI-compatible request shape, thinking controls, content parts, HTTP retry/backoff vs immediate 4xx, compact prompt, bounded move menus, JSONL logging, a real local HTTP round trip |
| `test_agent_turn.py` | 20 | move-shorthand and friendly option validation, legal-prefix retention, duplicate-order ignoring, a transport move-then-unload plan, repaired retries, exhausted-budget end turn, transport abort, victory without end_turn, replay, stepwise fresh-view play, step budget, one-step whole-turn plans, the async session and its abort |
| `test_agent_council.py` | 27 | unit assignment, filtered views and prompts, merge priority/dedupe/build rules, snapshot-level order filtering, commander brief, partial and total general failures, parallel merge, roster JSON, council replay |
| `test_agent_codec.py` | 25 | compact board text budget and canonical order, every legal order round-trip, engine path equivalence, loaded cargo flags, malformed text rejection, a stepwise agent whose orders travel through the codec |
| `test_amphibious.py` | 5 | landmass ids, stranded marking and ferry hints, a scripted board → sail → enemy-island landing, large-map ferry anchors, and the codec/LLM exposure of the crossing |
| `test_agent_match.py` | 4 | greedy self-play from the view alone replays exactly; the match CLI writes a save and order log; a real HTTP LLM side drives the full CLI; a council side consults commander and generals |
| `test_agent_league.py` | 8 | World Cup draw and pot split, five-round schedule with byes and balanced sides, group map rotation, standings and head-to-head tiebreaks, semifinal/final bracket, knockout board/seed resolution, dashboard metadata and report payload |
| `test_world_cup_runner.py` | 2 | the model-free 10-team setup, five-map group rotation, 24-match state and the asymmetric-knockout opt-in |
| `test_ui_layout.py` | 13 | hex/pixel round trip, centring, fit-to-view for every bundled map, HQ-focused opening camera for large maps, hit-testing, zoom, the reachable-area outline, and a guard that `engine/` imports neither pygame, the UI nor the agent layer |
| `test_ui_menu.py` | 8 | menu navigation skips disabled entries, wraps around, and scrolls a window |
| `test_ui_picker.py` | 5 | scenario titles/metadata, thumbnail sampling, graceful handling of broken maps |
| `test_ui_info.py` | 5 | hover text for terrain, owners, units, ammo, rearm and status |
| `test_large_maps.py` | 12 | the `_m` pack dimensions, ten-times-area claim, 50-unit cap, forty-turn replays, and the large-map prompt/codec budgets |
| `test_ui_buttons.py` | 3 | toolbar layout, overflow and hit-testing |
| `test_ui_interaction.py` | 22 | select/move/attack/build/action-menu/weapon-picker held to the engine's own options; cursor and attack region; move-then-attack for direct weapons; blocked-attack reasons; a UI-played turn replays from its orders |
| `test_ui_app.py` | 27 | the real pygame loop on SDL's dummy driver: draw, menus, the titled scenario picker with preview, hover, toolbar, click-to-move, cursor, action menu, status, pause, end turn, build menu, zoom, a local-LLM turn in a worker thread (thinking overlay, input guard, transport-error recovery, side picker), an injected stepwise agent getting a fresh view per batch, and a council menu/turn |

## Next

- **M8:** done — a local OpenAI-compatible model plays a full game. The earlier
  74B Nemotron run reached a turn-60 draw and replayed exactly; the current
  league baseline is Qwen3.8-27B IQ3_S. Play it in **New Game vs LLM**.
- **M10:** done — the council adds a commander pre-brief, five role generals with
  disjoint unit slices, a deterministic priority merger, and graceful partial
  failures. Play it in **New Game vs Council** or through
  `py -m tbs.agents.match --b council`.
- **M11:** done — five maps (three fair, one asymmetric, plus the intro),
  metadata and thumbnails in a scrolling picker, an explicit default scenario,
  fit-to-view cameras and per-map fairness tests.
- **M11b:** done — five `_m` maps at about ten times the tile area, the 50-unit
  per-team cap, an HQ-focused opening camera with overview fit, a bounded
  large-map LLM prompt, and a dedicated NCM persona league command.
- **NCM experiment:** done — the [NCM fork](https://github.com/marcoaurelioguerrap/NCM_fork)
  trains a 53M NCM player for this engine and releases a base checkpoint. The
  60-game all-AI league against Qwen3.8-27B IQ3_S is in
  [`docs/all_ai_league`](docs/all_ai_league); the league used persona checkpoints,
  not the base download.
- **M9:** fog of war and scenario configuration.
- **M12 (proposed):** map authoring tools — a `mapcheck` linter for supply
  connectivity and per-side economy, then a deterministic symmetric map
  generator exposed as "Random Skirmish".
- **M3 polish:** combat animations, snapshot undo and a minimap.

## License

MIT. See [LICENSE](LICENSE). The NCM fork is a separate MIT-licensed derivative
of [HazyResearch/ncm](https://github.com/HazyResearch/ncm); this project is not
an official Stanford or HazyResearch release.
