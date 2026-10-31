# TBS World Cup 2026

**10 teams · 2 groups · 24/24 matches played**

The five NCM personas, four randomized conventional AIs and the greedy baseline play a seeded two-group World Cup on the new `_m` maps. Group matches award 3/1/0 points; the top two in each group reach the cross-group semifinals. `PF`/`PA` are board scores (2 × sites + live units + HP/100); at the turn cap they are the goal difference, and a knockout game level on board score is decided by the better group seed.

- Generator: `mixed_generator.pt`
- Decode: `legal`
- Prefix length: `4096`
- Turn cap: `30` · steps per turn: `10`
- Tournament seed: `20261025`
- Dashboard batch: `personas/world_cup_20261025`

## Draw

| Group | Teams |
|---|---|
| A | NCM Anvil, NCM Line, Conventional 900002, Conventional 900001, Conventional 900000 |
| B | NCM Blitz, NCM Siege, NCM Vanguard, Conventional 900003, Conventional Greedy |

Group map order: `last_stand_m`, `border_skirmish_m`, `archipelago_m`, `urban_delta_m`, `mountain_pass_m`.

`last_stand_m` is the asymmetric attack/defence map; it appears in the group rotation with sides alternating, while the knockout uses the four hex-true mirrored maps.

## Group stage

### Group A

| Pos | Team | Kind | P | W | D | L | PF | PA | PD | Pts |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | NCM Anvil | ncm | 4 | 0 | 4 | 0 | 6907.8 | 6851.3 | +56.5 | 4 |
| 2 | Conventional 900002 | heuristic | 4 | 0 | 4 | 0 | 6709.1 | 6679.4 | +29.7 | 4 |
| 3 | Conventional 900001 | heuristic | 4 | 0 | 4 | 0 | 7420.4 | 7418.4 | +1.9 | 4 |
| 4 | Conventional 900000 | heuristic | 4 | 0 | 4 | 0 | 6945.3 | 6946.5 | -1.2 | 4 |
| 5 | NCM Line | ncm | 4 | 0 | 4 | 0 | 5984.8 | 6071.7 | -86.9 | 4 |

### Group B

| Pos | Team | Kind | P | W | D | L | PF | PA | PD | Pts |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | NCM Siege | ncm | 4 | 1 | 3 | 0 | 5974.9 | 5979.2 | -4.3 | 6 |
| 2 | NCM Vanguard | ncm | 4 | 0 | 4 | 0 | 6728.7 | 6593.1 | +135.6 | 4 |
| 3 | NCM Blitz | ncm | 4 | 0 | 4 | 0 | 6767.1 | 6790.3 | -23.2 | 4 |
| 4 | Conventional Greedy | greedy | 4 | 0 | 4 | 0 | 6815.6 | 6847.7 | -32.1 | 4 |
| 5 | Conventional 900003 | heuristic | 4 | 0 | 3 | 1 | 7340.9 | 7416.9 | -76.0 | 3 |

## Group results

### Group A

| Round | Map | Home | Score | Away | Winner | Decided by | Turns |
|---|---|---|---|---|---|---|---|
| 1 | last_stand_m | Conventional 900000 | 1638.1 - 1581.8 | NCM Line | draw | draw | 31 |
| 1 | last_stand_m | Conventional 900002 | 1631.6 - 1588.4 | Conventional 900001 | draw | draw | 31 |
| 2 | border_skirmish_m | Conventional 900000 | 1101.1 - 1122.9 | NCM Anvil | draw | draw | 31 |
| 2 | border_skirmish_m | NCM Line | 1085.3 - 1080.5 | Conventional 900002 | draw | draw | 31 |
| 3 | archipelago_m | Conventional 900001 | 2420.2 - 2438.0 | NCM Anvil | draw | draw | 31 |
| 3 | archipelago_m | Conventional 900002 | 2418.0 - 2424.0 | Conventional 900000 | draw | draw | 31 |
| 4 | urban_delta_m | NCM Anvil | 1581.7 - 1579.1 | Conventional 900002 | draw | draw | 31 |
| 4 | urban_delta_m | NCM Line | 1566.7 - 1587.9 | Conventional 900001 | draw | draw | 31 |
| 5 | mountain_pass_m | NCM Anvil | 1765.2 - 1751.0 | NCM Line | draw | draw | 31 |
| 5 | mountain_pass_m | Conventional 900001 | 1823.8 - 1782.1 | Conventional 900000 | draw | draw | 31 |

### Group B

| Round | Map | Home | Score | Away | Winner | Decided by | Turns |
|---|---|---|---|---|---|---|---|
| 1 | last_stand_m | Conventional Greedy | 1602.0 - 1568.9 | NCM Siege | draw | draw | 31 |
| 1 | last_stand_m | NCM Vanguard | 1645.2 - 1600.6 | Conventional 900003 | draw | draw | 31 |
| 2 | border_skirmish_m | Conventional Greedy | 1055.1 - 1079.3 | NCM Blitz | draw | draw | 31 |
| 2 | border_skirmish_m | NCM Siege | 1064.9 - 1086.0 | NCM Vanguard | draw | draw | 31 |
| 3 | archipelago_m | Conventional 900003 | 2420.6 - 2407.1 | NCM Blitz | draw | draw | 31 |
| 3 | archipelago_m | NCM Vanguard | 2433.0 - 2384.9 | Conventional Greedy | draw | draw | 31 |
| 4 | urban_delta_m | NCM Blitz | 1542.7 - 1564.5 | NCM Vanguard | draw | draw | 31 |
| 4 | urban_delta_m | NCM Siege | 1591.0 - 1553.3 | Conventional 900003 | NCM Siege | victory | 30 |
| 5 | mountain_pass_m | NCM Blitz | 1738.0 - 1750.1 | NCM Siege | draw | draw | 31 |
| 5 | mountain_pass_m | Conventional 900003 | 1766.5 - 1773.6 | Conventional Greedy | draw | draw | 31 |

## Knockout

### Semifinals

| Round | Map | Home | Score | Away | Winner | Decided by | Turns |
|---|---|---|---|---|---|---|---|
| 1 | archipelago_m | NCM Anvil | 2427.9 - 2453.3 | NCM Vanguard | NCM Vanguard | board | 31 |
| 1 | border_skirmish_m | NCM Siege | 1097.7 - 1085.6 | Conventional 900002 | NCM Siege | board | 31 |

### Third place

| Round | Map | Home | Score | Away | Winner | Decided by | Turns |
|---|---|---|---|---|---|---|---|
| 2 | urban_delta_m | NCM Anvil | 1574.8 - 1583.7 | Conventional 900002 | Conventional 900002 | board | 31 |

### Final

| Round | Map | Home | Score | Away | Winner | Decided by | Turns |
|---|---|---|---|---|---|---|---|
| 2 | mountain_pass_m | NCM Siege | 1739.3 - 1749.6 | NCM Vanguard | NCM Vanguard | board | 31 |

## Podium

- **Champion:** NCM Vanguard
- **Runner-up:** NCM Siege
- **Third place:** Conventional 900002

## NCM vs conventional AI

| Split | Games | NCM wins | Conventional wins | Draws | NCM score | Conventional score |
|---|---|---|---|---|---|---|
| NCM vs conventional | 14 | 2 | 1 | 11 | 22773.4 | 22692.5 |
| NCM vs NCM | 6 | - | - | 4 | - | - |
| Conventional vs conventional | 4 | - | - | 4 | - | - |

## NCM quality

| Team | W-D-L | Pts | Games | Orders | Legal% | Fallback% | Tactical% | Model% | Action mix |
|---|---|---|---|---|---|---|---|---|---|
| NCM Blitz | 0-4-0 | 4 | 4 | 1200 | 100% | 0% | 96% | 100% | attack 19%, build 1%, capture 0%, hold 1%, move 78% |
| NCM Anvil | 0-4-2 | 4 | 6 | 1800 | 100% | 0% | 36% | 100% | attack 6%, build 3%, hold 4%, load 0%, move 86% |
| NCM Line | 0-4-0 | 4 | 4 | 1200 | 100% | 0% | 99% | 100% | attack 12%, build 1%, capture 2%, hold 0%, move 84% |
| NCM Vanguard | 2-4-0 | 10 | 6 | 1790 | 100% | 0% | 99% | 100% | attack 18%, build 5%, capture 1%, hold 0%, move 76% |
| NCM Siege | 2-3-1 | 9 | 6 | 1791 | 100% | 0% | 99% | 100% | attack 20%, build 1%, capture 1%, hold 1%, move 78% |

## Conventional AI

| Team | Kind | W-D-L | Pts | Games | Orders | Action mix | Builds |
|---|---|---|---|---|---|---|---|
| Conventional 900000 | heuristic | 0-4-0 | 4 | 4 | 1190 | attack 4%, build 3%, hold 51%, load 0%, move 42%, unload 0% | tank 26, artillery 10, aa_missile 4 |
| Conventional 900001 | heuristic | 0-4-0 | 4 | 4 | 1200 | attack 8%, build 4%, hold 50%, load 0%, move 38%, unload 0% | artillery 35, aa_missile 2, infantry 6 |
| Conventional 900002 | heuristic | 1-4-1 | 7 | 6 | 1790 | attack 5%, build 1%, capture 0%, hold 63%, load 1%, move 28%, unload 1% | artillery 9, aa_missile 5, infantry 5 |
| Conventional 900003 | heuristic | 0-3-1 | 3 | 4 | 1190 | attack 2%, build 3%, hold 56%, load 1%, move 37%, unload 1% | artillery 11, tank 18, aa_missile 2 |
| Conventional Greedy | greedy | 0-4-0 | 4 | 4 | 1320 | attack 20%, move 80% | - |

## Maps

| Map | Games | Decisive | Draws | Avg turns | Avg board-score gap |
|---|---|---|---|---|---|
| archipelago_m | 5 | 1 | 4 | 31.0 | 22.1 |
| border_skirmish_m | 5 | 1 | 4 | 31.0 | 16.8 |
| last_stand_m | 4 | 0 | 4 | 31.0 | 44.3 |
| mountain_pass_m | 5 | 1 | 4 | 31.0 | 17.1 |
| urban_delta_m | 5 | 2 | 3 | 30.8 | 18.4 |

## Reproducibility

- Replay verified: **24/24** completed matches.
- Match saves: `C:\AIs\NCM\artifacts\tbs\matches`
- Dashboard report: `C:\AIs\NCM\artifacts\tbs\personas\world_cup_20261025`
- Resume state: `C:\Users\marco\OneDrive\Documentos\opencode_code\turn_based_strategy_vibe\docs\world_cup\tournament.json`

Use `--resume` to continue an interrupted tournament and `--report-only` to rebuild this report without loading the models.
