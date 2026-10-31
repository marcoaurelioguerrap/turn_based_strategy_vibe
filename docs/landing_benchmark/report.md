# Amphibious landing benchmark

Scripted `GreedyAgent` on both sides. A landing means a land unit ended on a
different landmass from where it started; an enemy-home landing means it
reached the opponent's HQ island.

| Map | Games | Games with landing | Games reaching enemy home | Avg first landing | Units landed (ever) | Enemy-home units (ever) | Loads | Unloads | Wins | Draws |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| archipelago | 3 | 3 | 3 | 2 | 29 | 29 | 26 | 26 | 0 | 3 |
| archipelago_m | 3 | 3 | 3 | 7 | 22 | 22 | 20 | 20 | 0 | 3 |

## Games

- `archipelago` seed 900000: draw, turn 21, first landing 2, landed-ever 9, enemy-home-ever 9, final-landed 1, loads 8, unloads 8
- `archipelago` seed 900001: draw, turn 21, first landing 2, landed-ever 10, enemy-home-ever 10, final-landed 4, loads 9, unloads 9
- `archipelago` seed 900002: draw, turn 21, first landing 2, landed-ever 10, enemy-home-ever 10, final-landed 1, loads 9, unloads 9
- `archipelago_m` seed 901000: draw, turn 21, first landing 7, landed-ever 7, enemy-home-ever 7, final-landed 2, loads 7, unloads 7
- `archipelago_m` seed 901001: draw, turn 21, first landing 7, landed-ever 8, enemy-home-ever 8, final-landed 2, loads 7, unloads 7
- `archipelago_m` seed 901002: draw, turn 21, first landing 7, landed-ever 7, enemy-home-ever 7, final-landed 2, loads 6, unloads 6
