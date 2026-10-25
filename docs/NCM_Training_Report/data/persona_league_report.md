# Persona league

10 games on 1 maps, fallback `persona`, generator `generator.pt`, prefix `2048`, retries `6`.

| persona | W-L-U | win% | score | attack% | build% | hold% | exact% | model% | fallback% | context% | enemy dist | style dist | model style |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aggressive | 0-0-4 | 0% | -2.7 | 15% | 4% | 50% | 10% | 82% | 18% | 0% | 3.1 | 0.20 | 0.16 |
| defensive | 0-0-4 | 0% | -22.2 | 18% | 4% | 44% | 14% | 75% | 25% | 0% | 3.5 | 0.14 | 0.11 |
| infantry | 0-0-4 | 0% | +12.5 | 13% | 4% | 48% | 12% | 79% | 21% | 0% | 3.7 | 0.18 | 0.14 |
| armor | 0-0-4 | 0% | +1.3 | 12% | 4% | 46% | 6% | 71% | 29% | 0% | 2.9 | 0.17 | 0.11 |
| artillery | 0-0-4 | 0% | +11.1 | 15% | 3% | 52% | 5% | 68% | 32% | 0% | 4.0 | 0.22 | 0.15 |

## Fallback reasons

- **aggressive**: `no-legal-sample` 248 (100%)
  repairs: action 547, same-unit 245, unit 149, nearest 44
- **defensive**: `no-legal-sample` 259 (100%)
  repairs: action 366, same-unit 138, unit 113, nearest 24
- **infantry**: `no-legal-sample` 340 (100%)
  repairs: action 571, same-unit 261, unit 178, nearest 72
- **armor**: `no-legal-sample` 396 (100%)
  repairs: action 379, same-unit 230, unit 161, nearest 117
- **artillery**: `no-legal-sample` 423 (100%)
  repairs: action 486, same-unit 168, unit 115, nearest 57

## Fallback reasons by map

- **border_skirmish**: aggressive: no-legal-sample 248; armor: no-legal-sample 396; artillery: no-legal-sample 423; defensive: no-legal-sample 259; infantry: no-legal-sample 340

## Build mix

- **aggressive**: {'infantry': 21, 'aa_missile': 39, 'transport_ship': 1}
  model-only: {'infantry': 21, 'aa_missile': 39, 'transport_ship': 1}
- **defensive**: {'aa_missile': 17, 'infantry': 23, 'artillery': 3}
  model-only: {'aa_missile': 17, 'infantry': 23, 'artillery': 3}
- **infantry**: {'infantry': 66}
  model-only: {'infantry': 66}
- **armor**: {'tank': 33, 'artillery': 3, 'infantry': 23}
  model-only: {'tank': 33, 'artillery': 3, 'infantry': 23}
- **artillery**: {'infantry': 10, 'aa_missile': 7, 'artillery': 24, 'tank': 2}
  model-only: {'infantry': 10, 'aa_missile': 7, 'artillery': 24, 'tank': 2}

## Pairwise wins (row vs column)

| | aggressive | defensive | infantry | armor | artillery |
|---|---|---|---|---|---|
| **aggressive** | - | 0-0 (1U) | 0-0 (1U) | 0-0 (1U) | 0-0 (1U) |
| **defensive** | 0-0 (1U) | - | 0-0 (1U) | 0-0 (1U) | 0-0 (1U) |
| **infantry** | 0-0 (1U) | 0-0 (1U) | - | 0-0 (1U) | 0-0 (1U) |
| **armor** | 0-0 (1U) | 0-0 (1U) | 0-0 (1U) | - | 0-0 (1U) |
| **artillery** | 0-0 (1U) | 0-0 (1U) | 0-0 (1U) | 0-0 (1U) | - |
