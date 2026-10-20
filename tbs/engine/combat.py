"""Damage maths: integer-only and fully deterministic.

The formula, restated in the README so agents can reason about it:

    base    = weapon.power * multiplier[defender class] // 100
    base    = base * attacker.hp // 100                # attacker condition
    damage  = base * (100 - defender terrain cover) // 100
    damage  = damage * (100 + luck) // 100             # seeded luck, -12..+12 %
    damage  = max(1, damage)

A weapon with a ``0`` multiplier for a damage class cannot hit that class at
all.  Counterattacks use the same maths with :data:`COUNTER_POWER_PCT` of the
defender's power and only happen at distance 1: a unit hit by a ranged
(indirect) weapon neither counterattacks nor is countered, exactly like in
Advance Wars.
"""

from __future__ import annotations

from .data import WeaponSpec
from .hexgrid import hex_distance
from .rng import luck_percent
from .state import GameState, Unit, remove_unit

#: A counterattack deals this percentage of the defender's normal damage.
COUNTER_POWER_PCT = 50


def compute_damage(
    state: GameState,
    attacker: Unit,
    weapon: WeaponSpec,
    defender: Unit,
    *,
    power_pct: int = 100,
    luck_pct: int | None = None,
    phase: str = "attack",
    attacker_hp: int | None = None,
) -> int:
    """Damage one unit deals to another with one weapon.

    Pass ``luck_pct`` to make the result exactly predictable (used by tests and
    by "what would this attack do?" previews); otherwise luck is derived from
    the game seed and the context of this attack.

    ``attacker_hp`` replaces the attacker's condition for this calculation; the
    legal-attack previews use it to describe the counterattack a defender would
    actually answer with once it has taken the predicted damage.
    """
    defender_spec = state.spec_of(defender)
    multiplier = weapon.multipliers.get(defender_spec.damage_class, 0)
    if multiplier <= 0 or defender.pos is None:
        return 0

    condition = attacker.hp if attacker_hp is None else max(0, attacker_hp)
    base = weapon.power * multiplier // 100
    base = base * condition // 100
    if power_pct != 100:
        base = base * power_pct // 100

    cover = state.terrain_of(state.map.tile(defender.pos)).defense_pct
    damage = base * (100 - cover) // 100

    if luck_pct is None:
        if not state.config.get("damage_luck", True):
            luck_pct = 0
        else:
            luck_pct = luck_percent(
                state.seed, "luck", state.turn_number, state.action_seq, phase, attacker.id, defender.id, weapon.id
            )
    damage = damage * (100 + luck_pct) // 100
    return max(1, damage)


def available_weapon(state: GameState, unit: Unit, target: Unit, dist: int) -> WeaponSpec | None:
    """The first weapon of ``unit`` that can legally hit ``target`` at ``dist``."""
    spec = state.spec_of(unit)
    target_class = state.spec_of(target).damage_class
    for weapon in spec.weapons:
        if not weapon.covers(dist):
            continue
        if not weapon.can_hit(target_class):
            continue
        if unit.ammo.get(weapon.id, 0) <= 0:
            continue
        return weapon
    return None


def weapon_options(state: GameState, unit: Unit, target: Unit) -> list[WeaponSpec]:
    """All legal weapons against ``target``, in spec order.

    A unit that has moved this turn may only use direct weapons: indirect
    weapons (``min_range > 1``) need a unit that stayed put, as in Advance Wars.
    """
    if target.pos is None or unit.pos is None:
        return []
    dist = hex_distance(unit.pos, target.pos)
    spec = state.spec_of(unit)
    target_class = state.spec_of(target).damage_class
    return [
        weapon
        for weapon in spec.weapons
        if weapon.covers(dist)
        and weapon.can_hit(target_class)
        and unit.ammo.get(weapon.id, 0) > 0
        and (not unit.moved or weapon.is_direct)
    ]


def preview_damage(
    state: GameState,
    attacker: Unit,
    weapon: WeaponSpec,
    defender: Unit,
    *,
    power_pct: int = 100,
    attacker_hp: int | None = None,
) -> int:
    """Damage without luck — used by "what would this do?" views."""
    return compute_damage(state, attacker, weapon, defender, power_pct=power_pct, luck_pct=0, attacker_hp=attacker_hp)


def resolve_attack(state: GameState, attacker: Unit, defender: Unit, weapon: WeaponSpec) -> list[dict]:
    """Carry out one attack (and the counterattack it provokes), returning events."""
    assert attacker.pos is not None and defender.pos is not None
    dist = hex_distance(attacker.pos, defender.pos)
    events: list[dict] = []

    damage = compute_damage(state, attacker, weapon, defender)
    attacker.ammo[weapon.id] = max(0, attacker.ammo.get(weapon.id, 0) - 1)
    defender.hp -= damage
    events.append(
        {
            "type": "attack",
            "attacker": attacker.id,
            "defender": defender.id,
            "weapon": weapon.id,
            "distance": dist,
            "damage": damage,
            "defender_hp": max(0, defender.hp),
        }
    )

    if defender.hp <= 0:
        events.extend(destroy_unit(state, defender, cause=f"attacked by {attacker.id}"))
        return events

    if dist == 1 and weapon.is_direct:
        counter = available_weapon(state, defender, attacker, 1)
        if counter is not None:
            counter_damage = compute_damage(
                state, defender, counter, attacker, power_pct=COUNTER_POWER_PCT, phase="counter"
            )
            defender.ammo[counter.id] = max(0, defender.ammo.get(counter.id, 0) - 1)
            attacker.hp -= counter_damage
            events.append(
                {
                    "type": "counterattack",
                    "attacker": defender.id,
                    "defender": attacker.id,
                    "weapon": counter.id,
                    "distance": 1,
                    "damage": counter_damage,
                    "defender_hp": max(0, attacker.hp),
                }
            )
            if attacker.hp <= 0:
                events.extend(destroy_unit(state, attacker, cause=f"counterattacked by {defender.id}"))
    return events


def destroy_unit(state: GameState, unit: Unit, *, cause: str) -> list[dict]:
    """Remove a unit from the game and describe what happened.

    Cargo carried by the unit is destroyed with it, as in Advance Wars.
    """
    destroyed = [
        {"type": "destroyed", "unit": unit.id, "owner": unit.owner, "cause": cause},
        *[
            {"type": "destroyed", "unit": cargo_id, "owner": unit.owner, "cause": f"was cargo of {unit.id}"}
            for cargo_id in unit.cargo
        ],
    ]
    remove_unit(state, unit.id)
    return destroyed
