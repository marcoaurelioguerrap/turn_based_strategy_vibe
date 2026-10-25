"""The turn coordinator: legal prefix, repair request, and an unconditional end."""

from __future__ import annotations

import json

import pytest

from tbs.agents.greedy import GreedyAgent
from tbs.agents.llm import LlmTransportError
from tbs.agents.protocol import AgentPlan, PlanFormatError
from tbs.agents.turn import (
    StepwiseSession,
    normalize_order,
    recent_events,
    run_turn,
    run_turn_stepwise,
)
from tbs.engine import rules
from tbs.engine.errors import ActionError
from tbs.engine.serialize import replay, state_hash
from tbs.engine.state import new_game

from .helpers import (
    SPECS,
    bundled_scenario,
    make_state,
    owned_cells,
    unit,
)


class ScriptedAgent:
    name = "scripted"

    def __init__(self, *plans):
        self.plans = list(plans)
        self.requests = []

    def plan(self, request):
        self.requests.append(request)
        result = self.plans.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def start_bundled():
    state = new_game(bundled_scenario(), SPECS)
    rules.start_game(state)
    return state


def movable_unit(state):
    for unit_id in sorted(state.units_of(state.current_player), key=lambda item: item.id):
        moves = rules.legal_moves(state, unit_id.id)
        if moves:
            return unit_id.id, moves[0]
    raise AssertionError("the bundled scenario has no movable unit")


def test_a_whole_turn_move_shorthand_is_expanded_and_the_turn_ends():
    state = start_bundled()
    player = state.current_player
    unit_id, move = movable_unit(state)
    agent = ScriptedAgent(
        AgentPlan(orders=[{"action": "move", "unit": unit_id, "to": list(move["pos"])}])
    )

    result = run_turn(state, agent)

    assert state.unit(unit_id).pos == tuple(move["pos"])
    assert result.attempts == 1
    assert result.ended_turn is True
    assert result.orders[0]["path"] == move["path"], "the accepted order is the engine order"
    assert result.orders[-1] == {"action": "end_turn"}
    assert state.current_player != player

    rebuilt = replay(bundled_scenario(), json.loads(json.dumps(result.orders)))
    assert state_hash(rebuilt) == state_hash(state), "accepted orders replay exactly"
    assert recent_events(state), "the log tail gives the next planner continuity"


def test_a_rejection_keeps_the_valid_prefix_and_retries_with_feedback():
    state = start_bundled()
    unit_id, move = movable_unit(state)
    other_id = next(
        unit.id for unit in state.units_of(state.current_player) if unit.id != unit_id
    )
    bad = AgentPlan(
        orders=[
            {"action": "move", "unit": unit_id, "to": list(move["pos"])},
            {"action": "attack", "unit": other_id, "target": "Z99"},
        ]
    )
    agent = ScriptedAgent(bad, AgentPlan(orders=[]))

    result = run_turn(state, agent, attempts=2)

    assert state.unit(unit_id).pos == tuple(move["pos"])
    assert len(agent.requests) == 2
    assert agent.requests[1].feedback and "Z99" in agent.requests[1].feedback[0]
    assert len(result.applied) == 1
    assert result.errors and "Z99" in result.errors[0]
    assert result.ended_turn is True


def test_duplicate_orders_for_one_unit_are_ignored_not_rejected():
    state = start_bundled()
    unit_id, move = movable_unit(state)
    order = {"action": "move", "unit": unit_id, "to": list(move["pos"])}
    result = run_turn(state, ScriptedAgent(AgentPlan(orders=[order, order])))
    assert len(result.applied) == 1
    assert result.errors == []
    assert len(result.ignored) == 1
    assert result.ended_turn is True


def test_no_orders_is_a_complete_turn_and_still_ends():
    state = start_bundled()
    player = state.current_player
    result = run_turn(state, ScriptedAgent(AgentPlan(orders=[])))
    assert result.applied == []
    assert result.attempts == 1
    assert result.ended_turn is True
    assert state.current_player != player


def test_format_failures_are_retried_then_end_the_turn():
    state = start_bundled()
    agent = ScriptedAgent(PlanFormatError("not json"), AgentPlan(orders=[]))
    result = run_turn(state, agent, attempts=2)
    assert len(agent.requests) == 2
    assert "not json" in agent.requests[1].feedback[0]
    assert result.attempts == 2
    assert result.ended_turn is True


def test_an_exhausted_retry_budget_still_ends_the_turn():
    state = start_bundled()
    agent = ScriptedAgent(PlanFormatError("not json"), PlanFormatError("still not json"))
    result = run_turn(state, agent, attempts=2)
    assert len(agent.requests) == 2
    assert len(result.errors) == 2
    assert result.ended_turn is True
    assert len(result.orders) == 1 and result.orders[0] == {"action": "end_turn"}


def test_a_transport_failure_propagates_without_advancing_the_game():
    state = start_bundled()
    player = state.current_player
    agent = ScriptedAgent(LlmTransportError("server down"))
    with pytest.raises(LlmTransportError, match="server down"):
        run_turn(state, agent)
    assert state.current_player == player
    assert not state.log, "a transport failure touches nothing"


def test_capturing_the_objective_ends_the_game_without_an_end_turn():
    state = make_state(
        rows=["H."],
        units=[unit("infantry", "A", 0, 0, unit_id="A1")],
        regions=[owned_cells("B", [(0, 0)])],
    )
    state.map.tile((0, 0)).capture_points = 10
    agent = ScriptedAgent(AgentPlan(orders=[{"action": "capture", "unit": "A1"}]))

    result = run_turn(state, agent)

    assert state.status == "victory"
    assert state.winner == "A"
    assert result.ended_turn is False
    assert result.orders == [{"action": "capture", "unit": "A1"}]


def test_move_shorthand_rejects_a_destination_the_engine_does_not_list():
    state = start_bundled()
    unit_id, _move = movable_unit(state)
    with pytest.raises(ActionError, match="no legal move"):
        normalize_order(state, {"action": "move", "unit": unit_id, "to": [99, 99]})


def test_an_engine_path_is_passed_through_unchanged():
    state = start_bundled()
    unit_id, move = movable_unit(state)
    order = {"action": "move", "unit": unit_id, "path": move["path"]}
    assert normalize_order(state, order) is order


# ---------------------------------------------------------------------------
# stepwise turns: fresh view after every batch
# ---------------------------------------------------------------------------


class OneOrderAgent:
    """Turns the greedy whole-turn plan into one order per request."""

    name = "one-order"

    def __init__(self):
        self.greedy = GreedyAgent()
        self.calls = 0

    def plan(self, request):
        self.calls += 1
        whole = self.greedy.plan(request)
        return AgentPlan(orders=whole.orders[:1], commentary="")


class HoldOneAgent:
    """Holds the first unit that may still act, one unit per request."""

    name = "hold-one"

    def __init__(self):
        self.calls = 0

    def plan(self, request):
        self.calls += 1
        for current in request.view["units"]:
            if (
                current["owner"] == request.player
                and current["pos"] is not None
                and not current["acted"]
                and not current["loaded"]
                and not current["new"]
            ):
                return AgentPlan(orders=[{"action": "hold", "unit": current["id"]}])
        return AgentPlan(orders=[])


class EndTurnAgent:
    """Says end_turn on the first request, without touching a unit."""

    name = "end-turn"

    def __init__(self):
        self.calls = 0

    def plan(self, request):
        self.calls += 1
        return AgentPlan(orders=[], end_turn=True)


def test_stepwise_one_order_agent_plays_a_whole_game_and_replays_exactly():
    state = start_bundled()
    agent = OneOrderAgent()
    orders: list[dict] = []
    turns = 0
    while state.status == "ongoing" and turns < 20:
        result = run_turn_stepwise(state, agent, max_steps=40)
        assert result.ended_turn or state.status != "ongoing"
        assert result.attempts <= 40
        orders.extend(result.orders)
        turns += 1

    assert turns >= 2, "the game did not get past the first turn"
    assert len(orders) > 20, "the one-order agent barely acted"
    rebuilt = replay(bundled_scenario(), json.loads(json.dumps(orders)))
    assert state_hash(rebuilt) == state_hash(state), "stepwise orders replay exactly"


class AllUnitsHoldAgent:
    """A whole-turn plan that spends every available unit in one response."""

    name = "all-hold"

    def plan(self, request):
        orders = [
            {"action": "hold", "unit": current["id"]}
            for current in request.view["units"]
            if current["owner"] == request.player
            and current["pos"] is not None
            and not current["acted"]
            and not current["loaded"]
            and not current["new"]
        ]
        return AgentPlan(orders=orders)


def test_stepwise_whole_turn_plan_applies_and_finishes_in_one_step():
    whole_state = start_bundled()
    step_state = start_bundled()
    whole = run_turn(whole_state, AllUnitsHoldAgent())
    step = run_turn_stepwise(step_state, AllUnitsHoldAgent(), max_steps=8)

    assert step.attempts == 1, "nothing was left for a second call"
    assert [item.order for item in step.applied] == [item.order for item in whole.applied]
    assert step.orders == whole.orders
    assert state_hash(step_state) == state_hash(whole_state)


def test_stepwise_budget_exhaustion_still_ends_the_turn():
    state = start_bundled()
    player = state.current_player
    agent = HoldOneAgent()

    result = run_turn_stepwise(state, agent, max_steps=2)

    assert agent.calls == 2
    assert result.attempts == 2
    assert len(result.applied) == 2
    assert all(item.order["action"] == "hold" for item in result.applied)
    assert result.ended_turn is True
    assert state.current_player != player


def test_stepwise_format_failure_is_retried_with_feedback():
    state = start_bundled()
    unit_id, move = movable_unit(state)
    agent = ScriptedAgent(
        PlanFormatError("bad step"),
        AgentPlan(orders=[{"action": "move", "unit": unit_id, "to": list(move["pos"])}]),
    )

    result = run_turn_stepwise(state, agent, max_steps=2)

    assert len(agent.requests) == 2
    assert "bad step" in agent.requests[1].feedback[0]
    assert result.attempts == 2
    assert len(result.applied) == 1
    assert state.unit(unit_id).pos == tuple(move["pos"])
    assert result.ended_turn is True


def test_stepwise_explicit_end_turn_stops_asking():
    state = start_bundled()
    agent = EndTurnAgent()

    result = run_turn_stepwise(state, agent, max_steps=8)

    assert agent.calls == 1
    assert result.applied == []
    assert result.ended_turn is True


def test_stepwise_an_empty_plan_stops_asking():
    state = start_bundled()
    agent = ScriptedAgent(AgentPlan(orders=[]))

    result = run_turn_stepwise(state, agent, max_steps=8)

    assert len(agent.requests) == 1, "an empty plan is a finished turn, not a prompt loop"
    assert result.applied == []
    assert result.ended_turn is True


def test_stepwise_session_driven_manually_matches_the_sync_loop():
    manual_state = start_bundled()
    auto_state = start_bundled()
    session = StepwiseSession(manual_state, AllUnitsHoldAgent(), max_steps=8)
    while True:
        request = session.next_request()
        if request is None:
            break
        session.record_plan(session.agent.plan(request))
        if session.done:
            break
    manual = session.finish()
    auto = run_turn_stepwise(auto_state, AllUnitsHoldAgent(), max_steps=8)

    assert manual.orders == auto.orders
    assert manual.attempts == auto.attempts == 1
    assert manual.ended_turn is True
    assert state_hash(manual_state) == state_hash(auto_state)


def test_stepwise_session_stops_at_the_budget_and_keeps_feedback():
    state = start_bundled()
    session = StepwiseSession(state, HoldOneAgent(), max_steps=2)
    first = session.next_request()
    assert first is not None
    session.record_failure("try again")
    second = session.next_request()
    assert second is not None and second.feedback == ["try again"]
    assert session.ready() is False, "the second call used the last budget slot"
    session.record_plan(HoldOneAgent().plan(second))
    assert session.next_request() is None, "the budget is two calls"
    assert session.steps == 2
    assert session.ready() is False

    result = session.finish()
    assert result.attempts == 2
    assert result.ended_turn is True


def test_stepwise_session_abort_leaves_the_turn_untouched():
    state = start_bundled()
    player = state.current_player
    session = StepwiseSession(state, HoldOneAgent())
    session.abort()

    result = session.finish()

    assert result.ended_turn is False
    assert state.current_player == player
    assert not state.log, "an aborted stepwise turn touches nothing"
