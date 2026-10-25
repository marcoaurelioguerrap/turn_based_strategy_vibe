"""The M10 council: assignment, filtered prompts, merge rules and failures."""

from __future__ import annotations

import json

import pytest

from tbs.agents.council import (
    CommanderAgent,
    CouncilAgent,
    build_commander_messages,
    merge_plans,
)
from tbs.agents.generals import (
    DEFAULT_GENERALS,
    GeneralSpec,
    assign_units,
    filter_view,
    general_role,
    load_generals,
    select_generals,
)
from tbs.agents.llm import (
    JsonlLogger,
    LlmClient,
    LlmConfig,
    LlmResponseError,
    LlmTransportError,
    build_messages,
)
from tbs.agents.protocol import Agent, AgentPlan, PlanFormatError, TurnRequest
from tbs.agents.turn import apply_plan, build_request, run_turn
from tbs.engine import rules
from tbs.engine.serialize import replay, state_hash
from tbs.engine.state import new_game

from .helpers import SPECS, bundled_scenario


def start_bundled():
    state = new_game(bundled_scenario(), SPECS)
    rules.start_game(state)
    return state


def request_for(state=None, *, player="A", feedback=None):
    state = state or start_bundled()
    return build_request(state, player, feedback)


class FakeMember:
    name = "fake-general"

    def __init__(self, planner=None, error=None):
        self.planner = planner
        self.error = error
        self.requests = []

    def plan(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if self.planner is not None:
            return self.planner(request)
        return AgentPlan()


class FakeCommander:
    name = "fake-commander"

    def __init__(self, text="take the centre", error=None):
        self.text = text
        self.error = error
        self.requests = []

    def brief(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.text


def hold_first(request):
    if not request.allowed_units:
        return AgentPlan()
    return AgentPlan(
        orders=[{"action": "hold", "unit": request.allowed_units[0]}],
        commentary=f"{request.role.split(',')[0]} waits",
    )


def council_with(members, commander=None, **kwargs):
    return CouncilAgent(
        members=members,
        commander=commander,
        generals=kwargs.pop("generals", DEFAULT_GENERALS),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# assignment and filtered views
# ---------------------------------------------------------------------------


def test_every_acting_unit_is_assigned_to_exactly_one_general():
    view = request_for().view
    assignments = assign_units(view, DEFAULT_GENERALS)
    acting = {
        unit_id
        for unit_id, summary in view["unit_actions"].items()
        if not summary.get("acted")
        and not summary.get("loaded_in")
        and not summary.get("new_unit")
    }
    assigned = [unit_id for unit_ids in assignments.values() for unit_id in unit_ids]
    assert sorted(assigned) == sorted(acting)
    assert len(assigned) == len(set(assigned)), "no unit may be commanded twice"
    assert "A4" in assignments["vanguard"], "tanks go to the armoured general"
    assert "A2" in assignments["anvil"], "infantry go to the line"
    assert "A8" in assignments["talon"], "air goes to the sky marshal"
    assert "A9" in assignments["fleet"], "naval goes to the admiral"
    assert "A6" in assignments["quartermaster"], "trucks go to logistics"


def test_unknown_unit_types_fall_to_the_reserve_general():
    view = request_for().view
    view = json.loads(json.dumps(view))
    view["units"].append({"id": "A99", "type": "prototype", "owner": "A"})
    view["unit_actions"]["A99"] = {"hold": True}
    assignments = assign_units(view, DEFAULT_GENERALS)
    assert "A99" in assignments["quartermaster"]


def test_a_general_only_sees_its_own_actions_and_builds_when_allowed():
    view = request_for().view
    filtered = filter_view(view, ["A4"], can_build=False)
    assert set(filtered["unit_actions"]) == {"A4"}
    assert "build_options" not in filtered
    assert filtered["units"] == view["units"], "the whole battlefield stays visible"
    builder = filter_view(view, [], can_build=True)
    assert builder["unit_actions"] == {}
    assert builder["build_options"] == view["build_options"]


def test_the_prompt_carries_doctrine_brief_and_only_the_general_s_actions():
    general = DEFAULT_GENERALS[0]
    view = request_for().view
    request = TurnRequest(
        player="A",
        view=filter_view(view, ["A4"]),
        role=general_role(general),
        allowed_units=["A4"],
        brief="break the left flank",
    )
    messages = build_messages(request)
    system = messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    assert "YOUR COMMAND" in system
    assert general.doctrine in system
    assert "break the left flank" in system
    assert "one of the generals" in system
    assert payload["your_units"] == ["A4"]
    assert payload["brief"] == "break the left flank"
    assert set(payload["view"]["unit_actions"]) == {"A4"}

    single = build_messages(TurnRequest(player="A", view=view))
    assert "YOUR COMMAND" not in single[0]["content"]
    assert "one of the generals" not in single[0]["content"]


def test_selecting_generals_keeps_priority_order_and_rejects_unknown_ids():
    selected = select_generals(DEFAULT_GENERALS, ["quartermaster", "vanguard"])
    assert [general.id for general in selected] == ["vanguard", "quartermaster"]
    with pytest.raises(ValueError, match="unknown general.*nobody"):
        select_generals(DEFAULT_GENERALS, ["nobody"])


# ---------------------------------------------------------------------------
# merge rules
# ---------------------------------------------------------------------------


def test_merge_keeps_priority_dedupes_units_and_keeps_focus_fire():
    view = request_for().view
    senior = GeneralSpec("senior", "Senior", "", "first", ("tank",), priority=1)
    junior = GeneralSpec("junior", "Junior", "", "second", ("infantry",), priority=2)
    results = [
        (senior, AgentPlan(orders=[{"action": "hold", "unit": "A4"}], commentary="hold")),
        (
            junior,
            AgentPlan(
                orders=[
                    {"action": "hold", "unit": "A4"},
                    {"action": "hold", "unit": "A2"},
                    {"action": "end_turn"},
                ],
                commentary="advance",
            ),
        ),
    ]
    merged, dropped = merge_plans(results, {"senior": ["A4"], "junior": ["A4", "A2"]}, view)
    assert merged.orders == [
        {"action": "hold", "unit": "A4"},
        {"action": "hold", "unit": "A2"},
    ]
    assert any("already ordered" in line for line in dropped)
    assert merged.end_turn is False
    assert "Senior: hold" in merged.commentary
    assert "Junior: advance" in merged.commentary


def test_merge_keeps_two_units_attacking_the_same_target():
    view = request_for().view
    left = GeneralSpec("left", "Left", "", "left", ("artillery",), priority=1)
    right = GeneralSpec("right", "Right", "", "right", ("destroyer",), priority=2)
    results = [
        (left, AgentPlan(orders=[{"action": "attack", "unit": "A5", "target": "B3"}])),
        (right, AgentPlan(orders=[{"action": "attack", "unit": "A9", "target": "B3"}])),
    ]
    merged, _dropped = merge_plans(results, {"left": ["A5"], "right": ["A9"]}, view)
    assert [order["unit"] for order in merged.orders] == ["A5", "A9"]


def test_merge_drops_orders_the_view_already_says_are_impossible():
    view = request_for().view
    general = GeneralSpec("vanguard", "Vanguard", "", "attack", ("tank",), priority=1)
    results = [
        (
            general,
            AgentPlan(
                orders=[
                    {"action": "attack", "unit": "A4", "target": "B4"},
                    {"action": "move", "unit": "A4", "to": [99, 99]},
                ]
            ),
        )
    ]
    merged, dropped = merge_plans(results, {"vanguard": ["A4"]}, view)
    assert merged.orders == []
    assert any("no legal attack" in line for line in dropped)
    assert any("cannot reach" in line for line in dropped)


def test_merge_keeps_a_legal_attack_from_the_view():
    view = request_for().view
    general = GeneralSpec("vanguard", "Vanguard", "", "attack", ("artillery",), priority=1)
    merged, dropped = merge_plans(
        [(general, AgentPlan(orders=[{"action": "attack", "unit": "A5", "target": "B3"}]))],
        {"vanguard": ["A5"]},
        view,
    )
    assert merged.orders == [{"action": "attack", "unit": "A5", "target": "B3"}]
    assert dropped == []


def test_only_a_builder_may_build_and_only_one_build_per_site():
    view = request_for().view
    option = view["build_options"][0]
    general = GeneralSpec("quartermaster", "Q", "", "build", (), priority=1, can_build=True)
    soldier = GeneralSpec("vanguard", "V", "", "fight", ("tank",), priority=2)
    build = {"action": "build", "site": option["site"], "unit": option["unit"]}
    merged, dropped = merge_plans(
        [
            (general, AgentPlan(orders=[build, dict(build)])),
            (soldier, AgentPlan(orders=[dict(build)])),
        ],
        {"quartermaster": [], "vanguard": ["A4"]},
        view,
    )
    assert merged.orders == [build]
    assert any("Quartermaster's job" in line for line in dropped)
    assert any("second build" in line for line in dropped)


# ---------------------------------------------------------------------------
# the council as an agent
# ---------------------------------------------------------------------------


def test_a_council_turn_consults_every_active_general_and_replays():
    state = start_bundled()
    commander = FakeCommander("push the left flank")
    members = {general.id: FakeMember(hold_first) for general in DEFAULT_GENERALS}
    council = council_with(members, commander=commander, use_commander=True)

    result = run_turn(state, council, attempts=1)

    assert result.ended_turn is True
    assert len(commander.requests) == 1
    assert len(result.applied) == len(DEFAULT_GENERALS)
    assert all(member.requests for member in members.values())
    for general in DEFAULT_GENERALS:
        request = members[general.id].requests[0]
        assert request.role is not None and general.doctrine in request.role
        assert request.allowed_units is not None
        assert request.brief == "push the left flank"
        assert set(request.view["unit_actions"]) <= set(request.allowed_units)
    assert council.last_stats["calls"] == 1 + len(DEFAULT_GENERALS)
    assert council.last_stats["failures"] == []
    assert any("Commander: push the left flank" in text for text in result.commentary)

    rebuilt = replay(bundled_scenario(), json.loads(json.dumps(result.orders)))
    assert state_hash(rebuilt) == state_hash(state), "accepted orders replay exactly"


def test_the_council_is_an_agent_by_the_protocol():
    members = {general.id: FakeMember() for general in DEFAULT_GENERALS}
    council = council_with(members, use_commander=False)
    assert isinstance(council, Agent)
    assert council.name == "council"


def test_a_general_format_failure_is_skipped_and_the_rest_still_play():
    state = start_bundled()
    members = {general.id: FakeMember(hold_first) for general in DEFAULT_GENERALS}
    members["talon"] = FakeMember(error=PlanFormatError("no JSON"))
    council = council_with(members, use_commander=False)

    result = run_turn(state, council, attempts=1)

    assert result.ended_turn is True
    assert council.last_stats["failures"] == ["talon"]
    assert "unavailable: Talon" in council.last_stats["dropped"] or any(
        "unavailable: Talon" in text for text in result.commentary
    )
    assert len(result.applied) == len(DEFAULT_GENERALS) - 1


def test_all_generals_failing_raises_a_retryable_error():
    members = {
        general.id: FakeMember(error=PlanFormatError("bad reply")) for general in DEFAULT_GENERALS
    }
    council = council_with(members, use_commander=False)
    with pytest.raises(PlanFormatError, match="bad reply"):
        council.plan(request_for())


def test_one_transport_failure_does_not_abort_a_working_council():
    state = start_bundled()
    members = {general.id: FakeMember(hold_first) for general in DEFAULT_GENERALS}
    members["fleet"] = FakeMember(error=LlmTransportError("server busy"))
    council = council_with(members, use_commander=False)

    result = run_turn(state, council, attempts=1)

    assert result.ended_turn is True
    assert council.last_stats["transport_failures"] == ["fleet"]
    assert len(result.applied) == len(DEFAULT_GENERALS) - 1


def test_all_transport_failures_propagate_to_the_ui():
    members = {
        general.id: FakeMember(error=LlmTransportError("server down"))
        for general in DEFAULT_GENERALS
    }
    council = council_with(members, use_commander=False)
    with pytest.raises(LlmTransportError, match="server down"):
        council.plan(request_for())


def test_a_commander_failure_is_ignored_and_generals_play_without_a_brief():
    state = start_bundled()
    commander = FakeCommander(error=LlmTransportError("brief failed"))
    members = {general.id: FakeMember(hold_first) for general in DEFAULT_GENERALS}
    council = council_with(members, commander=commander, use_commander=True)

    result = run_turn(state, council, attempts=1)

    assert result.ended_turn is True
    assert commander.requests
    assert all(member.requests[0].brief == "" for member in members.values())
    assert council.last_stats["calls"] == len(DEFAULT_GENERALS)


def test_no_commander_mode_saves_one_call():
    state = start_bundled()
    members = {general.id: FakeMember(hold_first) for general in DEFAULT_GENERALS}
    council = council_with(members, use_commander=False)

    run_turn(state, council, attempts=1)

    assert council.last_stats["calls"] == len(DEFAULT_GENERALS)


def test_a_build_only_quartermaster_is_still_consulted():
    view = request_for().view
    option = view["build_options"][0]
    view = json.loads(json.dumps(view))
    view["unit_actions"] = {"A4": view["unit_actions"]["A4"]}

    members = {
        "vanguard": FakeMember(hold_first),
        "quartermaster": FakeMember(
            lambda request: AgentPlan(
                orders=[{"action": "build", "site": option["site"], "unit": option["unit"]}]
            )
        ),
    }
    council = council_with(
        members,
        generals=select_generals(DEFAULT_GENERALS, ["vanguard", "quartermaster"]),
        use_commander=False,
    )
    plan = council.plan(TurnRequest(player="A", view=view))
    assert [order["action"] for order in plan.orders] == ["hold", "build"]
    assert members["quartermaster"].requests[0].allowed_units == []


def test_parallel_mode_merges_in_priority_order():
    state = start_bundled()
    members = {general.id: FakeMember(hold_first) for general in DEFAULT_GENERALS}
    council = council_with(members, use_commander=False, parallel=True)

    result = run_turn(state, council, attempts=1)

    assert result.ended_turn is True
    units = [item.order["unit"] for item in result.applied]
    assert len(units) == len(DEFAULT_GENERALS)
    assert units == [
        "A4",  # Vanguard
        "A1",  # Anvil
        "A8",  # Talon
        "A9",  # Fleet
        "A6",  # Quartermaster
    ], "the merge is deterministic despite thread completion order"


def test_no_active_units_returns_an_empty_plan_without_calling_anyone():
    called = []

    def should_not_run(request):
        called.append(request)
        return AgentPlan()

    members = {general.id: FakeMember(should_not_run) for general in DEFAULT_GENERALS}
    council = council_with(members, use_commander=False)
    view = {
        "turn": 1,
        "status": "ongoing",
        "current_player": "A",
        "units": [{"id": "A1", "type": "tank", "owner": "A"}],
        "unit_actions": {"A1": {"acted": True}},
    }
    plan = council.plan(TurnRequest(player="A", view=view))
    assert plan.orders == []
    assert called == []
    assert council.last_stats["calls"] == 0


# ---------------------------------------------------------------------------
# commander prompt, logging and roster loading
# ---------------------------------------------------------------------------


def test_the_commander_prompt_contains_the_view_and_a_short_instruction():
    request = request_for()
    messages = build_commander_messages(request)
    assert "supreme commander" in messages[0]["content"]
    assert "at most 60 words" in messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    assert payload["turn"] == request.view["turn"]
    assert "view" in payload


def test_the_real_commander_returns_a_trimmed_brief_and_logs_both_sides(tmp_path):
    def transport(*_args):
        return json.dumps(
            {"choices": [{"message": {"content": "  Push   the left.  "}}]}
        ).encode()

    logger = JsonlLogger(tmp_path / "council.jsonl")
    commander = CommanderAgent(
        LlmClient(LlmConfig(model="m"), transport=transport), logger=logger
    )
    assert commander.brief(request_for()) == "Push the left."
    records = [
        json.loads(line) for line in (tmp_path / "council.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["kind"] for record in records] == ["commander_request", "commander_response"]
    assert records[0]["agent"] == "commander"
    assert records[1]["brief"] == "Push the left."


def test_an_empty_commander_reply_is_a_retryable_output_error():
    def transport(*_args):
        return json.dumps({"choices": [{"message": {"content": "   "}}]}).encode()

    commander = CommanderAgent(LlmClient(LlmConfig(model="m"), transport=transport))
    with pytest.raises(LlmResponseError, match="empty message"):
        commander.brief(request_for())


def test_a_json_roster_can_replace_the_default_generals(tmp_path):
    path = tmp_path / "generals.json"
    path.write_text(
        json.dumps(
            {
                "generals": [
                    {
                        "id": "spear",
                        "name": "Spear",
                        "title": "the point",
                        "doctrine": "Attack.",
                        "unit_types": ["tank"],
                        "priority": 5,
                        "can_build": True,
                        "reserve": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    roster = load_generals(path)
    assert len(roster) == 1
    assert roster[0].id == "spear"
    assert roster[0].can_build is True


def test_a_bad_roster_names_the_problem(tmp_path):
    path = tmp_path / "generals.json"
    path.write_text('{"generals":[{"id":"a"},{"id":"a","doctrine":"x"}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="needs a non-empty doctrine"):
        load_generals(path)
    path.write_text('{"generals":[{"id":"a","doctrine":"x"},{"id":"a","doctrine":"y"}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate id"):
        load_generals(path)


def test_apply_plan_accepts_the_council_plan_and_ends_the_turn():
    state = start_bundled()
    members = {general.id: FakeMember(hold_first) for general in DEFAULT_GENERALS}
    council = council_with(members, use_commander=False)
    plan = council.plan(build_request(state, "A"))
    result = apply_plan(state, plan)
    assert not result.errors
    rules.perform(state, {"action": "end_turn"})
    assert state.current_player == "B"
