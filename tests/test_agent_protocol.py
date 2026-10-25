"""The agent-facing plan parser: model noise in, validated order lists out."""

from __future__ import annotations

import pytest

from tbs.agents.protocol import AgentPlan, PlanFormatError, parse_plan


def test_clean_plan_round_trips_orders_and_commentary():
    plan = parse_plan(
        '{"orders":[{"action":"move","unit":"A2","to":[3,4]},'
        '{"action":"attack","unit":"A3","target":"B2"}],"commentary":"advance"}'
    )
    assert plan == AgentPlan(
        orders=[
            {"action": "move", "unit": "A2", "to": [3, 4]},
            {"action": "attack", "unit": "A3", "target": "B2"},
        ],
        commentary="advance",
    )


def test_fenced_json_is_accepted():
    reply = '```json\n{"orders":[{"action":"hold","unit":"A1"}]}\n```'
    assert parse_plan(reply).orders == [{"action": "hold", "unit": "A1"}]
    one_line = '```json {"orders":[{"action":"hold","unit":"A1"}]}```'
    assert parse_plan(one_line).orders == [{"action": "hold", "unit": "A1"}]


def test_json_embedded_in_prose_is_accepted():
    reply = 'Sure. Here is my plan:\n{"orders":[{"action":"capture","unit":"A1"}]}\nGood luck!'
    assert parse_plan(reply).orders == [{"action": "capture", "unit": "A1"}]


def test_bare_list_and_single_order_are_accepted():
    assert parse_plan('[{"action":"hold","unit":"A1"}]').orders == [
        {"action": "hold", "unit": "A1"}
    ]
    assert parse_plan('{"action":"hold","unit":"A1"}').orders == [
        {"action": "hold", "unit": "A1"}
    ]


def test_end_turn_is_absorbed_into_the_plan_flag():
    plan = parse_plan('{"orders":[{"action":"end_turn"},{"action":"hold","unit":"A1"}]}')
    assert plan.end_turn is True
    assert plan.orders == [{"action": "hold", "unit": "A1"}]


def test_unknown_action_is_a_retryable_format_error():
    with pytest.raises(PlanFormatError, match="orders\\[0\\]"):
        parse_plan('{"orders":[{"action":"teleport","unit":"A1"}]}')


def test_a_reply_without_json_is_a_retryable_format_error():
    with pytest.raises(PlanFormatError, match="no JSON order plan"):
        parse_plan("I attack the enemy with everything I have!")


def test_orders_must_be_objects():
    with pytest.raises(PlanFormatError, match="must be an object"):
        parse_plan('{"orders":["attack"]}')
