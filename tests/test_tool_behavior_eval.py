# Assisted-by: Codex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch/dense_gr"))
from tool_behavior_eval import fixtures, parse_calls, score


def xml(name, **args):
    return "<tool_call><function=%s>%s</function></tool_call>" % (
        name, "".join("<parameter=%s>%s</parameter>" % pair for pair in args.items()))


def test_result_pairs_change_reference_and_wrong_variant_fails():
    cases = fixtures()
    for left, right in zip(cases[:32:2], cases[1:32:2]):
        assert left["pair"] == right["pair"]
        assert left["messages"][-1] != right["messages"][-1]
        if "expected_value" in left:
            assert score(left, left["expected_value"])["success"]
            assert not score(left, right["expected_value"])["success"]
        else:
            expected = left["expected_calls"][0]["function"]
            text = xml(expected["name"], **expected["arguments"])
            assert score(left, text)["success"]
            assert not score(right, text)["success"]


def test_malformed_and_wrong_schema_fail():
    case = fixtures()[8]
    assert not score(case, "<tool_call><function=get_ticket_details>")["success"]
    assert not score(case, xml("get_ticket_details", wrong="x"))["schema_valid"]
    assert not score(case, xml("undeclared", ticket_id="x"))["schema_valid"]
    assert parse_calls(xml("schedule_retry", retry_after_seconds=31))[0]["function"]["arguments"] == {"retry_after_seconds": 31}


def test_truncated_answer_never_passes_and_opposite_value_is_rejected():
    case = fixtures()[0]
    assert not score(case, case["expected_value"], True)["success"]
    assert not score(case, case["expected_value"] + " or " + case["forbidden_values"][0])["success"]


def test_request_missing_information_can_be_an_imperative():
    case = next(c for c in fixtures() if c["category"] == "missing_info")
    assert score(case, "Please provide your ticket number.")["success"]
    assert not score(case, "Your ticket number is PUBLIC-123.")["success"]


def test_numeric_xml_string_argument_uses_declared_type():
    case = next(c for c in fixtures() if c["category"] == "initial")
    case["expected_calls"][0]["function"]["arguments"]["ticket_number"] = "78701"
    assert score(case, xml("lookup_ticket", ticket_number="78701"))["success"]


def test_parallel_order_does_not_change_call_match():
    case = next(c for c in fixtures() if c["category"] == "initial")
    calls = [case["expected_calls"][0], {"type": "function", "function": {
        "name": "get_ticket_details", "arguments": {"ticket_id": "x"}}}]
    case["expected_calls"] = calls
    text = xml("get_ticket_details", ticket_id="x") + xml("lookup_ticket", **calls[0]["function"]["arguments"])
    assert score(case, text)["success"]
