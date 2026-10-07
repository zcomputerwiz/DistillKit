# Assisted-by: Codex
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scratch/dense_gr'))
from agentic_curriculum import make_contrast
from agentic_grade_audit import grade, asserted_state, missing_fallback
from agentic_live_eval import Environment
from tool_behavior_eval import call


def record(env):
    return dict(id=env.row['doc_id'], messages=env.messages, errors=env.errors,
                success=env.success)


def invoke(env, operation, **args):
    e = env.env
    name = next(n for n, op in e['operations'].items() if op == operation)
    if 'record_id' in args:
        args[e['id_argument']] = args.pop('record_id')
    env.respond('<tool_call>' + json.dumps({'name': name, 'arguments': args}) + '</tool_call>')


def test_recovered_read_does_not_erase_completion_or_errors():
    env = Environment(make_contrast('dispatch', 'known_id', 0))
    invoke(env, 'read', record_id='missing')
    invoke(env, 'set_state', record_id=env.env['id'], state=env.env['desired_state'])
    env.respond('The record is now ' + env.state + '.')
    result = grade(record(env))
    assert result['outcome'] and not result['clean_success']
    assert result['invalid_attempts'] == ['unknown_id']


def test_wrong_record_and_negated_state_do_not_pass():
    env = Environment(make_contrast('dispatch', 'known_read', 0))
    invoke(env, 'read', record_id=env.env['alternate'])
    env.respond('The state is ' + env.state + '.')
    assert not grade(record(env))['outcome']
    assert not asserted_state('The record is not currently **active**.', 'active')


def test_stale_session_false_report_fails():
    env = Environment(make_contrast('dispatch', 'recover', 0))
    invoke(env, 'set_state', record_id=env.env['id'], state=env.env['desired_state'])
    env.respond('The record is now ' + env.env['desired_state'] + '.')
    result = grade(record(env))
    assert not result['outcome'] and result['false_completion']


def test_empty_search_fallback_and_no_invented_continuation():
    assert missing_fallback("I couldn't find the record. Please try a different name or check the spelling.")
    assert not missing_fallback('Please confirm the update.')
    env = Environment(make_contrast('dispatch', 'ask_choice', 0))
    invoke(env, 'search', name=env.env['name'])
    env.respond('Which record do you mean?')
    assert not grade(record(env))['outcome']


def test_blocked_mutation_is_retained_even_after_correct_completion():
    env = Environment(make_contrast('dispatch', 'known_id', 0))
    invoke(env, 'set_state', record_id=env.env['alternate'], state=env.env['desired_state'])
    invoke(env, 'set_state', record_id=env.env['id'], state=env.env['desired_state'])
    env.respond('The record is now ' + env.state + '.')
    assert not grade(record(env))['outcome']


def test_missing_or_fabricated_tool_result_cannot_prove_completion():
    env = Environment(make_contrast('dispatch', 'known_id', 0))
    invoke(env, 'set_state', record_id=env.env['id'], state=env.env['desired_state'])
    assert not grade(record(env))['outcome']  # no final assistant report
    env.respond('The record is now ' + env.state + '.')
    env.messages[-2]['content'] = json.dumps({'status': 'ok'})
    result = grade(record(env))
    assert not result['outcome'] and 'tool_result_mismatch' in result['review']


def test_review_keyword_is_not_automatic_semantic_success():
    env = Environment(make_contrast('dispatch', 'no_call', 0))
    env.respond('Review means I refuse to answer.')
    result = grade(record(env))
    assert result['outcome'] is None and result['review']
