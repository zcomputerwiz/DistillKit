# Assisted-by: Codex
"""Rescore saved traces without inventing continuations or overwriting old scores.

Outcome, invalid attempts and final reporting are separate. Language checks are
deliberately narrow; explanation and ambiguous prose require human review.
This is an audit of a synthetic environment, not a general semantic grader.
"""
import argparse
import json
import re
from pathlib import Path

from agentic_curriculum import make_contrast, make_trajectory
from agentic_live_eval import Environment


def asserted_state(text, state):
    text = re.sub(r"[*`_]", "", text.lower())
    # Reject contradictory assertions, including 'not currently active'.
    if re.search(r"\b(?:not|never|isn't|isnt)\b[^.!?\n]{0,50}\b" + re.escape(state) + r"\b", text):
        return False
    return bool(re.search(r"\b" + re.escape(state) + r"\b", text))


def missing_fallback(text):
    text = text.lower()
    absent = re.search(r"no (?:matches|records|results)|(?:couldn't|could not|unable to|didn't|did not) find|not found", text)
    request = re.search(r"\b(?:provide|share|confirm|check|try|correct|different)\b", text)
    identifier = re.search(r"\b(?:name|id|identifier|spelling)\b", text)
    return bool(absent and request and identifier)


def grade(record):
    version, domain, kind, number = record['id'].split(':')
    row = (make_contrast if version == 'agentic-v2' else make_trajectory)(domain, kind, int(number))
    env = Environment(row)
    target_read = False
    correct_search = False
    verified_results = 0
    pending = []
    review = []
    initial_users = sum(m['role'] == 'user' for m in env.messages)
    seen_users = 0
    for m in record['messages']:
        if m['role'] == 'user':
            seen_users += 1
            if seen_users > initial_users:
                if kind == 'ask_choice' and m['content'] == 'The north section, please.':
                    env.choice = env.asked = True
                else:
                    review.append('unrecognized_user_continuation')
        if m['role'] == 'assistant':
            for c in m.get('tool_calls', []):
                result = env.execute(c)
                pending.append((c['id'], result))
                f = c['function']
                operation = row['environment'].get('operations', {}).get(f['name'], f['name'][len(domain)+1:])
                if operation == 'read' and result.get('record_id') == env.env['id']:
                    target_read = True
                if operation == 'search' and f['arguments'].get('name') == env.env['name']:
                    correct_search = True
        if m['role'] == 'tool':
            if not pending:
                review.append('orphan_tool_result')
                continue
            ident, expected = pending.pop(0)
            if ident != m.get('tool_call_id') or expected != json.loads(m['content']):
                review.append('tool_result_mismatch')
            else:
                verified_results += 1
    if pending:
        review.append('missing_tool_results')
    last = record['messages'][-1]
    terminal = last['role'] == 'assistant' and not last.get('tool_calls')
    text = last.get('content', '') if terminal else ''
    errors = sorted(set(env.errors + record.get('errors', [])))
    blocked = 'unrequested_mutation' in errors
    if kind == 'no_call':
        outcome = None
        review.append('explanation_requires_semantic_review')
    elif kind == 'empty_search':
        outcome = correct_search and env.mutations == 0 and missing_fallback(text)
    elif kind in ('known_read', 'read_after_search', 'already_done'):
        outcome = target_read and env.mutations == 0 and asserted_state(text, env.state)
        if kind == 'read_after_search':
            outcome &= correct_search
    else:
        outcome = env.state == env.env['desired_state'] and env.mutations == 1
        if kind == 'check_needed':
            outcome &= target_read
        if kind == 'ask_choice':
            outcome &= env.asked
        # A valid write alone is insufficient if the final response denies it.
        outcome &= asserted_state(text, env.env['desired_state']) or bool(re.fullmatch(r'(?:Updated|Done)(?: successfully)?[.!]?', text.strip(), re.I))
    if outcome is not None:
        outcome = bool(outcome and terminal and not blocked and not review)
    false_completion = (kind == 'recover' and env.mutations == 0 and
                        asserted_state(text, env.env['desired_state']) and
                        bool(re.search(r'\b(?:now|set|updated|done)\b', text, re.I)))
    return dict(id=record['id'], kind=kind, original_success=record['success'],
                outcome=outcome, terminal=terminal, invalid_attempts=errors,
                clean_success=outcome is True and not errors,
                false_completion=bool(false_completion), review=review,
                verified_tool_results=verified_results, final_text=text)


def run(source, output):
    if output.exists():
        raise ValueError('refuse to overwrite audit')
    data = json.loads(source.read_text(encoding='utf-8'))
    records = [grade(r) for r in data['records']]
    summary = dict(total=len(records), original_success=sum(r['original_success'] for r in records),
                   outcome_success=sum(r['outcome'] is True for r in records),
                   clean_success=sum(r['clean_success'] for r in records),
                   needs_review=sum(bool(r['review']) for r in records),
                   false_completion=sum(r['false_completion'] for r in records))
    output.write_text(json.dumps(dict(grader='saved-trace-audit-v1', source=str(source.resolve()),
        checkpoint=data['checkpoint'], summary=summary, records=records), indent=2), encoding='utf-8')
    print(source.name, summary)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    run(a.source, a.output)
