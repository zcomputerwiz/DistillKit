"""Regrade saved scenario runs under both numeric rules by replaying their turns.

The environments are deterministic, so replaying each saved assistant turn rebuilds the
episode; the strict replay must reproduce the saved grade exactly before the supported
grade is trusted. No model runs.

    python scratch/dense_gr/regrade_scenarios.py completion-v2/eval/*/scenarios-*.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from completion_scenarios import ConsistentScenarioEnvironment  # noqa: E402

ROWS = {r['doc_id']: r for r in json.loads((HERE / 'completion-v2/agent-scenarios.json').read_text())['rows']}


def replay(record, numeric):
    env = ConsistentScenarioEnvironment(ROWS[record['id']], numeric)
    turns = [m for m in record['messages'] if m['role'] == 'assistant']
    for n, m in enumerate(turns):
        text = m['content'] + ''.join('<tool_call>' + json.dumps(dict(name=c['function']['name'],
                                      arguments=c['function']['arguments'])) + '</tool_call>'
                                      for c in m.get('tool_calls') or [])
        last = n == len(turns) - 1
        env.respond(text, truncated=last and 'truncated' in record['errors'])
    return env


def regrade(path):
    data = json.loads(Path(path).read_text())
    strict = supported = mismatched = 0
    flips = []
    for r in data['records']:
        old = replay(r, 'strict')
        if old.success != r['success'] or (old.grade or {}).get('success') != (r['grade'] or {}).get('success'):
            mismatched += 1
            continue
        new = replay(r, 'supported')
        strict += old.success
        supported += new.success
        if new.success != old.success:
            flips.append((r['id'], old.success, new.success, r['messages'][-1]['content'][:160]))
    return dict(cases=len(data['records']), strict=strict, supported=supported,
                replay_mismatches=mismatched, flips=flips)


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(p, json.dumps(regrade(p), indent=1))
