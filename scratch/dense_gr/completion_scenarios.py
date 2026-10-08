# Assisted-by: Codex
"""Versioned correction of the independent scenario suite's no-match world."""
import argparse
import hashlib
import json
from pathlib import Path

from agentic_live_eval import run
from agentic_scenarios import ScenarioEnvironment, reference
from tool_tasks import call_problem

HERE = Path(__file__).resolve().parent
DATA = HERE/'completion-v2/agent-scenarios.json'


class ConsistentScenarioEnvironment(ScenarioEnvironment):
    def execute(self, call):
        if self.row['kind'] == 'empty' and not call_problem(call,self.definitions):
            name=call['function']['name']
            if name in ('resolve_alias','inspect_item','change_state'):
                self.calls += 1
                # A known alias can legitimately be absent; absence is not an
                # invalid call. Guessed IDs still count as invalid attempts.
                if name == 'resolve_alias':
                    if call['function']['arguments']['alias'] != self.row['query']:
                        self.errors.append('unknown_alias')
                else:
                    self.errors.append('unknown_id')
                return {'error':'NOT_FOUND'}
        return super().execute(call)


def freeze():
    if DATA.exists():
        raise ValueError('refusing to overwrite corrected scenario suite')
    source=HERE/'phase3-eval/agent-scenarios.json'
    payload=json.loads(source.read_text())
    # Preserve all prompts, schemas, IDs and positive reference continuations.
    for row in payload['rows']:
        env=ConsistentScenarioEnvironment(row)
        actions,final=reference(row)
        for name,args in actions:
            env.respond('<tool_call>'+json.dumps(dict(name=name,arguments=args))+'</tool_call>')
        env.respond(final)
        if not env.success or env.errors:
            raise ValueError('invalid corrected reference path: '+row['doc_id'])
    payload.update(version=2,environment_version='consistent-no-match-v2',
        inherited_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        correction='Empty worlds return NOT_FOUND from alias resolution, inspection and writes; no phantom record exists.')
    DATA.write_text(json.dumps(payload,indent=2)+'\n')
    print('Reference-validated',len(payload['rows']),'corrected scenario worlds.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','run'))
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--sample',action='store_true')
    parser.add_argument('--short-only',action='store_true')
    parser.add_argument('--prefill-query-chunk',type=int,default=0)
    args=parser.parse_args()
    if args.command == 'freeze':
        freeze()
        return
    if not args.checkpoint or not args.output:
        parser.error('run requires checkpoint and output')
    data=json.loads(args.data.read_text())
    if data.get('environment_version') != 'consistent-no-match-v2':
        raise ValueError('requires corrected, versioned scenario data')
    args.grading_version=2
    environments=[ConsistentScenarioEnvironment(row) for row in data['rows']
        if not args.short_only or row['padding_lines']==0]
    run(args,environments)
    result=json.loads(args.output.read_text())
    result.update(environment_version=data['environment_version'],
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
