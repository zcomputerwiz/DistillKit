# Assisted-by: Codex
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scratch/dense_gr'))
from agentic_scenarios import ScenarioEnvironment, scenario, reference
from completion_scenarios import ConsistentScenarioEnvironment
from completion_curriculum import CompletionWorld


def call(name,**arguments):
    return dict(function=dict(name=name,arguments=arguments))


def test_empty_world_is_absent_through_every_retrieval_path():
    for variant in range(3):
        row=scenario('empty',variant,0)
        env=ConsistentScenarioEnvironment(row)
        assert env.execute(call('directory',query=row['query']))==dict(entries=[],next_cursor=None)
        assert env.execute(call('resolve_alias',alias=row['query']))==dict(error='NOT_FOUND')
        for ident in (row['target'],row['alternate'],'invented-id'):
            assert env.execute(call('inspect_item',id=ident))==dict(error='NOT_FOUND')
            assert env.execute(call('change_state',id=ident,state='ready'))==dict(error='NOT_FOUND')
        assert not env.resolved and not env.reads and env.mutations==0
        assert 'unauthorized_write' not in env.errors


def test_correction_preserves_all_reference_paths_and_nonempty_worlds():
    from agentic_scenarios import KINDS
    for kind in KINDS:
        for variant in range(3):
            row=scenario(kind,variant,0)
            old=ScenarioEnvironment(copy.deepcopy(row))
            new=ConsistentScenarioEnvironment(copy.deepcopy(row))
            actions,final=reference(row)
            for name,args in actions:
                text='<tool_call>'+json.dumps(dict(name=name,arguments=args))+'</tool_call>'
                old.respond(text)
                new.respond(text)
            old.respond(final)
            new.respond(final)
            assert old.success and new.success and old.messages==new.messages
            assert old.grade==new.grade


def test_new_training_empty_world_has_no_phantom_record():
    env=CompletionWorld('empty',0,'train',0)
    for operation in ('read','write'):
        args=dict(ident=env.ident)
        if operation=='write':
            args['state']=env.desired
        assert env.execute(env.invocation(operation,**args))==dict(error='NOT_FOUND')
    assert env.reads==0 and env.mutations==0
