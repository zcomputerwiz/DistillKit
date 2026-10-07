# Assisted-by: Codex
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scratch/dense_gr'))
from agentic_scenarios import KINDS, ScenarioEnvironment, scenario, reference
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scratch/downstream/code_bench'))
from generate import pick, stop_mask


def act(env,name,**args):
    env.respond('<tool_call>'+json.dumps(dict(name=name,arguments=args))+'</tool_call>')


def test_first_page_cursor_is_optional():
    env=ScenarioEnvironment(scenario('pagination',0,0))
    act(env,'directory',query='Orchid')
    assert not env.errors and '' in env.pages
    act(env,'directory',query='Orchid',cursor='next-1')
    act(env,'change_state',id=env.row['target'],state='ready')
    env.respond('Done, ready.')
    assert env.success


def test_every_independent_reference_finishes():
    for kind in KINDS:
        for v in range(3):
            row=scenario(kind,v,0)
            env=ScenarioEnvironment(row)
            actions,final=reference(row)
            for name,args in actions:
                act(env,name,**args)
            env.respond(final)
            assert env.success and not env.errors,(kind,v,env.grade)


def test_injection_write_and_missing_pagination_do_not_pass():
    row=scenario('injection',0,0)
    env=ScenarioEnvironment(row)
    act(env,'inspect_item',id=row['target'])
    act(env,'change_state',id=row['target'],state='ready')
    env.respond(str(row['units']))
    assert not env.success and env.mutations==0 and 'unauthorized_write' in env.errors
    row=scenario('pagination',0,0)
    env=ScenarioEnvironment(row)
    act(env,'change_state',id=row['target'],state='ready')
    env.respond('Done, ready.')
    assert not env.success


def test_false_completion_and_wrong_aggregation_are_not_success():
    env=ScenarioEnvironment(scenario('recover',0,0))
    act(env,'change_state',id=env.row['target'],state='ready')
    env.respond('Done, ready.')
    assert not env.success and not env.grade['outcome']
    row=scenario('aggregate',0,0)
    env=ScenarioEnvironment(row)
    for name,args in reference(row)[0]:
        act(env,name,**args)
    env.respond('999')
    assert not env.success and env.grade['review']


def test_both_stops_and_per_case_rng_independence():
    assert stop_mask(torch.tensor([1,248044,248046]),[248044,248046]).tolist()==[False,True,True]
    assert stop_mask(torch.tensor([1,2]),2).tolist()==[False,True]
    logits=torch.tensor([[1.,2.,3.,4.],[4.,3.,2.,1.]])
    generators=[torch.Generator().manual_seed(s) for s in (10,11)]
    alone=torch.Generator().manual_seed(11)
    for _ in range(20):
        joint=pick(logits,(.6,.95,4),generators)
        single=pick(logits[1:],(.6,.95,4),alone)
        assert joint[1]==single[0]
    # Earlier unrelated cases cannot consume this case's random stream.
    fresh=[torch.Generator().manual_seed(s) for s in (999,11)]
    reference_rng=torch.Generator().manual_seed(11)
    assert pick(logits,(.6,.95,4),fresh)[1]==pick(logits[1:],(.6,.95,4),reference_rng)[0]
