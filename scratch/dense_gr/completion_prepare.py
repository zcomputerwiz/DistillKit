# Assisted-by: Codex
"""Student negatives and reference scores for completion-v2; never trains a model."""
import copy
import json
from pathlib import Path

import phase3_run
from completion_curriculum import CompletionWorld
from completion_eval import numeric_report, total_claims
from completion_plan import OUT, OLD, HERE, digest
from tool_behavior_eval import parse_calls


def prefix_world(row):
    _,split,kind,n,_,turn=row['pair_id'].split(':')
    n,turn=int(n),int(turn)
    w=CompletionWorld(kind,n,split,(0,32,96)[n%3])
    actions=w.reference_actions()
    for action in actions[:turn]:
        if not isinstance(action,dict):
            raise ValueError('nonterminal reference prefix contains a final report')
        w.exchange(action)
        if n%2:
            w.messages[-2]['content']="I'll make that call now."
    return w,actions[turn]


def certify(w,good,text):
    text=text.replace('<|im_end|>','').strip()
    try:
        calls=parse_calls(text,w.definitions)
    except (ValueError,KeyError,TypeError):
        return None,None
    branch=copy.deepcopy(w)
    for call in calls:
        branch.execute(call)
    if branch.errors:
        return branch.errors[-1],branch.proof()
    if calls:
        return None,branch.proof()  # valid alternative actions are not rejected
    if isinstance(good,dict):
        # These worlds have no unresolved user choice. Every remaining reference
        # path is already execution-verified; ending without a call leaves it undone.
        return ('stops_before_complete_retrieval' if w.family.startswith('aggregate')
                else 'stops_before_available_action'),branch.proof()
    if w.family.startswith('aggregate'):
        if numeric_report(text,w.total):
            return None,branch.proof()
        # Only a bare wrong answer or an explicit, uniquely stated wrong total
        # is certified; unknown prose is kept for review.
        import re
        bare=re.fullmatch(r'\s*\d+\s*[.!]?\s*',text)
        totals=total_claims(text)
        if bare or (totals and len(totals[0])==1 and (totals[0][0]!=w.total or not totals[1])):
            return 'wrong_observed_total',branch.proof()
    return None,branch.proof()


def integrate():
    output=OUT/'data/pairs-final.jsonl'
    if output.exists():
        raise ValueError('refusing to overwrite integrated pairs')
    rows=[json.loads(s) for s in (OUT/'data/pairs.jsonl').read_text().splitlines()]
    generated=[json.loads(s) for s in (OUT/'data/student-rollouts.jsonl').read_text().splitlines()]
    rollout={r['doc_id'].removeprefix('onpolicy:').removesuffix(':greedy'):r for r in generated}
    schedule=json.loads((OUT/'data/pair-schedule.json').read_text())
    if len(generated)!=80 or len(rollout)!=80 or set(rollout)!={rows[i]['pair_id'] for i in schedule['indices']}:
        raise ValueError('student rollout inventory differs from the 80 scheduled turns')
    audit=[]
    for i in schedule['indices']:
        row=rows[i]
        sample=rollout[row['pair_id']]
        if sample['text'][:sample['prompt_chars']]!=row['prompt']:
            raise ValueError('student prefix changed')
        answer=sample['text'][sample['prompt_chars']:]
        w,good=prefix_world(row)
        reason,proof=certify(w,good,answer) if sample['finished'] else (None,None)
        if reason and answer!=row['chosen']:
            row['rejected']=answer
            row['rejected_kind']=reason
        else:
            reason=None
        audit.append(dict(pair_id=row['pair_id'],reason=reason,finished=sample['finished'],
                          student_response=answer,before=w.proof(),student_after=proof,
                          origin='student_greedy' if reason else 'curated_counterfactual'))
    output.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    (OUT/'data/student-audit.json').write_text(json.dumps(dict(scheduled=80,
        replaced=sum(bool(r['reason']) for r in audit),decisions=audit),indent=2))


def main():
    phase3_run.OUT=OUT
    plan=json.loads((OUT/'plan.json').read_text())
    for path,sha in plan['input_sha256'].items():
        if digest(path)!=sha:
            raise ValueError('prepared asset changed: '+path)
    # Reuse stage receipts without inheriting the preceding run's training stages.
    protocol=OUT/'evaluation.json'
    if not protocol.exists():
        protocol.write_text(json.dumps(dict(status='preparation_only',
            inherited_protocol_sha256=digest(OLD/'evaluation.json'),
            live_heldout=str(HERE/'completion_eval.py'),cases=44),indent=2))
    phase3_run.invoke('prepare-student',plan['preparation_commands'][0],'0',
        environment_overrides={'TORCHINDUCTOR_CACHE_DIR':str(OUT/'compile-cache-preparation')})
    if not (OUT/'data/pairs-final.jsonl').exists():
        integrate()
    phase3_run.invoke('prepare-reference',plan['preparation_commands'][1],'0')
    rows=[json.loads(s) for s in (OUT/'data/pairs-ref.jsonl').read_text().splitlines()]
    original=[json.loads(s) for s in (OUT/'data/pairs.jsonl').read_text().splitlines()]
    if [r['pair_id'] for r in rows]!=[r['pair_id'] for r in original]:
        raise ValueError('reference scoring dropped or reordered pairs')
    import math
    for row in rows:
        a,b=row['chosen_start'],row['rejected_start']
        if a!=b or row['chosen_ids'][:a]!=row['rejected_ids'][:b]:
            raise ValueError('reference prefixes differ')
        if not all(0<a<len(row[s+'_ids']) and math.isfinite(row['ref_'+s]) for s in ('chosen','rejected')):
            raise ValueError('invalid reference scores or assistant spans')
    (OUT/'prepared.json').write_text(json.dumps(dict(status='reference_prepared_training_not_launched',
        pairs=len(rows),sha256={n:digest(OUT/'data'/n) for n in
            ('pairs-final.jsonl','pairs-ref.jsonl','student-audit.json','pair-schedule.json')},
        still_required=['manual negative review','longest-pair GPU backward preflight','final frozen launch protocol']),indent=2))
    print('Reference preparation complete. No training was launched.',flush=True)


if __name__=='__main__':
    main()
