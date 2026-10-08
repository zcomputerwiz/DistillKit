# Assisted-by: Codex
"""Prepare a provisional 40-step extension with complete-path, matched replay exposure."""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'completion-v2'
OLD=HERE/'execution-grounded'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    target=OUT/'plan.json'
    if target.exists():
        raise ValueError('refusing to overwrite provisional plan')
    rows=[json.loads(s) for s in (OUT/'data/pairs.jsonl').read_text().splitlines()]
    groups={}
    for i,row in enumerate(rows):
        groups.setdefault(row['trajectory_id'],[]).append(i)
    blocks=[
        [('recover_checked',0),('aggregate',1),('denied',0),('timeout_pending',0),('already',0),('no_tool',0)],
        [('recover',1),('aggregate_overlap',0),('timeout_committed',1),('readonly',1),('empty',1),('denied',1),('already',1)],
        [('recover_checked',2),('aggregate',2),('timeout_pending',2),('readonly',2),('no_tool',2),('no_tool',3)],
        [('recover',4),('aggregate_overlap',4),('timeout_committed',4),('denied',4),('empty',4),('already',4),('no_tool',4)]]
    indices=[]
    for block in blocks:
        selected=sum((groups[f'completion-v2:train:{kind}:{n}'] for kind,n in block),[])
        if len(selected)!=20:
            raise ValueError('each 10-step checkpoint block must contain 20 complete-path turns')
        indices+=selected
    assert len(indices)==len(set(indices))==80
    inventory=hashlib.sha256(json.dumps([r['pair_id'] for r in rows],separators=(',',':')).encode()).hexdigest()
    schedule=dict(version=1,indices=indices,pair_inventory_sha256=inventory,
                  semantics='Four 20-pair blocks; every selected trajectory includes all assistant turns in order.',
                  complete_trajectories=sum(map(len,blocks)),checkpoints=[10,20,30,40])
    (OUT/'data/pair-schedule.json').write_text(json.dumps(schedule,indent=2))
    chosen=[rows[i] for i in indices]
    (OUT/'data/scheduled-prompts.jsonl').write_text(''.join(json.dumps(dict(doc_id=r['pair_id'],
        prompt=r['prompt'],split='train',source=r['source'],domain='agent',reference=r['chosen']))+'\n' for r in chosen))
    prior=json.loads((OLD/'plan.json').read_text())
    start=OLD/'targeted/checkpoints/smoke-r1-1-gr-s25-csa2'
    plan=dict(version=2,status='prepared_not_launched_pending_review_and_GPU_preflight',start_checkpoint=str(start),
        start_policy='Experimental initialization, not promotion. Fresh optimizer; not resume of prior pair state.',
        steps=40,checkpoint_steps=[10,20,30,40],weighted_replay_targets=966679,
        trajectory_count=schedule['complete_trajectories'],pair_count=80,
        family_counts=dict(Counter(r['trajectory_id'].split(':')[2] for r in chosen)),
        pair_settings=prior['pair_config'],arms={},
        freeze_pending=['Current pilot semantic/safety and broader retention review',
                        'Student negatives on the new training prefixes, individually certified',
                        'Reference scores from the selected starting checkpoint; no dropped/reordered pairs',
                        'GPU preflight at longest scheduled pair and actual replay shape',
                        'Exact source/runtime/data/checkpoint hashes before launch'],
        input_sha256={str(OUT/'data/manifest.json'):digest(OUT/'data/manifest.json'),
                      str(OUT/'data/pair-schedule.json'):digest(OUT/'data/pair-schedule.json'),
                      str(OLD/'plan.json'):digest(OLD/'plan.json')})
    for arm in ('control','completion'):
        argv=list(prior['arms']['control'])
        argv[argv.index('--init-from')+1]=str(start)
        for flag,value in (('--save-every','10'),('--evaluate-every','10'),
                           ('--checkpoints',str(OUT/arm/'checkpoints')),('--output',str(OUT/arm/'train.json'))):
            argv[argv.index(flag)+1]=value
        if arm=='completion':
            argv[0]=str(HERE/'completion_train.py')
            argv+=['--pair-schedule',str(OUT/'data/pair-schedule.json'),
                   '--pairs',str(OUT/'data/pairs-ref.jsonl'),'--pairs-per-step','2',
                   '--pair-weight','0.1','--dpo-beta','0.1','--pair-sft-weight','1.0']
        plan['arms'][arm]=argv
    plan['preparation_commands']=[
        [str(HERE/'onpolicy_rollouts.py'),'--checkpoint',str(start),'--inputs',str(OUT/'data/scheduled-prompts.jsonl'),
         '--count','80','--width','4096','--new','256','--batch-size','2','--seed','25','--greedy',
         '--output',str(OUT/'data/student-rollouts.jsonl')],
        [str(HERE/'ref_logprobs.py'),'--reference',str(start),'--pairs',str(OUT/'data/pairs-final.jsonl'),
         '--max-length','4096','--output',str(OUT/'data/pairs-ref.jsonl')]]
    # Reference preparation scores the entire 222-row inventory; integration must
    # preserve it and may replace only independently certified scheduled negatives.
    plan['reference_inventory_policy']='All 222 rows retain order/IDs; scheduled student failures may replace negatives. No reference scoring until labels are audited.'
    plan['export_commands']={arm:{str(step):[str(HERE/'export_state.py'),'--state',
        str(OUT/arm/'checkpoints'/f'state-step-{step:08d}'),'--like',str(OUT/arm/'checkpoints/smoke-r1-1-gr-s25-csa2'),
        '--output',str(OUT/arm/'exported'/f'step-{step:02d}')] for step in (10,20,30)} for arm in plan['arms']}
    plan['evaluation_policy']='One uninterrupted 40-step stage saves states at 10/20/30/40. After that stage finishes, use existing export_state.py and evaluate all four milestones without GPU contention. This provides a learning curve, not early stopping inside the stage. Paired resume remains unsupported. Evaluate heldout closed-loop recovery/aggregation, frozen safety screens and math at every milestone; paired code/reasoning retention on milestone candidates. No automatic additional training stage or promotion.'
    target.write_text(json.dumps(plan,indent=2))
    print('Prepared',schedule['complete_trajectories'],'complete trajectories, 80 turns; training not launched.')


if __name__=='__main__':
    main()
