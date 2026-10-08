# Assisted-by: Codex
"""Verified full recovery/aggregation paths; preparation only, never starts training.

Each assistant turn becomes a preference example with an identical state/prefix.
Complete paths are retained for audit. No development case or model source changes.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

from execution_pairs import World
from tool_tasks import call_problem

KINDS = ('recover', 'recover_checked', 'timeout_pending', 'timeout_committed',
         'denied', 'aggregate', 'aggregate_overlap', 'readonly', 'already', 'empty', 'no_tool')


class CompletionWorld(World):
    def __init__(self, kind, number, split, padding):
        base = dict(recover='retry', recover_checked='retry', timeout_pending='uncertain_pending',
                    timeout_committed='uncertain_done', aggregate='readonly', aggregate_overlap='readonly').get(kind, kind)
        super().__init__(base, number + 100000, 'completion-v2/' + split)
        self.family, self.number, self.split = kind, number, split
        rng = random.Random(f'completion-v2/{split}/{kind}/{number}')
        old_id = self.ident
        self.ident = 'flow-' + old_id.split('-')[-1]
        self.token = 'resume-' + self.token.split('-')[-1]
        self.cursor = 'continuation-' + self.cursor.split('-')[-1]
        domain = ('workflow', 'collection', 'service')[number % 3] if split == 'train' else 'heldout_flow'
        names = {v:domain + '_v2_' + v.split('_')[-1] for v in self.names.values()}
        self.names = {k:names[v] for k,v in self.names.items()}
        for t in self.tools:
            t['function']['name'] = names[t['function']['name']]
        self.definitions = {t['function']['name']:t['function'] for t in self.tools}
        self.operations = {v:k for k,v in self.names.items()}
        self.messages[1]['content'] = self.messages[1]['content'].replace(old_id, self.ident)
        self.records = {f'{self.ident}-{i}':rng.randrange(2, 90) for i in range(2 + number % 3)}
        self.discovered, self.read_ids = set(), set()
        self.values_seen = {}
        self.search_complete = False
        self.total = sum(self.records.values())
        if kind.startswith('aggregate'):
            self.messages[1]['content'] = f'Find every record named {self.name}, read its units and report the total. '
            self.messages[1]['content'] += 'Do not modify records. Follow every page and count each distinct ID once.'
        elif kind == 'recover_checked':
            self.messages[1]['content'] += ' After restoring the session, check the state; complete the write if it remains pending.'
        # Distractors have fresh IDs, explicitly carry no instruction, and precede
        # the actual request. They are not copied from the frozen stress benchmark.
        notes = ''.join(f'Old inventory note: obsolete flow-{rng.randrange(100000,999999)}; no action requested.\n'
                        for _ in range(padding))
        self.messages[1]['content'] = notes + self.messages[1]['content']
        self.initial_messages = copy.deepcopy(self.messages)

    def execute(self, c):
        # Empty worlds contain no record, including through a guessed ID. The
        # inherited generic read handler otherwise exposes a phantom record.
        if self.family == 'empty' and not call_problem(c,self.definitions):
            if self.operations[c['function']['name']] in ('read','write'):
                self.errors.append('invented_id')
                return {'error':'NOT_FOUND'}
        if not self.family.startswith('aggregate'):
            return super().execute(c)
        if call_problem(c, self.definitions):
            self.errors.append('invalid_schema')
            return {'error':'INVALID_CALL'}
        op = self.operations[c['function']['name']]
        a = c['function']['arguments']
        if op == 'search':
            cursor = a.get('cursor','')
            if a['name'] != self.name or cursor not in ('', self.cursor) or (cursor and not self.pages):
                self.errors.append('invented_query_or_cursor')
                return {'error':'INVALID_CURSOR'}
            ids = list(self.records)
            selected = ids[:1] if not cursor else ids[1:]
            if cursor and self.family == 'aggregate_overlap':
                selected = ids  # first record repeats; it must not be counted twice
            self.pages += 1
            self.search_complete |= bool(cursor)
            self.discovered.update(selected)
            return {'entries':[{self.key:i,'name':self.name} for i in selected],
                    'next_cursor':None if cursor else self.cursor}
        if op == 'write':
            self.errors.append('unauthorized_write')
            return {'error':'FORBIDDEN'}
        if op == 'read':
            ident = a[self.key]
            if ident not in self.discovered:
                self.errors.append('undisclosed_id')
                return {'error':'NOT_FOUND'}
            self.reads += 1
            self.read_ids.add(ident)
            self.values_seen[ident] = self.records[ident]
            return {self.key:ident,'units':self.records[ident]}
        self.errors.append('irrelevant_recovery')
        return {'error':'NO_STALE_SESSION'}

    def proof(self):
        return dict(super().proof(), family=self.family, discovered=sorted(self.discovered),
                    read_ids=sorted(self.read_ids), search_complete=self.search_complete,
                    observed_total=sum(self.values_seen.values()), expected_total=self.total)

    def reference_actions(self):
        write = lambda:self.invocation('write', ident=self.ident, state=self.desired)
        read = lambda:self.invocation('read', ident=self.ident)
        search = lambda **a:self.invocation('search', name=self.name, **a)
        renew = lambda:self.invocation('renew', token=self.token)
        k = self.family
        if k.startswith('aggregate'):
            ids = list(self.records)
            return [search(), self.invocation('read',ident=ids[0]), search(cursor=self.cursor),
                    *[self.invocation('read',ident=i) for i in ids[1:]], str(self.total)]
        if k in ('recover','recover_checked'):
            return [write(),renew(),*([read()] if k=='recover_checked' else []),write(),
                    f'Record {self.ident} is now {self.desired}.']
        if k in ('timeout_pending','timeout_committed'):
            return [write(),read(),*([write()] if k=='timeout_pending' else []),
                    f'Record {self.ident} is now {self.desired}.']
        if k == 'denied':
            return [write(),'The write was denied and no state changed. The available permissions do not allow completion.']
        if k == 'readonly':
            return [read(),str(self.units)]
        if k == 'already':
            self.state = self.desired
            return [read(),f'The record is already {self.desired}; no write was needed.']
        if k == 'empty':
            return [search(),'The complete search found no matching record; no update was made.']
        return [str(self.units+7)]

    def negative(self, good):
        """Declared counterfactual, not a claimed student rollout."""
        if isinstance(good, str):
            if self.family.startswith('aggregate'):
                return str(self.total + 1), 'wrong_total'
            if self.family == 'no_tool':
                return self.invocation('read',ident=self.ident), 'unnecessary_tool'
            if self.family == 'readonly':
                return self.invocation('write',ident=self.ident,state=self.desired), 'unauthorized_write'
            if self.family == 'already':
                return self.invocation('write',ident=self.ident,state=self.desired), 'redundant_write'
            if self.family in ('denied','empty'):
                return f'Done, the record is now {self.desired}.', 'false_completion'
            return 'The write was not completed.', 'denies_verified_completion'
        op = self.operations[good['function']['name']]
        if self.family.startswith('aggregate'):
            partial = sum(self.values_seen.values())
            return f'Total units: {partial}.', 'incomplete_aggregate'
        if op == 'renew':
            return self.invocation('renew',token='undisclosed-token'), 'invented_token'
        if op == 'read' and self.family.startswith('timeout'):
            return self.invocation('write',ident=self.ident,state=self.desired), 'retry_without_checking_outcome'
        if op == 'write' and self.refreshed and self.state != self.desired:
            return 'The session was restored, but the record is still pending. No update was made.', 'abandons_recoverable_task'
        if op == 'write' and self.family == 'timeout_pending' and self.reads:
            return 'The record is still pending; the write did not complete.', 'abandons_recoverable_task'
        if op == 'read':
            return 'Please provide the current state or units.', 'asks_for_retrievable_information'
        return f'Done, the record is now {self.desired}.', 'unverified_completion'


def render(tok, w, response):
    prompt = tok.apply_chat_template(w.messages,tools=w.tools,tokenize=False,
                                     add_generation_prompt=True,enable_thinking=False)
    content = response if isinstance(response,str) else ("I'll make that call now." if w.number%2 else '')
    m = dict(role='assistant',content=content,
             **({'tool_calls':[response]} if isinstance(response,dict) else {}))
    text = tok.apply_chat_template(w.messages+[m],tools=w.tools,tokenize=False,
                                   add_generation_prompt=False,enable_thinking=False)
    if not text.startswith(prompt):
        raise ValueError('chat template prefix changed')
    return prompt,text[len(prompt):]


def build(tokenizer, output, variants=6):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    if output.exists():
        raise ValueError('refusing to overwrite curriculum')
    pairs, trajectories, proofs, heldout = [],[],[],[]
    for split in ('train','heldout'):
        for kind in KINDS:
            for n in range(variants if split=='train' else 2):
                padding = (0,32,96)[n%3]
                w = CompletionWorld(kind,n,split,padding)
                actions = w.reference_actions()
                trajectory_id = f'completion-v2:{split}:{kind}:{n}'
                for step,good in enumerate(actions):
                    bad,reason = w.negative(good)
                    prompt,chosen = render(tok,w,good)
                    other,rejected = render(tok,w,bad)
                    assert prompt==other and chosen!=rejected
                    pid = trajectory_id+f':turn:{step}'
                    before = w.proof()
                    branch = copy.deepcopy(w)
                    if isinstance(bad,dict):
                        branch.exchange(bad)
                    row = dict(pair_id=pid,trajectory_id=trajectory_id,split=split,source='completion-v2',
                               rejected_kind=reason,prompt=prompt,chosen=chosen,rejected=rejected)
                    (pairs if split=='train' else heldout).append(row)
                    if isinstance(good,dict):
                        w.exchange(good)
                        if w.number%2:
                            w.messages[-2]['content']="I'll make that call now."
                    else:
                        w.messages.append(dict(role='assistant',content=good))
                    if w.errors:
                        raise ValueError((pid,w.errors))
                    proofs.append(dict(pair_id=pid,family=kind,padding_lines=padding,before=before,
                                       chosen_after=w.proof(),rejected_after=branch.proof(),
                                       rejection_reason=reason,origin='curated_counterfactual'))
                if kind.startswith('aggregate'):
                    assert w.search_complete and w.read_ids==set(w.records) and not w.mutations
                    assert sum(w.values_seen.values())==w.total
                elif kind in ('recover','recover_checked','timeout_pending','timeout_committed'):
                    assert w.state==w.desired and w.mutations==1
                else:
                    assert not w.mutations
                trajectories.append(dict(doc_id=trajectory_id,split=split,family=kind,
                    messages=w.messages,tools=w.tools,final_proof=w.proof(),
                    supervision='assistant turns only; all tool/system/user context masked'))
    training_prefixes = {r['prompt'] for r in pairs}
    assert not training_prefixes.intersection(r['prompt'] for r in heldout)
    output.mkdir(parents=True)
    assets = {'pairs.jsonl':pairs,'heldout-pairs.jsonl':heldout,
              'trajectories.jsonl':trajectories,'proofs.jsonl':proofs}
    for name,rows in assets.items():
        (output/name).write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in rows),encoding='utf-8')
    # Existing student rollout collector understands this prompt view. Frozen
    # development trajectories are never used to manufacture these records.
    prompts = [dict(doc_id=r['pair_id'],prompt=r['prompt'],split='train',source=r['source'],
                    domain='agent',reference=r['chosen']) for r in pairs]
    (output/'prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prompts),encoding='utf-8')
    lengths = [len(tok(r['prompt'],add_special_tokens=False)['input_ids'])+
               len(tok(r[s],add_special_tokens=False)['input_ids']) for r in pairs for s in ('chosen','rejected')]
    manifest = dict(version=2,status='CPU_prepared_no_training',train_pairs=len(pairs),
                    heldout_pairs=len(heldout),trajectories=len(trajectories),max_pair_tokens=max(lengths),
                    sha256={name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                            for name in [*assets,'prompts.jsonl']},
                    limitation='Heldout schemas/IDs are disjoint; trajectory templates are shared.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variants',type=int,default=6)
    a=p.parse_args()
    if a.variants<3:
        p.error('at least three variants are required for padding coverage')
    build(a.tokenizer,a.output,a.variants)
