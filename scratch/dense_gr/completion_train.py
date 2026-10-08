# Assisted-by: Codex
"""Existing trainer with a finite, audited pair order covering complete trajectories."""
import argparse
import functools
import hashlib
import json
from pathlib import Path

import smoke_train


def inventory_hash(rows):
    return hashlib.sha256(json.dumps([r['pair_id'] for r in rows],separators=(',',':')).encode()).hexdigest()


class OrderedPairSource(smoke_train.PairSource):
    def __init__(self,*args,schedule,**kwargs):
        super().__init__(*args,**kwargs)
        self.schedule=json.loads(Path(schedule).read_text())
        if self.ftpo:
            raise ValueError('ordered completion curriculum requires assistant-turn pairs')
        if inventory_hash(self.rows)!=self.schedule['pair_inventory_sha256']:
            raise ValueError('ordered pair inventory changed')
        self.indices=self.schedule['indices']
        if not self.indices or any(type(i)!=int or i<0 or i>=len(self.rows) for i in self.indices):
            raise ValueError('invalid ordered pair indices')
        self.cursor=0

    def take(self,count):
        if count<1 or self.cursor+count>len(self.indices):
            raise ValueError('ordered pair schedule exhausted; it never cycles')
        indices=self.indices[self.cursor:self.cursor+count]
        self.cursor+=count
        return [self._record(self.rows[i]) for i in indices]

    def widths(self):
        return sorted({self._width(len(self.rows[i][s+'_ids'])) for i in self.indices for s in ('chosen','rejected')})


def main():
    p=argparse.ArgumentParser(description=__doc__,add_help=False)
    p.add_argument('--pair-schedule',type=Path,required=True)
    args,remaining=p.parse_known_args()
    schedule=json.loads(args.pair_schedule.read_text())
    # Prevent accidental partial or cycling exposure, as the replay trainer does.
    options={flag:remaining[remaining.index(flag)+1] for flag in ('--max-steps','--pairs-per-step') if flag in remaining}
    if '--max-steps' not in options or int(options['--max-steps'])*int(options.get('--pairs-per-step',2))!=len(schedule['indices']):
        raise ValueError('steps times pairs-per-step must consume the entire finite pair schedule')
    if '--resume' in remaining:
        raise ValueError('paired resume remains unsupported; start a separately recorded stage')
    smoke_train.PairSource=functools.partial(OrderedPairSource,schedule=args.pair_schedule)
    print('Ordered pairs:',args.pair_schedule,'sha256',hashlib.sha256(args.pair_schedule.read_bytes()).hexdigest(),flush=True)
    return smoke_train.main(remaining)


if __name__=='__main__':
    raise SystemExit(main())
