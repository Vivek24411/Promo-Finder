#!/usr/bin/env python3
"""Compare artifacts on selected positive moments without pretending these are human acceptance labels.

python scripts/evaluate_v8.py OLD_RUN NEW_RUN --gold eval/gold_moments.json --out outputs/v8/comparison.json
Optional --ratings CSV: columns run,clip_key,accepted (0|1),review_seconds. Only completed human ratings
are used; no model or generated label is substituted. A partial pipeline run is not scored as a quality result.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate import load_moments,score_run,to_jsonable


def compare(paths,gold,ratings=None):
    moments=load_moments(Path(gold))
    human=list(csv.DictReader(open(ratings))) if ratings else []
    results=[]
    for path in paths:
        path=Path(path);data=json.loads((path/'promos.json').read_text());meta=data['meta']
        valid=meta.get('status','complete')=='complete'
        scored=to_jsonable(score_run(path,moments)) if valid else None
        top=data['promos'][:10]
        keys={p.get('clip_key',p['id']) for p in top}
        rated=[r for r in human if r['run']==path.name and r['clip_key'] in keys and r.get('accepted') in ('0','1')]
        results.append({'run':path.name,'implementation':meta.get('implementation',meta.get('scoring_preset','legacy')),
            'status':meta.get('status','historical'),'positive_moment_overlap':scored,
            'top10_human_acceptance':{'accepted':sum(int(r['accepted']) for r in rated),'rated':len(rated)} if rated else None,
            'review_seconds':sum(float(r['review_seconds']) for r in rated if r.get('review_seconds')) if rated else None,
            'false_acceptance_known_failures':None,'cost':meta.get('budget',meta.get('llm_usage')),
            'runtime_seconds':meta.get('seconds'),
            'limitation':'Selected positive moments are not exhaustive human acceptance/rejection labels. Partial runs cannot establish ranking quality.'})
    return {'comparison':results,'improvement_claim':None,'heldout_shows_evaluated':0}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs',nargs='+');ap.add_argument('--gold',required=True);ap.add_argument('--ratings');ap.add_argument('--out',required=True)
    a=ap.parse_args();Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(compare(a.runs,a.gold,a.ratings),indent=2))
    print('Saved comparison:',a.out)
