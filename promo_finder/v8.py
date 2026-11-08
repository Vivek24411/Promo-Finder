"""Production v8 entry point shared by CLI and Streamlit; historical pipelines remain reproducible."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
import subprocess
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from .candidates import _overlap
from .config import Settings, ROOT
from .llm_stages import rerank
from .parse import parse_show, script_files
from .rank import rank_verified, apply_rerank
from .report import write_all, promo_record
from .review import (STRICT, JUDGE_VERSION, SYSTEM, EXTRA, neutral_signals, assess, judge_one, faithful, repair, check, update_decision)
from .runtime import VERSION, DEFAULT_MODELS, atomic_json, routes, configuration_errors, role_settings
from .semantic import discover, generate, chunks, DISCOVERY_SYSTEM, DISCOVERY_CONTRACT


def code_fingerprint():
    paths=sorted((ROOT/'promo_finder').glob('*.py'))+[ROOT/'app.py',ROOT/'config.toml']
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in paths)).hexdigest()


def identity(show):
    content=[(str(p.relative_to(show)),hashlib.sha256(p.read_bytes()).hexdigest()) for p in script_files(show)]
    fingerprint=hashlib.sha256(json.dumps(content).encode()).hexdigest()
    config={k:os.getenv(k) for k in ('GEMINI_MODEL','CLAUDE_MODEL','MAX_RUN_COST_USD','MAX_RUN_SECONDS',
        'GEMINI_INPUT_USD_PER_MILLION','GEMINI_OUTPUT_USD_PER_MILLION','CLAUDE_INPUT_USD_PER_MILLION',
        'CLAUDE_OUTPUT_USD_PER_MILLION','FINAL_REVIEW_LIMIT','AUDIT_SAMPLE_SIZE')}
    run_id=hashlib.sha256(json.dumps([fingerprint,VERSION,code_fingerprint(),config],sort_keys=True).encode()).hexdigest()[:24]
    return fingerprint,run_id


def estimate(show, overrides=None, public=False, budget_usd=None):
    eps,report=parse_show(show,settings=STRICT)
    count=sum(1 for ep in eps for _ in chunks(ep))
    reviews=int(os.getenv('FINAL_REVIEW_LIMIT','60'))+int(os.getenv('AUDIT_SAMPLE_SIZE','4'))+6
    # Broad bound before upstream responses; two transport retries + one JSON repair per call.
    faithfulness=min(40,reviews)+6
    plan={'discovery':count,'final_review':reviews,'boundary_repairs':6,'faithfulness':faithfulness,'comparison':1}
    errors=configuration_errors(overrides,public)
    price_errors=[e for e in errors if 'Pricing' in e]
    cost=None
    if not price_errors:
        g,c=role_settings('gemini',(overrides or {}).get('gemini')),role_settings('claude',(overrides or {}).get('claude'))
        gi,go,ci,co=g['price_in'],g['price_out'],c['price_in'],c['price_out']
        cost=((count+faithfulness)*(20000*gi+4000*go)+(reviews+7)*(16000*ci+4000*co))/1e6
    return {'episodes':len(eps),'beats':sum(len(e.beats) for e in eps),'parse_coverage':report.coverage,
            'planned_calls':plan,'estimated_cost_usd':cost,'maximum_attempts':sum(plan.values())*6,
            'budget_usd':float(budget_usd or os.getenv('MAX_RUN_COST_USD','2')),'configuration_errors':errors,
            'runtime':'Depends on provider latency, cache hits and rate limits; capped by MAX_RUN_SECONDS (default 1200).',
            'note':'Planning estimate before caching/retries; each attempt is reserved separately. Unknown prices are not free.'}


def route_pool(pool,ann,by,limit=60,audit=4):
    ordered=sorted(pool,key=lambda p:(-p.final,len(p.segments),p.key))
    best=[]
    for ep in sorted(by):
        p=next((p for p in ordered if p.first_episode==ep),None)
        if p: best.append(p)
    best.sort(key=lambda p:(-p.final,p.key))
    primary=(best+[p for p in ordered if p not in best])[:limit]
    def issues(p):
        events=[ann.get(b.id,{}) for s in p.segments for b in by[s.episode].beats_between(s.start,s.end)]
        return len(p.segments)>1 or not any(events) or any(a.get('uncertainties') for a in events) or any(a.get('resolution') and a.get('ending',0)>=6 for a in events)
    difficult=[p for p in ordered if p not in primary and issues(p)][:6]
    tail=[p for p in ordered if p not in primary+difficult]
    sample=sorted(tail,key=lambda p:hashlib.sha256(p.key.encode()).hexdigest())[:audit]
    for label,ps in [('priority',primary),('observable_uncertainty',difficult),('low_rank_audit',sample)]:
        for p in ps: p.rule['route']=label
    return primary+difficult+sample


def score(p):
    raw=p.judgment.get('raw',{})
    scores=raw.get('scores',{})
    keys=('hook','cliffhanger','standalone','stakes','economy','coherence')
    if all(type(scores.get(k)) in (int,float) and 0<=scores[k]<=10 for k in keys):
        p.final=sum(scores[k]*w for k,w in zip(keys,(.25,.25,.15,.15,.1,.1)))/10 - .01*(len(p.segments)-1)
        p.rule['editorial_scores']=scores
        if p.decision['status']=='recommended' and (min(scores[k] for k in ('hook','cliffhanger','standalone','coherence'))<6 or p.final<.6):
            p.checks['quality_floor']=check('uncertain','Pass verdict is below the editorial quality floor; review required.')
            update_decision(p)
    else:
        p.final=0.


def rank_explanations(ranked):
    names={'hook':'opening impact','cliffhanger':'unresolved ending','standalone':'standalone clarity',
           'stakes':'conflict and stakes','economy':'economy of setup','coherence':'coherence'}
    for i,p in enumerate(ranked):
        sc=p.rule.get('editorial_scores',{})
        strongest=max(names,key=lambda k:sc.get(k,0))
        text=f"{'First choice' if i==0 else 'Next distinct choice'}: {names[strongest]} is its strongest judged dimension. "
        text+=p.why+' '
        if i+1<len(ranked):
            other=ranked[i+1].rule.get('editorial_scores',{})
            adv=[names[k] for k in names if sc.get(k,0)>other.get(k,0)]
            text+=('Relative to the next clip, rated higher for '+', '.join(adv[:2])+'. ') if adv else 'The adjacent clips are close; source order breaks equal scores. '
        if p.rule.get('rerank_position') is not None:
            text+='Order includes a bounded comparison among already passing clips. '
        p.rule['formula']=text+'Concern: '+(p.limitations or 'Delivery and visual execution still need an editor’s review.')


def run(show, out=None, cache=None, progress=None, clients=None, inspection=False, show_name=None):
    """inspection=True explicitly saves an offline boundary review, never model recommendations."""
    show=Path(show); fingerprint,run_id=identity(show)
    out=Path(out) if out else ROOT/'outputs'/'v8'/run_id
    out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _run(show,out,cache,progress,clients,inspection,show_name,fingerprint,run_id)


def _run(show,out,cache,progress,clients,inspection,show_name,fingerprint,run_id):
    start=time.monotonic()
    previous=out/'input_manifest.json'
    if previous.exists() and json.loads(previous.read_text()).get('input_sha256')!=fingerprint:
        raise ValueError('This output folder belongs to different input. Choose a new output folder to preserve its results and decisions.')
    atomic_json(previous,{'input_sha256':fingerprint,'run_id':run_id})
    for path in script_files(show):
        target=out/'original_input'/path.relative_to(show)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    eps,report=parse_show(show,settings=STRICT)
    atomic_json(out/'parse_report.json',{**report.summary(),'warnings':report.all_warnings,'unparsed':report.unparsed})
    if not eps: raise ValueError('No timestamped episodes were parsed. Check the supported format.')
    if report.coverage<.99: raise ValueError(f'Only {report.coverage:.1%} of timestamp lines parsed. Fix input before running.')
    by={e.number:e for e in eps}; sig=neutral_signals(eps)
    atomic_json(out/'source_context.json',{str(ep.number):[b.render(precise=True) for b in ep.beats] for ep in eps})
    if clients is None:
        gemini,claude,budget=routes(out,cache or ROOT/'.cache'/'v8')
    else:
        gemini,claude,budget=clients
    if inspection:
        ann,connections,coverage={},[],[{'episode':ep.number,'chunk':i,'beat_ids':[b.id for b in bs],
            'status':'not_processed','error':'Explicit offline inspection: no model calls.'} for ep in eps for i,(bs,_) in enumerate(chunks(ep))]
        atomic_json(out/'discovery.json',{'annotations':ann,'connections':connections,'chunks':coverage})
    else:
        ann,connections,coverage=discover(eps,gemini,out,progress)
    pool,counts=generate(eps,ann,connections)
    selected=route_pool(pool,ann,by,max(1,min(200,int(os.getenv('FINAL_REVIEW_LIMIT','60')))),
                        max(0,min(20,int(os.getenv('AUDIT_SAMPLE_SIZE','4'))))) if not inspection else []
    selected_keys={p.key for p in selected}
    # Store unreviewed candidates too, so expensive-stage failure never discards discovery.
    for p in pool:
        if p.key not in selected_keys:
            p.judgment={'for_clip':p.key,'unavailable':True,'errors':['Not selected for final review; remains unverified.']}
        assess(p,by,sig)
    repairs=[]
    for i,p in enumerate(selected):
        judge_one(p,by,claude)
        if p.kind=='stitched': p.judgment['faithfulness']=faithful(p,by,gemini)
        assess(p,by,sig); score(p)
        if len(repairs)<6 and p.decision['status']!='recommended':
            q=repair(p,by)
            if q and q.key not in {x.key for x in pool+repairs}:
                judge_one(q,by,claude)
                if q.kind=='stitched': q.judgment['faithfulness']=faithful(q,by,gemini)
                assess(q,by,sig); score(q); repairs.append(q)
        atomic_json(out/'progress.json',{'run_id':run_id,'stage':'final_review','completed':i+1,'total':len(selected),
            'candidates':[promo_record(q,by) for q in pool+repairs], 'budget':budget.summary()})
        if progress: progress('Reviewing clips',i+1,len(selected))
    pool+=repairs
    S=replace(STRICT,top_k=25,max_per_episode=2,max_stitched_in_top=3)
    ranked,review,rejected,selection=rank_verified(pool,S)
    rerank_stats={}
    if len(ranked)>1 and abs(ranked[0].final-ranked[min(4,len(ranked)-1)].final)<.08 and not inspection:
        rr=rerank(claude,ranked[:5],by,rounds=1,workers=1,log=lambda *_:None)
        ranked=apply_rerank(ranked,rr['positions'],5)
        rerank_stats={k:v for k,v in rr.items() if k!='positions'}
    rank_explanations(ranked)
    states={s:sum(p.decision['status']==s for p in pool) for s in ('recommended','review','rejected')}
    processed={bid for c in coverage if c['status']=='processed' for bid in c['beat_ids']}
    attempted={bid for c in coverage for bid in c['beat_ids']}
    ep_coverage=[{'episode':ep.number,'beats':len(ep.beats),'included_in_chunks':sum(b.id in attempted for b in ep.beats),
        'processed_beats':sum(b.id in processed for b in ep.beats),
        'status':'processed' if all(b.id in processed for b in ep.beats) else 'incomplete'} for ep in eps]
    failures=[c for c in coverage if c['status']!='processed']
    try: revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except (OSError,subprocess.CalledProcessError): revision='unknown'
    usage={k:gemini.stats.get(k,0)+claude.stats.get(k,0) for k in gemini.stats}
    usage['est_cost_usd']=budget.summary()['provider_usage_cost_usd']
    meta={'implementation':VERSION,'scoring_preset':'v8','run_id':run_id,'input_sha256':fingerprint,
        'code_revision':revision,'code_sha256':code_fingerprint(),'show':show_name or show.name,
        'episodes':len(eps),'episode_range':[min(by),max(by)],'beats':sum(len(ep.beats) for ep in eps),
        'story_minutes':round(sum(ep.duration for ep in eps)/60,2),'model':gemini.name+' → '+claude.name,
        'models':{'discovery':gemini.name,'final_judge':claude.name},
        'generation_settings':{role:{'temperature':c.temperature,'max_output_tokens':c.max_tokens,'max_transport_retries':c.max_retries,'rpm':c.rpm} for role,c in [('gemini',gemini),('claude',claude)]},
        'usage_by_role':{'gemini':dict(gemini.stats),'claude':dict(claude.stats)},
        'prompt_versions':{'discovery':'discovery-v8.1','judge':JUDGE_VERSION},
        'prompt_sha256':hashlib.sha256((DISCOVERY_SYSTEM+DISCOVERY_CONTRACT+SYSTEM+EXTRA).encode()).hexdigest(),
        'coverage':ep_coverage,'coverage_claim':'All parsed beats included in chunks is not exhaustive semantic discovery.',
        'input_files':len(report.files),'failed_input_files':[Path(f.path).name for f in report.files if any('file skipped' in w for w in f.warnings)],
        'discovery_failures':failures,
        'unfinished_final_reviews':sum(p.judgment.get('unavailable',False) for p in selected),
        'unselected_candidates':len(pool)-len(selected)-len(repairs),'configuration_errors':configuration_errors() if clients is None else [],
        'candidates':{'continuous':sum(p.kind=='continuous' for p in pool),'stitched':sum(p.kind=='stitched' for p in pool),**counts},
        'routing':{'selected':len(selected),'audited':sum(p.rule.get('route')=='low_rank_audit' for p in selected),
                   'audit_recovered':sum(p.rule.get('route')=='low_rank_audit' and p.decision['status']=='recommended' for p in selected),
                   'repairs':len(repairs)},'rerank':rerank_stats,'returned':len(ranked),'review':len(review),'rejected':len(rejected),
        'seconds':round(time.monotonic()-start,3),'llm_usage':usage,'budget':budget.summary(),
        'parse':report.summary(),'eligibility':{**selection,'evaluated':len(pool),'status':states},
        'status':'inspection_only' if inspection else 'partial' if failures or any(p.judgment.get('unavailable') for p in selected) or any(any('file skipped' in w for w in f.warnings) for f in report.files) else 'complete',
        'inspection_only':inspection}
    result={'meta':meta,'promos':ranked,'review':review,'rejected':rejected,'needs_context':[],
        'episodes':eps,'ep_by_num':by,'signals':sig,'profile':{},'parse_report':report,'evaluated':pool}
    atomic_json(out/'evaluated.json',[promo_record(p,by) for p in pool])
    write_all(result,out)
    atomic_json(out/'progress.json',{'run_id':run_id,'stage':meta['status'],'budget':budget.summary()})
    return result,out
