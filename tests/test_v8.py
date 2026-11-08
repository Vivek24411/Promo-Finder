"""Deterministic regression tests; scripted provider replies are not model-quality evidence."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import pytest
from promo_finder.candidates import Promo,Segment
from promo_finder.llm import LLM,LLMError,_Retryable
from promo_finder.parse import Beat,Episode,fmt_ts
from promo_finder.llm_stages import clip_context
from promo_finder.runtime import Budget,GuardedLLM
from promo_finder.semantic import chunks,generate,discover
from promo_finder.review import deterministic,assess,neutral_signals,judge_one
from promo_finder.v8 import score,run,route_pool


def episode():
    return Episode(1,'synthetic',beats=[
        Beat('e01b000',1,'ACTION',0,2,'The clock reverses.'),
        Beat('e01b001',1,'DIALOGUE',2,10,'If it reaches zero, this room disappears.',speaker='MARA'),
        Beat('e01b002',1,'ACTION',10,20,'Mara opens the control panel.'),
        Beat('e01b003',1,'DIALOGUE',20,25,'Which wire do I cut?',speaker='MARA')])


def clip(): return Promo('P01','continuous',[Segment(1,0,25)])


def response(p,ep):
    ctx=clip_context(p,{1:ep})
    refs=ctx.refs
    def sec(ref,**kw):return {'verdict':'pass','evidence':[ref.ref],'quotes':[{'ref':ref.ref,'text':ref.beat.text}],**kw}
    return {'clip_id':ctx.clip_id,'hook':sec(refs[0],description='The clock reverses.',mechanism='impossible event',timing='within_3s'),
        'standalone':sec(refs[1],conflict='Mara must stop the clock to save the room.',blocking_confusion='',harmless_curiosity='Why is the clock reversing?'),
        'ending':sec(refs[-1],central_question='Can she stop it?',final_moment='She must choose a wire.',open_question='Which wire prevents disappearance?',
                     final_moment_type='suspense',answered_within_clip=False,answered_by=[]),
        'scores':{k:8 for k in ('hook','cliffhanger','standalone','stakes','economy','coherence')},
        'why_it_works':'Immediate anomaly, explicit stakes, and an unresolved choice.','limitations':'Actual delivery needs review.','outcome':'recommended'}


class Fake:
    name='test:scripted'
    def __init__(self,data):self.data=data;self.prompts=[]
    def json(self,system,user,**kw):self.prompts.append((system,user));return copy.deepcopy(self.data)


def assess_reply(ep,p,data):
    client=Fake(data);judge_one(p,{1:ep},client);assess(p,{1:ep},neutral_signals([ep]));score(p);return client


def test_keyword_free_moment_can_be_top_and_cold_judge_sees_no_context():
    ep=episode();ep.logline='SECRET_CONTEXT_NOT_FOR_JUDGE'
    pool,counts=generate([ep],{},[])
    assert counts['legal_windows'] and pool
    p=clip();client=assess_reply(ep,p,response(p,ep))
    assert p.decision['status']=='recommended' and p.final==pytest.approx(.8)
    assert 'SECRET_CONTEXT_NOT_FOR_JUDGE' not in client.prompts[0][1]
    assert p.checks['standalone']['status']=='pass'  # harmless curiosity is not confusion


@pytest.mark.parametrize('kind',['late_line','late_hook','bad_quote','invented_id','answered','missing_context','bad_outcome','missing_score','partial'])
def test_false_passes_do_not_become_recommendations(kind):
    ep=episode();p=clip()
    if kind=='late_line':ep.beats[0].end=8
    if kind=='partial':ep.beats[-1].end=28
    raw=response(p,ep)
    if kind=='late_hook':raw['hook']['timing']='late'
    if kind=='bad_quote':raw['hook']['quotes'][0]['text']='invented explosion'
    if kind=='invented_id':raw['hook']['evidence']=['B99','B01']
    if kind=='answered':raw['ending'].update(answered_within_clip=True,answered_by=['B03'])
    if kind=='missing_context':raw['standalone']['blocking_confusion']='The stakes depend entirely on an unknown object.'
    if kind=='bad_outcome':raw['outcome']='review'
    if kind=='missing_score':del raw['scores']['economy']
    assess_reply(ep,p,raw)
    assert p.decision['status']!='recommended'


def test_subtle_visual_identity_reveal_remains_eligible():
    ep=episode();ep.beats[0].text='The portrait has her identical crescent-shaped birthmark.'
    p=clip();raw=response(p,ep);raw['hook']['description']='A matching birthmark exposes an identity connection.'
    assess_reply(ep,p,raw)
    assert p.decision['status']=='recommended'


@pytest.mark.parametrize('segments',[
    [Segment(1,0,19.999)],[Segment(1,0,60.001)],[Segment(99,0,25)],
    [Segment(1,.01,25)],[Segment(1,3,25)],[Segment(1,0,20),Segment(1,10,25)],
    [Segment(1,0,float('nan'))]])
def test_invalid_boundaries(segments):assert deterministic(Promo('','continuous',segments),{1:episode()})


def test_exact_duration_and_assembled_timeline():
    assert Promo('','stitched',[Segment(1,.1,10.2),Segment(1,20.2,30.1)]).duration==20
    ep=episode();p=Promo('','stitched',[Segment(1,0,10),Segment(1,20,25)])
    ctx=clip_context(p,{1:ep})
    assert ctx.duration==15 and ctx.seg_offsets==[0,10] and ctx.refs[-1].rel_start==10
    assert fmt_ts(20.001)=='00:20.001'


def test_chunk_coverage_without_truncating_whole_beats():
    ep=episode();ep.beats=[replace(ep.beats[0],id=f'e01b{i:03}',start=i*2,end=i*2+2,text='x'*80,scene=i//5) for i in range(100)]
    batches=list(chunks(ep,max_chars=600,overlap=3))
    assert len(batches)>1
    assert {b.id for bs,_ in batches for b in bs}=={b.id for b in ep.beats}
    assert set(b.id for b in batches[0][0])&set(b.id for b in batches[1][0])


def test_failed_discovery_records_every_episode(tmp_path):
    class Down:
        def json(self,*a,**k):raise LLMError('unavailable')
    eps=[episode(),replace(episode(),number=2)]
    _,_,coverage=discover(eps,Down(),tmp_path)
    assert len(coverage)==2 and all(c['status']=='failed' for c in coverage)


def client(tmp_path,budget=1):
    c=GuardedLLM(provider='openai_compat',model='test',base_url='https://example.invalid',
        cache_dir=str(tmp_path/'cache'),rpm=600000,max_retries=0,max_tokens=100,price_in=1,price_out=2)
    c.budget=Budget(tmp_path/'budget.sqlite',budget)
    return c


def test_duplicate_calls_cache_schema_model_and_prompt_invalidation(tmp_path,monkeypatch):
    c=client(tmp_path);calls=[]
    def transport(self,*args):calls.append(args);return '{"ok":true}',{'in':10,'out':2}
    monkeypatch.setattr(LLM,'_openai_compat',transport)
    with ThreadPoolExecutor(max_workers=3) as ex:
        values=list(ex.map(lambda _:c.json('s','u',schema={'version':1}),range(3)))
    assert all(v=={'ok':True} for v in values) and len(calls)==1
    c.json('s','u',schema={'version':2});c.json('s2','u',schema={'version':2})
    c.model='new';c.json('s','u',schema={'version':1})
    assert len(calls)==4
    assert c.budget.summary()['attempts']==4


def test_budget_exhaustion_and_unknown_pricing(tmp_path,monkeypatch):
    c=client(tmp_path,budget=.00001);calls=[]
    monkeypatch.setattr(LLM,'_openai_compat',lambda *a:(calls.append(1) or ('{}',{'in':1,'out':1})))
    with pytest.raises(LLMError,match='budget exhausted'):c.json('s','u')
    assert not calls
    c.price_in=0
    with pytest.raises(LLMError,match='Pricing unknown'):c.json('s','u')


def test_failed_attempts_are_reserved_and_retries_bounded(tmp_path,monkeypatch):
    c=client(tmp_path);c.max_retries=1
    def down(*a):raise _Retryable('503',retry_after=.001)
    monkeypatch.setattr(LLM,'_openai_compat',down)
    with pytest.raises(LLMError):c.json('s','u')
    summary=c.budget.summary()
    assert summary['attempts']==2 and summary['usage_unknown_attempts']==2
    assert summary['reserved_or_usage_cost_usd']>0 and summary['provider_usage_cost_usd']==0
    assert Budget(tmp_path/'budget.sqlite',1).summary()==summary


def test_crashed_request_is_not_blindly_repeated(tmp_path):
    c=client(tmp_path);Path(c.cache_dir).mkdir()
    (Path(c.cache_dir)/(c.cache_key('s','u')+'.pending')).write_text('{}')
    with pytest.raises(LLMError,match='unknown completion'):c.json('s','u')


def test_missing_strong_model_preserves_review(tmp_path):
    ep=episode();p=clip()
    class Down:
        name='claude:unavailable'
        def json(self,*a,**kw):raise LLMError('Claude unavailable')
    judge_one(p,{1:ep},Down());assess(p,{1:ep},neutral_signals([ep]))
    assert p.decision['status']=='review'


def test_incoherent_multi_segment_and_source_faithfulness():
    ep=episode();p=Promo('P01','stitched',[Segment(1,0,20),Segment(1,20,25)])
    raw=response(p,ep)
    raw['transition']={'verdict':'fail','evidence':['B03','B04'],'quotes':[{'ref':'B03','text':ep.beats[2].text},{'ref':'B04','text':ep.beats[3].text}],
                       'connection':'','reason':'Unexplained jump.'}
    assess_reply(ep,p,raw)
    assert p.decision['status']=='rejected'


def test_audit_routes_low_ranked_and_zero_signal_candidates():
    eps=[replace(episode(),number=i) for i in range(1,20)]
    by={ep.number:ep for ep in eps}
    pool=[Promo('', 'continuous',[Segment(i,0,25)]) for i in by]
    for i,p in enumerate(pool):p.final=i/20
    selected=route_pool(pool,{},by,limit=2,audit=3)
    assert sum(p.rule.get('route')=='low_rank_audit' for p in selected)==3
    assert sum(p.rule.get('route')=='observable_uncertainty' for p in selected)==6


def test_end_to_end_routes_and_resumes_without_duplicate_spend(tmp_path,monkeypatch):
    source=tmp_path/'show';source.mkdir()
    ep=episode()
    (source/'Ep1.md').write_text('EPISODE NUMBER: 1\n'+''.join(
        f'**{b.speaker if b.kind=="DIALOGUE" else b.kind}:**\nStart: ({fmt_ts(b.start)}) | End: ({fmt_ts(b.end)})\n{b.text}\n\n' for b in ep.beats))
    gemini=client(tmp_path);claude=client(tmp_path);gemini.model='gemini-test';claude.model='claude-test'
    calls=[]
    def transport(self,system,user,schema):
        calls.append(self.model)
        if self.model=='gemini-test':
            result={'events':[{'beat':'e01b000','quote':ep.beats[0].text,'hook':9,'ending':0,'stakes':8,'resolution':False,'reason':'Impossible clock.'},
                              {'beat':'e01b003','quote':ep.beats[-1].text,'hook':0,'ending':9,'stakes':9,'resolution':False,'reason':'Unresolved choice.'}],'connections':[]}
        else:
            import re
            cid=re.search(r'Candidate promo (K[A-F0-9]+)',user)[1]
            result=response(clip(),ep);result['clip_id']=cid
            # The 20-second variant has no actual suspense evidence. Model reply cites a nonexistent beat -> review.
        return json.dumps(result),{'in':100,'out':100}
    monkeypatch.setattr(LLM,'_openai_compat',transport)
    res,out=run(source,out=tmp_path/'out',clients=(gemini,claude,gemini.budget))
    assert res['meta']['coverage'][0]['processed_beats']==4
    assert res['promos'] and all(p.decision['status']=='recommended' for p in res['promos'])
    assert 'gemini-test' in calls and 'claude-test' in calls
    assert (out/'evaluated.json').exists() and (out/'source_context.json').exists()
    before=len(calls)
    res2,_=run(source,out=out,clients=(gemini,claude,gemini.budget))
    assert len(calls)==before and [p.key for p in res2['promos']]==[p.key for p in res['promos']]


def test_inferred_source_timing_cannot_auto_pass():
    ep=episode();ep.beats[0].timing_inferred=True
    p=clip();assess_reply(ep,p,response(p,ep))
    assert p.decision['status']=='review'


@pytest.mark.parametrize('timestamp',['00:61.00','01:75:01','00:01.1234567890'])
def test_invalid_source_timestamps_reported_instead_of_invented(tmp_path,timestamp):
    from promo_finder.parse import parse_show
    (tmp_path/'ep1.md').write_text(f'EPISODE NUMBER: 1\n**ACTION:**\nStart: ({timestamp}) | End: (02:00.00)\nA door opens.\n')
    eps,report=parse_show(tmp_path)
    assert not eps and report.coverage<1 and report.all_warnings


def test_hook_beat_mostly_before_the_cut_is_not_credited():
    # v8.6 (owner decision): a hook beat cut by the cut-in counts only if at least half of it plays in the first 3 s.
    # Here it began 3 s before the cut, so only 2 of its 5 s play: not credited (the B2 'credited with a strike it
    # never shows' failure). A beat that began 0.1 s early (95% plays) IS credited: see the test below.
    ep=episode();ep.beats[0].start=-3
    # A second source beat makes 0 a source boundary, while the claimed hook started earlier.
    ep.beats.append(Beat('e01b004',1,'CAMERA',0,25,'Wide shot.'))
    p=clip();raw=response(p,ep)
    raw['ending']['evidence']=['B04'];raw['ending']['quotes']=[{'ref':'B04','text':ep.beats[3].text}]
    assess_reply(ep,p,raw)
    assert p.decision['status']!='recommended'


def test_completed_provider_reply_survives_cache_write_failure_without_repeat(tmp_path,monkeypatch):
    import os
    c=client(tmp_path);calls=[]
    monkeypatch.setattr(LLM,'_openai_compat',lambda *a:(calls.append(1) or ('{}',{'in':10,'out':10})))
    def disk_error(*a):raise OSError('disk full')
    monkeypatch.setattr(os,'replace',disk_error)
    with pytest.raises(OSError):c.json('s','u')
    with pytest.raises(LLMError,match='unknown completion'):c.json('s','u')
    assert len(calls)==1


def test_unknown_or_nan_pricing_never_makes_paid_call(tmp_path,monkeypatch):
    c=client(tmp_path);c.price_in=float('nan');calls=[]
    monkeypatch.setattr(LLM,'_openai_compat',lambda *a:calls.append(1))
    with pytest.raises(LLMError,match='Pricing unknown'):c.json('s','u')
    assert not calls


@pytest.mark.parametrize('shot_end,eligible', [(26, True), (40, False)])
def test_cited_shot_cut_by_the_clip_counts_only_when_half_of_it_plays(shot_end, eligible):
    """v8.5: a final shot that runs past the cut-out is valid ending evidence when at least half of it is inside
    the clip (22-26 s: 75% plays); a shot mostly outside the clip (22-40 s: 17% plays) stays uncertain."""
    ep = episode()
    ep.beats.append(Beat('e01b004', 1, 'CAMERA', 22, shot_end, "Close-up of Mara's hand hovering over the wires."))
    p = clip(); raw = response(p, ep)
    ref = clip_context(p, {1: ep}).refs[-1]
    assert ref.beat.id == 'e01b004' and ref.partial_end
    raw['ending']['evidence'] = [ref.ref]; raw['ending']['quotes'] = [{'ref': ref.ref, 'text': ref.beat.text}]
    assess_reply(ep, p, raw)
    assert (p.decision['status'] == 'recommended') is eligible
    assert (p.checks['ending']['status'] == 'pass') is eligible


def test_v86_hook_timing_rule_is_whole_beat_and_proportional():
    from promo_finder.review import lands_in_opening
    ep = episode(); p = clip()
    def lands(start, end):
        ep.beats[0].start, ep.beats[0].end = start, end
        return lands_in_opening(clip_context(p, {1: ep}).refs[0], p)
    assert lands(0, 2) and lands(-0.1, 2)          # whole beat inside; or 95% of it plays
    assert lands(1.0, 3.4)                          # starts in time, ends by 3.5 s
    assert lands(1.0, 4.0)                          # 2.0 of 3.0 s plays before 3.0 s
    assert not lands(2.5, 6.0)                      # mostly after the window: late
    assert not lands(-3, 2) and not lands(0, 8)     # mostly before the cut; or a long beat mostly after 3 s


def test_v86_unquoted_extra_citation_is_dropped_not_fatal():
    ep = episode(); p = clip(); raw = response(p, ep)
    raw['standalone']['evidence'] = ['B02', 'B03']   # B03 cited but not quoted
    assess_reply(ep, p, raw)
    assert p.decision['status'] == 'recommended'
    assert p.judgment['sections']['standalone']['unquoted_refs_dropped'] == ['B03']


def test_v86_altered_quote_still_blocks():
    ep = episode(); p = clip(); raw = response(p, ep)
    raw['standalone']['quotes'].append({'ref': 'B03', 'text': 'Mara rips out every wire.'})
    assess_reply(ep, p, raw)
    assert p.decision['status'] != 'recommended' and p.checks['standalone']['status'] == 'uncertain'


def test_v86_broken_reply_is_retried_once():
    ep = episode(); p = clip(); good = response(p, ep)
    class Flaky(Fake):
        def json(self, system, user, **kw):
            self.prompts.append((system, user))
            return {'beat': 'fragment'} if len(self.prompts) == 1 else copy.deepcopy(self.data)
    client = Flaky(good)
    judge_one(p, {1: ep}, client); assess(p, {1: ep}, neutral_signals([ep])); score(p)
    assert len(client.prompts) == 2 and 'could not be used' in client.prompts[1][1]
    assert p.decision['status'] == 'recommended'


def test_v86_nested_scores_block_is_retried_too():
    ep = episode(); p = clip(); good = response(p, ep)
    bad = copy.deepcopy(good); bad['standalone']['scores'] = bad.pop('scores')   # valid JSON, wrong place
    class Nested(Fake):
        def json(self, system, user, **kw):
            self.prompts.append((system, user))
            return copy.deepcopy(bad if len(self.prompts) == 1 else good)
    client = Nested(good)
    judge_one(p, {1: ep}, client); assess(p, {1: ep}, neutral_signals([ep])); score(p)
    assert len(client.prompts) == 2 and p.decision['status'] == 'recommended'


def test_v86_uncertain_ending_is_an_answer_not_a_retry():
    ep = episode(); p = clip(); raw = response(p, ep)
    raw['ending'].update(verdict='uncertain', final_moment_type='uncertain')
    client = Fake(raw); judge_one(p, {1: ep}, client)
    assert len(client.prompts) == 1 and not p.judgment.get('unavailable')
