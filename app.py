"""Promo Finder v8: load a screenplay, find promos, review editorial evidence."""
from __future__ import annotations
import hashlib
import io
import json
import os
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
import streamlit as st
from promo_finder import config as C, v8
from promo_finder.parse import parse_show, fmt_ts
from promo_finder.llm import PRESETS
from promo_finder.runtime import (VERSION, KNOWN_PRICES, PROVIDERS, ROLE_LABELS, configuration_errors,
                                   role_settings, routes)

HERE=Path(__file__).parent
SAMPLE=HERE/'tests'/'fixtures'/'synthetic_show'
C.load_dotenv()
PUBLIC=os.getenv('PROMO_FINDER_PUBLIC','').lower() in ('1','true','yes')
RUNS=HERE/'outputs'/'v8'
st.set_page_config(page_title='Promo Finder',layout='wide',initial_sidebar_state='collapsed')
st.html('''<style>
.block-container{max-width:1120px;padding-top:2rem}
h1{font-size:2rem!important;letter-spacing:-.035em}h2{font-size:1.3rem!important}
.pf-intro{color:#514c46;border-bottom:1px solid #d8d4cc;padding-bottom:18px;margin-bottom:22px}
.pf-steps{color:#514c46;font-size:14px;line-height:1.8}
button:focus-visible,a:focus-visible,input:focus-visible{outline:3px solid #795538!important;outline-offset:3px}
pre{white-space:pre-wrap!important;overflow-wrap:anywhere}
@media(max-width:600px){.block-container{padding:1rem}.pf-steps{font-size:13px}h1{font-size:1.7rem!important}}
</style>''')


@contextmanager
def materialized(source):
    if source['kind'] in ('sample','local'):
        yield Path(source['path']); return
    with tempfile.TemporaryDirectory(prefix='promo_') as folder:
        root=Path(folder)
        total=0
        for name,data in source['files']:
            if name.lower().endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    members=[f for f in z.infolist() if not f.is_dir() and Path(f.filename).suffix.lower() in ('.txt','.md') and not Path(f.filename).name.startswith('.')]
                    total+=sum(f.file_size for f in members)
                    if total>50_000_000 or len(members)>2000: raise ValueError('Archive exceeds 50 MB or 2,000 script files.')
                    for f in members:
                        target=root/Path(f.filename).name
                        if target.exists(): raise ValueError('Duplicate script filenames in upload. Give each episode a unique filename.')
                        target.write_bytes(z.read(f))
            else:
                total+=len(data)
                if total>50_000_000: raise ValueError('Upload exceeds 50 MB of screenplay text.')
                target=root/Path(name).name
                if target.exists(): raise ValueError('Duplicate script filenames.')
                target.write_bytes(data)
        yield root


def load_run(path):
    data=json.loads((path/'promos.json').read_text())
    decisions={}
    if (path/'feedback.jsonl').exists():
        for line in (path/'feedback.jsonl').read_text().splitlines():
            try:
                item=json.loads(line);decisions[item.get('clip_key',item['id'])]=item['verdict']
            except (ValueError,KeyError): pass
    return {'data':data,'out':str(path),'decisions':decisions}


def run_label(path):
    try:
        meta=json.loads(path.read_text())['meta']
        when=time.strftime('%d %b %H:%M',time.localtime(path.stat().st_mtime))
        return f"{meta.get('show','Screenplay')} · {when} · {meta.get('status','saved run').replace('_',' ')}"
    except (ValueError,KeyError,OSError):
        return path.parent.name


def set_verdict(R,row,verdict):
    key=row.get('clip_key',row['id'])
    with (Path(R['out'])/'feedback.jsonl').open('a') as f:
        f.write(json.dumps({'id':row['id'],'clip_key':key,'verdict':verdict,'ts':time.time()})+'\n')
    R['decisions'][key]=verdict


def detail(R,row):
    st.subheader('Clip '+row['id'])
    key=row.get('clip_key',row['id'])
    human=R['decisions'].get(key,'not reviewed')
    st.caption(f"System: {row['status']} · Editor: {human} · {row['duration_sec']:g} seconds")
    for i,s in enumerate(row['segments'],1):
        st.markdown(f"**{i}. Episode {s['episode']} · {s['start']} → {s['end']}**")
    for label,field in [('Opening hook','hook'),('Ending','final_moment'),('Unresolved question','cliffhanger'),
                        ('Why consider it','why_it_works'),('Why it ranks here','rank_explanation'),('Concern','limitations')]:
        if row.get(field): st.markdown(f"**{label}**  \n{row[field]}")
    for reason in row.get('failure_reasons',[]): st.error(reason)
    for reason in row.get('review_reasons',[]): st.warning(reason)
    with st.expander('Opening, ending and validation evidence'):
        for name,c in row['checks'].items():
            st.markdown(f"**{name.replace('_',' ').capitalize()} — {c['status']}**  \n{c['reason']}")
            for ev in c.get('evidence',[]):
                st.caption(f"{ev['beat']} · {ev['time']} · clip {ev['clip_time']}")
                st.text(ev['text'])
    with st.expander('Read screenplay excerpt',expanded=True):
        st.caption('Source text in playback order. Screenplay timing does not establish acting or actual delivery.')
        for s in row['segments']:
            st.code('\n'.join(s['script']),language=None,wrap_lines=True)
    with st.expander('Source context'):
        context=Path(R['out'])/'source_context.json'
        if context.exists():
            source=json.loads(context.read_text())
            for ep in sorted({s['episode'] for s in row['segments']}):
                st.code('\n'.join(source.get(str(ep),[])),language=None,wrap_lines=True)
        else: st.caption('Source context is unavailable for this older run.')
    a,b=st.columns(2)
    a.button('Approve',key='approve_'+row['id'],on_click=set_verdict,args=(R,row,'approved'),width='stretch')
    b.button('Reject',key='reject_'+row['id'],on_click=set_verdict,args=(R,row,'rejected'),width='stretch')
    st.caption('Editor decisions are saved separately. Approval does not change the system’s original judgment.')


st.title('Promo Finder')
st.html('<div class="pf-intro">Find precise promotional cuts in timestamped screenplays.<br><span class="pf-steps">1. Load screenplay &nbsp; → &nbsp; 2. Find promos &nbsp; → &nbsp; 3. Review recommendations</span></div>')
with st.sidebar:
    st.caption('Editorial workspace · '+VERSION)
    if PUBLIC:
        st.info('Public demo. Uploaded screenplays are processed for your session only; results are not listed to '
                'other visitors. Server folders are not accessible.')
    else:
        runs=sorted(RUNS.glob('*/promos.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        if runs:
            selected=st.selectbox('Previous runs',runs,index=None,format_func=run_label)
            if selected and st.button('Open run'):
                st.session_state['result']=load_run(selected.parent)
        st.caption('Confidential source and outputs stay on this server. Share only with the permitted audience.')

source=None
current_fingerprint=None
mode=st.radio('Load screenplay',['Upload files','Sample show'] if PUBLIC else ['Upload files','Sample show','Local folder'],horizontal=True)
if mode=='Sample show':
    st.caption('Original synthetic test drama. It does not establish quality on other shows.')
    source={'kind':'sample','path':str(SAMPLE)}
elif mode=='Local folder':
    folder=st.text_input('Folder path',placeholder='/path/to/screenplays')
    if folder:
        if Path(folder).is_dir(): source={'kind':'local','path':folder}
        else: st.error('That folder does not exist.')
else:
    uploads=st.file_uploader('Episode screenplays: select the files, a whole folder, or one .zip',type=['txt','md','zip'],
                             accept_multiple_files='directory' if st.toggle('Upload a whole folder',key='folder_mode') else True)
    if uploads: source={'kind':'upload','files':[(f.name,f.getvalue()) for f in uploads]}
with st.expander('Supported input format'):
    st.write('One UTF-8 file per episode, with an episode number and timestamped dialogue or actions. Decimal seconds are supported. Missing end times are flagged for review.')
    st.code('EPISODE NUMBER: 1\n**ACTION:**\nStart: (00:00.00) | End: (00:02.50)\nThe clock begins counting backwards.\n\n**MARA:**\nStart: (00:02.50) | End: (00:05.00)\nWe have one minute to stop it.',language=None)

def model_panel():
    """Per-session provider/model/key for each stage. Keys stay in this browser session's memory only."""
    overrides,budget={},None
    default_budget=float(os.getenv('MAX_RUN_COST_USD','2'))
    with st.expander('Models and API keys (optional)'):
        st.caption('By default the server\'s models run: '
                   f"{role_settings('gemini')['model']} for the first pass and {role_settings('claude')['model']} for the final review. "
                   'You can swap either stage to another provider or model, or use your own key. Keys stay in this browser '
                   'session only and are never saved, logged or written into results.')
        for col,role in zip(st.columns(2),('gemini','claude')):
            d=role_settings(role)
            with col:
                st.markdown(f'**{ROLE_LABELS[role]}**')
                names=['server']+[p for p in PROVIDERS if not (PUBLIC and p=='ollama')]
                choice=st.selectbox('Provider',names,key=f'prov_{role}',
                                    format_func=lambda p,d=d:f"Server default ({d['model']})" if p=='server' else PROVIDERS[p])
                if choice=='server':
                    key=st.text_input('Your API key (optional)',type='password',key=f'key_{role}',
                                      help='Leave empty to use the server key for the default model.')
                    if key:
                        overrides[role]={'provider':d['provider'],'model':d['model'],'api_key':key,
                                         'price_in':d['price_in'],'price_out':d['price_out']}
                    continue
                preset='openai' if choice=='custom' else choice
                model=st.text_input('Model',value='' if choice=='custom' else PRESETS[preset][1],key=f'model_{role}_{choice}')
                base=st.text_input('Endpoint URL',placeholder='https://your-gateway.example/v1',key=f'url_{role}') if choice=='custom' else ''
                key=st.text_input('API key',type='password',key=f'key_{role}_{choice}',help='Kept in this browser session only.')
                pin,pout=KNOWN_PRICES.get(model.strip(),(0.0,0.0))
                a,b=st.columns(2)
                price_in=a.number_input('Input $ / 1M tokens',min_value=0.0,value=float(pin),step=0.05,format='%.2f',key=f'pin_{role}_{model}')
                price_out=b.number_input('Output $ / 1M tokens',min_value=0.0,value=float(pout),step=0.05,format='%.2f',key=f'pout_{role}_{model}')
                if not pin: st.caption('Enter this model\'s prices from the provider\'s pricing page; unknown prices block paid calls.')
                overrides[role]={'provider':choice,'model':model,'base_url':base,'api_key':key,'price_in':price_in,'price_out':price_out}
        own_keys=len(overrides)==2 and all(o.get('api_key') for o in overrides.values())
        if own_keys or not PUBLIC:
            value=st.number_input('Run budget (USD)',min_value=0.10,max_value=50.0,value=default_budget,step=0.50,key='budget')
            budget=value if abs(value-default_budget)>1e-9 else None
        else:
            st.caption(f'Run budget: ${default_budget:.2f}. Use your own keys for both stages to change it.')
    return overrides,budget


overrides,budget=model_panel()
# Results for different model settings never share a run folder (the key itself is never part of this tag).
settings_tag=('-m'+hashlib.sha256(json.dumps({r:{k:v for k,v in o.items() if k!='api_key'} for r,o in overrides.items()},
                                             sort_keys=True,default=str).encode()+str(budget).encode()).hexdigest()[:8]
              ) if overrides or budget else ''

if source:
    try:
        with materialized(source) as folder:
            current_fingerprint,_=v8.identity(folder)
            eps,report=parse_show(folder,settings=C.Settings())
            estimate=v8.estimate(folder,overrides,PUBLIC,budget)
        st.subheader('Screenplay ready' if eps else 'Check your screenplay')
        st.write(f"{len(eps)} episodes · {sum(len(e.beats) for e in eps):,} beats · {report.coverage:.1%} timestamp lines parsed")
        with st.expander('Detected episodes and parsing notes'):
            st.dataframe([{'Episode':e.number,'Duration':fmt_ts(e.duration),'Beats':len(e.beats)} for e in eps],hide_index=True,width='stretch')
            for warning in report.all_warnings: st.warning(warning)
        errors=configuration_errors(overrides,PUBLIC)
        if errors:
            st.warning('Finding recommendations is unavailable until the model settings are complete. An offline '
                       'inspection can still show source boundaries for manual review.')
            for error in errors: st.caption('• '+error)
        price=estimate['estimated_cost_usd']
        st.caption((f"Planning estimate: ${price:.2f} before retries and cache reuse. " if price is not None else 'Cost estimate unavailable: token prices are not configured. ')+f"Run budget: ${estimate['budget_usd']:.2f}. Runtime varies with provider latency.")
        ready=bool(eps) and report.coverage>=.99
        find=st.button('Find promos',type='primary',disabled=not ready or bool(errors))
        inspect=st.button('Inspect boundaries without models',disabled=not ready) if errors else False
        if find or inspect:
            status=st.empty()
            def progress(stage,done,total): status.info(f'{stage} · {done}/{total} completed')
            with materialized(source) as folder:
                fingerprint,run_id=v8.identity(folder)
                # Private runs use stable IDs for resume. Demo sessions do not list each other's results.
                if PUBLIC:
                    st.session_state.setdefault('demo_scope',hashlib.sha256(os.urandom(32)).hexdigest()[:12])
                run_dir=RUNS/(run_id+settings_tag+('-'+st.session_state['demo_scope'] if PUBLIC else '')+('-inspection' if inspect else ''))
                clients=routes(run_dir,HERE/'.cache'/'v8',overrides,budget) if (overrides or budget) and not inspect else None
                result,out=v8.run(folder,out=run_dir,progress=progress,inspection=inspect,clients=clients,
                                  show_name='Synthetic sample' if source['kind']=='sample' else 'Uploaded screenplay' if source['kind']=='upload' else None)
            st.session_state['result']=load_run(out)
            status.empty()
    except (ValueError,OSError,zipfile.BadZipFile) as e:
        st.error(str(e))

R=st.session_state.get('result')
if R:
    data=R['data'];meta=data['meta']
    st.subheader('Review recommendations')
    st.caption('Results for '+meta.get('show','screenplay'))
    if current_fingerprint and current_fingerprint!=meta.get('input_sha256'):
        st.info('These saved results belong to a different input from the screenplay currently loaded above.')
    if meta.get('status')!='complete': st.info('This run is incomplete or an offline inspection. Unfinished checks remain in Needs review.')
    tabs=st.tabs([f"Recommended ({len(data['promos'])})",f"Needs review ({len(data.get('review',[]))})",
                  f"Rejected ({len(data.get('rejected',[]))})",'Run details','Downloads'])
    for tab,field in zip(tabs[:3],['promos','review','rejected']):
        with tab:
            rows=data.get(field,[])
            if not rows: st.caption('No clips in this view.')
            else:
                st.dataframe([{'Rank':r['rank'],'Episode':', '.join(str(s['episode']) for s in r['segments']),
                    'Seconds':r['duration_sec'],'Description':r['hook'] or r['final_moment'] or 'Unreviewed source window',
                    'Editor':R['decisions'].get(r.get('clip_key',r['id']),'not reviewed')} for r in rows],hide_index=True,width='stretch')
                choice=st.selectbox('Review clip',range(len(rows)),format_func=lambda i,rows=rows:rows[i]['id']+' · '+(rows[i]['hook'] or 'Source window')[:100],key='select_'+field)
                detail(R,rows[choice])
    with tabs[3]:
        models=meta.get('models',{})
        st.caption(f"First pass: {models.get('discovery','?')} reads every episode and proposes candidates. Final review: "
                   f"{models.get('final_judge','?')} judges each shortlisted clip independently. Code validates evidence and timing. "
                   'No marketing performance is predicted.')
        st.json(meta)
    with tabs[4]:
        for name in ('promos.json','promos.csv','report.html','run_meta.json','evaluated.json','discovery.json','llm_calls.jsonl','feedback.jsonl'):
            path=Path(R['out'])/name
            if path.exists(): st.download_button('Download results · '+name,path.read_bytes(),file_name=name,key='download_'+name)
