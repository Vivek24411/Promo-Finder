"""Editor workflow driven in-process; no sockets or paid model calls."""
import json
import pytest
from conftest import ROOT,SYNTHETIC
AppTest=pytest.importorskip('streamlit.testing.v1').AppTest


def app(monkeypatch,tmp_path):
    monkeypatch.setenv('ANTHROPIC_API_KEY','')
    monkeypatch.setenv('GEMINI_INPUT_USD_PER_MILLION','0')
    # Keep test outputs outside actual user runs.
    import promo_finder.v8 as v8
    actual=v8.run
    monkeypatch.setattr(v8,'run',lambda show,**kw:actual(show,**{**kw,'out':tmp_path/'run'}))
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=120).run()
    assert not at.exception
    return at


def button(at,label):return next(b for b in at.button if b.label==label)


def test_model_panel_defaults_to_server_settings_and_missing_configuration(monkeypatch,tmp_path):
    at=app(monkeypatch,tmp_path)
    providers=[s for s in at.selectbox if s.label=='Provider']
    assert len(providers)==2 and all(s.value=='server' for s in providers)   # both stages default to the server's models
    assert not any(t.label=='Model' for t in at.text_input)                  # only shown when a stage is switched
    at.radio[0].set_value('Sample show').run()
    assert button(at,'Find promos').disabled                                 # server key/prices missing in this test
    assert not button(at,'Inspect boundaries without models').disabled
    assert not at.exception


def test_inspection_review_persistence_and_downloads(monkeypatch,tmp_path):
    at=app(monkeypatch,tmp_path)
    at.radio[0].set_value('Sample show').run()
    button(at,'Inspect boundaries without models').click().run()
    assert not at.exception
    assert at.tabs[0].label=='Recommended (0)' and 'Downloads' in [t.label for t in at.tabs]
    out=tmp_path/'run';before=(out/'promos.json').read_bytes()
    button(at,'Approve').click().run()
    assert not at.exception and json.loads((out/'feedback.jsonl').read_text().splitlines()[-1])['verdict']=='approved'
    assert (out/'promos.json').read_bytes()==before
    button(at,'Reject').click().run()
    assert json.loads((out/'feedback.jsonl').read_text().splitlines()[-1])['verdict']=='rejected'
    assert (out/'promos.json').read_bytes()==before


def test_public_demo_allows_uploads_but_never_server_folders(monkeypatch,tmp_path):
    monkeypatch.setenv('PROMO_FINDER_PUBLIC','1')
    at=app(monkeypatch,tmp_path)
    assert at.radio[0].options==['Upload files','Sample show']
    assert at.get('file_uploader') and not [t for t in at.text_input if t.label=='Folder path']
    assert not [s for s in at.selectbox if s.label=='Previous runs']
    prov=[s for s in at.selectbox if s.label=='Provider'][1]
    assert 'ollama' not in prov.options                                     # nothing that runs on the server's own network
    prov.set_value('anthropic').run()
    at.radio[0].set_value('Sample show').run()
    text=' '.join(c.value for c in at.caption)
    assert 'Enter your own API key for the final review model' in text      # server key is never spent on a swapped model
    assert button(at,'Find promos').disabled


def test_public_custom_endpoint_must_be_public_https(monkeypatch,tmp_path):
    monkeypatch.setenv('PROMO_FINDER_PUBLIC','1')
    at=app(monkeypatch,tmp_path)
    [s for s in at.selectbox if s.label=='Provider'][0].set_value('custom').run()
    next(t for t in at.text_input if t.label=='Endpoint URL').set_value('http://127.0.0.1:8080/v1').run()
    next(t for t in at.text_input if t.label=='API key').set_value('sk-visitor-0000').run()
    at.radio[0].set_value('Sample show').run()
    assert 'https:// URL' in ' '.join(c.value for c in at.caption) and button(at,'Find promos').disabled


def test_swapped_models_with_own_keys_run_and_keys_are_never_written(monkeypatch,tmp_path,mock_server):
    """Both stages pointed at another provider with typed keys (the interview scenario). The run uses them,
    run_meta names the swapped models, and the typed keys appear in no output file."""
    monkeypatch.setattr('promo_finder.llm.RateLimiter.acquire',lambda self,*a,**k:None)
    at=app(monkeypatch,tmp_path)
    for i,role_key in enumerate(('sk-first-SECRET-1111','sk-final-SECRET-2222')):
        [s for s in at.selectbox if s.label=='Provider'][i].set_value('custom').run()
        [t for t in at.text_input if t.label=='Endpoint URL'][i].set_value(mock_server.url).run()
        [t for t in at.text_input if t.label=='Model'][i].set_value('mock').run()
        [t for t in at.text_input if t.label=='API key'][i].set_value(role_key).run()
        ins=[n for n in at.number_input if n.label=='Input $ / 1M tokens'];outs=[n for n in at.number_input if n.label=='Output $ / 1M tokens']
        ins[i].set_value(0.10).run();outs[i].set_value(0.50).run()
    at.radio[0].set_value('Sample show').run()
    assert not button(at,'Find promos').disabled
    button(at,'Find promos').click().run()
    assert not at.exception
    out=tmp_path/'run'
    meta=json.loads((out/'run_meta.json').read_text())
    assert meta['models']=={'discovery':'openai_compat:mock','final_judge':'openai_compat:mock'}
    for f in out.rglob('*'):
        if f.is_file():
            data=f.read_bytes()
            assert b'SECRET-1111' not in data and b'SECRET-2222' not in data, f
