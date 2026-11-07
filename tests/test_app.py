"""The web app, driven headlessly (Streamlit AppTest) against the mock LLM and the synthetic show."""
import json
from pathlib import Path

import pytest

from conftest import ROOT, SYNTHETIC

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def _by_label(widgets, label):
    for w in widgets:
        if w.label.startswith(label):
            return w
    raise AssertionError(f"no widget labelled {label!r}: {[w.label for w in widgets]}")


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")            # never pick a real provider by default
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_app_rules_only_flow(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    _by_label(at.sidebar.selectbox, "Provider").set_value("none").run()
    _by_label(at.text_input, "Folder path").set_value(str(SYNTHETIC)).run()
    text = " ".join(h.proto.body for h in at.get("html"))
    assert "100.0%" in text and "episodes" in text          # coverage + stats shown before running
    _by_label(at.button, "Find promos").click().run()
    assert not at.exception, at.exception
    tabs = [t.label for t in at.tabs]
    assert any(t.startswith("Main ranking") for t in tabs) and "Downloads" in tabs
    assert len(at.get("html")) > 5


def test_app_llm_flow_with_mock(tmp_path, monkeypatch, mock_server):
    at = _app(tmp_path, monkeypatch)
    _by_label(at.sidebar.selectbox, "Provider").set_value("openai").run()
    _by_label(at.sidebar.text_input, "Model").set_value("mock").run()
    _by_label(at.sidebar.text_input, "Base URL").set_value(mock_server.url).run()
    _by_label(at.sidebar.text_input, "API key").set_value("sk-test-key-0000").run()
    _by_label(at.sidebar.selectbox, "Scoring").set_value("v5").run()
    _by_label(at.text_input, "Folder path").set_value(str(SYNTHETIC)).run()
    notes = " ".join(h.proto.body for h in at.get("html"))
    assert "new model calls" in notes or "no new model calls" in notes     # the estimate is shown first
    _by_label(at.button, "Find promos").click().run()
    assert not at.exception, at.exception
    approve = [b for b in at.button if b.label == "Approve"]
    assert approve, "no promo cards rendered"
    approve[0].click().run()
    runs = sorted((ROOT / "outputs" / "ui_runs").glob("*"), key=lambda p: p.stat().st_mtime)
    out = runs[-1]
    assert (out / "feedback.jsonl").exists() and (out / "show_profile.json").exists()
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["scoring_preset"] == "v5" and "sk-test-key-0000" not in json.dumps(meta)
    for f in out.iterdir():
        assert "sk-test-key-0000" not in f.read_text(errors="ignore")


def test_app_public_mode_hides_server_features(tmp_path, monkeypatch):
    """On a shared server: no server folders, no other visitors' runs, no cache wipe, no custom URL for the key."""
    monkeypatch.setenv("PROMO_FINDER_PUBLIC", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-key-0000")
    at = _app(tmp_path, monkeypatch)
    labels = [w.label for w in list(at.text_input) + list(at.sidebar.text_input) + list(at.sidebar.selectbox)]
    assert not any(l.startswith(("Folder path", "Previous runs", "Base URL")) for l in labels), labels
    _by_label(at.sidebar.selectbox, "Provider").set_value("openai").run()
    assert not any(w.label.startswith("Base URL") for w in at.sidebar.text_input)   # key can't be pointed elsewhere
    assert not [p for p in at.sidebar.get("popover") if "Clear cache" in str(p.proto)]
    assert "Public demo" in " ".join(c.value for c in at.sidebar.caption)
