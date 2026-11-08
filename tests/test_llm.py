"""Model-swap robustness (2d), rate limiting (2a) and key safety (Phase 1) - all against the mock server."""
import hashlib
import json
import time
import types

import pytest

from conftest import SYNTHETIC
from promo_finder.llm import LLM, LLMError, RateLimiter, extract_json
from promo_finder.llm_stages import (JUDGE_TEMPLATE, JUDGE_TEMPLATE_V5, MINER_TEMPLATE, MINER_TEMPLATE_V5,
                                     mine_moments)
from promo_finder.parse import parse_show

FAKE_KEY = "sk-test-SECRET-1234567890abcdef"


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Sure!\n```\n{"a": 1}\n```\nhope this helps', {"a": 1}),
    ('<think>maybe {x} or {"b": 2}?</think>\n{"a": 1}', {"a": 1}),
    ('<thinking>{"b": 2}</thinking>{"a": 1}', {"a": 1}),
    ('{"a": {"nested": [1, 2]}} and then {"b": 2} trailing {junk}', {"a": {"nested": [1, 2]}}),
    ('Answer: [{"clip": 1}]', [{"clip": 1}]),
    ('reasoning with {broken json then {"a": 1}', {"a": 1}),
])
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_extract_json_fails_cleanly():
    with pytest.raises(ValueError):
        extract_json("no json here at all")


@pytest.mark.parametrize("model", ["mock-fenced", "mock-think", "mock-trailing"])
def test_messy_replies_from_a_server(make_llm, model):
    eps, _ = parse_show(SYNTHETIC)
    hooks, cliffs, st = mine_moments(make_llm(model), "Test", eps[:2], log=lambda *_: None)
    assert st["episodes"] == 2 and hooks and cliffs


def test_invalid_ids_are_dropped_and_counted(make_llm):
    eps, _ = parse_show(SYNTHETIC)
    _, _, st = mine_moments(make_llm("mock-badids"), "Test", eps[:3], log=lambda *_: None)
    assert st["invalid_ids"] == 3            # one invented ID per episode, all rejected


def test_429_and_503_are_retried(make_llm):
    for model in ("mock-429x2", "mock-503x1"):
        llm = make_llm(model, max_retries=3)
        assert llm.json("s", "Describe this show: abcdefgh", tag="t")
        assert llm.stats["retries"] >= 1 and llm.stats["calls"] == 1


def test_timeout_gives_a_clear_error_naming_the_item(make_llm):
    llm = make_llm("mock-timeout", timeout=0.5, max_retries=1)
    with pytest.raises(LLMError) as e:
        llm.json("s", "u", tag="miner:ep07")
    assert "miner:ep07" in str(e.value) and e.value.kind in ("unavailable", "rate_limit")


def test_cache_hit_and_cache_only(make_llm, mock_server):
    llm = make_llm()
    before = mock_server.calls
    llm.json("s", "Describe this show: abcdefgh", tag="a")
    llm.json("s", "Describe this show: abcdefgh", tag="b")
    assert mock_server.calls - before == 1 and llm.stats["cache_hits"] == 1
    from promo_finder.llm import CacheMiss
    dry = make_llm(cache_only=True)
    with pytest.raises(CacheMiss):
        dry.json("s", "never asked before", tag="x")


def test_v4_cache_key_format_is_unchanged():
    """v4 answers stay valid: for v4 parameters the key is the original sha256(provider|model|system|user)."""
    llm = LLM(provider="openai_compat", model="m")
    old = hashlib.sha256("openai_compat|m|S|U".encode()).hexdigest()[:24]
    assert llm.cache_key("S", "U") == old
    assert LLM(provider="openai_compat", model="m", temperature=0.7).cache_key("S", "U") != old


def test_api_key_never_written_anywhere(make_llm, tmp_path, mock_server):
    """The key may only travel in the Authorization header to the chosen provider."""
    llm = make_llm(api_key=FAKE_KEY)
    llm.json("s", "Describe this show: abcdefgh", tag="t")
    assert FAKE_KEY not in repr(llm)
    assert mock_server.bodies[-1]["headers"].get("Authorization") == f"Bearer {FAKE_KEY}"
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert FAKE_KEY not in p.read_text(errors="ignore"), p
    assert FAKE_KEY not in llm._redact(f"error: invalid key {FAKE_KEY} key=abcdefghijklmnop")


def test_rate_limiter_spaces_requests():
    lim = RateLimiter(rpm=600)            # one request every 0.1 s
    t0 = time.monotonic()
    for _ in range(4):
        lim.acquire()
    assert time.monotonic() - t0 >= 0.29


def test_rate_limiter_token_budget():
    lim = RateLimiter(rpm=6000, tpm=1000)
    lim.acquire(900)
    lim.window[0] = (time.monotonic() - 59.8, 900)     # pretend the first call was 59.8 s ago
    t0 = time.monotonic()
    lim.acquire(300)                                    # would exceed 1000/min -> waits ~0.2 s
    assert time.monotonic() - t0 >= 0.15


def test_v5_prompts_really_differ():
    assert "emotional_reveal" in MINER_TEMPLATE_V5 and "emotional_reveal" not in MINER_TEMPLATE
    assert '"unanswered": true' in JUDGE_TEMPLATE_V5 and "unanswered" not in JUDGE_TEMPLATE
    JUDGE_TEMPLATE_V5.format(clips="x")                 # still a valid format string
    MINER_TEMPLATE_V5.format(show="s", num=1, logline="l", beats="b")


# ---------------------------------------------------------------------------- Anthropic adapter (mocked SDK)
def _fake_anthropic(monkeypatch, behaviour):
    anthropic = pytest.importorskip("anthropic")

    class Messages:
        def create(self, **kw):
            Messages.last = kw
            return behaviour(kw)

    class Client:
        def __init__(self, **kw):
            Client.kw = kw
            self.messages = Messages()

    monkeypatch.setattr(anthropic, "Anthropic", Client)
    return Client, Messages


def _resp(text, stop="end_turn"):
    return types.SimpleNamespace(stop_reason=stop, content=[types.SimpleNamespace(type="text", text=text)],
                                 usage=types.SimpleNamespace(input_tokens=10, output_tokens=5))


def test_anthropic_adapter_happy_path(monkeypatch, tmp_path):
    Client, Messages = _fake_anthropic(monkeypatch, lambda kw: _resp('```json\n{"ok": true}\n```'))
    llm = LLM(provider="anthropic", model="claude-haiku-5-5", api_key=FAKE_KEY, cache_dir=str(tmp_path))
    assert llm.json("sys", "user", tag="t") == {"ok": True}
    assert Messages.last["model"] == "claude-haiku-5-5" and Messages.last["system"] == "sys"
    assert "fallbacks" not in Messages.last and "betas" not in Messages.last
    assert "temperature" not in Messages.last          # sampling params are rejected by current Claude models
    assert llm.stats["in_tokens"] == 10


def test_anthropic_refusal_is_reported(monkeypatch, tmp_path):
    _fake_anthropic(monkeypatch, lambda kw: _resp("", stop="refusal"))
    llm = LLM(provider="anthropic", model="m", api_key="k", cache_dir=str(tmp_path))
    with pytest.raises(LLMError) as e:
        llm.json("s", "u")
    assert e.value.kind == "refusal"


def test_anthropic_rate_limit_is_retried(monkeypatch, tmp_path):
    import anthropic
    import httpx2 as httpx_mod
    calls = {"n": 0}

    def behaviour(kw):
        calls["n"] += 1
        if calls["n"] == 1:
            req = httpx_mod.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.RateLimitError("slow down", response=httpx_mod.Response(429, request=req), body=None)
        return _resp('{"ok": 1}')
    _fake_anthropic(monkeypatch, behaviour)
    llm = LLM(provider="anthropic", model="m", api_key="k", cache_dir=str(tmp_path), max_retries=2, rpm=6000)
    assert llm.json("s", "u") == {"ok": 1} and llm.stats["retries"] == 1
