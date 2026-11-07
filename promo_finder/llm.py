"""One small interface over many model providers, so the model is a config line.

    provider = "openai_compat"  -> OpenAI, Gemini, Groq, OpenRouter, Ollama, vLLM, any gateway (one HTTP shape)
    provider = "anthropic"      -> Claude via the official `anthropic` SDK (optional dependency)
    provider = "none"           -> rules only, no network

Every answer is cached on disk (re-runs are free and resumable) and logged verbatim to the run
folder. Calls are thread-safe: one shared rate limiter per provider keeps all workers under the
provider's requests/minute and tokens/minute limits. API keys live only in memory: they are never
part of a cache key, a log line, run_meta.json or an error message.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

PRESETS = {
    # name: provider, model, base_url, api key env var
    "gemini":     ("openai_compat", "gemini-3.5-flash-lite", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "groq":       ("openai_compat", "llama-3.3-70b-versatile", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("openai_compat", "meta-llama/llama-3.3-70b-instruct:free", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai":     ("openai_compat", "gpt-4o-mini", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "ollama":     ("openai_compat", "qwen2.5:7b", "http://localhost:11434/v1", ""),
    "anthropic":  ("anthropic", "claude-opus-5-5", "", "ANTHROPIC_API_KEY"),
    "none":       ("none", "", "", ""),
}
# v4 call parameters. Cache keys for exactly these parameters keep the original (v4) key format,
# so the v4 cache stays valid; any other parameters are added to the key.
V4_PARAMS = {"temperature": 0.2, "max_tokens": 8000, "structured": "json_object"}


class LLMError(RuntimeError):
    """A call failed for good. `kind` drives the readable message in the UI."""
    def __init__(self, msg: str, kind: str = "error", status: int | None = None):
        super().__init__(msg)
        self.kind, self.status = kind, status


class CacheMiss(Exception):
    """Raised in cache-only mode instead of spending money."""


class Cancelled(Exception):
    """The user pressed Cancel."""


class _Retryable(Exception):
    def __init__(self, msg, retry_after=None):
        super().__init__(msg)
        self.retry_after = retry_after


class RateLimiter:
    """Shared by all threads that call one provider: spaces request starts by 60/rpm seconds and
    keeps estimated tokens in any 60-second window under tpm (0 = no token limit)."""

    def __init__(self, rpm: float, tpm: float = 0):
        self.gap = 60.0 / max(rpm, 0.1)
        self.tpm = tpm
        self.lock = threading.Lock()
        self.next_ok = 0.0
        self.window: deque = deque()

    def acquire(self, est_tokens: int = 0, cancel: threading.Event | None = None):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.window and self.window[0][0] < now - 60:
                    self.window.popleft()
                wait = self.next_ok - now
                if self.tpm and self.window and sum(t for _, t in self.window) + est_tokens > self.tpm:
                    wait = max(wait, self.window[0][0] + 60 - now)
                if wait <= 0:
                    self.next_ok = now + self.gap
                    self.window.append((now, est_tokens))
                    return
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            time.sleep(min(wait, 1.0))


_LIMITERS: dict = {}
_LIMITERS_LOCK = threading.Lock()


def shared_limiter(key: str, rpm: float, tpm: float) -> RateLimiter:
    with _LIMITERS_LOCK:
        lim = _LIMITERS.get(key)
        if lim is None or lim.gap != 60.0 / max(rpm, 0.1) or lim.tpm != tpm:
            lim = _LIMITERS[key] = RateLimiter(rpm, tpm)
        return lim


@dataclass
class LLM:
    provider: str = "none"
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    api_key: str = field(default="", repr=False)   # typed in the UI: memory only, never logged
    rpm: float = 12.0
    tpm: float = 0.0
    workers: int = 3
    timeout: float = 120.0
    max_retries: int = 6
    structured: str = "json_object"     # json_object | json_schema | none
    temperature: float = 0.2
    max_tokens: int = 8000
    price_in: float = 0.0               # USD per 1M input tokens (0 = free/unknown)
    price_out: float = 0.0
    cache_dir: str = ".cache/llm"
    cache_readonly_dirs: tuple = ()
    cache_only: bool = False            # True: never call the API (dry runs, cost estimates)
    log_path: str | None = None
    cancel: threading.Event | None = field(default=None, repr=False)
    stats: dict = field(default_factory=lambda: {"calls": 0, "cache_hits": 0, "cache_misses": 0, "errors": 0,
                                                 "retries": 0, "in_tokens": 0, "out_tokens": 0, "seconds": 0.0})
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "LLM":
        prov, model, base, env = PRESETS[name]
        kw = dict(provider=prov, model=model, base_url=base, api_key_env=env)
        kw.update({k: v for k, v in overrides.items() if v not in (None, "")})
        return cls(**kw)

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}" if self.enabled else "rules-only"

    def key(self) -> str:
        return self.api_key or (os.environ.get(self.api_key_env, "") if self.api_key_env else "")

    def has_key(self) -> bool:
        return bool(self.key()) or not self.api_key_env

    def est_cost(self) -> float:
        return self.stats["in_tokens"] / 1e6 * self.price_in + self.stats["out_tokens"] / 1e6 * self.price_out

    # ------------------------------------------------------------------ public
    def cache_key(self, system: str, user: str, schema: dict | None = None) -> str:
        params = {"temperature": self.temperature, "max_tokens": self.max_tokens,
                  "structured": self.structured if schema else "json_object"}
        base = f"{self.provider}|{self.model}|{system}|{user}"
        if params != V4_PARAMS or (schema and self.structured == "json_schema"):
            base += "|" + json.dumps({**params, "schema": schema if self.structured == "json_schema" else None},
                                     sort_keys=True)
        return hashlib.sha256(base.encode()).hexdigest()[:24]

    def cached(self, system: str, user: str, schema: dict | None = None) -> str | None:
        h = self.cache_key(system, user, schema)
        for d in (self.cache_dir, *self.cache_readonly_dirs):
            p = Path(d) / f"{h}.json"
            if p.exists():
                return json.loads(p.read_text())["text"]
        return None

    def json(self, system: str, user: str, tag: str = "", schema: dict | None = None):
        """Call the model and return parsed JSON (dict or list). Cached by (model, prompt, params)."""
        h = self.cache_key(system, user, schema)
        text = self.cached(system, user, schema)
        if text is not None:
            with self._lock:
                self.stats["cache_hits"] += 1
            self._log(tag, h, text, cached=True)
            return extract_json(text)
        with self._lock:
            self.stats["cache_misses"] += 1
        if self.cache_only:
            raise CacheMiss(tag)
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled()

        last_err = None
        for attempt in range(2):                       # one repair call if the reply isn't JSON
            prompt = user if attempt == 0 else user + "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object."
            text, usage = self._call(system, prompt, tag, schema)
            try:
                data = extract_json(text)
            except ValueError as e:
                last_err = e
                self._log(tag, h, text, error=str(e))
                continue
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.cache_dir) / f"{h}.json").write_text(json.dumps({"model": self.name, "text": text, "usage": usage}))
            self._log(tag, h, text)
            return data
        with self._lock:
            self.stats["errors"] += 1
        raise LLMError(f"{tag}: the model's reply was not valid JSON twice ({last_err})", kind="parse")

    # ------------------------------------------------------------------ internals
    def _limiter(self) -> RateLimiter:
        return shared_limiter(f"{self.provider}|{self.base_url}", self.rpm, self.tpm)

    def _call(self, system: str, user: str, tag: str, schema: dict | None) -> tuple[str, dict]:
        t0 = time.time()
        est = (len(system) + len(user)) // 4 + 500
        last = "unknown error"
        for attempt in range(self.max_retries + 1):
            self._limiter().acquire(est, self.cancel)
            try:
                if self.provider == "openai_compat":
                    out = self._openai_compat(system, user, schema)
                elif self.provider == "anthropic":
                    out = self._anthropic(system, user, schema)
                else:
                    raise LLMError("LLM disabled", kind="config")
                with self._lock:
                    self.stats["calls"] += 1
                    self.stats["seconds"] += time.time() - t0
                    self.stats["in_tokens"] += out[1].get("in", 0)
                    self.stats["out_tokens"] += out[1].get("out", 0)
                return out
            except _Retryable as e:
                last = str(e)
                if attempt == self.max_retries:
                    break
                delay = e.retry_after or min(60.0, 2.0 * 2 ** attempt)
                delay *= 0.5 + random.random()          # jitter: workers don't retry in lockstep
                with self._lock:
                    self.stats["retries"] += 1
                end = time.time() + delay
                while time.time() < end:
                    if self.cancel is not None and self.cancel.is_set():
                        raise Cancelled()
                    time.sleep(min(0.5, end - time.time()) if end > time.time() else 0)
        with self._lock:
            self.stats["errors"] += 1
        kind = "rate_limit" if "429" in last else "unavailable"
        raise LLMError(f"{tag}: gave up after {self.max_retries} retries ({last})", kind=kind)

    def _redact(self, text: str) -> str:
        k = self.key()
        if k and len(k) > 6:
            text = text.replace(k, "***")
        return re.sub(r"(key=|api[_-]?key[\"':= ]+)[A-Za-z0-9_\-\.]{8,}", r"\1***", text, flags=re.I)

    def _openai_compat(self, system: str, user: str, schema: dict | None) -> tuple[str, dict]:
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if schema and self.structured == "json_schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "reply", "schema": schema, "strict": False}}
        elif self.structured in ("json_object", "json_schema"):
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json", "User-Agent": "promo-finder/1.0"}
        if key := self.key():
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = self._redact(e.read().decode(errors="replace")[:300])
            if e.code in (429, 500, 502, 503, 504):
                ra = e.headers.get("Retry-After")
                raise _Retryable(f"HTTP {e.code}", float(ra) if ra and ra.replace('.', '', 1).isdigit() else None)
            kind = "auth" if e.code in (401, 403) else "bad_request"
            raise LLMError(f"HTTP {e.code} from {self.base_url}: {msg}", kind=kind, status=e.code) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise _Retryable(f"network/timeout: {self._redact(str(e))[:120]}") from None
        try:
            text = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected reply shape from {self.base_url}", kind="bad_request") from None
        u = data.get("usage") or {}
        return text, {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}

    def _anthropic(self, system: str, user: str, schema: dict | None) -> tuple[str, dict]:
        try:
            import anthropic
        except ImportError:
            raise LLMError("pip install anthropic  (needed for provider=anthropic)", kind="config") from None
        client = anthropic.Anthropic(api_key=self.key() or None, timeout=self.timeout, max_retries=0)
        output_config = {"effort": "low"}            # scoring task: low effort is plenty
        if schema and self.structured == "json_schema":
            output_config["format"] = {"type": "json_schema", "schema": schema}
        try:
            resp = client.beta.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=system,
                messages=[{"role": "user", "content": user}], output_config=output_config,
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            )
        except anthropic.RateLimitError as e:
            raise _Retryable("HTTP 429 (anthropic)") from e
        except anthropic.AuthenticationError as e:
            raise LLMError("anthropic rejected the API key", kind="auth", status=401) from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise _Retryable(f"HTTP {e.status_code} (anthropic)") from e
            raise LLMError(self._redact(str(e))[:300], kind="bad_request", status=e.status_code) from e
        except anthropic.APIConnectionError as e:
            raise _Retryable("network/timeout (anthropic)") from e
        if resp.stop_reason == "refusal":
            raise LLMError("model declined the request (stop_reason=refusal)", kind="refusal")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}

    def _log(self, tag, h, text, cached=False, error=None):
        if not self.log_path:
            return
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"tag": tag, "hash": h, "model": self.name, "cached": cached,
                           "error": self._redact(error) if error else None, "response": text})
        with self._lock, open(self.log_path, "a") as f:
            f.write(line + "\n")


_REASONING_RE = re.compile(r"<(think|thinking|reasoning|reflection)>.*?</\1>", re.S | re.I)


def extract_json(text: str):
    """Tolerant JSON extraction, in order: strip reasoning blocks, prefer a fenced block, then take
    the FIRST complete JSON object/array anywhere in the text (trailing commentary is ignored)."""
    t = _REASONING_RE.sub("", text or "").strip()
    t = re.sub(r"^.*?</(think|thinking|reasoning|reflection)>", "", t, flags=re.S | re.I)  # unopened close tag
    candidates = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", t, re.S)] + [t]
    dec = json.JSONDecoder()
    for c in candidates:
        for i, ch in enumerate(c):
            if ch in "{[":
                try:
                    obj, _ = dec.raw_decode(c[i:])
                except ValueError:
                    continue
                if isinstance(obj, (dict, list)):
                    return obj
    raise ValueError("no JSON object found")


def parse_json(text: str) -> dict:
    """Backwards-compatible name."""
    return extract_json(text)


def prompt_hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:12]
