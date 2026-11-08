"""v8 server routing, persistent reservations and process-safe call coalescing."""
from __future__ import annotations
import fcntl
import hashlib
import json
import math
import os
import sqlite3
import time
from pathlib import Path
from .llm import LLM, LLMError, PRESETS

VERSION = 'v8.0'
DEFAULT_MODELS = {'gemini': 'gemini-3.5-flash-lite', 'claude': 'claude-sonnet-5-5'}


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


class Budget:
    """Reserve before EACH attempt. Unknown/failed usage retains its full allowance.

    UTF-8 bytes + 1024 framing tokens is a deliberately conservative input allowance,
    not a tokenizer measurement. Output uses the configured hard maximum. Billing is
    never inferred from successful-call count. Interrupted reservations remain charged.
    """
    def __init__(self, path, limit=2.0, max_seconds=1200):
        self.path, self.limit, self.max_seconds = str(path), float(limit), int(max_seconds)
        if not math.isfinite(self.limit) or self.limit <= 0 or self.max_seconds <= 0:
            raise LLMError('Budget and runtime limit must be positive finite values.', kind='config')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, model TEXT, tag TEXT, allowance REAL, charged REAL, usage TEXT, state TEXT)')
        self.started = time.monotonic()

    def db(self):
        return sqlite3.connect(self.path, timeout=30)

    def reserve(self, client, system, user, tag):
        if time.monotonic() - self.started > self.max_seconds:
            raise LLMError('Run time limit reached; completed work is preserved.', kind='budget')
        if any(not math.isfinite(v) or v <= 0 for v in (client.price_in, client.price_out)):
            raise LLMError(f'Pricing unknown for {client.model}; configure both token prices before paid calls.', kind='config')
        tin = len((system + user).encode()) + 1024
        allowance = (tin * client.price_in + client.max_tokens * client.price_out) / 1e6
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            used = db.execute('SELECT COALESCE(SUM(charged),0) FROM attempts').fetchone()[0]
            if used + allowance > self.limit:
                raise LLMError('Run budget exhausted; remaining clips need review.', kind='budget')
            cur = db.execute('INSERT INTO attempts(model,tag,allowance,charged,state) VALUES (?,?,?,?,?)',
                             (client.name, tag, allowance, allowance, 'reserved'))
            return cur.lastrowid

    def settle(self, ident, client, usage, failed=False):
        known = not failed and usage and usage.get('reported', True)
        cost = ((usage.get('in', 0) * client.price_in + usage.get('out', 0) * client.price_out) / 1e6) if known else None
        with self.db() as db:
            db.execute('UPDATE attempts SET charged=COALESCE(?,allowance),usage=?,state=? WHERE id=?',
                       (cost, json.dumps(usage) if known else None, 'reported' if known else 'usage_unknown', ident))

    def summary(self):
        with self.db() as db:
            rows = db.execute('SELECT allowance,charged,usage,state FROM attempts').fetchall()
        return {'limit_usd': self.limit, 'reserved_or_usage_cost_usd': sum(r[1] for r in rows),
                'provider_usage_cost_usd': sum(r[1] for r in rows if r[3] == 'reported'),
                'usage_unknown_attempts': sum(r[3] != 'reported' for r in rows), 'attempts': len(rows),
                'billing_cost_usd': None, 'runtime_limit_seconds': self.max_seconds}


class GuardedLLM(LLM):
    """Same adapter contract as LLM, with per-request disk locks and a shared run budget."""
    budget = None
    def cache_key(self, system, user, schema=None):
        # v8 always includes schema AND endpoint (legacy keys intentionally retained in LLM).
        raw = [VERSION, self.provider, self.model, self.base_url, system, user, schema,
               self.temperature, self.max_tokens, self.structured]
        return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()

    def json(self, system, user, tag='', schema=None):
        h = self.cache_key(system, user, schema)
        root = Path(self.cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        with (root / (h + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            marker = root / (h + '.pending')
            if self.cached(system, user, schema) is not None:
                return super().json(system, user, tag, schema)
            if marker.exists():
                raise LLMError('An interrupted request has unknown completion; inspect the pending record before retrying.', kind='unavailable')
            if self.cache_only:
                return super().json(system, user, tag, schema)
            if not self.has_key():
                raise LLMError(f'Server configuration missing {self.api_key_env}.', kind='config')
            if any(not math.isfinite(v) or v <= 0 for v in (self.price_in, self.price_out)):
                raise LLMError(f'Pricing unknown for {self.model}; configure both token prices before paid calls.', kind='config')
            attempts_before = self.budget.summary()['attempts']
            marker.write_text(json.dumps({'hash': h, 'model': self.name, 'tag': tag}))
            try:
                value = super().json(system, user, tag, schema)
            except Exception as exc:
                # A transport/parse/disk failure after an attempt can hide a completed paid
                # request. Keep the marker; resume must not silently repeat that work.
                if self.budget.summary()['attempts'] == attempts_before or (isinstance(exc, LLMError) and exc.kind == 'auth'):
                    marker.unlink(missing_ok=True)
                raise
            marker.unlink(missing_ok=True)
            return value

    def _meter(self, operation, system, user, schema):
        if not self.has_key():
            raise LLMError(f'Server configuration missing {self.api_key_env}.', kind='config')
        ident = self.budget.reserve(self, system, user, self._tag)
        try:
            text, usage = operation(system, user, schema)
        except Exception as e:
            self.budget.settle(ident, self, None, failed=True)
            self._log(self._tag, self.cache_key(system, user, schema), '', error=f'{type(e).__name__}: provider attempt failed')
            raise
        self.budget.settle(ident, self, usage)
        # Save every provider response, including malformed replies, before parsing/caching.
        self._log(self._tag + ':raw', self.cache_key(system, user, schema), text)
        return text, usage

    def _call(self, system, user, tag, schema):
        self._tag = tag  # v8 serial calls; concurrency is intentionally one per route
        return super()._call(system, user, tag, schema)

    def _openai_compat(self, system, user, schema):
        return self._meter(super()._openai_compat, system, user, schema)

    def _anthropic(self, system, user, schema):
        return self._meter(super()._anthropic, system, user, schema)


# --- per-stage model settings ------------------------------------------------------------------------
# Role keys stay 'gemini' (first pass: reads every episode) and 'claude' (final review) for compatibility with
# run_meta; the provider behind each role is configurable per session. API keys typed in the app live in memory
# only (LLM.api_key has repr=False and is never written to cache keys, logs, run_meta or outputs).
SERVER_ROLES = {'gemini': ('gemini', 'GEMINI_API_KEY'), 'claude': ('anthropic', 'ANTHROPIC_API_KEY')}
ROLE_LABELS = {'gemini': 'First pass (reads every episode)', 'claude': 'Final review (judges each shortlisted clip)'}
PROVIDERS = {'gemini': 'Google Gemini', 'anthropic': 'Anthropic Claude', 'openai': 'OpenAI', 'groq': 'Groq',
             'openrouter': 'OpenRouter', 'custom': 'Other OpenAI-compatible endpoint', 'ollama': 'Ollama (local)'}
# USD per 1M tokens (input, output). Checked on the providers' pricing pages on 2026-11-07; the app pre-fills these
# and the user can edit them. Unknown models need prices typed in: unknown is never treated as free.
KNOWN_PRICES = {'gemini-3.5-flash-lite': (0.30, 2.50), 'claude-haiku-5-5': (0.10, 0.50),
                'claude-sonnet-5-5': (2.00, 10.00), 'claude-opus-5-5': (4.00, 20.00)}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def role_settings(role, override=None):
    """Effective provider settings for a role: the server's .env defaults, or a per-session override."""
    o = override or {}
    if o.get('provider'):
        prov = o['provider']
        preset = 'openai' if prov == 'custom' else prov
        return {'role': role, 'override': True, 'provider': prov, 'preset': preset,
                'model': (o.get('model') or '').strip() or PRESETS[preset][1],
                'base_url': (o.get('base_url') or '').strip() if prov == 'custom' else '',
                'api_key': o.get('api_key') or '', 'key_env': PRESETS[preset][3],
                'price_in': _f(o.get('price_in')), 'price_out': _f(o.get('price_out')), 'rpm': _f(o.get('rpm')) or 12.0}
    preset, env = SERVER_ROLES[role]
    return {'role': role, 'override': False, 'provider': preset, 'preset': preset,
            'model': os.getenv(role.upper() + '_MODEL', DEFAULT_MODELS[role]), 'base_url': '', 'api_key': '', 'key_env': env,
            'price_in': _f(os.getenv(role.upper() + '_INPUT_USD_PER_MILLION')),
            'price_out': _f(os.getenv(role.upper() + '_OUTPUT_USD_PER_MILLION')),
            'rpm': _f(os.getenv(role.upper() + '_RPM', '12')) or 12.0}


def public_url_error(url):
    """A visitor-supplied endpoint must be HTTPS on a public host, so the server can't be pointed at its own
    network (SSRF). Returns an error string or ''."""
    import ipaddress
    import socket
    from urllib.parse import urlparse
    u = urlparse(url or '')
    if u.scheme != 'https' or not u.hostname:
        return 'Custom endpoint must be an https:// URL.'
    try:
        addrs = {a[4][0] for a in socket.getaddrinfo(u.hostname, u.port or 443)}
    except OSError:
        return 'Custom endpoint host could not be resolved.'
    for a in addrs:
        ip = ipaddress.ip_address(a.split('%')[0])
        if not ip.is_global:
            return 'Custom endpoint must be a public internet address.'
    return ''


def configuration_errors(overrides=None, public=False):
    """What blocks paid calls. With no overrides this checks the server .env exactly as before."""
    errors = []
    for role in ('gemini', 'claude'):
        s = role_settings(role, (overrides or {}).get(role))
        label = 'first pass' if role == 'gemini' else 'final review'
        if s['override']:
            if public and not s['api_key']:
                errors.append(f'Enter your own API key for the {label} model (the public demo only uses its own keys with its default models).')
            elif not (s['api_key'] or (s['key_env'] and os.getenv(s['key_env'])) or s['preset'] == 'ollama'):
                errors.append(f'Missing API key for the {label} model.')
            if s['provider'] == 'custom':
                if not s['base_url']:
                    errors.append(f'Enter the endpoint URL for the {label} model.')
                elif public and (err := public_url_error(s['base_url'])):
                    errors.append(f'{label.capitalize()}: {err}')
            if public and s['provider'] == 'ollama':
                errors.append('Ollama runs on your own machine; it is not available on the public demo.')
            names = (f'{label} input price', f'{label} output price')
        else:
            if not os.getenv(s['key_env']):
                errors.append(f'Missing server setting: {s["key_env"]}')
            names = tuple(f'{role.upper()}_{side}_USD_PER_MILLION' for side in ('INPUT', 'OUTPUT'))
        for name, value in zip(names, (s['price_in'], s['price_out'])):
            if not math.isfinite(value) or value <= 0:
                errors.append(f'Pricing unknown or invalid: {name}')
    return errors


def routes(out_dir, cache_dir, overrides=None, budget_usd=None):
    budget = Budget(Path(out_dir) / 'budget.sqlite', budget_usd or os.getenv('MAX_RUN_COST_USD', '2'),
                    os.getenv('MAX_RUN_SECONDS', '1200'))
    clients = []
    for role in ('gemini', 'claude'):
        s = role_settings(role, (overrides or {}).get(role))
        client = GuardedLLM.from_preset(s['preset'], model=s['model'], base_url=s['base_url'] or None,
            api_key=s['api_key'] or None,
            max_tokens=4000,  # Claude's thinking shares this cap; 2400 truncated replies
            max_retries=2, workers=1, rpm=s['rpm'], timeout=60, price_in=s['price_in'], price_out=s['price_out'],
            cache_dir=str(cache_dir), log_path=str(Path(out_dir) / 'llm_calls.jsonl'))
        if s['override'] and s['provider'] == 'custom' and not s['api_key']:
            client.api_key_env = ''        # never send a server key to a visitor-supplied URL
        client.budget = budget
        clients.append(client)
    return *clients, budget
