#!/usr/bin/env python3
"""One tiny call per provider preset (or a custom endpoint) -> PASS/FAIL. Run it with YOUR keys.

  python scripts/smoke_test_provider.py                     # every preset whose key is set
  python scripts/smoke_test_provider.py --preset gemini
  python scripts/smoke_test_provider.py --base-url https://gateway/v1 --model their-model --api-key-env THEIR_KEY
Each call costs a few tokens. Checks: the call works, the reply parses as JSON, and (optionally)
whether the endpoint accepts structured output (json_schema).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from promo_finder import config as C  # noqa: E402
from promo_finder.llm import PRESETS, LLMError  # noqa: E402

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}, "word": {"type": "string"}},
          "required": ["ok", "word"]}


def check(llm, structured: bool) -> str:
    llm.cache_dir = str(Path(C.ROOT) / ".cache" / "smoke")       # never mixes with real caches
    llm.structured = "json_schema" if structured else "json_object"
    t = time.time()
    data = llm.json("Reply with JSON only.", f'Return {{"ok": true, "word": "promo"}} ({time.time()})',
                    tag="smoke", schema=SCHEMA if structured else None)
    if not (isinstance(data, dict) and data.get("ok") is True):
        raise LLMError(f"unexpected reply: {str(data)[:80]}")
    return f"{time.time() - t:.1f}s"


def main():
    C.load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=[p for p in PRESETS if p != "none"])
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--api-key-env")
    ap.add_argument("--structured", action="store_true", help="also test json_schema structured output")
    a = ap.parse_args()
    cfg = C.load()
    targets = [a.preset] if a.preset else (["openai"] if a.base_url else [p for p in PRESETS if p != "none"])
    ok = True
    for name in targets:
        llm = C.make_llm(cfg, name, a.model, a.base_url, api_key_env=a.api_key_env)
        llm.max_retries = 2
        if not llm.has_key():
            print(f"  SKIP  {name:<11} no key ({llm.api_key_env})")
            continue
        for structured in ([False, True] if a.structured else [False]):
            label = f"{name}{' +schema' if structured else ''}"
            try:
                print(f"  PASS  {label:<18} {llm.name:<55} {check(llm, structured)}")
            except Exception as e:  # noqa: BLE001 - report every failure kind
                ok = False
                print(f"  FAIL  {label:<18} {llm.name:<55} {type(e).__name__}: {str(e)[:120]}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
