import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from mock_llm import MockLLM  # noqa: E402

SYNTHETIC = ROOT / "tests" / "fixtures" / "synthetic_show"


@pytest.fixture(scope="session")
def mock_server():
    try:
        server = MockLLM()
    except PermissionError:
        pytest.skip("HTTP integration requires sockets, which this sandbox disables; in-process adapter tests still run")
    with server as m:
        yield m


@pytest.fixture
def make_llm(tmp_path, mock_server):
    """An LLM pointed at the mock server, with its own empty cache."""
    from promo_finder.llm import LLM

    def _make(model="mock", **kw):
        kw.setdefault("rpm", 6000)
        kw.setdefault("workers", 4)
        kw.setdefault("cache_dir", str(tmp_path / "cache"))
        kw.setdefault("log_path", str(tmp_path / "out" / "llm_calls.jsonl"))
        return LLM(provider="openai_compat", model=model, base_url=mock_server.url, **kw)
    return _make


def write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p
