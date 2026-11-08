"""One place for every tunable number, shared by the CLI, the web app and the evaluation.

config.toml selects a named *scoring preset* (v4 = the submitted behaviour, v5 = the improved
pipeline). Every knob below has a v4 default, so `scoring_preset = "v4"` reproduces v4 exactly.
See DECISIONS.md for why each number has the value it has.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.toml"


@dataclass(frozen=True)
class Settings:
    # --- candidate windows (code) -------------------------------------------------
    min_len: float = 20.0                 # brief: promos are 20-60 s
    max_len: float = 60.0
    hook_window: float = 3.0              # brief: hook in the first 3 s
    hook_lead: float = 0.3                # a hook beat may start this much before the cut (rounding slack)
    cliff_window: float = 4.0             # last N seconds checked for the cliffhanger
    per_episode: int = 6                  # continuous candidates kept per episode
    candidate_overlap: float = 0.4        # NMS: max overlap between two kept candidates
    end_of_episode_bonus: float = 0.25    # the writers put a cliffhanger at every episode end
    resolve_penalty: float = 0.5          # how much a "resolved" ending cancels cliffhanger score
    resolve_on_calm_tone: bool = True     # v4: a calm/tender *tone tag* counts as resolved
    emotional_reveal_exempt: bool = False # v5: AI-tagged emotional reveals (>=7) skip the resolve penalty
    # --- stitched (two-moment) promos ---------------------------------------------
    stitched_enabled: bool = True
    stitched_max_episode_gap: int = 8
    stitched_top_moments: int = 30
    stitched_limit: int = 40
    stitched_penalty: float = 0.95        # every cross-episode cut costs some clarity
    stitched_min_standalone: float = 0.0  # v5: stitched promos must clear the standalone threshold
    max_stitched_in_top: int = 0          # v5: cap in the final list (0 = no cap)
    # --- rules / AI blend ---------------------------------------------------------
    lexicon: str = "v4_legacy"            # v4_legacy | generic  (show-specific words come from the show profile)
    show_profile: bool = False            # v5: one LLM call reads the loglines -> show-specific words
    miner_prompt: str = "v4"              # v4 | v5 (v5 adds the emotional_reveal type)
    w_llm: float = 0.6                    # miner tags vs rule scores
    judge_enabled: bool = True
    judge_prompt: str = "v4"              # v4 | v5 (v5 adds yes/no cliffhanger checks)
    judge_top_n: int = 60
    judge_batch: int = 5
    shortlist_overlap: float = 0.5
    w_judge: float = 0.6                  # cold-viewer judge vs rule scores in the final score
    temperature: float = 0.2
    # --- final ranking ------------------------------------------------------------
    standalone_mode: str = "additive"     # additive (v4) | geometric | gate
    standalone_threshold: float = 0.5     # gate: below this -> "needs context" list
    cliff_mode: str = "score"             # score (v4) | checks | blend
    rerank: bool = False                  # v5: listwise rerank of the top N
    rerank_top: int = 10
    rerank_rounds: int = 3
    rerank_seed: int = 7
    final_overlap: float = 0.3
    max_per_episode: int = 2
    top_k: int = 25
    funnel_strength: float = 0.7          # early-episode bonus falls from 1.0 to 0.3 across the show
    w_hook: float = 0.30
    w_cliff: float = 0.30
    w_standalone: float = 0.20
    w_intensity: float = 0.10
    w_funnel: float = 0.10
    # --- v7: verified eligibility (see promo_finder/eligibility.py) --------------------
    eligibility: str = "legacy"           # legacy (v4-v6: checks are informational) | verified (v7: one authoritative decision)
    count_production_cues: bool = True    # v7 False: "cliffhanger", "cut to black", music swells are not story evidence
    len_tolerance: float = 0.05           # rounding slack on the 20-60 s rule (source times have 0.01-0.1 s precision)
    ending_window: float = 5.0            # v7: ending evidence must be heard/seen in the final N seconds
    boundary_search: float = 0.0          # v7 4.0: seconds searched around a cut-in for an opening that shows the action
    max_boundary_variants: int = 0        # v7: cap on re-cut openings sent to the judge
    repair_rounds: int = 0                # v7 2: re-cut endings the judge shows are answered inside the clip, then re-judge
    max_repairs: int = 15                 # cap per repair round
    judge_retries: int = 2                # v7: extra calls when a judge reply fails validation
    # --- parser checks --------------------------------------------------------------
    min_coverage: float = 0.99
    episode_min_sec: float = 20.0
    episode_max_sec: float = 600.0
    frame_rate: int = 30

    def weights(self) -> dict:
        return {"hook": self.w_hook, "cliff": self.w_cliff, "standalone": self.w_standalone,
                "intensity": self.w_intensity, "funnel": self.w_funnel}

    def with_weights(self, w: dict | None) -> "Settings":
        if not w:
            return self
        return replace(self, **{f"w_{k}": float(v) for k, v in w.items() if f"w_{k}" in _FIELDS})


_FIELDS = {f.name for f in fields(Settings)}


@dataclass
class RunConfig:
    """Everything that is not scoring: provider limits, paths, eval files."""
    scoring_preset: str = "v4"
    settings: Settings = field(default_factory=Settings)
    llm_preset: str = "none"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    provider: dict = field(default_factory=dict)       # rpm, tpm, workers, timeout, price_in, price_out ...
    cache_dir: Path = ROOT / ".cache" / "llm"
    cache_readonly_dirs: tuple = (ROOT / ".cache" / "v4_snapshot",)
    outputs_dir: Path = ROOT / "outputs"
    gold_path: Path = ROOT / "eval" / "gold_moments.json"
    heldout_path: Path = ROOT / "eval" / "heldout_moments.json"
    raw: dict = field(default_factory=dict)


PROVIDER_DEFAULTS = {"rpm": 12, "tpm": 0, "workers": 3, "timeout": 120, "max_retries": 6,
                     "price_in": 0.0, "price_out": 0.0, "structured": "json_object"}


def _path(v, default: Path) -> Path:
    if not v:
        return default
    p = Path(os.path.expanduser(str(v)))
    return p if p.is_absolute() else ROOT / p


def load(path: Path | str | None = None, preset: str | None = None, overrides: dict | None = None) -> RunConfig:
    """Read config.toml -> RunConfig. `preset` / `overrides` win over the file."""
    p = Path(path) if path else CONFIG_PATH
    raw = tomllib.loads(p.read_text()) if p.exists() else {}
    name = preset or raw.get("scoring_preset", "v4")
    presets = raw.get("scoring_presets", {})
    if name not in presets and name != "v4":
        raise SystemExit(f"Unknown scoring preset '{name}'. Defined: {', '.join(['v4', *presets])}")
    vals = dict(presets.get(name, {}))
    vals.update(overrides or {})
    unknown = set(vals) - _FIELDS
    if unknown:
        raise SystemExit(f"Unknown setting(s) in preset '{name}': {', '.join(sorted(unknown))}")
    settings = Settings(**vals)

    llm = raw.get("llm", {})
    llm_preset = llm.get("preset", "none")
    prov = {**PROVIDER_DEFAULTS, **raw.get("providers", {}).get("default", {}),
            **raw.get("providers", {}).get(llm_preset, {})}
    paths = raw.get("paths", {})
    ev = raw.get("eval", {})
    return RunConfig(
        scoring_preset=name, settings=settings, llm_preset=llm_preset, model=llm.get("model"),
        base_url=llm.get("base_url"), api_key_env=llm.get("api_key_env"), provider=prov,
        cache_dir=_path(paths.get("cache_dir"), ROOT / ".cache" / "llm"),
        cache_readonly_dirs=tuple(_path(d, ROOT) for d in paths.get("cache_readonly_dirs", [".cache/v4_snapshot"])),
        outputs_dir=_path(paths.get("outputs_dir"), ROOT / "outputs"),
        gold_path=_path(ev.get("gold_path"), ROOT / "eval" / "gold_moments.json"),
        heldout_path=_path(ev.get("heldout_path"), ROOT / "eval" / "heldout_moments.json"),
        raw=raw,
    )


def provider_limits(cfg: RunConfig, llm_preset: str) -> dict:
    """Limits/prices for a given provider preset (config [providers.<name>] over defaults)."""
    pr = cfg.raw.get("providers", {})
    return {**PROVIDER_DEFAULTS, **pr.get("default", {}), **pr.get(llm_preset, {})}


def scoring_presets(cfg: RunConfig) -> list[str]:
    return ["v4", *[k for k in cfg.raw.get("scoring_presets", {}) if k != "v4"]]


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (KEY=value lines) so keys stay out of code and git. Empty values are
    skipped, and real environment variables always win. Shared by the CLI and the web app."""
    p = path or ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if k.strip() and v:
                os.environ.setdefault(k.strip(), v)


def make_llm(cfg: RunConfig, llm_preset: str | None = None, model: str | None = None, base_url: str | None = None,
             api_key: str = "", api_key_env: str | None = None, workers: int | None = None,
             log_path: str | None = None):
    """Build the LLM client from config + overrides (one place, used by CLI, app and evaluation)."""
    from .llm import LLM
    name = llm_preset or cfg.llm_preset
    lim = provider_limits(cfg, name)
    S = cfg.settings
    return LLM.from_preset(
        name, model=model or (cfg.model if name == cfg.llm_preset else None),
        base_url=base_url or (cfg.base_url if name == cfg.llm_preset else None),
        api_key_env=api_key_env or (cfg.api_key_env if name == cfg.llm_preset else None), api_key=api_key,
        rpm=float(lim["rpm"]), tpm=float(lim["tpm"]), workers=int(workers or lim["workers"]),
        timeout=float(lim["timeout"]), max_retries=int(lim["max_retries"]), structured=str(lim["structured"]),
        price_in=float(lim["price_in"]), price_out=float(lim["price_out"]), temperature=S.temperature,
        cache_dir=str(cfg.cache_dir), cache_readonly_dirs=tuple(str(d) for d in cfg.cache_readonly_dirs),
        log_path=log_path)
