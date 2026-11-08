# Promo Finder

An editorial workspace for finding source-grounded 20–60 second promotional clips in timestamped screenplays. **v8.0** is the current local implementation. Earlier v4–v7 code, cached runs and experiments remain available for historical comparison.

Load screenplay → Find promos → Review recommendations. No model selectors or credential fields appear in the editor workflow. Each clip includes exact ordered segments, a screenplay excerpt, opening/ending evidence, ranking rationale, limitations, and separate human approval. This finds cuts; it does not render video or predict marketing performance.

## Run locally

Python 3.11+:

```bash
cd '/home/vivek/Documents/Rajan Kuku/promo-finder'
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
# Populate server variables from .env.example in your EXISTING .env.
# Do not overwrite an existing environment file; never commit credentials.
.venv/bin/streamlit run app.py
```

Server configuration:

- `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`
- `GEMINI_MODEL` (default `gemini-3.5-flash-lite`), `CLAUDE_MODEL` (default `claude-sonnet-5-5`)
- `GEMINI_INPUT_USD_PER_MILLION`, `GEMINI_OUTPUT_USD_PER_MILLION`, `CLAUDE_INPUT_USD_PER_MILLION`, `CLAUDE_OUTPUT_USD_PER_MILLION`: set current prices for the selected model. Unknown prices block paid calls; they are never reported as free.
- `MAX_RUN_COST_USD=2.00`, `MAX_RUN_SECONDS=1200`
- `FINAL_REVIEW_LIMIT=60`, `AUDIT_SAMPLE_SIZE=4`, `GEMINI_RPM=12`, `CLAUDE_RPM=12`

Model names/prices are server settings. Consult [Google pricing](https://ai.google.dev/gemini-api/docs/pricing) and [Claude models and pricing](https://platform.claude.com/docs/en/models/overview). Model access and prices depend on the account; changing the model requires checking its price. The compatible HTTP adapter and Claude Messages adapter remain internal interfaces. Claude calls have **no automatic fallback model**.

```bash
.venv/bin/python run.py --show /path/to/show --estimate
.venv/bin/python run.py --show /path/to/show
# Explicit offline inspection: always unverified; no paid calls.
.venv/bin/python run.py --show tests/fixtures/synthetic_show --inspect
.venv/bin/python -m pytest tests -q
```

The v8 CLI uses the configured budget without an interactive confirmation. Missing services produce a partial artifact with reasons; the UI disables Find promos until configuration is present. Historical experiments explicitly select `--scoring-preset v7 --preset gemini` (or v4/v6); legacy experiments do not use v8's budget guard. They are not the normal editor route.

## Input and timing

One UTF-8 `.txt` or `.md` file per episode; uploads also accept ZIP. Include `EPISODE NUMBER: 1` or an episode number in the filename. Beat example:

```text
EPISODE NUMBER: 1
**ACTION:**
Start: (00:00.00) | End: (00:02.50)
The clock begins counting backwards.

**MARA:**
Start: (00:02.50) | End: (00:05.00)
We have one minute to stop it.
```

Dialogue, actions, camera and sound cues are preserved with stable beat IDs. Times mean decimal seconds, not frames. Export precision is retained (up to nine decimal places); durations use decimal subtraction and must actually lie within 20–60 seconds. Source-derived boundaries must not bisect dialogue. Missing/invalid ends are flagged and cannot support automatic recommendations. Long action descriptions do not supply word- or action-level timing: the meaningful hook remains uncertain unless its complete supporting beat finishes within three seconds.

## Actual pipeline

1. Parse all files and report failures. A parse coverage below 99% stops the run.
2. Gemini reads every episode in whole-beat chunks with overlap and preceding source context. Compact annotations cite original beat IDs and exact quotes. No show profile, title, genre vocabulary or keyword score gates eligibility.
3. Code enumerates legal continuous windows. Memory/search limits: 24 retained variants per episode, at most 80 sampled legal ends per start; optional model-suggested two-segment, chronological, same-episode cuts (40 maximum). Deduplication is delayed until after review. These bounds can miss good clips and are recorded.
4. Semantic scores prioritize final review, including episode representatives. Claude reviews up to 60 primary candidates, six observable uncertainty cases, and four deterministic low-ranked audit samples. Selection uses evidence/context/timing/ending/transition signals, not confidence. Unselected clips remain unverified and are exported in `evaluated.json`.
5. Claude sees one clip only, without the show summary, generation rationale, or other candidates. Code checks source references, exact quotes, strict boundaries, assembled timing, and contradictions. Stitched clips receive a separate Gemini source-context faithfulness check. Missing/malformed/conflicting evidence means review, not a pass.
6. Up to six failed endings can get one source-supported boundary repair each; each repaired clip is judged afresh. No repair inherits a pass.
7. Only passing clips above the quality floor enter ranking. Weights: opening 25%, ending 25%, standalone clarity 15%, stakes 15%, setup economy 10%, coherence 10%; a small cut penalty favors continuous clips. No early-episode bonus. Near-duplicates and repeated moments are removed; maximum two per episode and three stitched promos. One optional Claude comparison of up to five close passing clips replaces an all-pairs tournament. Return at most 25, never fill a quota.

Coverage distinguishes source beats included in chunks from successfully processed beats. Neither proves exhaustive discovery or correct narrative judgment.

## Cost, persistence and privacy

Before each provider attempt, SQLite reserves a conservative allowance using UTF-8 input bytes plus framing allowance and the maximum output tokens. Provider-reported usage settles reservations; errors or missing usage retain the allowance. Two transport retries and one invalid-JSON repair are bounded, with rate limiting and backoff. Calls are serial per route. Runtime prevents new calls after the limit; an in-flight request/rate-limit wait can extend wall time.

Disk locks coalesce identical calls across reruns/processes. Cache identity includes content, boundaries, prompt, model, endpoint, schema and generation parameters. Completed calls are reused; ambiguous interrupted calls leave a `.pending` record and are not automatically repeated. Inspect the provider record before an operator clears such a marker. Run directories use content/code/configuration fingerprints. A SQLite ledger survives resume; unknown attempts never become zero-cost attempts. Reports distinguish usage-based cost calculations from actual billing (unavailable).

Outputs: `outputs/v8/<run-id>/` contains `promos.json/csv`, `report.html/md`, `run_meta.json`, `parse_report.json`, `discovery.json`, `evaluated.json`, `source_context.json`, a byte-for-byte `original_input/` snapshot, `progress.json`, `budget.sqlite`, and raw `llm_calls.jsonl` when calls occur. All evaluated candidates are preserved even when the UI shows a smaller selection. `feedback.jsonl` stores human decisions by exact clip identity, independently of model output. Local exports/caches contain confidential content and remain gitignored.

`PROMO_FINDER_PUBLIC=1` (the public demo) allows uploads of files, a folder or a `.zip` plus the sample show, hides server folders and previous runs, and isolates each session's results. **Models and API keys** (in the app) swaps the provider, model and key of either stage per session: keys live only in that browser session; on the public demo a swapped model needs the visitor's own key, and a custom endpoint must be public HTTPS (no server key is ever sent to it). v8.6 is deployed at https://promo.expediva.in (server settings in the server's `.env`, public mode on). The Docker/nginx configuration is unchanged.

## Verification and submission

See the [two-page solution doc](docs/Promo_Finder_Solution_Doc.pdf), the [verification report](docs/verification/v8.md) and [design decisions](DECISIONS.md).

**Completed real run (v8.6, 8 Nov 2026):** all 50 episodes (3,727 beats) parsed and model-processed; Gemini 3.5 Flash-Lite discovery on every episode (free tier), Claude Haiku 5.5 final review of 72 clips; about 11 minutes and $0.10 of provider usage per full run. **45 clips passed every check with exact quoted evidence; 25 are returned (the cap), from 25 different episodes**; 19 went to review and 8 were rejected, each with reasons. Against the 20 hand-picked gold moments the top 25 cover 9/20 (v6: 12/20; 95% intervals overlap), so no ranking improvement is claimed. Unedited artifacts (confidential, gitignored): `outputs/v8/fd1e6aebd5ec7e477650e108/`. The first real run recommended only 3 clips because of prompt/checker contract bugs, not model errors; see BUGS_FOUND B28–B35.

The existing gold set contains selected positive moments, not exhaustive acceptance/rejection labels. `evaluate.py` now calls the top-list metric **gold overlap**, not precision. Human acceptance, false acceptance, ranking improvement and held-out-show quality have not been established. Synthetic regressions verify constraints; they do not establish model quality. The UI was exercised through Streamlit AppTest; real browser, narrow-screen, contrast and keyboard checks remain unverified because this sandbox cannot launch its listening socket.
