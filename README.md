# Promo Finder

Reads a microdrama's timestamped screenplays and returns **ranked 20–60 s promos**. For each promo you get:
- the episode and start/end time of every segment;
- the hook (what grabs a stranger in the first 3 s);
- the cliffhanger it cuts on;
- one line on why it works.

It comes with a web app for editors, a CLI, an evaluation harness and tests.

**Live app:** https://promo.expediva.in. It runs in public mode; open the *Sample show* to try it without uploading anything.

**The scripts are confidential, so none are in this repo.** Point the tool at your own folder. An original 10-episode test show (`tests/fixtures/synthetic_show/`, not Kuku content) is included for trying it out.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # streamlit (+ anthropic, only for Claude)
pip install -r requirements-dev.txt        # pytest, for the tests
```

The core pipeline and CLI use only the Python 3.11+ standard library.

## Web app

```bash
streamlit run app.py                        # http://localhost:8501
```

The app follows the editor's steps: **Load → Check → Run → Review.**

1. **Load.** Upload `.txt`/`.md` files, a folder or a `.zip`; type a folder path; open the sample show; or reopen a previous run from the sidebar.
2. **Check.** Before anything is spent, the app shows:
   - how many episodes, beats and minutes it found;
   - the share of timestamp lines it could read (below 99% it stops, unless you tick "Run anyway");
   - parser warnings;
   - the number of **new model calls and tokens** the run will need.
3. **Run.** Live progress (stage, calls, tokens, time, estimated cost) and a Cancel button. A rules-only preview appears within seconds.
4. **Review.**
   - Promo cards with Approve/Reject. Verdicts are saved per run, in `feedback.jsonl`.
   - A *Needs context* tab, the report, a table, run details, and downloads of every output file.

**Sidebar:**
- **Provider:** Rules only (no key), Gemini, Claude, OpenAI or any OpenAI-compatible base URL, Groq, OpenRouter, Ollama.
- **Model id, API key, parallel calls.**
- **Scoring preset:** `v6` (the default: v4 plus the changes that held up in the ablation), `v4` (as submitted, still reproducible from cache) or `v5` (every change tried).
- **Episode range**, **number of promos**, **previous runs**, and **Clear cache**. The read-only v4 snapshot survives Clear cache.

**Keys and privacy:**
- A key typed into the app stays in that browser session's memory. It's never written to disk, logs, cache keys or `run_meta.json`.
- Uploaded scripts are written to a temporary folder that's deleted right after each step.
- Streamlit's usage statistics (telemetry) are switched off in `.streamlit/config.toml`.

## Command line

```bash
python run.py --show /path/to/show                                  # rules only, no key, ~1 s
python run.py --show /path/to/show --preset gemini                  # key from .env (GEMINI_API_KEY=...)
python run.py --show /path/to/show --preset gemini --scoring-preset v4   # the submitted version
python run.py --show /path/to/show --preset gemini --estimate       # how many NEW calls? spends nothing
python run.py --show /path/to/show --preset gemini --episodes 26-50 --workers 4
python run.py --show /path/to/show --preset anthropic --model claude-haiku-5-5
python run.py --show /path/to/show --preset openai --base-url https://your-gateway/v1 --model your-model --api-key-env YOUR_KEY_VAR
```

If a run needs uncached model calls, the CLI shows the estimate and asks before spending (`--yes` skips the question). `--force` runs even when parse coverage is below 99%.

**Presets** (base URL and key variable built in; rate limits and prices in `config.toml [providers.*]`):

| `--preset` | Default model | Key variable |
|---|---|---|
| `none` | rules only | none |
| `gemini` | gemini-3.5-flash-lite | `GEMINI_API_KEY` |
| `anthropic` | claude-opus-5-5 | `ANTHROPIC_API_KEY` |
| `openai` | gpt-4o-mini (or any `--base-url`) | `OPENAI_API_KEY` |
| `groq` | llama-3.3-70b-versatile | `GROQ_API_KEY` |
| `openrouter` | llama-3.3-70b-instruct:free | `OPENROUTER_API_KEY` |
| `ollama` | qwen2.5:7b (local) | none |

**Outputs** go to `outputs/<show>/<model>[__preset]/`:
- `promos.json` and `promos.csv` (the main list plus the needs-context list);
- `report.html` and `report.md`;
- `run_meta.json`: model, workers, settings, prompt hashes, stage timings, token use, parse coverage, held-out hash;
- `llm_calls.jsonl`: every raw model reply;
- `show_profile.json` and `parse_report.json`.

## Deploy on a shared server (Docker + nginx)

The image runs the app in **public mode** (`PROMO_FINDER_PUBLIC=1`), which hides everything that's only safe on your own machine:
- the "folder on this computer" input, so visitors can't read the server's files;
- *Previous runs*, so visitors never see each other's results;
- *Clear cache*;
- the custom **Base URL** field. With a key configured on the server, a custom URL would send that key to whoever runs the URL, and it would let visitors make the server call internal addresses.

```bash
# on the server, in a folder holding a clean copy of this repo plus a .env with GEMINI_API_KEY=...
sudo docker build -t promo-finder .
sudo docker run -d --name promo-finder --restart unless-stopped --memory 1g --cpus 1 \
  --env-file .env -p 127.0.0.1:8511:8501 \
  -v promo-finder-cache:/app/.cache -v promo-finder-outputs:/app/outputs promo-finder
sudo cp deploy/promo.expediva.in.nginx /etc/nginx/sites-available/promo.expediva.in
sudo ln -s /etc/nginx/sites-available/promo.expediva.in /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d promo.expediva.in        # HTTPS, once DNS points at the server
```

If port 80 is closed to the internet (true on the E-Cell server: only 443 is open), certbot can't verify over HTTP. Use the DNS check instead: run `sudo certbot run -a manual -i nginx --preferred-challenges dns -d promo.expediva.in --redirect`, add the TXT record it shows (`_acme-challenge.promo`), wait until `dig` shows it, then press Enter. That certificate doesn't renew by itself: repeat the steps before it expires (90 days).

- The container listens on localhost only; nginx proxies to it, including Streamlit's WebSocket.
- The memory and CPU caps protect the other sites on the server.
- To update: copy the new code, rebuild, then `sudo docker rm -f promo-finder` and run the same `docker run` again. The cache and outputs volumes are kept.
- Anyone with the link can run the model on your key, so watch your quota.

## Run on a new show (the interview flow)

1. `python run.py --show NEW_SHOW` checks the format in about a second, with no key. Look at the coverage line and the warnings.
2. `python run.py --show NEW_SHOW --preset gemini --estimate` shows the cost (v6 scoring is the default).
3. Run it, or use the web app. On the Gemini free tier (about 12 requests/minute) 50 episodes take 6 minutes or more. A paid key with `--workers 8` takes about a minute.
4. To rehearse, use the sample show in the app, or `python run.py --show REAL_PACK --episodes 26-50` as a "new" show.

## Quality checks

```bash
python evaluate.py outputs/<show>/<run> ...          # dev set: hits/total, rate, 95% Wilson interval
python evaluate.py --heldout outputs/<show>/<run>    # held-out set, kept separate and logged
python evaluate.py --ablation --show <folder>        # each change on its own; prints new-call counts first
python evaluate.py --ablation --show <folder> --allow-api   # actually spend the calls
python scripts/validate_moments.py eval/heldout_moments.json --show <folder>
```

**Evaluation sets.** The dev set is `eval/gold_moments.json` (20 moments, gitignored; the schema is in `eval/gold_moments.example.json`). For a held-out set:
1. Copy `eval/heldout_moments.template.json` to `eval/heldout_moments.json` and write the moments yourself, *before* looking at the tool's picks for those episodes.
2. Every held-out evaluation appends a line to `eval/heldout_ledger.log`, and you get a loud warning if the file ever changes.

**Human check of the AI judge** (run it on the synthetic show only):
1. `python scripts/rating_sheet.py <run>` builds a blind, printable sheet.
2. Raters fill in `ratings_template.csv`.
3. `python scripts/judge_agreement.py <run> ratings_*.csv` reports the Spearman correlation, the mean absolute difference, the share of ratings within ±2 points, and any bias.

**Provider smoke test** (one tiny call each, with your keys): `python scripts/smoke_test_provider.py [--structured]`.

**Tests:** `python -m pytest tests` runs about 60 tests with a mock LLM (no network, no cost), on the parser, the model layer, the pipeline end to end, and the app. A real-data v4 regression test runs only if the private input pack is present.

**Speed:** `python scripts/bench_mock.py --latency 1.5` measures 50 episodes against the mock server: 84 s with 1 worker, 23 s with 4, 14 s with 8.

## How it works

```
scripts → PARSE (code) → RULES (code) → SHOW PROFILE (LLM, v5) → MOMENT MINER (LLM, 1 call/episode, beat IDs only)
        → CANDIDATES (code: shot-boundary cuts, 20–60 s, two-moment trailers) → COLD-VIEWER JUDGE (LLM, sees only the clip)
        → RANK + CHECKS (code) → LISTWISE RERANK of the top 10 (LLM, v5) → outputs + editor review
```

**What the AI does, and what code does:**
- **The model never writes a timestamp.** It answers with beat IDs, code validates every ID, and code owns all times, cut points and length rules.
- **Rules make the tool free to run and explainable.** The AI handles the judgement calls (is this a hook? would a stranger follow it?), and an editor makes the final pick.

See `DECISIONS.md` for every number and design choice, and `BUGS_FOUND.md` for every bug found so far.

## Results on the input pack (dev set, 20 moments; one moment = 5 points)

| Run | recall@25 | precision@10 |
|---|---|---|
| Rules only | 6/20 = 0.30 [0.15–0.52] | 3/10 = 0.30 |
| Rules + AI miner, no judge | 8/20 = 0.40 [0.22–0.61] | 4/10 = 0.40 |
| Full v4 | 13/20 = 0.65 [0.43–0.82] | 6/10 = 0.60 |
| v5 (every change tried) | 11/20 = 0.55 [0.34–0.74] | 5/10 = 0.50 |
| **v6** (v4 + show profile + needs-context + stitched safeguards) | 12/20 = 0.60 [0.39–0.78] | 7/10 = 0.70 |

**Reading the table:**
- Going from rules alone to rules + AI is a real gain, and most of it comes from the cold-viewer judge.
- Smaller differences fall inside the 95% intervals.
- Several v5 ideas made results worse, and v6 keeps only the ones that didn't. See the ablation section of `DECISIONS.md`.
- A fresh v4 run matched 6 of the top 10. Every call is cached, so a given run is exactly reproducible.

## Known limits

- **It reads the script, not the video**, so it can't see acting, music or pacing.
- **The rule lists are English.**
- **The dev set is small** and was drafted by the AI, then reviewed by the author.
- **The early-episode bonus is an untested assumption.**
- **The judge is generous with cliffhanger scores.** v5 adds yes/no checks and a head-to-head rerank to address that.
