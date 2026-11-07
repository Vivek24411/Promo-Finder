# Bugs found

A running log of every bug found in this project, including bugs in AI-written code (most of the code was pair-written with Claude Code). Each entry covers what went wrong, how it was caught, and the fix. Wording is kept generic: no script text or plot details.

Status: **Fixed**, **Open** (found, not fixed yet; the planned phase is noted), or **Process** (a mistake in how a tool was used, not a product bug).

## Fixed

| # | Bug | How it was caught | Fix |
|---|---|---|---|
| B1 | The default Gemini model (`gemini-2.5-flash`) was chosen by the AI from memory, but Google had retired it for new API keys (HTTP 404). | Listed the live models with the real key before the first run. | Changed the preset default to `gemini-3.5-flash-lite`, which was verified to answer on the free tier. |
| B2 | **Hook window.** A beat already *in progress* at the cut-in counted as "the hook in the first 3 s". A clip starting ~1 s after a strike sound effect was credited with the strike it never shows. | Evaluation: a promo for a gold moment was counted as a miss, and reading its timestamps showed the clip opened just after the key beat. | A hook beat must *start* in `[cut − 0.3 s, cut + 3 s)` (`candidates.window_parts`, and the same rule in `rank.quality_checks`). |
| B3 | **Shortlist starvation.** Only the top-N clips by rule score went to the AI judge, so whole episodes were never judged. A key early-episode moment missed the cut-off by 0.02. | Diagnosed two evaluation misses by printing their candidate scores against the shortlist cut-off. | Every episode's best clip is always judged, then the shortlist is filled by score (`pipeline.run`). |
| B4 | **Additive scoring.** With hook and cliffhanger simply added, a great ending could rescue a weak opening; a clip the judge rated 6/10 on hook ranked #1. | Read the #1 promo and its component scores. | Hook and cliffhanger combined as a geometric mean. Only partly effective: the clip fell to #2, because the rules still over-score one keyword (see O9). |
| B5 | **Show-name cache mismatch.** The CLI cleaned the folder name ("…-scripts (2)" → "…") but the app used the raw name. The prompts differed, so the cache missed and **61 real API calls were made silently**. | An app test showed "61 calls" where it should have shown "0 / 62 cached". | Shared `pipeline.clean_show_name()`, used by both. Still open: uploads with no typed show name (see O13). |
| B6 | **Episode-number collisions.** Files with no number in the name or header all parsed as episode 0, giving duplicate beat IDs. | Code review while adding uploads; tested with renamed copies. | `load_show` renumbers by natural file order when numbers are missing or duplicated. |
| B7 | Deprecated Streamlit calls (`components.v1.html`, `use_container_width`) were scheduled for removal, which would break a cloud deploy. | Warnings in the automated app test. | Switched to `st.iframe` (with a sandboxed `data:` URL) and `width="stretch"`. |
| B8 | Evaluation metric v1 (does the promo contain the gold window's midpoint?) was unfair to long moments. Metric v2 (≥ half of the window, or 8 s) penalised promos that open *exactly* on the key line. | Read individual misses that were clearly good picks. | Added `key@K`: does the promo contain the single key instant? All three metrics are reported side by side. |
| B9 | The "Model calls" metric label was truncated in the app. | Browser screenshot. | Shorter label: "new / cached". |

## Found while mapping (Phase 0), fixed in the improvement round

| # | Bug / risk | Fix |
|---|---|---|
| O1 | `funnel_fit` assumed episodes start at 1, so the bonus went **negative** on subsets (−0.43 for episode 50 of 26–50). | Uses the position within the loaded range. Identical for 1..N, so v4 is unchanged. Test: `test_episode_subset_keeps_funnel_in_range`. |
| O2 | The `LLM` repr included the API key. | `repr=False`, plus redaction of the key in every error and log line. Test: `test_api_key_never_written_anywhere`. |
| O3 | The cache key ignored temperature, max_tokens and output mode. | They're now part of the key, except for exactly v4's parameters (so v4 answers stay valid). Test: `test_v4_cache_key_format_is_unchanged`. |
| O4 | `parse_json` broke when reasoning text or trailing commentary contained braces. | `extract_json`: strips reasoning blocks, prefers fenced blocks, then takes the first *complete* JSON value. 8 tests. |
| O5 | Blocks with no Start line were dropped silently. | A parse-coverage report runs on every load and stops the run below 99% (unless forced), listing every unread line. |
| O6 | The app ignored `config.toml`. | The app and CLI both use `config.load()` and `make_llm()`, and the weights come only from the scoring preset. |
| O7 | Uploaded scripts stayed in `/tmp` forever. | Each step writes them to a temporary folder that is deleted when the step ends. |
| O10 | `evaluate.py` hard-coded the gold path. | Paths come from `config.toml [eval]`. |
| O11 | Test clicks sat in a global `editor_feedback.jsonl`. | Feedback is saved per run (`<run>/feedback.jsonl`); the old file was deleted. |
| O12 | Plot details appeared in comments, a docstring, the README and the keyword list. | Scrubbed. The v4 keyword list moved to a gitignored `*.private.toml`, and the real-data paths to `eval/paths.private.json`. |
| O13 | Uploading the pack without the exact show name caused new calls. | Not silent any more: the app and CLI show the number of new calls before running, and the CLI asks for confirmation. |
| O14 | The throttle wasn't thread-safe. | A shared `RateLimiter` per provider (requests and tokens per minute) with a lock. Tests cover the spacing and the token budget. |
| O15 | Two `.env` loaders behaved differently. | One `config.load_dotenv()`. |
| O16 | The app showed raw stack traces. | Readable messages per error kind (bad key, rate limit, provider down, bad request, refusal, unreadable reply), and `client.showErrorDetails = "none"`. |

## Still open

| # | Issue | Notes |
|---|---|---|
| O8 | **Judge batch effect:** a clip's score depends on its 4 batch-mates, and the cache key includes the whole batch. | v5's head-to-head rerank reduces the impact on the top 10. The full fix is one clip per call (about 5× the judge calls), or fixed batches. |
| O9 | The rules over-score show-specific words. | Mitigated in v6 (generic list plus show-profile words). Ablation 5h: precision@10 7/10 vs v4's 6/10, inside the noise. The profile's code check dropped 6 invented words. |

## Found during the improvement round

| # | Bug | How it was caught | Fix |
|---|---|---|---|
| B10 | **The v5 judge prompt edit didn't apply.** The prompt is a `.format()` template, so its JSON braces are doubled (`}}`), but the replacement looked for single braces. v5 would have silently sent the v4 prompt. | Re-reading the code before running it. | Matched the doubled braces; `test_v5_prompts_really_differ` now asserts every v5 change is present. |
| B11 | **Streamlit sends usage statistics by default** (`browser.gatherUsageStats = true`). That's telemetry, which the plan rules out. | Checked `streamlit config show` while designing the theme. | Switched off in `.streamlit/config.toml`. |
| B12 | The real-data test named the show in a file path, and the v4 keyword list contained show-specific words: confidential information that would have been committed. | A grep for names and terms across the files a fresh clone would contain. | Both moved to gitignored private files; tests skip when they're absent. |
| B13 | A test expected the unread line on line 7; it's on line 9. | The test failed. | The test's arithmetic was wrong, not the parser. Fixed the expectation and documented the count. |
| B14 | The benchmark repeats the same 10 episodes, so some judge prompts are identical. Parallel workers can both miss the cache for the same prompt (56 vs 61 calls). | The call count changed with the number of workers. | Benign: it only happens with duplicate prompts, which real shows don't have. Documented. |

## Found while deploying to a shared server

| # | Bug / risk | How it was caught | Fix |
|---|---|---|---|
| B15 | **The app assumed it runs on the editor's own machine.** On a public server: (1) the custom Base URL field would send the *server's* API key to any URL a visitor typed, and would let visitors make the server call internal addresses (SSRF); (2) "Folder path" read the server's folders; (3) "Previous runs" showed every visitor's results, including script text; (4) anyone could clear the cache. | Reviewing the app's inputs before deploying, asking "what could a stranger do with this field?" | Public mode (`PROMO_FINDER_PUBLIC=1`, on in the Docker image) hides all four. Test: `test_app_public_mode_hides_server_features`, confirmed to fail with public mode off. |

## Process mistakes (how the tools were used, not the product)

| # | Mistake | Consequence | Lesson |
|---|---|---|---|
| P1 | A diagnostic script passed a different show name than the cached run. | ~25 unplanned real API calls before it was stopped. | Check cache hits on one call before any batch (now ground rule 6). |
| P2 | `pkill -f <pattern>` matched the shell running it. | The command killed itself (exit 144). | Kill by PID. |
| P3 | A zsh glob with no match aborted a whole `rm` line. | A clean-up silently did nothing. | Check the result after deleting. |
| P4 | Playwright was installed onto an almost-full disk. | A half-installed package (116 MB) had to be removed. | Check free space before installing. |
| P5 | `ps \| grep pattern \| kill` again matched the shell that was running it (its own command line contained the pattern). | The command killed itself (exit 144). | Match on the process *name* column (`$2 ~ /chrome/`), not the full command line. |
| P6 | The browser test of the deployed app tried to switch the provider to "Rules only" but didn't check that the switch worked, then clicked *Find promos*. The switch had silently failed. | **17 unapproved Gemini calls** (~40k tokens) on the synthetic sample show (no Kuku content). | Any automated click on a spend button must first assert the provider and the "new model calls" estimate shown on the page. |
