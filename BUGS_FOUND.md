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
| O8 | **Judge batch effect:** a clip's score depends on its 4 batch-mates, and the cache key includes the whole batch. | v5's head-to-head rerank reduces the impact on the top 10. **Fixed in v7:** one clip per call. v4–v6 still batch. |
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

## Found in the v6 run review (fixed in v7)

An external review of the v6 outputs reported inconsistent outcomes: `judge.checks_answered = 0`, 8 of 25 main promos failing an exported check, 17 of 25 carrying cold-viewer confusion notes, and every main promo passing `ends_unresolved`, including clips that visibly answer their own cliffhanger. Reading the code found these causes. v4 and v6 are left as they were, so they stay reproducible; v7 fixes them.

| # | Bug | Root cause | Fix (v7) |
|---|---|---|---|
| B16 | **The needs-context gate let a 4/10 stranger rating through.** 5 of the 25 v6 main promos had a judge standalone score of 4/10. | The gate compared the *blended* standalone (`0.4 × rule + 0.6 × judge`) with 0.5. The rule proxy is almost always 1.0, so a 4/10 judge score became 0.64. | Standalone is a semantic check (pass/fail/uncertain with evidence). A verdict that contradicts the judge's own score (pass at ≤ 4/10) is not trusted. Scores only rank. |
| B17 | **`ends_unresolved` passed 100% of main promos.** | A rule check (`max cliff > 0.6 × max resolve`) run on clips the generator had already selected for high cliff scores, so it passed by construction. It never looked at the judge's answer. | The ending check needs the judge's open question, evidence from the final 5 s, and a whole-clip "is it answered inside the clip?" pass. A clip that answers its own question fails and is re-cut earlier. |
| B18 | **Clear visual hooks failed `hook_in_first_3s`.** | After blending, a beat the miner didn't tag maxes at `0.4 × rule` = 0.36, below the 0.45 threshold. Every untagged hook failed. | Rule opinions use the unblended strength (`max(rule, miner tag)`). A strong rule hook can't be overruled by a judge "fail": the clip goes to review instead of being rejected. |
| B19 | **`judge.checks_answered = 0` in v6.** | v6 runs the v4 judge prompt, which never asks the yes/no checks. Not a parser bug: the counter reported something the prompt never requested, and nothing downstream noticed the checks were absent. | The v7 contract asks for every check. Diagnostics separate API calls, cache hits, parsed replies, valid replies, retries, and explicit pass/fail/uncertain/missing answers per check. |
| B20 | **Production notes scored as cliffhangers.** | The lexicon gave `cliff = 1.0` to the literal word "cliffhanger" and to "cut to black", and 0.8 to music swells, so stage directions counted as story evidence. | `count_production_cues = false`: those phrases are stripped from stage directions, and music-only sound cues carry no cliff score. A music cue is not accepted as ending evidence. |
| B21 | **The judge was credited with action the clip doesn't show.** | `render_clip` printed a beat already under way at the cut-in as `+00.0s`, even when only 0.4 s of a 1.2 s beat was inside the clip. | The v7 clip text marks partial beats ("under way before the promo starts; 33% shown"). Hook evidence must start in the first 3 s or be at least half visible. Openings that start just after a strong action are re-cut to include it. |
| B22 | **`cliff_score_spread.score_0_10` was on a 0–1 scale.** | The mean was divided by 10 (0.74 meant 7.4/10). | Reported as `judge_cliffhanger` with `scale: "0-10"`, plus hook and standalone. |
| B23 | **Exported checks were informational.** 8 of 25 v6 main promos failed one. | Only length and clean cuts gated the list. | One authoritative decision (`eligibility.decide`): any fail → rejected, any uncertain → review, all pass → recommended. Only recommended clips are ranked, and `top_k` is a maximum. |
| B24 | `fmt()` would print 59.96 s as `00:60.0`. | Formatting rounded the seconds after splitting off the minutes. | The minute now rolls over. Every other value formats byte-identically (checked on 200k random times), so cache keys are unchanged. Exports use `MM:SS.ss`, the scripts' own precision. |
| B25 | Explanations used a slangy persona voice and unsupported performance claims ("always … on TikTok"). | The v4 judge persona ("a 24-year-old scrolling TikTok"). | A professional editor persona that must name the concrete mechanism. Code replaces any text with platform/performance claims or dismissive words with a template built from the cited evidence. |
| B26 | The 20–60 s check had no rounding tolerance. | `60.000000001` failed. | ±0.05 s tolerance (`len_tolerance`). |
| B27 | **First v7 judge contract (`judge-v7.1`) passed 72 of 72 endings**, including a clip that ends on the object of a search being found and grasped. | Real run on the input pack. The contract asked "what question does the ending leave open?", and the judge can always find *some* question: 14 of 72 answers were follow-on questions like "what will she do now that she has found it?". Valid JSON with valid evidence still can't discriminate if the question can't. | `judge-v7.2` also asks for the clip's central question and classifies the final moment as suspense / resolution / neutral *by what happens, not tone*. Only "suspense" can pass. A resolution's `answered_by` beats drive the earlier re-cut. Code rejects an open question that presupposes a resolution ("now that…"). The v7.1 run is kept in `outputs/…/gemini-3.5-flash-lite__v7_judge-v7.1/` as the record. |

## Process mistakes (how the tools were used, not the product)

| # | Mistake | Consequence | Lesson |
|---|---|---|---|
| P1 | A diagnostic script passed a different show name than the cached run. | ~25 unplanned real API calls before it was stopped. | Check cache hits on one call before any batch (now ground rule 6). |
| P2 | `pkill -f <pattern>` matched the shell running it. | The command killed itself (exit 144). | Kill by PID. |
| P3 | A zsh glob with no match aborted a whole `rm` line. | A clean-up silently did nothing. | Check the result after deleting. |
| P4 | Playwright was installed onto an almost-full disk. | A half-installed package (116 MB) had to be removed. | Check free space before installing. |
| P5 | `ps \| grep pattern \| kill` again matched the shell that was running it (its own command line contained the pattern). | The command killed itself (exit 144). | Match on the process *name* column (`$2 ~ /chrome/`), not the full command line. |
| P6 | The browser test of the deployed app tried to switch the provider to "Rules only" but didn't check that the switch worked, then clicked *Find promos*. The switch had silently failed. | **17 unapproved Gemini calls** (~40k tokens) on the synthetic sample show (no Kuku content). | Any automated click on a spend button must first assert the provider and the "new model calls" estimate shown on the page. |

## v8 investigation and changes (2026-11-07)

- Confirmed saved v7 hook/standalone passes: 73/73. This is evidence of a permissive judge on that shortlist, not evidence that every check was correct.
- Episode 13 is not wholly absent: one fragment occurs in a recommended stitched clip; its long continuous cut is in review. The claim that a particular stronger boundary never reached recommendations needs a separately labeled target before recovery can be measured.
- Episode 16 differs across artifacts: the v7.1 artifact recommended it with an ending pass; the newer saved v7 artifact has a raw judge ending pass but code marks it uncertain. Thus the reported issue depends on version. This is one concrete model/code disagreement behind the new strict evidence route.
- Existing UI required editors to manage provider/key/model settings. v8 moves them to server configuration and reduces review to a table plus selected detail.
- v7 accepted rounding tolerance around duration limits, and two-decimal exports lost finer source precision. v8 uses decimal duration subtraction and strict source boundaries; exports preserve finer decimals.
- A line starting before three seconds previously sufficed for hook timing. v8 requires a fully included supporting beat ending within three seconds, or review.
- Successful-only usage counting and unknown-as-zero prices could hide spend. The new ledger reserves every attempted call and retains allowances when provider usage is unavailable.
- Review selector callbacks initially captured the loop's final rows; the AppTest approval flow exposed this. Bound each selector to its own rows and verified persisted decisions without model-output mutation.

## v8.1–v8.6: found in the first real Gemini + Claude runs (2026-11-07/08)

Codex wrote v8 without Claude credentials, so its prompt/checker contract had never met a real model. The first real 50-episode run (Claude Haiku 5.5) recommended **3** clips. Every cause below was found by replaying the saved replies, not by guessing; Claude's 1,110 quotes in that run were **all exact** substrings, so the losses were contract bugs, not model errors.

| # | Bug | How it was caught | Fix |
|---|---|---|---|
| B28 | **Quote rule existed only in prose.** The checker demanded one exact quote per cited beat, but the example JSON in the prompt had no `quotes` field. Claude followed the example and omitted quotes in 91 sections. | Every failing section had `quotes = []`; all supplied quotes matched exactly. | `judge_shape_v8`: the example JSON now carries `quotes` (one per evidence ref), `timing`, `outcome` and the extra scores (v8.4). |
| B29 | **One unquoted citation sank a whole section.** Claude cited 5 beats and quoted 4 exactly; the checker treated that as "missing or invented". | Residual failures were all "N cited, N−1 quoted, 0 invalid". | v8.6: evidence = the beats quoted **exactly**; unquoted refs are dropped (`unquoted_refs_dropped`). An altered/invented quote still blocks. |
| B30 | **Harmless curiosity put in `blocking_confusion`.** 14 standalone passes said "names aren't explained, though the core conflict is clear" in the blocking field, contradicting the pass. | Read the 14 values. | Prompt and example JSON say: empty on pass; names/backstory belong in `harmless_curiosity` (v8.5). |
| B31 | **Any partly-outside shot sank the evidence.** A citation of a shot that runs past the cut made the section uncertain even when other cited beats fully supported it. | 48 sections; replay showed 13 clear with a visibility rule. | Owner decision, v8.5–v8.6: a cited beat counts when ≥ 50 % of it plays inside the clip; the section is uncertain only if **no** cited beat does. |
| B32 | **Hook timing: whole beat had to end by 3.0 s**, in both prompt and code. A strong line from +1.0 to +3.8 s failed; Claude marked many hooks "uncertain" because the prompt told it to. | Replay: 10 sections; plus many judge-side "uncertain" hooks. | Owner decision, v8.6 (`lands_in_opening`, same text in the prompt): a beat lands when it starts by 3.0 s and ends by 3.5 s, or ≥ 50 % of the whole beat plays before 3.0 s. A beat mostly before the cut-in is still not credited (B2). |
| B33 | **Malformed JSON cached as a valid fragment.** A stray token inside a reply made the tolerant parser return an inner object; it was cached, so reruns replayed it (Gemini: episodes 2 and 45; Claude: 1 review). | Reply text failed `json.loads` at the stray token. | One retry with a different prompt (new cache key) when discovery lacks `events` or a review lacks `clip_id`/sections. |
| B34 | **Claude replies truncated** at `max_tokens=2400` (thinking shares the cap; replies averaged ~1,800). | 2 reviews ended mid-string. | Claude cap 4,000; the estimate formula updated to match. |
| B35 | **Interrupted call blocked an episode forever.** Episode 15's `.pending` marker (by design) stopped retries. | `run_meta.discovery_failures`. | Operator check per README: no reply was ever logged (free-tier Gemini, no charge), so the marker was cleared. |
