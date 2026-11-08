# Decisions: study notes

These notes explain every number in `config.toml` and every design choice: **what it does, why it has this value, and the evidence for it.** They are study notes, not the submission doc.

"Evidence" means a row in `outputs/ablation/ablation.md`, or an honest *"hand-picked, and here's how I'd tune it."* The tuning plan is the same for every hand-picked number: collect editors' Approve/Reject verdicts (`feedback.jsonl` per run), then choose the value that maximises the approval rate of the top 10 on shows that weren't used for tuning.

## Core design

| Decision | Why (first principles) |
|---|---|
| **The model never outputs timestamps, only beat IDs.** | Timestamps are facts that already exist in the script. A model that copies or recalls them can only get them wrong. With IDs, code looks the time up, and an invalid ID is caught and counted (0 of 350 in v4). |
| **Code owns every cut: shot boundaries only, never inside a spoken line.** | An editor can't cut mid-sentence, and a cut between shots looks natural. These are hard rules, so they're enforced in code, not left to a model. |
| **Rules plus AI, blended.** | Rules are free, instant and explainable, and they're the fallback when there's no key. The AI understands meaning (sarcasm, subtext, an emotional twist), but it costs money and varies from run to run. The blend gets the best of both. |
| **A separate cold-viewer judge that never sees the show's context.** | "Makes sense on its own to someone who has never seen the show" can only be tested by something that genuinely hasn't seen it. The judge gets clip text only. |
| **A cache keyed on the exact prompt and parameters.** | Re-runs are free and reproducible, and a run that hits a rate limit resumes where it stopped. v4's key format is kept for v4's parameters, so old answers stay valid. |
| **Every tunable number lives in config, with v4 as a named preset.** | Before/after comparisons stay honest, and v4 can always be reproduced (the v4 regression test checks this). |
| **One `pipeline.run()` used by the CLI, the app and the evaluation.** | No duplicated logic, so the app can't drift from what was evaluated. |

## Candidate windows (`[scoring_presets]`; defaults are in `promo_finder/config.py`)

| Setting | Value | What it does / why | Evidence |
|---|---|---|---|
| `min_len`, `max_len` | 20, 60 s | Promo length. | Given by the brief. |
| `hook_window` | 3.0 s | The hook beat must *start* within 3 s of the cut-in. | Given by the brief. Bug B2 showed that "in progress" isn't enough. |
| `hook_lead` | 0.3 s | Tolerance for a beat that starts slightly before the cut. | Script timestamps are rounded to about 0.1–0.3 s. Hand-picked. |
| `cliff_window` | 4.0 s | Where to look for the cliffhanger: the last 4 s. | A cliffhanger beat plus a reaction shot is typically 2–4 s. Hand-picked. |
| `per_episode` | 6 | Continuous candidates kept per episode after de-duplication. | Enough variety without flooding the judge. Hand-picked; tune it by checking whether misses were ever generated. |
| `candidate_overlap` | 0.4 | Two candidates may share at most 40% of the shorter one. | Stops 10 near-copies of the same scene. Hand-picked. |
| `end_of_episode_bonus` | +0.25 | Bonus for ending where the writers ended the episode. | Writers place a cliffhanger at every episode end. Hand-picked. |
| `resolve_penalty` | 0.5 | How much a "resolved" ending (embrace, relief) cancels the cliffhanger score. | A resolved ending is the opposite of a cliffhanger. Hand-picked. |
| `resolve_on_calm_tone` | v4 true, v5 false | Whether a calm or tender *tone tag* alone counts as resolved. | v4 missed an emotional reunion because "tender" read as resolved. Ablation row 5d. |
| `emotional_reveal_exempt` | v5 true | AI-tagged emotional reveals with strength ≥ 7 skip the resolve penalty. | Same miss as above. Row 5g. |

## Two-moment (stitched) trailers

| Setting | Value | Why | Evidence |
|---|---|---|---|
| `stitched_max_episode_gap` | 8 | An opener and a closer at most 8 episodes apart. | Further apart, the story jump is too big for a stranger. Hand-picked. |
| `stitched_penalty` | ×0.95 | Every cut costs a little clarity. | The cold viewer scored stitched clips lower on standalone (4.8 vs 5.8). |
| `stitched_min_standalone` | v5 0.5 | A stitched promo must clear the standalone threshold, or it moves to the *needs context* list. | Same evidence. Row 5c. |
| `max_stitched_in_top` | v5 3 | Cap on stitched promos in the final list. | Keep the slate mostly self-contained clips. Hand-picked. |
| Continuity flag | always | "Check visual continuity (costume/location)" on every stitched promo. | The script can't show whether costumes or locations match across a cut; a human must check. |

## Blending and the AI stages

| Setting | Value | Why | Evidence |
|---|---|---|---|
| `w_llm` | 0.6 | Miner tags vs. rule scores, per beat. | The AI is better at meaning, so it gets a little more weight; the rules keep beats the AI ignored alive. Rows 2 vs 3. |
| `w_judge` | 0.6 | Cold-viewer judge vs. rules in the final score. | Same reasoning. Row 3 → 4 is the largest gain in the table (8 → 13 of 20). |
| `judge_top_n` | 60 | Clips sent to the judge. | Bounds the cost per show (≈ 12 calls). The best clip of *every* episode is always included (bug B3). |
| `judge_batch` | 5 | Clips per judge call. | Fewer calls, but batch-mates can affect scores (open issue O8). One clip per call would cost about 5× more. |
| `temperature` | 0.2 | Low randomness. | We want consistent scores, not creative writing. Some variance remains: a fresh run shared 6 of the top 10. |
| `show_profile` | v5 on | One call reads the loglines and returns characters plus show-specific reveal/threat words. Words not found in the scripts are dropped by code. | Replaces hand-written show words (like this show's title word) with per-show words. Row 5h. |
| `lexicon` | v4 `v4_legacy`, v5 `generic` | Keyword lists in `lexicon.toml`. The v4 list mixed in show-specific words and is kept in a private file. | O9: one show-specific word pushed a weak clip to #2. |
| `judge_prompt` v5 / `cliff_mode` | blend | Adds three yes/no checks (a question left unanswered? stakes? anything resolved?). The check score is mixed 50/50 with the 0–10 score. | The 0–10 cliffhanger score saturated (21 of 25 scored ≥ 8). Yes/no questions are harder to inflate. `run_meta.cliff_score_spread` compares how spread out the two are. Row 5e. |
| `rerank` | v5 on, 3 rounds, top 10 | Asks the model to *order* the top 10 head-to-head, 3 times with shuffled order and random letter labels, then averages the positions. Invalid orderings are dropped. | Comparing clips is more reliable than absolute scores, and shuffling cancels position and label bias. Row 5f. |

## Final ranking

| Setting | Value | Why | Evidence |
|---|---|---|---|
| Hook × cliffhanger | geometric mean | A promo needs **both**; a great ending can't rescue a weak opening. | Bug B4. |
| `standalone_mode` | v4 additive, v5 gate | *additive:* standalone is added on top. *geometric:* standalone is multiplied in like hook and cliffhanger. *gate:* clips below `standalone_threshold` (0.5) go to the separate *needs context* list. | Rows 5a and 5b. In the gate, those clips aren't wasted: they suit retargeting viewers who already know the show. |
| Weights | hook 0.30, cliff 0.30, standalone 0.20, intensity 0.10, funnel 0.10 | The brief's three qualities dominate; intensity and the early-episode bonus are tie-breakers. | Hand-picked. Tune them with editor approvals. |
| `funnel_strength` | 0.7 | The early-episode bonus falls from 1.0 (first episode) to 0.3 (last). It uses the position within the loaded range (bug O1). | Assumption: viewers land on Ep1, so early promos pay off sooner and spoil less. Kept mild, because a show's title often gives away its big twist. Test it with real ad data (CTR, cost per install by source episode). |
| `final_overlap`, `max_per_episode` | 0.3, 2 | Diversity in the final list. | An editor wants a slate, not 5 cuts of one scene. Hand-picked. |
| `top_k` | 25 | Promos returned. | Enough for a week of posting. |

## Parser

| Setting | Value | Why |
|---|---|---|
| `min_coverage` | 0.99 | If more than 1% of the lines that look like timestamps can't be read, stop before spending money. Silently dropped beats (O5) are the worst failure, because nobody notices them. |
| `episode_min/max_sec` | 20–600 s | Sanity warning for implausible episode lengths. |
| `frame_rate` | 30 | If every fraction is below 30, ".30" might mean frames, not hundredths of a second. The tool warns and records its assumption (decimal seconds). |

## Providers (`[providers.*]`)

| Setting | Why |
|---|---|
| `rpm` 12 (Gemini) | The free tier is strict and often returns 503 under load. One shared, thread-safe limiter per provider means extra workers never break the limit. |
| `workers` 3 | Overlaps network latency. On the free tier `rpm` is the bottleneck anyway (62 calls at 12/min ≥ 5 min). On paid tiers workers dominate (mock benchmark: 84 s → 14 s for 1 → 8 workers). |
| `timeout` 120 s, `max_retries` 6 | Backoff of 2, 4, 8 … s with ±50% jitter, so workers don't retry in lockstep. After the last retry, a clear error names the episode or batch. |
| `price_in/out` | Only for the cost estimates. 0 means free tier or unknown. Claude prices are from Anthropic's list (Opus 5.5 $4/$20, Haiku 5.5 $0.10/$0.50 per 1M tokens). |

## Ablation results (dev set, 20 moments; one moment = 5 points), and what was decided

107 real Gemini calls; all variants are in `outputs/ablation/ablation.md`. Every difference below except
rules → AI is inside the 95% intervals, and each variant swaps a few moments in and out. That pattern fits
noise plus the judge batch effect (O8), so read the table as "no evidence of improvement", not "proven worse".

| Change | recall@25 | Moments gained / lost vs v4 | Decision |
|---|---|---|---|
| v4 (reference) | 13/20 | n/a | still selectable (`--scoring-preset v4`) and reproducible from cache |
| 5a standalone geometric | 13/20 | none | optional; no effect |
| 5b standalone gate | 13/20 | none (4 clips move to *needs context*) | **kept in v6**: better UX, no cost |
| 5c stitched safeguards | 13/20 | none (3 clips move) | **kept in v6** |
| 5d tender tone ≠ resolved | 11/20 | loses 2 | dropped |
| 5e yes/no cliffhanger checks | 12/20 | +2 / −3 | dropped. The checks saturated *more* than the 0–10 score (mean 0.96 vs 0.74), so the hypothesis failed |
| 5f listwise rerank | 13/20 | none (same top 10, reordered) | optional; 3 calls/show |
| 5g emotional reveals | 10/20 | +1 / −4; still misses the one emotional moment (G16) | dropped. The model tagged 24 "emotional reveals" in 50 episodes, so the penalty exemption spread to ordinary tender scenes |
| 5h show profile + generic words | 12/20 | +2 / −3; best precision@10 (7/10) and key@25 (13/20) | **kept in v6**: needed for unseen shows |
| v5 (everything) | 11/20 | +3 / −5 | not recommended |
| **v6** (v4 + 5b + 5c + 5h) | 12/20 | precision@10 7/10, key@25 13/20 | **the default** (`scoring_preset = "v6"`): ties v4 within noise and doesn't depend on one show's vocabulary |

What the table really says: the cold-viewer judge is the big lever (8 → 13 of 20). The next gains are
more likely to come from a stronger judge model, or one clip per judge call (fixing O8), than from more rules.

## v7: verified eligibility (the default)

v4–v6 computed checks but didn't act on them, so a clip could fail a check and still be ranked #3 (BUGS_FOUND B16–B23). v7 separates **eligibility** (may this clip be recommended?) from **quality** (how good is it?), and decides eligibility in one place: `promo_finder/eligibility.py`.

| Decision | Why |
|---|---|
| **Statuses: recommended / review / rejected.** Any fail → rejected; any uncertain → review; all pass → recommended. | A clip that fails the brief isn't a promo, however high it scores. "Uncertain" exists so that a missing or unsupported answer never turns into a pass. |
| **Scores rank only recommended clips; `top_k` is a maximum.** | Weighted scores can't override a failed check. Returning 9 verified promos beats padding to 25 with failures. |
| **Deterministic checks (code):** duration (±0.05 s rounding tolerance), source bounds, segment order, cuts between spoken lines. | These are facts about timestamps, so code decides them. |
| **Semantic checks (judge + code):** hook in the first 3 s, unresolved ending, a new viewer follows, and the cut (two-segment promos). | These need understanding, so the judge answers each one as pass/fail/uncertain and cites beat refs. Code then checks that the evidence holds up. |
| **A verdict counts only if its evidence holds up.** Hook: a cited beat starts in the first 3 s, or is at least half visible. Ending: a non-music beat in the final 5 s, a stated open question, and the clip mustn't answer that question itself. Standalone: a stated conflict plus a cited beat. Cut: a stated connection plus a beat from each side. | Turns "the model said so" into something checkable. A judge that cites a mid-clip line for the ending, or a music swell, gets "uncertain". |
| **The ending is judged by the final moment's type** (`judge-v7.2`): suspense (something pending), resolution (tension released: a goal achieved, a search succeeds, a danger escaped, a scene closed in affection) or neutral, classified by what happens, not by tone. Only suspense passes. A resolution with cited `answered_by` beats is a confirmed fail and is re-cut before them. An open question that presupposes a resolution ("now that…") can't pass. | The `judge-v7.1` contract asked only for "the question the ending leaves open" and passed 72/72 endings (B27): a follow-on question can be written for any ending. Asking what the last moment *does* is harder to satisfy by rewording. |
| **A verdict that contradicts the judge's own score is not trusted** (pass at ≤ 4/10, fail at ≥ 8/10). | This is the B16 failure: a 4/10 standalone rating passing as "understandable". |
| **Rules can only send a clip to review, never pass it.** A strong rule hook (≥ 0.8, unblended) turns a judge "fail" into review. A visible resolution in the last beat (embrace, relief; never tone alone) turns a judge "pass" into review. A cut whose two sides share no named character turns a "pass" into review. | Disagreement goes to a human. Sharing a character is necessary but not sufficient, so it can't pass a cut on its own. |
| **Judge v7: one clip per call.** Prompt `judge:v7`, schema `judge-v7.2`. | Fixes the batch effect (O8) and scopes beat refs to one clip. Costs about 5× v6's judge calls (≈ 85 calls for 50 episodes). The schema version is in the prompt text, so a changed contract changes every cache key: old judgments can't masquerade as new ones. |
| **The judge sees only the selected footage**, in playback order (CAMERA, ACTION, dialogue, SFX), with partial beats marked. No logline, continuity notes or show profile. | A cold viewer has only the clip. Marking partial beats fixes B21. |
| **Validation and bounded retries.** Structural problems (wrong clip id, missing section, bad verdict, unknown ref, non-numeric score) trigger up to `judge_retries = 2` extra calls, each listing the problems. What still fails stays missing and becomes "uncertain". | A malformed answer never silently passes. Retries use a different prompt, so they're cached separately and re-runs are deterministic. |
| **Opening re-cuts** (`boundary_search = 4 s`, max 20 per run). For a shortlisted clip whose opening is weak, or starts after a strong physical beat has begun or just ended, search clean event boundaries (any shot, action, sound or line start not inside dialogue). Score each by "strongest beat starting in the first 3 s − 0.5 × share of a strong action already missed". Keep a re-cut only for a material gain (+0.1). | Captures the advertised action instead of its aftermath (B21). Both versions are judged normally; neither is forced into the ranking. |
| **Ending repairs** (`repair_rounds = 2`, max 15 per round). When the judge cites the beat that answers the clip's own question, re-cut to the last clean cut-out before it. When the ending fails without such a beat, cut after the strongest earlier tension beat. | "Search for earlier unresolved endings." Two rounds, because the first re-cut can still show the answer (an object found, then grasped). |
| **Every re-cut is a new clip.** Rule parts, duration and hook/cliff beats are recomputed. It carries no judgment, and a judgment is bound to the exact footage key (`Promo.key`). | Never reuse a judgment for different footage. `decide()` refuses a judgment whose key doesn't match. |
| **Production cues aren't evidence** (`count_production_cues = false`). Calm tone alone isn't resolution (`resolve_on_calm_tone = false`). | The writers' "cliffhanger" note describes intent, not what the viewer sees (B20). Ablation 5d dropped the calm-tone change in v5 because it lost moments *under score-based ranking*. In v7 the ending is decided by the semantic check, so the rule no longer has to stand in for it. |
| **Lexicon `generic_v7`** = `generic` + snatch / fling / wrench / yank / toss. | A snatched or thrown object is a visual hook the generic list missed. |
| **Explanations.** "Why it works" and "limitations" are separate fields. Code replaces text with platform or performance claims or dismissive words with a template built from the cited evidence (`explanations_replaced` in `run_meta`). | B25. |
| **One set of records behind every export.** `report.build_records` feeds JSON, CSV, Markdown and HTML. `run_meta.eligibility` is computed from the same `p.decision` / `p.checks`. | Reports and metrics can't disagree. A test checks that the IDs and statuses in the CSV and JSON match. |
| **Rules-only runs recommend nothing.** Every clip goes to review, marked "not verified". | Without a model, nothing semantic has been verified, and the brief's semantic criteria can't be checked by keywords. |

**Cost of v7** on the 50-episode pack (measured, gemini-3.5-flash-lite): 74 judge calls (72 clips + 1 validation retry + 1 ending repair), 148k input + 34k output tokens, 408 s wall clock on the free tier (12 requests/min). v6 used 12 batched judge calls. Profile and miner calls are unchanged and stay cached.

**Measured outcome** (same pack): 73 clips evaluated → 56 eligible (25 returned: `top_k` is the cap), 7 review, 10 rejected (8 incoherent cuts, 2 endings the clip resolves itself). The judge passed hook and standalone for 73/73: the remaining weakness is judge leniency, not the contract. Dev-set recall@25 is unchanged from v6 (12/20); precision@10 is 4/10 against 7/10, inside the 95% intervals.

## v8.0 — current implementation (2026-11-07)

Earlier sections are historical experiments; README and docs/verification/v8.md describe current behavior. The v7 working-tree changes were preserved, with the new production orchestration in v8.py. A separate route avoids silently changing prior experimental results.

- No keyword dependency in discovery, eligibility or ranking. Gemini selects grounded beat IDs and code retains multiple source-supported variants. This replaces v7's vocabulary-weighted shortlist and early NMS.
- Cold-viewer judgment uses Claude per clip. A pass requires literal evidence and a complete supporting hook beat within three seconds; coarse screenplay timing therefore sends more clips to review. This is intentionally conservative, not a claim that those clips are poor.
- Search limits: 14k-character target chunks, six overlapping beats plus four preceding factual beats; no truncation inside a beat. At most 80 candidate ends per start and 24 retained windows per episode; at most two segments, same episode, chronological. Cross-episode construction is deferred because it needs more source-context checks. The bounds are coverage limitations, not quality guarantees.
- Default budget $2/run, unknown prices block payment, serial provider calls, 60 priority reviews plus six difficult cases and four low-rank samples, up to six one-time ending repairs. A single bounded comparison only ranks already passing clips. Sampling audits the bounded candidate pool, not all omitted windows.
- Persistent SQLite reservations survive interruptions. Actual billing is unknown. File locks/cache keys prevent successful request duplication; ambiguous crashes preserve a pending marker instead of blindly retrying.
- Human review uses immutable clip identities in an append-only decision file. Public mode accepts only synthetic sample material. Confidential source, excerpts, caches and raw outputs remain ignored.
- Existing SDK fallback arguments were removed so Claude cannot silently route to a different model.
- No claimed quality improvement: the stronger-model live evaluation is blocked by configuration/network, and the positive-only gold set cannot measure true precision. The earlier phrase "inside the noise" did not establish equivalent ranking quality.


## v8.6 — owner decisions after the first real runs (2026-11-08)

| Decision | Why |
|---|---|
| **Evidence = beats quoted exactly.** An unquoted citation is dropped (`unquoted_refs_dropped`), not fatal; any altered or invented quote still makes the section uncertain. | The quote check exists to catch fabricated evidence. A citation without a quote is unproven, so it is ignored; it is not fabricated. |
| **One adequately visible cited beat is enough** (≥ 50 % of it plays inside the clip). | Claude often also cites the shot that runs past the cut-out; that extra citation sank otherwise well-supported endings. All cited beats mostly outside → still uncertain. |
| **Hook timing** (`review.lands_in_opening`, same wording in the prompt): a beat lands when it starts by 3.0 s and ends by 3.5 s, or ≥ 50 % of the whole beat plays before 3.0 s. | The brief asks for a hook *in the first 3 s*, i.e. something that starts and registers there. v8's "the whole beat must end by 3.0 s" rejected a line from +1.0 to +3.8 s. A beat mostly before the cut-in is still not credited (bug B2). |
| **The example JSON carries every field the checker requires**; one retry for any reply that fails validation. | Models copy the example shape (B28). A retry uses a different prompt, so a cached malformed reply is never replayed (B33). |
| Claude `max_tokens` 4,000 (was 2,400). | Haiku's thinking shares the cap; 2 replies were truncated (B34). Cost impact at $0.50/M output is negligible. |
