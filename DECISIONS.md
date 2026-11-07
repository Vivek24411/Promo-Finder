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
