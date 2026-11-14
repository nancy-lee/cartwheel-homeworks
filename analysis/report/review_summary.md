# HW4 review summary

Counts come from `analysis/state/annotations.json` (active notes only; a superseded "no failure observed" marker kept as history is not counted), `analysis/state/sample_manifest.json` and `analysis/state/labels/`.

## 1. Reviewed sample: size and composition

I reviewed **122 distinct conversations** (69 with a failure note, 53 marked "no failure observed"). Two of the 122 (final-183, final-205) belong to scenarios I later set aside as unrealistic, so 120 are usable.

| Batch | Method | Traces | Failure notes | No failure observed |
|---|---|---|---|---|
| 1 | 15 uniform random + 15 cluster representatives (seed 7, `analysis/helpers/selection.py`) | 30 | 17 | 13 |
| 2 | One product dimension chosen before seeing outcomes (`intent`), 3 to 4 traces per value across its 9 values | 30 | 11 | 19 |
| 3 | Depth searches for candidate modes plus close negatives (25 drawn at random from what the searches retrieved) | 25 | 19 | 6 |
| 4 | 15 uniform random, after the first taxonomy draft | 15 | 7 | 8 |
| 5 | 15 more uniform random (second stability check) | 15 | 11 | 4 |
| Outside the batches | Reviewed by hand while investigating (final-004, 014, 040, 051, 176, 205, 218) | 7 | 4 | 3 |

No trace is counted in more than one batch. Batches 1 to 5 use 115 distinct traces; the 7 extras bring the total to 122.

**How the sample was built.** The review list is the 225 realistic scenarios of the 250 in `scenarios/support_scenarios.jsonl`. I set 25 aside before sampling as unrealistic (role and intent mismatch, a merchant written as a buyer, no record identified, the same request repeated 5 or more times); the reasons are in `analysis/state/excluded_scenarios.json`. I excluded one more (final-183) while reviewing batch 3, for a total of 26. I read batch 2's scenarios in full and swapped out any that were unrealistic. I never selected a trace because a model or heuristic predicted a failure.

**Data-quality notes.** About 82% of scenario openings begin with hi, hey or hello, and many share phrasing. The way the scenarios are generated needs to improve.

## 2. Final taxonomy (5 modes) and sample fractions

I labeled every one of the 120 usable conversations present or absent for each of the 5 final modes (600 decisions, none left open). Each decision is saved in `analysis/state/labels/<mode>.jsonl` and written to Langfuse as a score named after the mode (1 present, 0 absent). A conversation is labeled against its last trace. **These are sample fractions from a deliberately mixed sample, not prevalence estimates**, because clustering, depth searches and filtering change the mix.

| Mode | Present | Fraction of 120 | First-failure notes | Requirement source |
|---|---|---|---|---|
| `platform_policy_not_store_policy` | 21 | 18% | 21 | Gap (new rule needed, see section 6) |
| `right_lookup_tool_not_used` | 29 | 24% | 18 | Gap (TOOL-5 and TOOL-6 say what the tools return, not when to use each) |
| `write_without_enough_context` | 13 | 11% | 10 | Gap |
| `reply_not_direct_or_actionable` | 42 | 35% | 8 | Extends RESP-3 |
| `wrong_human_handoff` | 7 | 6% | 4 | ESC-3, plus Gap for the support-caller case |

68 conversations have at least one mode present, 39 have two or more, and 52 have none. The "present" counts are higher than the first-failure counts because open coding stops at the first failure, while the labeling pass records every mode present in the trace.

`wrong_human_handoff` merges two earlier candidates, `support_caller_sent_to_support` (a support caller pointed back at support) and `missing_escalation_to_human` (a dispute or request not escalated). Both are fixed by the same escalation rules, and neither reached 3 clear positives on its own. Not final modes: `tool_lookup_mismatch` (1 note) is on a watch list below the 3-positive minimum; `merchant_advice_not_actionable` (192), `user_role_not_used_in_reply` (194) and `reply_reveals_inaccessible_order_info` (046, source RESP-4) are single notes.

**Close negatives (at least 3 where the data permits).** For the lookup, write, irrelevant and handoff modes I found 7, 6, 6 and 8 candidates by text and tool checks, and I confirmed each as "no failure observed" in my review. The policy mode has **2** (final-004, final-218). No other reviewed trace shows a store's policy used correctly, so the data does not permit a third.

## 3. Stability check

- Batch 4 (the first 15 uniform traces, after the taxonomy draft): **1** previously unseen mode (final-194, the reply ignored the caller's role).
- Batch 5 (the final 15): **1** previously unseen mode (final-046, the reply hinted that a denied order belonged to someone else). I grouped final-115 into an existing mode.
- Conclusion: the set of modes is stable enough. Each of the last two batches (15 traces each) produced one new mode, and none added a sixth final mode.
- A caveat on that conclusion: while I reviewed the labels and counted which traces fall into which modes, I changed the modes themselves to be more specific, so that each one has a clear pass or fail. I rewrote every definition as a "Fail if…" rule, renamed `write_action_beyond_request` to `write_without_enough_context`, merged "several matches not clarified" into `right_lookup_tool_not_used`, merged `support_caller_sent_to_support` and `missing_escalation_to_human` into `wrong_human_handoff`, and changed `irrelevant_results_or_offers` to `reply_not_direct_or_actionable` to match how I had actually labeled it. Two renames were also needed to fit Langfuse's 35-character limit on score names. So the stability check shows that new kinds of failure stopped appearing, but the definitions kept sharpening until I applied them trace by trace.

## 4. One taxonomy revision

`write_action_beyond_request` became `write_without_enough_context`. After batch 3, five notes (final-091, 129, 130, 017, 184) said the agent should have asked for the refund reason or confirmed before the write, which did not fit "a write the user did not ask for". My rule is that the agent does not always need to confirm, but it needs enough context to decide. I widened the definition to match, and the old name no longer fit. Motivating notes: 983950553f0c (final-091), 1b966ccf7a5e (final-129), 1efb30084757 (final-130). Other revisions along the way: I judged the negative-price notes (027, 028, 053) not to be a mode and folded them into `reply_not_direct_or_actionable`; I folded "several matches not clarified" (4 notes) into `right_lookup_tool_not_used`; and I dropped the reply-tone mode (note 094 stays).

## 5. Search suggestions

13 suggestions from the depth searches: 11 accepted, 2 rejected. Rejected: final-042 (negative price; my note described a different failure, a reply that did not match the tool lookup, not a negative value returned) and final-049 (write without context; the agent followed its instructions correctly and sent the return request to a human agent for review, so the write was not made without context). `suggestions.json` keeps each mode name as it was when the suggestion was made: `write_action_beyond_request` is now `write_without_enough_context`, `lookup_not_used_before_replying` is now `right_lookup_tool_not_used`, and `negative_price_presented_as_valid` was folded into `reply_not_direct_or_actionable`.

## 6. SPEC.md revisions and the motivating annotation

I added these to `SPEC.md` as TOOL-10 (under a new "Tool use guidance" heading in section 4) and RESP-6 to RESP-9 (section 6). Editing `SPEC.md` does not change the running agent; the prompt in `agent/agent.py` is unchanged.

| Revision | Motivating annotation |
|---|---|
| Use `find_order` when the user names a product; confirm among several matches | 8b094e57576b (final-148), a9a9d3e67695 (final-030) |
| Store policy over platform policy, and name the source | aa985a6b36d1 (final-103), bdf616329e1b (final-212) |
| Enough context before a write | 983950553f0c (final-091), 929b29ea0985 (final-002) |
| Show only what was asked for; flag invalid values | c7e3de569ede (final-016), 87fb863f88dd (final-027) |
| Treat support callers as the handler | bfe4aec2047a (final-051), f102c59954fd (final-237) |

## 7. Findings about the system prompt (not modes)

The agent's prompt (`agent/agent.py`, version 5bc7336ec81c) never names `find_order` or `list_my_orders`, and says nothing about store policy beating platform policy. For final-046 the agent did what the prompt says: "offer to help with the user's own orders", which is the wording that hints the order is someone else's (RESP-4 already exists in `SPEC.md`). These traces describe this prompt version; I did not change the prompt during HW4.

## 8. Limitations

- I did not do Part C (Workshop).
- Close-negative candidates came from text and tool-call checks, then my own judgment.
- The sample is not representative, so the fractions are not prevalence.
- A multi-turn conversation gets one label per mode, stored against its last trace.
