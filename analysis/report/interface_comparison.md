# HW4 interface comparison

Compared against the reference (`analysis/server.py`, `analysis/ui/index.html`) and the interface I built in `analysis/review_app/`.

## One design retained from the reference

The reference organizes review into three views: **trace** (read a trace and annotate the first failure in place), **map** (a 2D projection of the whole store with clusters, the sampled traces and the annotated ones), and **progress** (the failure-mode treemap and the suggestion queue). My interface keeps those three views and the same state files under `analysis/state/` (`annotations.json`, `samples.json`, `patterns.json`, `suggestions.json`), so the review loop's code can read my state unchanged. I added a fourth view, **labels**, for the present or absent decision per conversation and final mode.

## One design changed after inspecting the traces

Cartwheel creates one Langfuse trace per user turn, so a multi-turn conversation spans several traces. My interface lists **conversations grouped by `cartwheel.session_id`**, in order, so a followup is never read alone. I write notes in a **right-hand rail beside the evidence** (the message or tool call they refer to), not in a popover over the conversation. The agent's narration before a tool call (gray) is shown apart from the reply the user actually receives (green), and the scenario information and answer key are hidden until requested. Review is keyboard driven (`j`/`k` between conversations, `a` for a note, `o` for no failure observed). Free-text notes come first, before any score or label.

What I saw in the standard Langfuse view (from reviewing the first traces):

- **Score configs first.** The built-in human annotation queue needs score configs (the dimensions annotators score) defined before review can start. I had not defined any, so this stalled open coding. Langfuse suits a process that is already established, less so a greenfield one.
- **Every span is its own step.** It was not simple to annotate a whole message or to add comments on it, and there was no annotation queue defined that supported that workflow.
- **No conversation view.** Trace ids are not a conversation. Looking up a scenario and reading several messages together to follow how the agent performs over multiple turns was hard. For final-006 the messages were not in one view, and it was not trivial to navigate between them.
- **Tool calls live inside each trace.** I had to open each individual trace to find the tool calls behind a reply, instead of seeing them right away next to the response.
- **Too slow for 100 traces.** Taken together this would make reviewing 100 traces terribly inefficient.

These are why my interface groups traces into one conversation per `cartwheel.session_id`, shows tool calls and results inline beside each reply, takes free-text notes beside the evidence, and leaves any scoring to a later labeling step.

## One limitation remaining

The taxonomy cannot be edited in the page: accepting, renaming, merging or splitting a mode means editing `patterns.json` by hand. Also, the Progress view's batch table and the row tags treat every batch as batch 1 ("uniform"), and a conversation with several turns gets one label per mode stored against its last trace.
