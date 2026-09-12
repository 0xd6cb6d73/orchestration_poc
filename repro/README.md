# GPT-OSS reasoning-only response reproducer

`gpt_oss_empty_response.py` isolates the failure from Phoenix trace
`9f9e40103a168ef2812e2ba72fd83d10` in experiment `ea28f9d7a983`.
It needs no application imports, task generator, orchestrator, Phoenix, or database
server: one Pydantic AI agent, a tiny in-memory SQLite query tool, and a typed answer.

From the repository root, using the locked environment (`uv sync --extra llm` if needed):

```sh
# Deterministic, offline replay of the observed response shape.
.venv/bin/python repro/gpt_oss_empty_response.py

# Same adapter and tool round trip, but with a proper final_result response.
.venv/bin/python repro/gpt_oss_empty_response.py --mode control

# Real provider; loads OPENAI_BASE_URL and OPENAI_API_KEY from .env.
.venv/bin/python repro/gpt_oss_empty_response.py --mode live
```

Live mode has a six-model-request/six-tool-call ceiling and 60-second overall timeout.
The SDK can retry transport failures within that deadline. Output reports response
shape and synthetic-task SQL, not credentials, headers, or reasoning content.
Exit 0 means the expected replay failure or a correct control/live answer; read the
printed outcome to distinguish them. Exit 1 means a failed assertion/answer check;
exit 2 is inconclusive (another failure or budget exhaustion).

## What it demonstrates

The offline replay supplies a successful query tool call, followed by two responses
with `finish_reason=stop`, no content/tool calls, and a reasoning field containing
SQL-shaped text. It uses a synthetic HTTP fixture, not a byte-for-byte provider
capture. Pydantic AI exhausts `retries={"output": 1}` and raises
`UnexpectedModelBehavior: Exceeded maximum output retries (1)`.
The control supplies a real `final_result` tool call and returns `task-0000: 0`.
Both pass through the real OpenAI SDK and Pydantic AI response parser.

This distinguishes reasoning text from executable tool calls. The fuller Phoenix
agent transcript contains reasoning on the failed turn, even though the normalized
LLM output table appears empty. No adapter should execute SQL extracted from that
reasoning as an implicit tool call.

## Verification on 2026-09-12

Environment: Python 3.11, `pydantic-ai-slim==2.36.0`, `openai==3.13.0`,
`httpx2==2.12.0` (available through the locked LLM dependencies).

- Offline replay: reproduced output-retry exhaustion after three model responses.
- Offline control: returned the expected mapping after two responses.
- Initial live probe hit a SQLite authorization error during schema discovery;
  the reproducer now returns SQL errors as tool results, as the original suite does.
- A subsequent live probe returned a reasoning-only stop response, then recovered
  to another tool call and exhausted the earlier four-request cap. This reproduced
  the intermediate symptom, not the terminal exception.
- The final minimal prompt explicitly requested the two SELECTs. It completed
  correctly in three responses: two query calls, then nonempty answer content.
  **The terminal failure is deterministic in replay, but was not reproduced by
  the final live minimal task.** The original 32-job scheduling workload and provider
  variability may matter; these checks do not establish a provider root cause.
- The installed model profile warns that temperature is ignored with reasoning
  enabled. The script retains the experiment's `temperature=0` setting for parity.

Offline async execution stalled in the restricted sandbox; replay and control were
verified outside it. Ruff lint and formatting checks pass.

The separate [review draft recovery plan](../docs/review-draft-recovery-plan.md)
addresses the first finding from the experiment analysis. It does not change the
baseline checkpoint or scoring policy.
