# SwapCheck project rules

- Python 3.12, Pydantic v2, strict typing. Every change must keep `ruff check .`, `ruff format --check .`, `mypy swapcheck` and `pytest` green.
- All tests run offline with the scripted mock providers. Never require API keys or network access in tests.
- Never hardcode model names in source code; model IDs live only in YAML configs. API keys come only from environment variables.
- Never fabricate human labels, benchmark numbers or sample results. Anything illustrative must be clearly labeled as an example.
- Read the existing code before adding anything, and reuse existing models, the cache, the config module and the scoring code instead of duplicating them.
- Follow PLAN.md and POLICY.md. Scenario expectations are written by hand from POLICY.md; never compute them with the policy code.
- Policy and time logic must never read the wall clock; use the scenario's `current_time`.
- Do not weaken or delete a test to make it pass. If a test is wrong, explain why and fix it.
- Keep changes small and focused, and list the files you changed and the commands you ran.
