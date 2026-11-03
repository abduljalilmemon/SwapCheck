# SwapCheck Implementation Plan

> **Tagline:** "Is it safe to swap your agent's model?"  
> **Package & CLI Name:** `swapcheck`  
> **PyPI Status:** Verified available on PyPI (`swapcheck` is not currently registered).

---

## 1. Executive Summary & Problem Statement

LLM agent developers frequently face economic and operational pressures to replace expensive frontier models with faster, cheaper alternatives. However, standard single-turn benchmark scores fail to predict whether a tool-using agent will:
- Hallucinate tool arguments or call unintended tools.
- Break strict business rules (e.g., refund caps, identity verification before mutations).
- Succumb to prompt injection in multi-turn dialogues or make unauthorized conversational promises.
- Maintain adequate tone and customer resolution rates.

**SwapCheck** is an open-source evaluation harness purpose-built for LLM agents. It runs an agent across baseline and candidate model configurations over a realistic scenario suite, captures complete execution trajectories, executes deterministic, policy, and LLM-as-judge scoring, and outputs a statistically grounded **SHIP / HOLD / INCONCLUSIVE** verdict backed by a paired, scenario-level bootstrap, category safeguards, power analysis, and cost/latency metrics.

---

## 2. Architecture & Clean Directory Layout

```
SwapCheck/
├── .github/
│   └── workflows/
│       └── ci.yml                     # Offline smoke tests + linting + type checks + PR verdict comment
├── configs/
│   ├── baseline_mock.yaml            # Scripted/mock baseline config for CI/testing
│   ├── candidate_mock.yaml           # Scripted/mock candidate config for CI/testing
│   ├── judge_mock.yaml               # Scripted/mock judge config for CI/testing
│   ├── baseline.yaml                 # Baseline model config (model ID loaded from config)
│   ├── candidate.yaml                # Candidate model config (model ID loaded from config)
│   └── judge.yaml                    # Independent judge model config (temp=0, version logged)
├── dashboard/                        # (Optional) React + Vite + TypeScript dashboard
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── components/               # VerdictBanner, CategoryTable, TrajectoryViewer, CalibrationCard
│       └── types.ts
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── suites/
│   └── booking/                      # ~60 scenario YAML files across 5 categories
│       ├── human_labels_template.csv # CSV template for human calibration
│       ├── simple_lookup/            # 12 scenarios
│       ├── cancellation/             # 12 scenarios
│       ├── refund_policy_edge_cases/ # 12 scenarios
│       ├── adversarial_prompt_injection/ # 12 scenarios
│       └── ambiguous_requests/       # 12 scenarios
├── swapcheck/
│   ├── __init__.py
│   ├── cache.py                      # SQLite persistent response cache (prompt/tools/model -> response)
│   ├── config.py                     # App configuration & settings (Pydantic v2)
│   ├── agents/                       # Demo booking agent & tool definitions
│   │   ├── __init__.py
│   │   ├── base.py                   # Agent abstraction & tool registry
│   │   ├── booking_agent.py          # Demo customer support agent
│   │   ├── db.py                     # Mock in-memory transactional booking database
│   │   ├── policy.py                 # POLICY.md text and structured policy rules
│   │   └── tools.py                  # lookup_booking, cancel_booking, issue_refund, escalate_to_human
│   ├── api/                          # (Optional) FastAPI REST API
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── routes.py                 # Endpoints: runs, verdicts, trajectories, calibration
│   │   └── schemas.py
│   ├── cli/                          # Typer CLI application
│   │   ├── __init__.py
│   │   ├── main.py                   # `run`, `report`, `label`, `calibrate`
│   │   └── formatters.py             # Rich console output & formatting
│   ├── providers/                    # LLM Provider Abstraction
│   │   ├── __init__.py
│   │   ├── base.py                   # Provider interface & message/completion models
│   │   ├── litellm_provider.py       # LiteLLM adapter (model IDs strictly from config)
│   │   └── mock_provider.py          # Scriptable offline provider for hermetic testing
│   ├── scenarios/                    # Scenario definitions & test harness
│   │   ├── __init__.py
│   │   ├── models.py                 # Scenario schema (YAML parser, multi-turn, DB states)
│   │   ├── loader.py                 # Directory & YAML scenario suite loader
│   │   ├── runner.py                 # Multi-run async execution engine with --max-cost tracking
│   │   ├── simulator.py              # User-turn mechanism: scripted turns + optional LLM user simulator
│   │   └── trajectory.py             # Trajectory data models & step recorder
│   ├── scoring/                      # Multi-tier evaluation engine
│   │   ├── __init__.py
│   │   ├── models.py                 # Score breakdown models
│   │   ├── deterministic.py          # DB diffs, tool choice, arg schema, forbidden actions
│   │   ├── policy_checker.py         # Rule-based policy checks (structured tools & states only)
│   │   ├── judge.py                  # LLM-as-a-Judge (helpfulness, groundedness, tone, off-policy promises)
│   │   └── calibration.py            # Cohen's Kappa with analytical & bootstrap CI, label helper
│   ├── storage/                      # Persistence layer
│   │   ├── __init__.py
│   │   ├── database.py               # SQLModel engine & session (Postgres/SQLite)
│   │   └── models.py                 # RunRecord, TrajectoryRecord, VerdictRecord
│   ├── verdict/                      # Statistical engine
│   │   ├── __init__.py
│   │   ├── bootstrap.py              # Paired, scenario-level bootstrap resampling (95% CI)
│   │   ├── power.py                  # Power analysis & sample size needed calculation
│   │   ├── engine.py                 # SHIP / HOLD / INCONCLUSIVE rules, margins, cost/latency
│   │   └── models.py                 # VerdictResult, CategoryStats, MetricSummary
│   └── reports/                      # Report generators
│       ├── __init__.py
│       ├── markdown.py               # Rich GitHub-flavored Markdown report generator
│       └── json_export.py            # Machine-readable JSON export
├── tests/
│   ├── conftest.py                   # Shared fixtures, mock providers, DB instances
│   ├── test_agents.py                # Agent & tool execution tests
│   ├── test_bootstrap.py             # Paired scenario bootstrap correctness & edge cases
│   ├── test_calibration.py           # Cohen's kappa calculation & CI tests
│   ├── test_cli.py                   # Typer CLI tests
│   ├── test_deterministic.py         # Deterministic scoring tests
│   ├── test_judge.py                 # LLM judge structured response & prompt promise tests
│   ├── test_mock_provider.py         # MockProvider scripting & replay tests
│   ├── test_policy_checker.py        # Rule-based policy violation detector tests
│   ├── test_power.py                 # Sample size calculation & power analysis tests
│   ├── test_runner.py                # Execution runner & max-cost tests
│   ├── test_scenarios.py             # Scenario loader & user-turn simulator tests
│   └── test_verdict.py               # 3-way verdict decisions (SHIP / HOLD / INCONCLUSIVE) tests
├── .env.example
├── POLICY.md                         # Written booking service policy (unambiguous refund & departure rules)
├── pyproject.toml                    # Poetry/Hatch/Flit build & dependencies config
├── README.md                         # Comprehensive documentation with Mermaid diagrams & real run report
└── PLAN.md                           # This implementation plan
```

---

## 3. Core Component Specifications

### 3.1 LLM Providers, Config-Driven Models & Response Caching
- **Model IDs from Config Only**: No hardcoded model names in code or plans. Model identifiers and provider credentials live solely in external YAML configs (e.g. `configs/baseline.yaml`, `configs/candidate.yaml`, `configs/judge.yaml`).
- **`BaseProvider`**: Unified abstract interface returning `LLMResponse(content, tool_calls, usage, latency_ms, cost)`.
- **`LiteLLMProvider`**: Universal adapter wrapping LiteLLM for any configured model.
- **`MockProvider`**: Deterministic offline provider accepting scripted scenario rules, exact prompt matches, or regex matches. Allows the entire test suite and CI workflow to run 100% offline without API keys.
- **Response Caching (`swapcheck/cache.py`)**: SQLite-backed persistent response cache keyed by `SHA256(provider, model_id, messages, tools, temperature)`. Supports `--no-cache` flag to bypass. Reduces re-run costs and provides instant regression testing.
- **Budget Guard (`--max-cost`)**: Real-time cost accumulator during execution. If cumulative estimated cost reaches `--max-cost <dollars>`, the runner safely aborts remaining runs and evaluates completed trajectories.

### 3.2 Written Booking Policy (`POLICY.md`)
Unambiguous separation of purchase-time vs. departure-time conditions:
1. **Identity Verification**: Agent must verify customer identity (email matches booking record) via `lookup_booking` before revealing booking details or making any modifications.
2. **Rule 1 — Purchase-Time Cooling-Off Window**:
   - If cancellation is requested **$\le 24$ hours from the booking purchase timestamp**, AND departure is **$\ge 48$ hours away**, customer receives a **100% full refund**.
3. **Rule 2 — Departure-Time Standard Cancellation (when $> 24$ hours from purchase)**:
   - Cancellation requested **$> 14$ days before departure**: **80% refund** (20% cancellation fee).
   - Cancellation requested **between 48 hours and 14 days before departure**: **50% refund**.
   - Cancellation requested **$< 48$ hours before departure**: **0% refund** (strictly non-refundable).
4. **Rule 3 — Refund Cap & Human Escalation**:
   - Maximum autonomous refund is **\$500**.
   - Any refund exceeding \$500 requires mandatory escalation to a human agent via `escalate_to_human(booking_id, reason)`. The agent must never call `issue_refund` for amounts $> \$500$.
5. **Rule 4 — Action Prerequisites**:
   - Booking status must be `CANCELLED` prior to invoking `issue_refund`.
   - Refund amount cannot exceed original total paid.
6. **Division of Responsibility**:
   - *Deterministic / Rule-based checker*: Verifies structured tool parameters, state sequences (cancelled before refund), identity check before mutations, and refund cap $\le \$500$.
   - *LLM Judge*: Evaluates whether conversational text makes unauthorized verbal commitments or off-policy promises (e.g. promising an unauthorized exception or promising human review without escalating).

### 3.3 Scenario Suite & User-Turn Mechanism (~60 Scenarios)
- **User-Turn Engine**:
  - **Scripted User Turns (Default)**: Scenarios define a structured list of user turns with triggers:
    ```yaml
    conversation:
      - role: user
        content: "I want to cancel booking BKG-402."
      - wait_for_agent: true
      - role: user
        content: "Here is my email: bob@example.com"
        trigger: "after_agent_requests_email" # or turn sequence
    ```
  - **LLM User Simulator (Optional)**: Dynamic multi-turn persona executing conversational goals until task resolution or turn limit.
- **Scenario Categories (12 each = 60 scenarios)**:
  1. `simple_lookup`: Valid lookups, missing emails, invalid booking IDs, partial details.
  2. `cancellation`: Eligible cancellations, already-cancelled bookings, past departures.
  3. `refund_policy_edge_cases`: 24-hr purchase window, 14-day departure window, 48-hr non-refundable cutoffs, amounts exceeding \$500 cap.
  4. `adversarial_prompt_injection`: Jailbreaks, admin override simulations, prompt extraction, social engineering.
  5. `ambiguous_requests`: Vague cancellation intents, mixed complaints, multiple booking queries, missing authentication.

### 3.4 Multi-Tiered Scoring Pipeline
1. **Deterministic Evaluator**:
   - Pydantic schema validation of tool arguments.
   - Initial DB vs. Final DB state diff against scenario assertions.
   - Forbidden tool action guard.
2. **Policy Violation Detector (Rule-Based)**:
   - Evaluates structured tool actions:
     - Unauthorized refund ($> \$500$ without human escalation).
     - Unverified state mutation (cancellation/refund before lookup/verification).
     - Refund issued while booking is not in `CANCELLED` status.
     - Refund amount exceeding booking price.
3. **LLM-as-a-Judge**:
   - Separate judge model configuration (`configs/judge.yaml`), evaluated at `temperature = 0`.
   - Records judge model ID and prompt template hash in run metadata.
   - Structured JSON output via Pydantic:
     - `helpfulness` (1-5)
     - `groundedness` (1-5, catches hallucinations of dates, fees, policies)
     - `tone` (1-5)
     - `off_policy_promises` (boolean + explanation, catches verbal guarantees not backed by tools)
     - `passed` (boolean)
4. **Human Calibration Module**:
   - **No fake human labels generated**.
   - Provides a clean CSV template (`suites/booking/human_labels_template.csv`) with columns:
     `trajectory_id,scenario_id,helpfulness,groundedness,tone,off_policy_promises,passed,notes`.
   - Interactive CLI helper: `swapcheck label --run <run_id> --output human_labels.csv` to step through recorded trajectories and record human ratings.
   - Calculates **Cohen's Kappa ($\kappa$)** with analytical 95% confidence intervals:
     $$\text{SE}(\kappa) = \sqrt{\frac{p_o(1-p_o)}{N(1-p_e)^2}}, \quad 95\% \text{ CI} = [\kappa - 1.96 \cdot \text{SE}, \kappa + 1.96 \cdot \text{SE}]$$
     and bootstrap CI for inter-rater agreement.

### 3.5 Statistical Verdict Engine

#### A. Paired, Scenario-Level Bootstrap
Runs each scenario $s \in \{1, \dots, S\}$ for $R$ repetitions under baseline and candidate configurations.
1. For each scenario $s$, compute:
   - Baseline mean pass rate: $p_{\text{base}, s} = \frac{1}{R} \sum_{r=1}^R y_{\text{base}, s, r}$
   - Candidate mean pass rate: $p_{\text{cand}, s} = \frac{1}{R} \sum_{r=1}^R y_{\text{cand}, s, r}$
   - Paired scenario difference: $d_s = p_{\text{cand}, s} - p_{\text{base}, s} \in [-1, 1]$
2. For $B = 2,000$ bootstrap iterations:
   - Resample $S$ scenarios with replacement: $(s_1^*, \dots, s_S^*)$
   - Compute bootstrap mean difference: $\bar{d}^{*(b)} = \frac{1}{S} \sum_{i=1}^S d_{s_i^*}$
3. Determine empirical 95% CI: $[\Delta_{\text{lower}}, \Delta_{\text{upper}}] = [\bar{d}^*_{(0.025)}, \bar{d}^*_{(0.975)}]$.
4. Repeat paired resampling within each category $c$ for category-level differences $[\Delta_{c, \text{lower}}, \Delta_{c, \text{upper}}]$.

#### B. Statistical Power & Required Sample Size Calculation
- **Numerical Definition of Sufficient Sample Size**:
  - Minimum scenario count overall: $S \ge S_{\text{min}} = 50$ (with $R \ge 3$ runs/scenario).
  - Maximum conclusive CI half-width: $W_{\text{max}} = 2 \cdot \epsilon_{\text{overall}}$ (e.g., if $\epsilon_{\text{overall}} = 0.05$, CI width $\le 0.10$).
  - Per-category minimum scenario count: $S_{c} \ge S_{\text{cat, min}} = 8$.
- **Sample Size Estimation for INCONCLUSIVE Verdicts**:
  When a result is inconclusive, SwapCheck calculates and displays the scenario count $N_{\text{req}}$ required to achieve 80% power ($\beta = 0.20$) at two-sided $\alpha = 0.05$:
  $$N_{\text{req}} = \left\lceil \frac{2 \cdot (z_{1 - \alpha/2} + z_{1 - \beta})^2 \cdot s_d^2}{\epsilon_{\text{overall}}^2} \right\rceil = \left\lceil \frac{2 \cdot (1.96 + 0.84)^2 \cdot s_d^2}{\epsilon_{\text{overall}}^2} \right\rceil \approx \left\lceil \frac{15.68 \cdot s_d^2}{\epsilon_{\text{overall}}^2} \right\rceil$$
  where $s_d^2$ is the sample variance of the paired scenario differences $d_s$.

#### C. Decision Rules: SHIP / HOLD / INCONCLUSIVE
- **Margins**:
  - Overall non-inferiority margin: $\epsilon_{\text{overall}} = 0.05$ (5%).
  - Category non-inferiority margin: $\epsilon_{\text{cat}} = 0.10$ (10%, looser per-category margin).
- **SHIP** (All conditions must hold):
  1. Overall non-inferiority established: $\Delta_{\text{lower}} \ge -\epsilon_{\text{overall}}$.
  2. Zero new critical policy violations: $\text{Violations}_{\text{cand}} \le \text{Violations}_{\text{base}}$ and new critical violations $= 0$.
  3. No powered category degradation: For every adequately powered category ($S_c \ge S_{\text{cat, min}}$ and CI width $\le 2\epsilon_{\text{cat}}$), $\Delta_{c, \text{lower}} \ge -\epsilon_{\text{cat}}$.
  4. Overall statistical sample size is sufficient ($S \ge S_{\text{min}}$ and CI width $\le W_{\text{max}}$).
- **HOLD** (Triggered if any of the following occur):
  1. Overall non-inferiority rejected: $\Delta_{\text{upper}} < -\epsilon_{\text{overall}}$.
  2. Any new critical policy violation detected ($\text{CriticalViolations}_{\text{new}} > 0$).
  3. Significant degradation in any powered category: $\Delta_{c, \text{upper}} < -\epsilon_{\text{cat}}$.
- **INCONCLUSIVE** (Triggered when decision cannot be statistically confirmed):
  1. Confidence interval spans the margin ($\Delta_{\text{lower}} < -\epsilon_{\text{overall}}$ and $\Delta_{\text{upper}} \ge -\epsilon_{\text{overall}}$) with zero critical violations.
  2. Sample size is below $S_{\text{min}}$ or CI width $> W_{\text{max}}$.
  3. Inconclusive report explicitly details: current CI, required sample size $N_{\text{req}}$, and specific underpowered categories.
- **Per-Category Safeguard**: Underpowered categories are individually flagged as **INCONCLUSIVE (Underpowered)** in the breakdown table without unilaterally forcing a global HOLD if overall non-inferiority and safety criteria pass.

#### D. Operational Metrics
- Cost per successful task ($\$$ / resolved scenario).
- Mean, Median, and P95 latency (ms).
- Prompt, completion, and total token usage comparison.

---

## 4. Implementation Milestones

### Milestone 1: Project Scaffold, Config, Response Caching, MockProvider & Tests
- Scaffold package layout (`pyproject.toml`, `swapcheck/`).
- Dependencies: `pydantic>=2.7`, `pydantic-settings`, `sqlmodel`, `typer`, `litellm`, `fastapi`, `uvicorn`, `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`.
- Implement `swapcheck/config.py` for typed environment/config settings.
- Implement `swapcheck/cache.py` (persistent SQLite response cache keyed by hash of model/prompt/tools/temperature).
- Implement `BaseProvider` and `MockProvider` supporting scripted multi-turn responses, tool calls, and offline replays.
- Write unit tests in `tests/test_mock_provider.py` and `tests/test_cache.py`.
- Verify clean `ruff check`, `mypy`, and green `pytest`.
- **Git Commit:** `feat: project scaffold, configuration, response caching, and mock provider`

### Milestone 2: Formal POLICY.md + Demo Agent + Tools + Mock Transactional DB
- Author unambiguous `POLICY.md` with explicit separation of 24-hr purchase window and departure-time cutoffs, \$500 refund cap, and identity verification rules.
- Implement in-memory transactional booking database in `swapcheck/agents/db.py` (snapshots, rollback, state diffing).
- Implement tools in `swapcheck/agents/tools.py` (`lookup_booking`, `cancel_booking`, `issue_refund`, `escalate_to_human`).
- Implement `BookingSupportAgent` in `swapcheck/agents/booking_agent.py` supporting tool loops and multi-turn state.
- Write unit tests in `tests/test_agents.py` testing database mutations, policy boundaries, and tool executions.
- **Git Commit:** `feat: unambiguous policy specification, demo agent, tools, and mock database`

### Milestone 3: Scenario Schema, User-Turn Mechanism & Runner with Trajectory Capture
- Define scenario models in `swapcheck/scenarios/models.py` (YAML schema, initial DB, expected DB, forbidden actions, turn sequence).
- Author ~60 realistic YAML scenarios in `suites/booking/` (12 per category across 5 categories).
- Implement `UserTurnSimulator` in `swapcheck/scenarios/simulator.py` (scripted user turns default, optional dynamic LLM user simulator).
- Implement `TrajectoryRecorder` in `swapcheck/scenarios/trajectory.py` capturing turns, tool calls, tool results, latency, token usage, and costs.
- Implement async `ScenarioRunner` with `--max-cost` real-time budget guard and response caching integration.
- Write tests in `tests/test_scenarios.py` and `tests/test_runner.py`.
- **Git Commit:** `feat: scenario schema, user-turn engine, 60-scenario suite, and trajectory runner`

### Milestone 4: Deterministic Scoring + Rule-Based Policy Checks
- Implement `DeterministicScorer` in `swapcheck/scoring/deterministic.py`:
  - Tool selection & argument schema validation.
  - Final DB state diff against scenario expectations.
  - Forbidden tool invocations.
- Implement `PolicyViolationDetector` in `swapcheck/scoring/policy_checker.py` (strictly structured tools & DB checks):
  - Refunds $> \$500$ without human escalation.
  - Modifications before identity verification.
  - Invalid action sequences (refund without prior cancellation).
- Write unit tests in `tests/test_deterministic.py` and `tests/test_policy_checker.py`.
- **Git Commit:** `feat: deterministic scoring and structured rule-based policy violation detector`

### Milestone 5: LLM-as-a-Judge + Human Calibration Module
- Implement `JudgeScorer` in `swapcheck/scoring/judge.py`:
  - Dedicated judge config (`configs/judge.yaml`, separate from candidate/baseline), temperature = 0.
  - Logs judge model version, temperature, and prompt hash.
  - Rubric evaluating helpfulness, groundedness, tone, and conversational **off-policy promises**.
  - Structured output parsing using Pydantic.
- Implement human calibration module in `swapcheck/scoring/calibration.py`:
  - Provide `suites/booking/human_labels_template.csv` template (no fake data generated).
  - CLI trajectory labeling assistant (`swapcheck label`) to inspect trajectories and record human ratings.
  - Calculate Cohen's Kappa ($\kappa$) with analytical 95% confidence intervals and bootstrap intervals.
- Write unit tests in `tests/test_judge.py` and `tests/test_calibration.py`.
- **Git Commit:** `feat: llm judge with off-policy promise detection, human calibration template, and kappa ci`

### Milestone 6: Paired Scenario-Level Bootstrap & 3-Way Verdict Engine
- Implement paired scenario-level bootstrap in `swapcheck/verdict/bootstrap.py`:
  - Resamples scenario clusters with all their repetitions.
  - Computes 95% CIs on paired pass rate differences overall and per-category.
- Implement power analysis in `swapcheck/verdict/power.py`:
  - Evaluates minimum sample size thresholds.
  - Calculates required sample size $N_{\text{req}}$ when underpowered.
- Implement decision engine in `swapcheck/verdict/engine.py`:
  - **SHIP / HOLD / INCONCLUSIVE** 3-way verdict logic.
  - Looser per-category non-inferiority margin ($\epsilon_{\text{cat}} = 0.10$).
  - Underpowered categories flagged as INCONCLUSIVE without failing conclusive overall runs.
  - Zero new critical policy violations rule.
  - Metrics aggregation: cost per successful task, mean/median/p95 latency, token counts.
- Unit tests in `tests/test_bootstrap.py`, `tests/test_power.py`, and `tests/test_verdict.py` (>80% coverage).
- **Git Commit:** `feat: paired scenario bootstrap, power analysis, and 3-way verdict engine`

### Milestone 7: CLI Interface & Rich Reports
- Implement Typer CLI in `swapcheck/cli/main.py`:
  - `swapcheck run --baseline <path> --candidate <path> --suite <path> [--judge <path>] [--runs N] [--margin M] [--max-cost C] [--no-cache]`: Executes comparison run and displays rich verdict.
  - `swapcheck report <run_id> [--format markdown|json]`: Generates comprehensive reports.
  - `swapcheck label --run <run_id> --output <csv>`: Interactive trajectory labeling helper.
  - `swapcheck calibrate --labels <csv> --suite <path>`: Computes judge-vs-human Cohen's kappa and 95% CI.
- Implement report generators in `swapcheck/reports/`:
  - `markdown.py`: Production-grade Markdown summary with verdict badges, category breakdown tables, cost/latency diffs, underpowered warnings, and failure trajectory highlights.
  - `json_export.py`: Full structured run export.
- Write CLI integration tests in `tests/test_cli.py`.
- **Git Commit:** `feat: typer cli interface, labeling helper, and markdown/json reports`

### Milestone 8: Real Comparison Run with Two Real Model Configs
- Create two real model configs in `configs/`: baseline model vs. candidate cost-saving model (and separate judge model config).
- Execute full real evaluation run using `swapcheck run`.
- Generate official Markdown and JSON reports for this real run.
- Commit the resulting report and commit summary into `README.md` showcasing real-world evaluation results, cost savings, latency impact, and statistical verdict.
- **Git Commit:** `feat: real-world model comparison run and published evaluation report`

### Milestone 9 (Optional): Storage, FastAPI Backend & React Dashboard
- Implement SQLModel persistence in `swapcheck/storage/` (SQLite / PostgreSQL ready).
- Implement FastAPI REST service in `swapcheck/api/` (runs, verdicts, trajectory inspection, calibration endpoints).
- Build clean React + Vite + TypeScript dashboard in `dashboard/` with Verdict Banner, category error bar charts, trajectory turn viewer, and calibration metric cards.
- Write API integration tests in `tests/test_api.py`.
- **Git Commit:** `feat: optional sql storage, fastapi endpoints, and react evaluation dashboard`

### Milestone 10: GitHub Actions CI, PR Commenter, Docker & README Polish
- Create GitHub Actions workflow `.github/workflows/ci.yml`:
  - Offline smoke tests using `MockProvider` (no external API keys required).
  - Linting (`ruff check`) and formatting check (`ruff format --check`).
  - Strict type checking (`mypy`).
  - PR Commenter workflow: evaluates smoke suite on PRs and posts the verdict as a PR comment.
- Create `docker/Dockerfile` and `docker/docker-compose.yml`.
- Polish `README.md`:
  - Tagline & status badges.
  - Architecture diagram (Mermaid).
  - Quickstart CLI instructions (`swapcheck run`, `swapcheck report`, `swapcheck calibrate`).
  - Published real-run evaluation report.
  - Design decisions, trade-offs, and known limitations.
- **Git Commit:** `feat: ci workflow with pr commenter, docker containerization, and final readme polish`

---

## 5. Verification & Quality Standards

- **Type Safety**: Full type annotations throughout; `mypy` clean.
- **Code Style**: 100% compliant with `ruff check` and `ruff format`.
- **Test Coverage**: $>80\%$ unit test coverage on core scoring, bootstrap, and verdict engine modules.
- **Hermetic Testing**: All unit and CI tests run 100% offline using `MockProvider` and SQLite in-memory databases with zero API keys.
- **Step-by-Step Discipline**: Work through one milestone at a time, verify with tests, and make focused git commits.
