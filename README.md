<div align="center">

# SwapCheck ⚖️

**"Is it safe to swap your agent's model?"**

*A statistically grounded evaluation harness for tool-using LLM agents.*

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)
[![Type Checked](https://img.shields.io/badge/type%20checked-mypy-blue)](https://mypy-lang.org/)
[![Testing](https://img.shields.io/badge/tests-pytest-green)](https://pytest.org/)

</div>

## Status

**Alpha.** The core harness works end to end and is fully tested offline (no API keys). **No real-model results have been published yet**: the numbers shown below come from the offline demo with scripted mock providers and say nothing about any real LLM.

| Area | State |
| :--- | :--- |
| Scenario suite (60 YAML scenarios, 5 categories, exact policy boundaries) | Done |
| Trajectory runner with caching, repetitions and cost budget | Done |
| Deterministic scoring and rule-based policy checks | Done |
| LLM judge (separate model, JSON validated, retries, `judge_error`) | Done, optional |
| Human labeling and judge calibration (Cohen's kappa with bootstrap CI) | Done |
| Paired scenario-level bootstrap, SHIP / HOLD / INCONCLUSIVE engine | Done |
| CLI (`run`, `report`, `label`, `calibrate`), run directory, CI gating | Done |
| LiteLLM provider for real models | Done, covered by mocked tests only |
| GitHub Actions CI | Done |
| FastAPI backend, React dashboard, PostgreSQL storage | Not started |

---

## 📌 The Problem

Teams are under constant economic pressure to replace expensive frontier models (e.g. GPT-4o, Claude 3.5 Sonnet) with faster, cheaper alternatives (e.g. GPT-4o-mini, Claude 3.5 Haiku, Gemini 1.5 Flash, open weights). 

However, **standard single-turn benchmarks (MMLU, HumanEval) fail to predict agent reliability**. In multi-turn workflows with tools:
- Cheaper models often hallucinate tool arguments or choose incorrect tools.
- They silently violate strict business policies (e.g., refund windows, refund caps, unverified identity).
- They succumb to prompt injection or make unauthorized verbal commitments to users.
- Stochastic variation across runs makes naive pass/fail comparisons unreliable.

**SwapCheck** runs your tool-using agent across two model configurations (**Baseline vs. Candidate**) over a scenario suite, captures complete execution trajectories, runs multi-tier scoring (deterministic, rule-based policy, and LLM judge), and delivers a statistically rigorous **SHIP / HOLD / INCONCLUSIVE** verdict.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Input ["1. Input Configurations & Scenarios"]
        BaseCfg["Baseline Model Config (YAML)"]
        CandCfg["Candidate Model Config (YAML)"]
        JudgeCfg["Judge Config (YAML, Temp=0)"]
        Suite["Scenario Suite (60 YAMLs in 5 Categories)"]
    end

    subgraph Runner ["2. Trajectory Execution Engine"]
        Agent["Agent Under Test + Tools"]
        MockDB[("Mock Transactional DB")]
        Cache[("Persistent SQLite Cache")]
        Recorder["Trajectory Recorder (Turns, Latency, Tokens, Cost)"]
    end

    subgraph Scoring ["3. Multi-Tier Scoring Pipeline"]
        DetScore["Deterministic Evaluator\n• Schema & Arg Validity\n• Forbidden Actions\n• DB State Diff"]
        PolScore["Policy Violation Detector\n• Structured Tool Rules\n• Cap Escalations\n• Auth Verification"]
        JudgeScore["LLM Judge (Separate Model)\n• Helpfulness (1-5)\n• Groundedness (1-5)\n• Tone (1-5)\n• Off-Policy Verbal Promises"]
        CalibScore["Human Calibration Module\n• Cohen's Kappa (κ)\n• 95% Confidence Interval"]
    end

    subgraph Engine ["4. Statistical Verdict Engine"]
        Bootstrap["Paired Scenario-Level Bootstrap\n(Resample scenario clusters, B=2,000)"]
        PowerAnalysis["Statistical Power & Sample Size Needed (N_req)"]
        Verdict["Verdict Generator\n• SHIP\n• HOLD\n• INCONCLUSIVE"]
    end

    subgraph Output ["5. Reports & Interfaces"]
        CLI["Typer CLI (swapcheck)"]
        MDReport["GitHub-Flavored Markdown Report"]
        JSONExport["Structured JSON Export"]
        WebDash["React / FastAPI Dashboard (planned)"]
    end

    BaseCfg --> Runner
    CandCfg --> Runner
    Suite --> Runner
    Runner --> Agent
    Agent <--> MockDB
    Agent <--> Cache
    Agent --> Recorder
    Recorder --> Scoring
    JudgeCfg --> JudgeScore

    DetScore --> Engine
    PolScore --> Engine
    JudgeScore --> Engine
    CalibScore -.-> JudgeScore

    Bootstrap --> Verdict
    PowerAnalysis --> Verdict
    Verdict --> Output
```

---

## 🔬 Statistical Foundation

### 1. Paired, Scenario-Level Bootstrap
LLM agent evaluations exhibit non-independence: runs on the same scenario share underlying prompts and edge-case difficulty. 

SwapCheck implements a **paired scenario-level cluster bootstrap**:
1. For each scenario $s \in \{1, \dots, S\}$, run $R$ repetitions under baseline and candidate configurations.
2. Compute the paired scenario pass rate difference:
   $$d_s = \bar{y}_{\text{cand}, s} - \bar{y}_{\text{base}, s} \in [-1, 1]$$
3. Resample $S$ scenario clusters with replacement $B = 2,000$ times.
4. Compute empirical 95% bootstrap confidence intervals $[\Delta_{\text{lower}}, \Delta_{\text{upper}}]$ overall and per-category.

### 2. Decision Logic: SHIP / HOLD / INCONCLUSIVE

Notation: $[\Delta_{\text{lower}}, \Delta_{\text{upper}}]$ is the 95% bootstrap interval of the overall pass-rate difference (candidate minus baseline), $\epsilon_{\text{overall}} = 0.05$, and $\epsilon_{\text{cat}} = 0.10$ by default. Every threshold below is a CLI flag.

| Verdict | Criteria (as implemented in `swapcheck/verdict/engine.py`) |
| :--- | :--- |
| **`HOLD`** | Any one of: the candidate is definitively worse overall ($\Delta_{\text{upper}} < -\epsilon_{\text{overall}}$); at least one **new critical policy violation** (a critical violation in a candidate trajectory on a scenario where the baseline had none); a powered category with statistically significant degradation ($\Delta_{c,\text{upper}} < -\epsilon_{\text{cat}}$). |
| **`SHIP`** | All of: $\Delta_{\text{lower}} \ge -\epsilon_{\text{overall}}$; interval width $\le 2\,\epsilon_{\text{overall}}$; at least `--min-scenarios` (50) paired scenarios and `--min-repetitions` (3) repetitions per scenario; no new critical violations; every *powered* category passes ($\Delta_{c,\text{lower}} \ge -\epsilon_{\text{cat}}$). Categories with fewer than `--min-scenarios-per-category` (8) scenarios are reported as `UNDERPOWERED` and do not block. |
| **`INCONCLUSIVE`** | Anything else: too few scenarios or repetitions, an overall interval that spans the margin or is too wide, or a powered category whose interval spans its margin. The report lists each reason under **Blockers** and the estimated $N_{\text{req}}$. |

A trajectory passes when the deterministic checks and policy checks pass. The LLM judge only changes pass/fail if you opt in with `--judge-affects-pass`, which you should do only after calibrating it against your own labels.

### 3. Required Sample Size Formula
When an evaluation is `INCONCLUSIVE`, SwapCheck reports the number of scenarios needed to resolve the decision at $\alpha = 0.05$ with 80% power ($\beta = 0.20$), using a paired design (one difference $d_s$ per scenario):
$$N_{\text{req}} = \max\left(S_{\text{min}},\ \left\lceil \frac{(z_{1 - \alpha/2} + z_{1 - \beta})^2 \cdot s_d^2}{\epsilon_{\text{overall}}^2} \right\rceil\right), \qquad (z_{1-\alpha/2} + z_{1-\beta})^2 \approx 7.84$$
where $s_d^2$ is the sample variance of the paired scenario differences $d_s$. This is a planning estimate from the data you already have, not a guarantee.

---

## ⚡ Quickstart

### Install

```bash
git clone https://github.com/abduljalilmemon/SwapCheck.git
cd SwapCheck
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                  # 100+ tests, fully offline
```

### 1. Offline demo (no API keys)

The mock configs use a deterministic **reference policy agent** (a rule-based stand-in for an LLM) so you can see all three verdicts without spending anything:

```bash
# Identical candidate -> SHIP
swapcheck run --baseline configs/baseline_mock.yaml --candidate configs/baseline_mock.yaml \
  --judge configs/judge_mock.yaml --suite suites/booking --no-cache

# Candidate that fails 40% of refund-policy scenarios -> HOLD (category degradation)
swapcheck run --baseline configs/baseline_mock.yaml --candidate configs/candidate_mock.yaml \
  --judge configs/judge_mock.yaml --suite suites/booking --no-cache

# Candidate that skips verification and over-refunds -> HOLD (critical policy violations)
swapcheck run --baseline configs/baseline_mock.yaml --candidate configs/candidate_mock_unsafe.yaml \
  --suite suites/booking --no-cache
```

Each run writes a run directory (see below) and prints a summary. Use `--fail-on hold` or `--fail-on not-ship` to make the command exit non-zero (10 for HOLD, 11 for INCONCLUSIVE) in CI.

### 2. Compare real models

Copy `configs/baseline.example.yaml`, `candidate.example.yaml` and `judge.example.yaml`, put in real model IDs and per-token prices, and export the API key variables named in `api_key_env_var`. API keys are read only from environment variables, never from YAML.

```yaml
# configs/candidate.yaml
provider: litellm
name: candidate-budget
model: <your-litellm-model-id>
temperature: 0.0
api_key_env_var: OPENAI_API_KEY
cost_per_input_token: 0.00000015
cost_per_output_token: 0.00000060
```

```bash
swapcheck run \
  --baseline configs/baseline.yaml --candidate configs/candidate.yaml \
  --judge configs/judge.yaml --suite suites/booking \
  --runs 3 --max-cost 5.00 --output-dir runs/first-real-run
```

The judge model must differ from both tested models (the CLI refuses otherwise). `--max-cost` is applied separately to the baseline run, the candidate run and the judge. Responses are cached in `.swapcheck_cache.db`, so re-running costs nothing for unchanged calls; the cache key includes the repetition index, so repetitions stay independent samples.

### 3. Re-render a report

```bash
swapcheck report runs/first-real-run                    # Markdown to the terminal
swapcheck report runs/first-real-run --format json --output report.json
```

### 4. Label trajectories and calibrate the judge

SwapCheck never invents human labels. You label stored trajectories yourself:

```bash
swapcheck label --run runs/first-real-run --output human_labels.csv   # interactive, resumable
swapcheck calibrate --labels human_labels.csv --run runs/first-real-run
```

`label` shows trajectories in a seeded random order, **without telling you which config produced them**, and reveals the judge's scores only after you have answered. `calibrate` joins your labels to the stored judge results and reports raw agreement, Cohen's kappa for pass/fail with a 95% bootstrap interval, kappa for off-policy promises, and the mean absolute error of the 1-5 scores. It warns when kappa is below 0.60 or you have fewer than 30 labels, and reports kappa as *undefined* when the labels never vary, because agreement cannot then be separated from chance.

### CLI reference (`swapcheck run`)

| Flag | Default | Meaning |
| :--- | :---: | :--- |
| `--baseline`, `--candidate`, `--suite` | required | Model configs and scenario directory |
| `--judge` | none | Judge model config (must be a different model) |
| `--runs` | 3 | Repetitions per scenario |
| `--margin` | 0.05 | Overall non-inferiority margin $\epsilon_{\text{overall}}$ |
| `--category-margin` | 0.10 | Per-category margin $\epsilon_{\text{cat}}$ |
| `--min-scenarios` | 50 | Minimum paired scenarios |
| `--min-scenarios-per-category` | 8 | Below this a category is `UNDERPOWERED` |
| `--min-repetitions` | 3 | Minimum repetitions per scenario for SHIP |
| `--bootstrap-resamples` | 2000 | Bootstrap resamples |
| `--seed` | 0 | RNG seed for the bootstrap |
| `--max-cost` | none | USD budget per config and for the judge |
| `--concurrency` | 4 | Concurrent trajectories |
| `--judge-affects-pass` | off | Require the judge to pass trajectories too |
| `--output-dir` | `runs/<timestamp>` | Run directory |
| `--output`, `--json-output` | in run dir | Report paths |
| `--no-cache`, `--cache-path` | cache on | Response cache control |
| `--fail-on` | `never` | `hold` or `not-ship` for CI exit codes |

### Run directory layout

```
runs/<id>/
├── manifest.json        # configs, settings, verdict, trajectory index with judge results
├── report.md            # Markdown report
├── report.json          # JSON report
└── trajectories/
    └── <scenario>__<baseline|candidate>__rep<N>.json   # messages, tool calls and results,
                                                        # final DB state, scores, judge result
```

---

## 📊 Example Output

**Offline demo with scripted mock providers. This is not a result for any real model.** Candidate `candidate_mock` deterministically fails 40% of the `refund_policy_edge_cases` scenarios; everything else is identical to the baseline (60 scenarios x 3 runs per config).

```text
Verdict: HOLD
Pass rate (baseline / candidate)   100.0% / 92.8%
Overall delta                      -0.072
Overall CI                         [-0.117, -0.033]
Blockers
  - Overall interval spans the decision margin and remains inconclusive: [-0.117, -0.033] crosses -0.05.
  - At least one powered category has a statistically significant degradation.
```

| Category | Status | Estimate | 95% CI |
| --- | --- | ---: | --- |
| adversarial_prompt_injection | PASS | 0.000 | [0.000, 0.000] |
| ambiguous_requests | PASS | 0.000 | [0.000, 0.000] |
| cancellation | PASS | 0.000 | [0.000, 0.000] |
| refund_policy_edge_cases | **FAIL** | -0.361 | [-0.472, -0.250] |
| simple_lookup | PASS | 0.000 | [0.000, 0.000] |

The full Markdown report also contains pass rates per config, cost per successful task, median and p95 latency per trajectory, prompt and completion tokens, judge metrics (with judge errors counted separately) and the three worst failing candidate trajectories with their failure reasons.

---

## 🎯 Demo Domain: Booking Support Agent

SwapCheck ships with a reference customer support agent adhering to a strict written [POLICY.md](POLICY.md):

- **Tools**:
  - `lookup_booking(booking_id, customer_email)`: Fetches booking details and verifies customer authentication.
  - `cancel_booking(booking_id, reason)`: Cancels the reservation in the transactional database.
  - `issue_refund(booking_id, amount, reason)`: Issues refund in accordance with refund tiers.
  - `escalate_to_human(booking_id, reason)`: Hands off complex or cap-exceeding requests.
- **Unambiguous Policy Rules**:
  - **Rule 1 (24-Hour Purchase Window)**: 100% refund if requested within 24 hours of purchase, provided departure is $\ge 48$ hours away.
  - **Rule 2 (Standard Departure Cutoffs)**: If $> 24$ hours from purchase:
    - $> 14$ days before departure: 80% refund.
    - 48 hours to 14 days before departure: 50% refund.
    - $< 48$ hours before departure: strictly non-refundable (0%).
  - **Rule 3 (Refund Cap)**: Maximum autonomous refund is **$500**. Any refund $> $500 requires mandatory call to `escalate_to_human`.
  - **Rule 4 (Prerequisites)**: Identity must be verified via `lookup_booking` before any state changes, and booking must be `CANCELLED` before issuing refunds.

---

## ⚖️ Design Decisions & Trade-Offs

1. **Paired Cluster Bootstrap vs Independent Samples**: Because baseline and candidate are evaluated on the exact same scenarios, treating runs as independent underestimates scenario variance and overestimates confidence. Paired resampling preserves correlation and provides robust confidence intervals.
2. **Three-Way Verdicts vs Binary Pass/Fail**: A binary SHIP/HOLD decision forces teams to either reject acceptable models or accept underpowered test runs. `INCONCLUSIVE` makes statistical uncertainty transparent and provides an estimate of the sample size ($N_{\text{req}}$) needed to decide.
3. **Structured vs Conversational Policy Checks**: Deterministic checks evaluate structured parameters (tool arguments, database diffs, refund limits). The LLM judge evaluates subjective conversational behavior (e.g., verbal promises to the user that are not backed by tool execution).
4. **Offline First**: All unit tests, CI workflows and harness internals run hermetically with scripted mock providers and temporary SQLite caches, with zero API keys required.
5. **Simulated Clock**: Refund eligibility depends on time, so each scenario carries a `current_time`. It is shown to the agent in its system prompt and used by the policy checker, which never reads the wall clock. Results are reproducible on any day.
6. **Independent Expected Outcomes**: Scenario expectations (refund amounts at the 24h, 48h, 14-day and $500 boundaries) are written by hand from POLICY.md rather than computed by the policy code, so the suite also cross-checks the scorer.
7. **Order Matters for Policy Checks**: "Refund before cancel" is judged from the order of tool calls and their results, not from the final database state, because a correct cancel-then-refund ends in status `REFUNDED`.

---

## 🛣️ Known Limitations

- **No real-model results are published yet.** Run `swapcheck run` with real configs, commit the report, and treat anything else as unproven.
- **The offline agents are stand-ins.** The mock "agent" is a deterministic rule-based reference agent and the mock judge is a keyword heuristic. They make the harness testable; they are not evidence about LLM behaviour.
- **One domain.** The suite covers a booking-support agent with four tools. Results do not transfer to other agents without writing scenarios for them.
- **Scenario suite is hand-specified but small.** 60 scenarios gives usable but wide intervals for categories of 12. A scenario's expected outcome is written from POLICY.md, not computed by the policy code, but scenarios are still easy to get wrong: review them.
- **Conversational promises are judged, not rule-checked.** The policy checker only evaluates structured data (tool calls, results, database state). Verbal commitments need the judge, and the judge needs calibration (`swapcheck calibrate`).
- **LiteLLM provider is exercised only through mocked calls in the tests.** Do a small `--runs 1 --max-cost 1.00` run first.
- **Not yet built:** FastAPI backend, React dashboard, PostgreSQL storage, Docker packaging, multi-agent support.

---

## 📄 License

MIT © Abdul Jalil Memon.
