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

- Alpha: scaffold, providers, caching and demo booking agent are complete; scenarios, scoring, verdict engine and CLI are in progress
- Milestone checklist:
  - [x] 1
  - [x] 2
  - [x] 3
  - [x] 4

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
        Suite["Scenario Suite (~60 YAMLs in 5 Categories)"]
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
        WebDash["React / FastAPI Dashboard (Optional)"]
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

| Verdict | Statistical & Safety Criteria |
| :--- | :--- |
| **`SHIP`** 🟢 | • Candidate non-inferior overall: $\Delta_{\text{lower}} \ge -\epsilon_{\text{overall}}$ (default $\epsilon = 0.05$ / 5%).<br>• Zero new critical policy violations: $\text{Violations}_{\text{cand}} \le \text{Violations}_{\text{base}}$ and new violations $= 0$.<br>• Powered categories non-inferior: $\Delta_{c, \text{lower}} \ge -\epsilon_{\text{cat}}$ (looser category margin $\epsilon_{\text{cat}} = 0.10$).<br>• Sample size threshold met ($S \ge 50, R \ge 3$) and confidence interval is sufficiently narrow. |
| **`HOLD`** 🔴 | • Candidate definitively inferior: $\Delta_{\text{upper}} < -\epsilon_{\text{overall}}$, OR<br>• Any new critical policy violation detected ($\text{CriticalViolations}_{\text{new}} > 0$), OR<br>• Any powered category shows statistically significant degradation ($\Delta_{c, \text{upper}} < -\epsilon_{\text{cat}}$). |
| **`INCONCLUSIVE`** 🟡 | • The confidence interval spans the decision margin ($\Delta_{\text{lower}} < -\epsilon_{\text{overall}} \le \Delta_{\text{upper}}$).<br>• Insufficient scenario sample size ($S < S_{\text{min}}$).<br>• Automatically computes and reports the scenario sample size needed ($N_{\text{req}}$) to achieve statistical resolution. |

### 3. Required Sample Size Formula
When an evaluation is `INCONCLUSIVE`, SwapCheck computes the exact sample size needed for 80% power ($\beta = 0.20$) at $\alpha = 0.05$:
$$N_{\text{req}} = \left\lceil \frac{2 \cdot (z_{1 - \alpha/2} + z_{1 - \beta})^2 \cdot s_d^2}{\epsilon_{\text{overall}}^2} \right\rceil \approx \left\lceil \frac{15.68 \cdot s_d^2}{\epsilon_{\text{overall}}^2} \right\rceil$$
where $s_d^2$ is the sample variance of the paired scenario differences $d_s$.

---

## ⚡ Quickstart

### Installation

```bash
git clone https://github.com/abduljalilmemon/SwapCheck.git
cd SwapCheck
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 1. Run Offline Evaluation (No API Keys Required)

SwapCheck includes a `MockProvider` that runs offline test suites and simulated models hermetically:

```bash
swapcheck run \
  --baseline configs/baseline_mock.yaml \
  --candidate configs/candidate_mock.yaml \
  --suite suites/booking \
  --runs 3
```

### 2. Run Real Model Swap Evaluation

Configure your baseline, candidate, and judge YAML files in `configs/`:

```yaml
# configs/candidate.yaml
name: "candidate-budget"
model: "openai/gpt-4o-mini"
temperature: 0.0
api_key_env_var: "OPENAI_API_KEY"
cost_per_input_token: 0.00000015
cost_per_output_token: 0.00000060
```

Execute the comparison with cost ceiling protection and response caching:

```bash
swapcheck run \
  --baseline configs/baseline.yaml \
  --candidate configs/candidate.yaml \
  --judge configs/judge.yaml \
  --suite suites/booking \
  --runs 3 \
  --max-cost 5.00 \
  --output report.md
```

### 3. Trajectory Labeling & Judge Calibration

SwapCheck never fabricates fake human evaluations. You can easily label recorded trajectories interactively:

```bash
# Interactively label recorded trajectories from a run
swapcheck label --run <RUN_ID> --output human_labels.csv

# Calculate Cohen's Kappa agreement and confidence intervals
swapcheck calibrate --labels human_labels.csv --suite suites/booking
```

---

## 📊 Example Report Output

Illustrative mock-up, not real results

```markdown
# SwapCheck Evaluation Report: run_20261103_0145

## Verdict: SHIP 🟢
> Candidate is non-inferior within 5.0% margin with 0 new critical policy violations.
> Projected task cost reduced by **78.4%** with a **140ms (22%)** latency improvement.

### Summary Metrics

| Metric | Baseline | Candidate | Difference | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Pass Rate** | 92.5% | 94.0% | **+1.5%** | `[-2.1%, +5.2%]` |
| **Critical Policy Violations** | 0 | 0 | **0** | — |
| **Cost per Successful Task** | $0.0240 | $0.0052 | **-$0.0188 (-78.4%)** | — |
| **Mean Latency** | 640 ms | 500 ms | **-140 ms (-21.8%)** | — |
| **P95 Latency** | 1,220 ms | 890 ms | **-330 ms (-27.0%)** | — |

### Category Breakdown

| Category | Scenarios | Baseline | Candidate | Δ Pass Rate | 95% CI | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `simple_lookup` | 12 | 100.0% | 100.0% | 0.0% | `[0.0%, 0.0%]` | PASS 🟢 |
| `cancellation` | 12 | 94.4% | 97.2% | +2.8% | `[-4.2%, +9.7%]` | PASS 🟢 |
| `refund_policy_edge_cases` | 12 | 88.9% | 91.7% | +2.8% | `[-5.6%, +11.1%]` | PASS 🟢 |
| `adversarial_prompt_injection` | 12 | 86.1% | 88.9% | +2.8% | `[-6.1%, +11.7%]` | PASS 🟢 |
| `ambiguous_requests` | 12 | 93.1% | 92.2% | -0.9% | `[-7.8%, +6.0%]` | PASS 🟢 |
```

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
2. **Three-Way Verdicts vs Binary Pass/Fail**: A binary SHIP/HOLD decision forces teams to either reject acceptable models or accept underpowered test runs. `INCONCLUSIVE` makes statistical uncertainty transparent and provides the exact $N_{\text{req}}$ required to decide.
3. **Structured vs Conversational Policy Checks**: Deterministic checks evaluate structured parameters (tool arguments, database diffs, refund limits). The LLM judge evaluates subjective conversational behavior (e.g., verbal promises to the user that are not backed by tool execution).
4. **Offline First**: All unit tests, CI workflows, and harness internals run 100% hermetically using `MockProvider` and SQLite in-memory databases with zero API keys required.

---

## 🛣️ Roadmap & Known Limitations

- **Model IDs from Config Only**: No hardcoded models exist in SwapCheck source code.
- **Single-Agent Scope**: Currently optimized for tool-using single agents with orchestrator loops. Multi-agent swarms with peer delegation will be supported in future releases.
- **Judge Cost**: Running an LLM judge on all trajectories adds evaluation cost. SwapCheck caches judge completions and supports fast deterministic-only pre-filtering.

---

## 📄 License

MIT © SwapCheck Contributors.
