# CARVE: budgeted causal verification of LLM failure attributions

Code and paper for *Which Fix Should We Test? Budgeted Causal Verification of LLM
Failure Attributions for Self-Improving Agents* (ACL submission draft, `paper/`).

An analyst LLM attributes agent failures to categories and to modifiable
components; CARVE decides which (category, component, patch) attributions to
verify by intervention, and whether to use a cheap single-step check or a full
replay, by maximising expected decision value per unit cost.

## Layout

| Path | What it is |
|---|---|
| `carve/model.py` | Bayesian model over the category -> component graph (edge vs. patch quality, label noise, two fidelities) |
| `carve/policies.py` | CARVE selection rule and baselines (LLM-only, Replay-each, Uncertainty, Thompson) |
| `sim/` | Synthetic testbed with known ground truth, experiment driver, figures and tables |
| `results/` | JSON results behind every number in the paper |
| `agent_exp/` | Real-LLM study: ShopDesk environment, componentised agent, fault injection, analyst prompts, pipeline |
| `paper/` | LaTeX source (ACL style), figures, tables, compiled `main.pdf` |
| `tests/` | Unit tests for the model and a mock-LLM end-to-end test of the real pipeline |

## Reproduce the synthetic results

```bash
pip install -r requirements.txt
python -m sim.experiments all     # writes results/*.json (about 30 min on 4 cores)
python -m sim.plots               # writes paper/figures and paper/tables
cd paper && latexmk -pdf main.tex
```

## Run the real-agent study

Needs `ANTHROPIC_API_KEY`. Models default to `claude-opus-5-5`; override with
`AGENT_MODEL` and `ANALYST_MODEL`. Server-side refusal fallbacks are deliberately
off so that every call in an experiment is served by the model being studied.

```bash
python -m agent_exp.pipeline --out runs/r1 --stage taxonomy          # collect, attribute, induce taxonomy
for m in LLM-only Replay-each Uncertainty CARVE-full-only CARVE; do
  python -m agent_exp.pipeline --out runs/r1 --method $m --budget 60   # verify + held-out evaluation
done
```

Stages are cached under `--out`, so all methods share the same traces,
attributions and taxonomy.

## Status

Synthetic experiments: complete. Real-agent experiments: implemented and tested
with a mock model, not yet run (the paper's Section 7 is marked TODO).

## Reproduce the tau2 tables (review revision)

Needs tau2-bench installed (for task splits and component names); no API calls.
The banks and decisions are in `runs/tau2_{retail,airline}/`.

```bash
for d in tau2_retail tau2_airline; do
  python -m agent_exp.reanalysis --out runs/$d --domain $d --stage decide   # policies on the bank (cached)
  python -m agent_exp.reanalysis --out runs/$d --domain $d --stage report   # all scorings -> reanalysis.json
done
python -m agent_exp.heldout_breakdown --out runs/tau2_retail                # held-out by episode type
python -m agent_exp.heldout_breakdown --out runs/tau2_airline --domain tau2_airline
python -m agent_exp.review2_report --out runs/tau2_retail                  # step-patch removal, leave-one-out, clean episodes
python -m agent_exp.report_tau2                                             # paper/tables/*.tex
cd paper && latexmk -pdf main.tex
```

Second-review runs (API calls, about 3,800 episodes): `runs/review2_runs.sh` runs the task-matched
unpatched baseline (`agent_exp.review2_runs --stage baseline`, both domains), the airline held-out sets
(`agent_exp.bank_heldout`), and the retail step-patch removal, leave-one-out and clean-episode runs
(`--stage nostep|loo|clean`).

Re-collecting a bank: `python -m agent_exp.bank --out runs/tau2_retail --domain tau2_retail --stage bank`
(prompts in `agent_exp/analyst.py`, environment and faults in `agent_exp/tau2_env.py`).
