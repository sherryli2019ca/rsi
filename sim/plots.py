"""Figures and LaTeX tables for the paper from results/*.json."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")
F = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
T = os.path.join(os.path.dirname(__file__), "..", "paper", "tables")

STYLE = {
    "CARVE": ("#1b5e9e", "-", "o"),
    "CARVE-full-only": ("#6aa5d8", "--", "s"),
    "Uncertainty": ("#d9822b", "-", "^"),
    "Uncertainty-MF": ("#8e44ad", "-", "x"),
    "Replay-each": ("#7a7a7a", "-", "v"),
    "LLM-only": ("#b03a2e", ":", None),
    "CARVE-uniform-prior": ("#2e8b57", "-.", "d"),
}
LABEL = {"CARVE": "CARVE (ours)", "CARVE-full-only": "CARVE, full replay only",
         "Uncertainty": "Uncertainty sampling", "Uncertainty-MF": "Uncertainty + cheap checks",
         "Replay-each": "Replay-each (do-then-verify)",
         "LLM-only": "LLM attribution only", "CARVE-uniform-prior": "CARVE, flat prior"}

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 200})


def load(name):
    return json.load(open(os.path.join(R, f"{name}.json")))


def budget_curves():
    d = load("main")
    cps = d["checkpoints"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3))
    for m in ["LLM-only", "Replay-each", "Uncertainty", "Uncertainty-MF", "CARVE-full-only", "CARVE"]:
        c, ls, mk = STYLE[m]
        r = d["methods"][m]
        for ax, key in zip(axes, ["gain", "f1"]):
            mu, se = np.array(r[f"{key}_mean"]), np.array(r[f"{key}_se"])
            ax.plot(cps, mu, color=c, ls=ls, marker=mk, ms=3, lw=1.3, label=LABEL[m])
            ax.fill_between(cps, mu - se, mu + se, color=c, alpha=0.15, lw=0)
    axes[0].set_ylabel("Net repaired failures\n(fraction of oracle)")
    axes[1].set_ylabel("Edge F1")
    for ax in axes:
        ax.set_xlabel("Intervention budget (full-replay units)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(os.path.join(F, "budget_curves.pdf"))


def prior_sweep():
    d = load("prior")
    fig, ax = plt.subplots(figsize=(3.1, 2.2))
    for m in ["LLM-only", "Replay-each", "CARVE-uniform-prior", "CARVE"]:
        c, ls, mk = STYLE[m]
        mu = [x["gain_mean"][1] for x in d["methods"][m]]
        se = [x["gain_se"][1] for x in d["methods"][m]]
        ax.errorbar(d["r"], mu, yerr=se, color=c, ls=ls, marker=mk, ms=3, lw=1.3,
                    capsize=1.5, label=LABEL[m])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("LLM attribution accuracy $r$")
    ax.set_ylabel(f"Net gain at budget {d['budget']}\n(fraction of oracle)")
    ax.legend(fontsize=6.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.28))
    fig.set_size_inches(3.1, 2.8)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "prior_sweep.pdf"))


def fidelity():
    d = load("fidelity")
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.1), sharey=True)
    for m in ["CARVE-full-only", "CARVE"]:
        c, ls, mk = STYLE[m]
        mu = [x["gain_mean"][0] for x in d["cost"][m]]
        se = [x["gain_se"][0] for x in d["cost"][m]]
        axes[0].errorbar(d["cost_values"], mu, yerr=se, color=c, ls=ls, marker=mk, ms=3,
                         capsize=1.5, label=LABEL[m])
        mu = [x["gain_mean"][0] for x in d["quality"][m]]
        se = [x["gain_se"][0] for x in d["quality"][m]]
        gap = [s - f for s, f in d["quality_values"]]
        axes[1].errorbar(gap, mu, yerr=se, color=c, ls=ls, marker=mk, ms=3, capsize=1.5)
    axes[0].set_xlabel("Single-step cost / full-replay cost")
    axes[1].set_xlabel("Single-step check quality (sensitivity $-$ FPR)")
    axes[0].set_ylabel(f"Net gain at budget {d['budget']}")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fidelity.pdf"))


def rsi():
    d = load("rsi")
    fig, ax = plt.subplots(figsize=(3.1, 2.8))
    for m in ["LLM-only", "Replay-each", "Uncertainty", "Uncertainty-MF", "CARVE-full-only", "CARVE"]:
        c, ls, mk = STYLE[m]
        r = d["methods"][m]
        x = np.arange(len(r["success_mean"]))
        mu, se = np.array(r["success_mean"]), np.array(r["success_se"])
        ax.plot(x, mu, color=c, ls=ls, marker=mk, ms=3, lw=1.3, label=LABEL[m])
        ax.fill_between(x, mu - se, mu + se, color=c, alpha=0.15, lw=0)
    ax.axhline(d["oracle_success"], color="k", lw=0.7, ls="--", label="Oracle ceiling")
    ax.set_xlabel("RSI round (budget 40 per round)")
    ax.set_ylabel("Agent success rate")
    ax.legend(fontsize=6.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(os.path.join(F, "rsi.pdf"))


def fmt(m, s):
    return f"{m:.3f}\\,{{\\scriptsize$\\pm${s:.3f}}}"


def tables():
    d = load("main")
    cps = d["checkpoints"]
    cols = [20, 50, 100, 200]
    idx = [cps.index(c) for c in cols]
    rows = [("LLM attribution only", "LLM-only"), ("Replay-each (do-then-verify)", "Replay-each"),
            ("Replay-each, 3 patches", "Replay-each-3patch"),
            ("Sequential-each (adaptive stopping)", "Sequential-each"),
            ("Thompson sampling", "Thompson"),
            ("Uncertainty sampling", "Uncertainty"),
            ("Uncertainty + cheap checks", "Uncertainty-MF"), (None, None),
            ("CARVE (all components)", "CARVE"), ("\\quad full replay only", "CARVE-full-only"),
            ("\\quad myopic ($n{=}1$ look-ahead)", "CARVE-myopic"),
            ("\\quad uninformed prior", "CARVE-uniform-prior"),
            ("\\quad one patch per cell", "CARVE-one-patch"),
            ("\\quad no label-noise term", "CARVE-no-label-noise")]
    lines = []
    for name, key in rows:
        if key is None:
            lines.append("\\midrule")
            continue
        r = d["methods"][key]
        lines.append(name + " & " + " & ".join(fmt(r["gain_mean"][i], r["gain_se"][i])
                                              for i in idx) + " \\\\")
    open(os.path.join(T, "main.tex"), "w").write("\n".join(lines) + "\n")

    m = load("misspec")
    lines = [f"{k} & {fmt(v['gain_mean'][0], v['gain_se'][0])} \\\\" for k, v in m["cases"].items()]
    open(os.path.join(T, "misspec.tex"), "w").write("\n".join(lines) + "\n")

    s = load("scale")
    cps = s["checkpoints"]
    lines = []
    for key in ["LLM-only", "Replay-each", "Uncertainty", "Uncertainty-MF", "CARVE-full-only", "CARVE"]:
        r = s["methods"][key]
        lines.append(LABEL[key] + " & " + " & ".join(
            fmt(r["gain_mean"][cps.index(c)], r["gain_se"][cps.index(c)]) for c in [50, 100, 300])
            + " \\\\")
    open(os.path.join(T, "scale.tex"), "w").write("\n".join(lines) + "\n")

    j = load("joint")
    cps = j["checkpoints"]
    lines = []
    for key, name in [("LLM-only", "LLM attribution only"), ("Replay-each", "Replay-each"),
                      ("Thompson", "Thompson sampling"), ("Uncertainty", "Uncertainty sampling"),
                      ("Uncertainty-MF", "Uncertainty + cheap checks"), ("CARVE", "CARVE (ours)")]:
        r = j["methods"][key]
        lines.append(name + " & " + " & ".join(
            fmt(r["gain_mean"][cps.index(c)], r["gain_se"][cps.index(c)]) for c in [20, 50, 100, 200])
            + " \\\\")
    open(os.path.join(T, "joint.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    os.makedirs(F, exist_ok=True)
    os.makedirs(T, exist_ok=True)
    budget_curves()
    prior_sweep()
    fidelity()
    rsi()
    tables()


def stress_tables():
    st, fo, p1 = load("stress"), load("fooled"), load("prop1")
    g = lambda r: fmt(r["gain_mean"][0], r["gain_se"][0])
    rows = [
        ("Default testbed", st["cases"]["default"]["Uncertainty-MF"], st["cases"]["default"]["CARVE"]),
        ("Label accuracy $\\lambda=0.5$", st["cases"]["labels lam=0.5"]["Uncertainty-MF"],
         st["cases"]["labels lam=0.5"]["CARVE"]),
        ("Analyst overconfident ($\\hat r{+}0.3$)",
         st["cases"]["analyst overconfident (r_hat+0.3)"]["Uncertainty-MF"],
         st["cases"]["analyst overconfident (r_hat+0.3)"]["CARVE"]),
        ("Analyst underconfident ($\\hat r{-}0.3$)",
         st["cases"]["analyst underconfident (r_hat-0.3)"]["Uncertainty-MF"],
         st["cases"]["analyst underconfident (r_hat-0.3)"]["CARVE"]),
        ("Judge fooled by bad patches", fo["cases"]["judge fooled"]["Uncertainty-MF"],
         fo["cases"]["judge fooled"]["CARVE"]),
    ]
    for c in ["0.005", "0.05", "0.1"]:
        rows.append((f"Regression cost $c_{{\\mathrm{{fp}}}}={c}$", st["c_fp"][c]["Uncertainty-MF"],
                     st["c_fp"][c]["CARVE"]))
    lines = [f"{n} & {g(a)} & {g(b)} \\\\" for n, a, b in rows]
    open(os.path.join(T, "stress.tex"), "w").write("\n".join(lines) + "\n")
    lines = []
    for r in p1["rows"]:
        lines.append(f"${r['ell']:+.0f}$ & {r['e']} & {r['pred']:.2f} & {r['sim']:.2f} & {r['wrong']:.3f} \\\\")
    open(os.path.join(T, "prop1.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    stress_tables()
