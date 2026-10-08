"""Who&When (Zhang et al., ICML 2025): the judge methods on its public
multi-agent failure logs, to align our judges with published numbers.

Data: the Who&When folder of github.com/mingyin1/Agents_Failure_Attribution
(MIT licence): Hand-Crafted (58 logs) and Algorithm-Generated (126 logs), each
with the responsible agent and step labelled by people. Prompts: the authors'
all-at-once, step-by-step and binary-search prompts (Automated_FA/Lib/utils.py),
verbatim, in the setting with the problem's ground-truth answer; the agent key
is "role" for Hand-Crafted and "name" for Algorithm-Generated, as there.
Answers are parsed as there (agent and step lines; "1. yes" / "1. no";
"upper half" / "lower half", a random half otherwise, seeded per log).

  python -m attrib.whowhen run --data <Who&When dir> --out <dir> \
      --methods all_at_once:flash,all_at_once:pro,step_by_step:pro,binary_search:pro
  python -m attrib.whowhen score --data <Who&When dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attrib.methods import MODELS  # noqa: E402

SUBSETS = {"Hand-Crafted": "role", "Algorithm-Generated": "name"}


def _gen(system: str, prompt: str, method: str, model: str) -> str:
    from rrsi import llm
    mdl, think = MODELS[model]
    role = f"ww_{method}:{model}"
    if think:
        llm.THINKING_ROLES.add(role)
    try:
        return llm.generate(prompt, system=system, model=mdl, role=role,
                            max_tokens=24000 if think else 2000, max_retries=3 if think else 6)
    except RuntimeError:
        return ""


def all_at_once(d: dict, key: str, model: str) -> dict:
    chat = "\n".join(f"{e.get(key, 'Unknown Agent')}: {e.get('content', '')}" for e in d["history"])
    prompt = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
        f"The problem is:  {d.get('question', '')}\n"
        f"The Answer for the problem is: {d.get('ground_truth', '')}\n"
        "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation:\n\n" + chat +
        "\n\nBased on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
        """
            {
                "agent a": "xx",
                "agent b": "xxxx",
                "agent c": "xxxxx",
                "agent a": "xxxxxxx"
            },
            """
        "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction."
        "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n")
    out = _gen("You are a helpful assistant skilled in analyzing conversations.", prompt, "all_at_once", model)
    agent = re.search(r"Agent Name:\s*\**\s*([\w_]+)", out, re.IGNORECASE)
    step = re.search(r"Step Number:\s*\**\s*(\d+)", out, re.IGNORECASE)
    return {"agent": agent.group(1).strip() if agent else None,
            "step": int(step.group(1)) if step else None, "calls": 1, "raw": out[:2000]}


def step_by_step(d: dict, key: str, model: str) -> dict:
    hist = ""
    for idx, e in enumerate(d["history"]):
        name = e.get(key, "Unknown Agent")
        hist += f"Step {idx} - {name}: {e.get('content', '')}\n"
        prompt = (
            f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {d.get('question', '')}. "
            f"The Answer for the problem is: {d.get('ground_truth', '')}\n"
            f"Here is the conversation history up to the current step:\n{hist}\n"
            f"The most recent step ({idx}) was by '{name}'.\n"
            "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
            "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
            "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
            "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]")
        ans = _gen("You are a precise step-by-step conversation evaluator.", prompt, "step_by_step", model)
        if not ans or ans.lower().strip().startswith("1. yes"):
            return {"agent": name, "step": idx, "calls": idx + 1, "api_error": not ans}
    return {"agent": None, "step": None, "calls": len(d["history"])}


def binary_search(d: dict, key: str, model: str, seed: str) -> dict:
    hist, rng = d["history"], random.Random(seed)
    start, end, calls = 0, len(hist) - 1, 0
    while start < end:
        mid = start + (end - start) // 2
        chat = "\n".join(f"{e.get(key, 'Unknown Agent')}: {e.get('content', '')}" for e in hist[start:end + 1])
        prompt = (
            "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
            "Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user’s query.\n"
            f"The problem to address is as follows: {d.get('question', '')}\n"
            f"The Answer for the problem is: {d.get('ground_truth', '')}\n"
            f"Review the following conversation segment from step {start} to step {end}:\n\n{chat}\n\n"
            f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half (from step {start} to step {mid}) or the lower half (from step {mid + 1} to step {end}) of this segment.\n"
            "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'. Remember, your answer should be based on identifying the mistake that directly contributes to the failure in resolving the user's query. If no single clear error is evident, consider the step you believe is most responsible for the failure, allowing for subjective judgment, and base your answer on that.")
        r = _gen("You are an AI assistant specializing in localizing errors in conversation segments.",
                 prompt, "binary_search", model).lower()
        calls += 1
        if "upper half" in r:
            end = mid
        elif "lower half" in r:
            start = min(mid + 1, end)
        elif rng.randint(0, 1) == 0:
            end = mid
        else:
            start = min(mid + 1, end)
    return {"agent": hist[start].get(key, "Unknown Agent"), "step": start, "calls": calls}


def logs(data: Path):
    for sub, key in SUBSETS.items():
        for p in sorted((data / sub).glob("*.json"), key=lambda p: (len(p.stem), p.stem)):
            yield sub, key, p


def run(data: Path, out: Path, specs: list[str], workers: int) -> None:
    jobs = []
    for spec in specs:
        name, _, model = spec.partition(":")
        for sub, key, p in logs(data):
            o = out / spec.replace(":", "_") / sub / p.name
            if not o.exists():
                jobs.append((o, name, model, sub, key, p))

    def one(job):
        o, name, model, sub, key, p = job
        d = json.loads(p.read_text())
        if name == "all_at_once":
            res = all_at_once(d, key, model)
        elif name == "step_by_step":
            res = step_by_step(d, key, model)
        else:
            res = binary_search(d, key, model, f"{sub}/{p.name}")
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    with ThreadPoolExecutor(workers) as ex:
        for _ in ex.map(one, jobs):
            pass


def score(data: Path, out: Path) -> dict:
    rows = {}
    for mdir in sorted(p for p in out.iterdir() if p.is_dir()):
        for sub, key, p in logs(data):
            o = mdir / sub / p.name
            if not o.exists():
                continue
            d, r = json.loads(p.read_text()), json.loads(o.read_text())
            row = rows.setdefault(f"{mdir.name}/{sub}", {"n": 0, "agent": 0, "step": 0})
            row["n"] += 1
            row["agent"] += str(r.get("agent") or "").strip().lower() == str(d["mistake_agent"]).strip().lower()
            row["step"] += r.get("step") is not None and int(r["step"]) == int(d["mistake_step"])
    for row in rows.values():
        row["agent_acc"], row["step_acc"] = round(row["agent"] / row["n"], 3), round(row["step"] / row["n"], 3)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "score"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="all_at_once:pro")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "run":
        os.environ.setdefault("RRSI_USAGE_LOG", str(out / "usage.jsonl"))
        run(Path(args.data), out, args.methods.split(","), args.workers)
    res = score(Path(args.data), out)
    (out / "scores.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
