"""Admissibility of the oracle's corrections (post hoc, third review of paper 2).

The oracle sees the grading criteria and is told to use them only to know the
right outcome, never to supply ids, values or answers the agent had not yet
learned, and to correct only the step at hand (attrib.groundtruth.ORACLE_SYSTEM,
attrib.aw.ORACLE_SYSTEM). This script checks both instructions mechanically for
every correction that entered a registered rescue profile (gt/a: verdict
mistake, applied in both corrected replays, null replays present), re-scores
the methods with flagged corrections counted as not proposed, and draws a
stratified sample for a human audit.

  python -m attrib.admissibility /home/user/attrib_runs/main [--out <json>] [--audit <dir>]

information  every data value the correction uses must appear in what the agent
             had seen before step k, or in the policy, tool schemas or API docs:
             tau2: the opening message, earlier assistant turns, tool results and
             user replies; AppWorld: the instruction, the supervisor's profile,
             earlier code and outputs.
             tau2 tool calls: every argument value (strings; numbers above 10;
             free-text summaries skipped, numbers inside calculate expressions);
             tau2 messages: ids, emails, dates, flight numbers and amounts in the
             text; AppWorld: constants passed to apis.* calls: numbers above 10
             or any number passed as answer=; whole strings for arguments that
             look a value up (emails, passwords, ids, phone numbers, answers,
             app and API names) or are positional; dates for date arguments;
             emails and 10-digit numbers inside content the agent writes
             (titles, messages, file paths). A date counts as seen if the
             text names its month and day (or m/d), or names the month for the
             first or last day of a month; in airline, today and tomorrow
             also count when the conversation says so.
             "unseen id" = a non-numeric value not found; "unseen number" = a
             number not found (often a sum or difference the agent could compute,
             so this flag over-counts).
scope        API or tool calls in the correction against the observed step:
             "expanded" = at least two calls more than max(observed, 1);
             "strict" = more calls than max(observed, 1).

Re-scoring: R'_k = (1/K) sum_i c_ki g_ki with c_ki = 0 for flagged corrections;
rescuable = max R' >= 0.5; methods scored as in attrib.second_oracle.
"""
from __future__ import annotations

import argparse
import ast
import csv
import importlib
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

from attrib.analyze import _mean, load
from attrib.robustness import METHODS, kendall
from attrib.second_oracle import _ranking

DOMAINS = ("tau2_retail", "tau2_airline", "appworld")
K = 4
EXTRA = ("all_at_once_gain_pro", "binary_search_gain_pro")
FILTERS = ("unseen_id", "unseen_any", "expanded", "strict", "unseen_id_or_expanded")

_ID_PATTERNS = (r"#w\d{7}", r"\b\d{10}\b", r"[\w.+-]+@[\w-]+\.[\w.]+", r"\b[a-z]+_[a-z]+_\d{3,5}\b",
                r"\b(?:credit_card|gift_card|paypal|certificate)_\d+\b", r"\bhat\d{3}\b",
                r"\b\d{4}-\d{2}-\d{2}\b")
_LOOKUP = re.compile(r"email|password|username|_id$|^id$|phone|token|answer|account|user$|recipient|receiver|"
                     r"app_name|api_name")
_RES_ID = re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}\b")   # airline reservation ids
_AMOUNT = re.compile(r"\$?\d[\d,]*\.\d{1,2}\b|\$\d[\d,]*")


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _num_forms(x) -> set:
    f = float(x)
    out = {str(x).lower()}
    if f == int(f):
        out |= {str(int(f)), f"{f:.1f}", f"{f:.2f}"}
    else:
        out |= {f"{f:.2f}", f"{f:g}", f"{f:,.2f}"}
    return out


_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december")


def _date_seen(t: str, text: str) -> bool:
    """A yyyy-mm-dd value counts as seen if the text has it literally, as m/d,
    or as month and day ('may 24', '24th of may'); the first or last day of a
    month also counts if the month is named (range queries such as 'in
    February')."""
    y, m, d = (int(x) for x in t.split("-"))
    if t in text or not 1 <= m <= 12:
        return t in text
    mn = _MONTHS[m - 1]
    pats = [rf"\b0?{m}/0?{d}\b", rf"\b(?:{mn}|{mn[:3]})\.? 0?{d}(?:st|nd|rd|th)?\b",
            rf"\b0?{d}(?:st|nd|rd|th)? (?:of )?(?:{mn}|{mn[:3]})\b"]
    if any(re.search(p, text) for p in pats):
        return True
    cur = re.search(r"current time is (\d{4}-\d{2}-\d{2})", text)       # tau2 airline policy
    if cur:
        import datetime as dt
        delta = (dt.date(y, m, d) - dt.date.fromisoformat(cur.group(1))).days
        if (delta == 0 and "today" in text) or (delta == 1 and "tomorrow" in text):
            return True
    import calendar
    return d in (1, calendar.monthrange(y, m)[1]) and re.search(rf"\b(?:{mn}|{mn[:3]})\b", text) is not None


def _seen(s: str, prefix: str, static: str = "") -> bool:
    if s in prefix or (static and s in static):
        return True
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", s)) and _date_seen(s, prefix + " " + static)


def _has_num(x, text: str) -> bool:
    return any(re.search(r"(?<![\d.])" + re.escape(v) + r"(?![\d])", text) for v in _num_forms(x))


# ------------------------------------------------------------ visible text --
def tau2_static(d: str) -> str:
    os.environ["TAU2_DOMAIN"] = d.split("_", 1)[1]
    import agent_exp.tau2_env as m
    m = importlib.reload(m)
    return _norm(m.system_prompt(m.BASE_COMPONENTS) + "\n" + json.dumps(m.tool_defs(m.BASE_COMPONENTS)))


def aw_static() -> str:
    from attrib import aw
    docs = Path(aw.AW_ROOT) / "data" / "api_docs" / "standard"
    return _norm(aw.context() + "\n" + "\n".join(p.read_text() for p in sorted(docs.glob("*.json"))))


def tau2_prefix(rec: dict, k: int) -> str:
    parts = [rec.get("opening") or ""]
    for st in rec["steps"][:k]:
        for b in st["assistant"]:
            parts.append(b["text"] if b["type"] == "text" else json.dumps(b.get("input"), ensure_ascii=False))
        parts += [str(r.get("content")) for r in st.get("tool_results") or []]
        if st.get("user") is not None:
            parts.append(str(st["user"]))
    return _norm("\n".join(parts))


def aw_prefix(rec: dict, k: int) -> str:
    parts = [rec.get("instruction") or "", json.dumps(rec.get("supervisor"), ensure_ascii=False)]
    for st in rec["steps"][:k]:
        parts += [st.get("assistant") or "", st.get("output") or ""]
    return _norm("\n".join(parts))


# ------------------------------------------------------------------ checks --
def _leaves(x, key=""):
    if isinstance(x, dict):
        for kk, v in x.items():
            yield from _leaves(v, kk)
    elif isinstance(x, list):
        for v in x:
            yield from _leaves(v, key)
    else:
        yield key, x


def _text_tokens(text: str) -> tuple[list, list]:
    ids = [t for p in _ID_PATTERNS for t in re.findall(p, text.lower())]
    ids += [t.lower() for t in _RES_ID.findall(text)]
    amounts = [t.replace("$", "").replace(",", "") for t in _AMOUNT.findall(text)]
    return ids, [a for a in amounts if a]


def check_tau2(action: dict, prefix: str, static: str) -> dict:
    unseen_id, unseen_num = [], []
    if "message" in action:
        ids, nums = _text_tokens(str(action["message"]))
        unseen_id += [t for t in ids if not _seen(t, prefix, static)]
        unseen_num += [n for n in nums if not _has_num(n, prefix)]
        return {"unseen_id": unseen_id, "unseen_num": unseen_num, "calls": 0}
    calls = action.get("tool_calls") or []
    for c in calls:
        for key, v in _leaves(c.get("arguments") or {}):
            if key == "summary" or isinstance(v, bool) or v is None:
                continue
            if key == "expression":
                unseen_num += [n for n in re.findall(r"\d+(?:\.\d+)?", str(v))
                               if float(n) > 10 and not _has_num(n, prefix)]
                continue
            if isinstance(v, (int, float)):
                if abs(v) > 10 and not _has_num(v, prefix):
                    unseen_num.append(str(v))
                continue
            s = _norm(v)
            if not s:
                continue
            if len(s) > 40 and s.count(" ") >= 5:      # free text written by the agent
                ids, _ = _text_tokens(str(v))
                unseen_id += [t for t in ids if not _seen(t, prefix, static)]
                continue
            if re.fullmatch(r"-?\d+(?:\.\d+)?", s) and len(s) < 6:
                if float(s) > 10 and not _has_num(s, prefix):
                    unseen_num.append(s)
                continue
            if not _seen(s, prefix, static):
                unseen_id.append(s)
    return {"unseen_id": unseen_id, "unseen_num": unseen_num, "calls": len(calls)}


def _is_apis_call(node) -> bool:
    f = node.func
    while isinstance(f, ast.Attribute):
        f = f.value
    return isinstance(f, ast.Name) and f.id == "apis"


def check_aw(code: str, prefix: str, static: str) -> dict:
    from attrib.aw import _CALL
    unseen_id, unseen_num = [], []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"unseen_id": [], "unseen_num": [], "calls": len(_CALL.findall(code)), "unparsable": True}
    consts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_apis_call(node):
            consts += [("", a.value) for a in node.args if isinstance(a, ast.Constant)]
            consts += [(kw.arg or "", kw.value.value) for kw in node.keywords if isinstance(kw.value, ast.Constant)]
    for key, v in consts:
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            if (abs(v) > 10 or key == "answer") and not _has_num(v, prefix):
                unseen_num.append(str(v))
            continue
        s = _norm(v)
        if not s:
            continue
        if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
            if (float(s) > 10 or key == "answer") and not _has_num(s, prefix):
                unseen_num.append(s)
            continue
        if _LOOKUP.search(key) or not key:
            if not _seen(s, prefix, static):          # a value looked up, not written
                unseen_id.append(s)
        elif re.search(r"date|time", key):
            unseen_id += [t for t in re.findall(r"\d{4}-\d{2}-\d{2}", s) if not _seen(t, prefix, static)]
        else:                                          # content the agent writes: ids inside it
            ids = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+|\b\d{10}\b", s)
            unseen_id += [t for t in ids if not _seen(t, prefix, static)]
    return {"unseen_id": unseen_id, "unseen_num": unseen_num, "calls": len(_CALL.findall(code)),
            "lines": sum(bool(x.strip()) for x in code.splitlines())}


def _obs_calls(d: str, st: dict) -> tuple[int, int]:
    if d == "appworld":
        from attrib.aw import _CALL
        code = st.get("code") or ""
        return len(_CALL.findall(code)), sum(bool(x.strip()) for x in code.splitlines())
    return sum(b["type"] == "tool_use" for b in st["assistant"]), 0


# ----------------------------------------------------------------- collect --
def collect(main: Path, d: str, v: dict) -> list[dict]:
    """One row per effective correction of the registered profile."""
    from attrib.report_paper import aw_kind, tau2_kind
    aw = d == "appworld"
    static = aw_static() if aw else tau2_static(d)
    kind = aw_kind if aw else tau2_kind
    odir = main / d / "gt" / "a" / "oracle"
    rows = []
    for fid, g in v["gt"].items():
        rec = json.loads(Path(v["fails"][fid]["trace"]).read_text())
        for st in g["steps"]:
            if st["null"] is None:
                continue
            k = st["k"]
            obs = rec["steps"][k]
            prefix = aw_prefix(rec, k) if aw else tau2_prefix(rec, k)
            oc, ol = _obs_calls(d, obs)
            for i, s in enumerate(st["samples"]):
                if not (s.get("verdict") == "mistake" and s.get("corrected") is not None and s.get("n") == 2):
                    continue
                o = json.loads((odir / f"{fid}_k{k}_o{i}.json").read_text())
                act = o.get("action") or {}
                chk = check_aw(str(act.get("code") or ""), prefix, static) if aw else check_tau2(act, prefix, static)
                cc = chk["calls"]
                rows.append({"domain": d, "fid": fid, "k": k, "i": i, "kind": kind(obs),
                             "gain": s["corrected"] - st["null"], "rescuable": g["decisive"] is not None,
                             "decisive": g["decisive"] == k, "obs_calls": oc, "corr_calls": cc,
                             "obs_lines": ol, "corr_lines": chk.get("lines", 0),
                             "unseen_id": chk["unseen_id"], "unseen_num": chk["unseen_num"],
                             "unparsable": chk.get("unparsable", False),
                             "flags": {"unseen_id": bool(chk["unseen_id"]),
                                       "unseen_any": bool(chk["unseen_id"] or chk["unseen_num"]),
                                       "expanded": cc >= max(oc, 1) + 2,
                                       "strict": cc > max(oc, 1)},
                             "why": o.get("why"), "action": act})
                rows[-1]["flags"]["unseen_id_or_expanded"] = rows[-1]["flags"]["unseen_id"] or rows[-1]["flags"]["expanded"]
    return rows


def refiltered(v: dict, rows: list[dict], flt: str) -> dict:
    drop = {(r["fid"], r["k"], r["i"]) for r in rows if r["flags"][flt]}
    prof = {}
    for fid, g in v["gt"].items():
        R = []
        for st in g["steps"]:
            tot = 0.0
            for i, s in enumerate(st["samples"]):
                if s.get("verdict") == "mistake" and s.get("corrected") is not None and s.get("n") == 2 \
                        and st["null"] is not None and (fid, st["k"], i) not in drop:
                    tot += s["corrected"] - st["null"]
            R.append(tot / K)
        prof[fid] = R
    return prof


def _rates(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for f in FILTERS:
        out[f] = round(_mean([r["flags"][f] for r in rows]), 4) if rows else None
    out["unparsable"] = sum(r["unparsable"] for r in rows)
    out["mean_gain_flagged_id"] = round(_mean([r["gain"] for r in rows if r["flags"]["unseen_id"]]), 4) \
        if any(r["flags"]["unseen_id"] for r in rows) else None
    out["mean_gain_clean"] = round(_mean([r["gain"] for r in rows if not r["flags"]["unseen_id_or_expanded"]]), 4) \
        if rows else None
    return out


# ------------------------------------------------------------------- audit --
def _ctx_text(d: str, rec: dict, k: int, last: int = 4) -> str:
    """Everything the agent had seen before step k, untruncated; steps before
    the last `last` go in a collapsed block."""
    from attrib.groundtruth import _render_step

    def step(st):
        if d == "appworld":
            return f"--- step {st['index']} ---\n{st.get('assistant')}\n{st.get('output') or ''}"
        return f"--- step {st['index']} ---\n{_render_step({**st, 'tool_results': []})}\n" + "\n".join(
            f"  tool result: {r.get('content')}" for r in st.get("tool_results") or [])

    head = (f"TASK: {rec.get('instruction')}\nSUPERVISOR: {rec.get('supervisor')}" if d == "appworld"
            else f"USER (opening): {rec.get('opening')}")
    lo = max(0, k - last)
    early = "\n".join(step(st) for st in rec["steps"][:lo])
    recent = "\n".join(step(st) for st in rec["steps"][lo:k])
    out = ["```", head, "```"]
    if early:
        out += [f"<details><summary>steps 0-{lo - 1}</summary>", "", "```", early, "```", "", "</details>"]
    out += ["```", recent or "(no earlier steps)", "```"]
    return "\n".join(out)


def _obs_text(d: str, st: dict) -> str:
    if d == "appworld":
        return st.get("assistant") or ""
    from attrib.groundtruth import _render_step
    return _render_step({**st, "tool_results": [], "user": None})


def audit_sample(rows: list[dict], data: dict, n_per_domain: int, seed: int) -> list[dict]:
    """Corrections with a positive gain on rescuable failures, per domain: half
    with a mechanical flag (unseen id or expanded scope), half without, spread
    over step kinds."""
    rng = random.Random(seed)
    out = []
    for d in DOMAINS:
        pool = [r for r in rows if r["domain"] == d and r["rescuable"] and r["gain"] > 0]
        for flagged in (True, False):
            sub = [r for r in pool if r["flags"]["unseen_id_or_expanded"] == flagged]
            by_kind = defaultdict(list)
            for r in sub:
                by_kind[r["kind"]].append(r)
            for x in by_kind.values():
                rng.shuffle(x)
            want = n_per_domain // 2
            pick = []
            while len(pick) < want and any(by_kind.values()):
                for kk in sorted(by_kind):
                    if by_kind[kk] and len(pick) < want:
                        pick.append(by_kind[kk].pop())
            out += pick
        # top up from the other half if one half ran short
        short = n_per_domain - sum(r["domain"] == d for r in out)
        rest = [r for r in pool if r not in out]
        rng.shuffle(rest)
        out += rest[:max(short, 0)]
    rng.shuffle(out)
    return out


def write_audit(sample: list[dict], data: dict, adir: Path) -> None:
    adir.mkdir(parents=True, exist_ok=True)
    md = ["# Audit of oracle corrections (paper 2)", "",
          "For each item: the task, the conversation or episode before the step (last steps only), "
          "the agent's observed action, and the oracle's correction. Answer in audit_sheet.csv. Answer Q2 "
          "before reading the oracle's reason, which may mention the grading.",
          "",
          "- **Q1 information**: does the correction use any id, value or fact the agent had not seen "
          "before this step and could not get from the policy or the tool/API documentation? (yes/no)",
          "- **Q2 justified by what the agent could see**: without knowing how the task is graded, would a "
          "careful agent have good reason, from the conversation or episode so far and the policy, to take "
          "this action rather than the observed one? (yes/no/unclear)",
          "- **Q3 scope**: does the correction do substantially more than one careful step, for example "
          "several actions the agent would normally take over several turns? (yes/no)",
          "- **Q4 policy**: is the correction a step a careful agent following the policy could take here? (yes/no)",
          "", "The mechanical flags are hidden here so that they do not prime the audit.", ""]
    sheet = []
    for n, r in enumerate(sample, 1):
        d = r["domain"]
        rec = json.loads(Path(data[d]["fails"][r["fid"]]["trace"]).read_text())
        item = f"A{n:02d}"
        corr = r["action"].get("code") if d == "appworld" else json.dumps(r["action"], ensure_ascii=False, indent=1)
        md += [f"## {item} ({d}, step {r['k']})", "", "**Before the step**", "", _ctx_text(d, rec, r["k"]),
               "", "**Observed action**", "", "```", _obs_text(d, rec["steps"][r["k"]]), "```",
               "", "**Oracle's correction**", "", "```", str(corr), "```", "",
               f"Oracle's reason: {r['why']}", ""]
        sheet.append({"item": item, "domain": d, "fid": r["fid"], "step": r["k"], "sample": r["i"],
                      "Q1_unavailable_info": "", "Q2_justified_by_visible_prefix": "", "Q3_too_much_work": "",
                      "Q4_policy_valid": "", "note": ""})
    (adir / "audit_items.md").write_text("\n".join(md))
    with open(adir / "audit_sheet.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    key = [{"item": s["item"], **{k: r[k] for k in ("domain", "fid", "k", "i", "kind", "gain", "unseen_id",
                                                    "unseen_num", "obs_calls", "corr_calls")},
            "flags": r["flags"]} for s, r in zip(sheet, sample)]
    (adir / "audit_key_mechanical.json").write_text(json.dumps(key, indent=1))


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--out")
    ap.add_argument("--audit", help="directory for the stratified human-audit sample")
    ap.add_argument("--n-audit", type=int, default=20, help="items per domain")
    args = ap.parse_args()
    main_dir = Path(args.main)
    rng = random.Random(0)
    data = {d: load(main_dir / d) for d in DOMAINS}
    rows = {d: collect(main_dir, d, data[d]) for d in DOMAINS}
    allr = [r for d in DOMAINS for r in rows[d]]
    res = {"rates": {d: _rates(rows[d]) for d in DOMAINS},
           "rates_pooled": _rates(allr),
           "rates_rescuable_positive": _rates([r for r in allr if r["rescuable"] and r["gain"] > 0]),
           "rates_decisive_step": _rates([r for r in allr if r["decisive"]]),
           "by_kind": {d: {kk: _rates([r for r in rows[d] if r["kind"] == kk])
                           for kk in sorted({r["kind"] for r in rows[d]})} for d in DOMAINS},
           "calls": {d: {"obs_mean": round(_mean([r["obs_calls"] for r in rows[d]]), 3),
                         "corr_mean": round(_mean([r["corr_calls"] for r in rows[d]]), 3),
                         "obs_lines_mean": round(_mean([r["obs_lines"] for r in rows[d]]), 2),
                         "corr_lines_mean": round(_mean([r["corr_lines"] for r in rows[d]]), 2)} for d in DOMAINS},
           "examples_unseen_id": {d: [{"fid": r["fid"], "k": r["k"], "values": r["unseen_id"][:4]}
                                      for r in rows[d] if r["flags"]["unseen_id"]][:8] for d in DOMAINS}}
    methods = [m for m in METHODS + EXTRA if all(m in v["picks"] for v in data.values())]
    reg = {m: _mean([v["gt"][f]["R"][k] if isinstance(k := v["picks"][m].get(f), int) and 0 <= k < len(v["gt"][f]["R"])
                     else 0.0 for v in data.values() for f in v["gt"] if v["gt"][f]["decisive"] is not None])
           for m in METHODS}
    res["rescored"] = {}
    for flt in FILTERS:
        prof = {d: refiltered(data[d], rows[d], flt) for d in DOMAINS}
        r = _ranking(data, prof, rng, methods)
        r["kendall_tau_vs_registered"] = round(kendall({m: r["mean"][m] for m in METHODS}, reg), 3)
        res["rescored"][flt] = r
    if args.audit:
        sample = audit_sample(allr, data, args.n_audit, seed=0)
        write_audit(sample, data, Path(args.audit))
        res["audit"] = {"items": len(sample), "dir": args.audit,
                        "flagged_in_sample": sum(r["flags"]["unseen_id_or_expanded"] for r in sample)}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("rates", "rates_pooled", "rates_rescuable_positive", "rates_decisive_step",
                                          "calls")}, indent=1))
    for flt, r in res["rescored"].items():
        print(flt, r["n_rescuable"], "fw", r["mean"]["first_write"], "rank", r["first_write_rank"], "best", r["best_llm"],
              r["mean"][r["best_llm"]], "fw-bs", r["first_write_minus_binary_search"],
              "fw-bsgain", r.get("first_write_minus_binary_search_gain_pro"), "tau", r["kendall_tau_vs_registered"])


if __name__ == "__main__":
    main()
