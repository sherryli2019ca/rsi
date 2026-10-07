"""tau2-bench retail as an RRSI domain. Everything shared with the other tau2
domain (driver call, scoring, trace rendering, gates) is in
domains/tau2/common.py; this file adds the retail-specific leakage patterns."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from domains.tau2.common import Tau2Domain  # noqa: E402

DOMAIN = Tau2Domain("tau2_retail", "retail", [
    (r"#W\d{7}", "order id in diff"),
    (r"\b\d{10}\b", "item or product id in diff"),
])
