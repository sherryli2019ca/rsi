"""tau2-bench airline as an RRSI domain. Everything shared with the other tau2
domain (driver call, scoring, trace rendering, gates) is in
domains/tau2/common.py; this file adds the airline-specific leakage patterns."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from domains.tau2.common import Tau2Domain  # noqa: E402

DOMAIN = Tau2Domain("tau2_airline", "airline", [
    (r"\bHAT\d{3}\b", "flight number in diff"),
    (r"\b(?=[A-Z0-9]{6}\b)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}\b", "reservation code in diff"),
])
