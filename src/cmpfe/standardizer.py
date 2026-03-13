from __future__ import annotations

from typing import Dict

from rapidfuzz import fuzz

STANDARD_MAP: Dict[str, str] = {
    "ped": "tree",
    "jad": "root",
    "current ka flow": "electric current",
    "voltage ka pressure": "electric potential",
    "memory ka dabba": "memory block",
    "queue line": "queue",
    "stack of plates": "stack",
    "gati": "velocity",
    "bal": "force",
    "urja": "energy",
    "samikaran": "equation",
    "algorithm ka step": "algorithm step",
    "o of n": "time complexity",
}


def standardize_term(raw: str, threshold: int = 85) -> str:
    raw_l = raw.strip().lower()
    if raw_l in STANDARD_MAP:
        return STANDARD_MAP[raw_l]

    best = raw_l
    best_score = 0
    for k, v in STANDARD_MAP.items():
        score = fuzz.partial_ratio(raw_l, k)
        if score > best_score:
            best_score = score
            best = v

    if best_score >= threshold:
        return best
    return raw.strip().lower()
