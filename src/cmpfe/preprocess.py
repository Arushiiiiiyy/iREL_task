from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple

from langdetect import detect

from .models import TranscriptSegment


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def normalize_segments(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    out = []
    for seg in segments:
        txt = clean_text(seg.text)
        lang = seg.language
        if not lang:
            try:
                lang = detect(txt)
            except Exception:
                lang = "unknown"
        out.append(TranscriptSegment(start=seg.start, end=seg.end, text=txt, language=lang))
    return out


def dominant_languages(segments: List[TranscriptSegment], k: int = 3) -> List[str]:
    counts = Counter(seg.language or "unknown" for seg in segments)
    return [lang for lang, _ in counts.most_common(k)]


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text) if s.strip()]


def find_dependency_cues(text: str) -> List[Tuple[str, str]]:
    cues = [
        ("before", "must understand"),
        ("first", "then"),
        ("pehle", "fir"),
        ("pahle", "phir"),
        ("prerequisite", "for"),
        ("needed for", ""),
    ]
    lower = text.lower()
    hits = []
    for a, b in cues:
        if a in lower and (not b or b in lower):
            hits.append((a, b))
    return hits
