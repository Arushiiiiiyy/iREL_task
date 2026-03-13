from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

from .models import ConceptMention, TranscriptSegment
from .standardizer import standardize_term

TECH_TERM_RE = re.compile(
    r"\b(data structure|algorithm|array|linked list|stack|queue|tree|graph|"
    r"recursion|dynamic programming|complexity|os|thread|process|memory|"
    r"voltage|current|resistance|force|energy|equation|class|object|inheritance)\b",
    flags=re.IGNORECASE,
)


def extract_concepts(segments: List[TranscriptSegment], min_freq: int = 2) -> Tuple[List[str], List[ConceptMention], Dict[str, str]]:
    mentions: List[ConceptMention] = []
    norm_map: Dict[str, str] = {}
    freq = Counter()

    for i, seg in enumerate(segments):
        matches = TECH_TERM_RE.findall(seg.text)
        for raw in matches:
            raw_clean = raw.strip().lower()
            standard = standardize_term(raw_clean)
            norm_map[raw_clean] = standard
            freq[standard] += 1
            mentions.append(
                ConceptMention(
                    concept_raw=raw_clean,
                    concept_standard=standard,
                    segment_index=i,
                    confidence=0.75,
                )
            )

    concepts = sorted([c for c, ct in freq.items() if ct >= min_freq])
    if not concepts:
        concepts = sorted(freq.keys())
    return concepts, mentions, norm_map
