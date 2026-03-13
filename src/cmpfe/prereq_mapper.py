from __future__ import annotations

from typing import Dict, List

from .models import PrerequisiteEdge, TranscriptSegment
from .preprocess import find_dependency_cues


ORDER_WORDS = ["first", "then", "next", "finally", "pehle", "phir", "baad mein"]


def map_prerequisites(segments: List[TranscriptSegment], concepts: List[str]) -> List[PrerequisiteEdge]:
    edges: List[PrerequisiteEdge] = []

    concept_pos: Dict[str, int] = {c: 10**9 for c in concepts}
    for i, seg in enumerate(segments):
        low = seg.text.lower()
        for c in concepts:
            if c in low and concept_pos[c] == 10**9:
                concept_pos[c] = i

    ordered = sorted(concepts, key=lambda c: concept_pos.get(c, 10**9))
    for a, b in zip(ordered, ordered[1:]):
        edges.append(
            PrerequisiteEdge(
                prereq=a,
                target=b,
                reason="Introduced earlier in pedagogical flow",
                evidence_segment_indices=[concept_pos.get(a, -1), concept_pos.get(b, -1)],
                confidence=0.6,
            )
        )

    for i, seg in enumerate(segments):
        hits = find_dependency_cues(seg.text)
        if not hits:
            continue
        low = seg.text.lower()
        present = [c for c in concepts if c in low]
        if len(present) >= 2:
            edges.append(
                PrerequisiteEdge(
                    prereq=present[0],
                    target=present[1],
                    reason=f"Dependency cue(s) in sentence: {hits}",
                    evidence_segment_indices=[i],
                    confidence=0.8,
                )
            )

    dedup = {}
    for e in edges:
        dedup[(e.prereq, e.target)] = e
    return list(dedup.values())
