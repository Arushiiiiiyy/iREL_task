from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class VideoItem(BaseModel):
    id: str
    title: str
    url: str
    domain: str
    declared_code_mix: List[str] = Field(default_factory=list)


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    language: Optional[str] = None


class ConceptMention(BaseModel):
    concept_raw: str
    concept_standard: str
    segment_index: int
    confidence: float = 0.5


class PrerequisiteEdge(BaseModel):
    prereq: str
    target: str
    reason: str
    evidence_segment_indices: List[int] = Field(default_factory=list)
    confidence: float = 0.5


class VideoKnowledgeGraph(BaseModel):
    video_id: str
    title: str
    source_url: str
    domain: str
    languages_detected: List[str] = Field(default_factory=list)
    concept_nodes: List[str] = Field(default_factory=list)
    lexical_normalization_map: Dict[str, str] = Field(default_factory=dict)
    concept_mentions: List[ConceptMention] = Field(default_factory=list)
    prerequisite_edges: List[PrerequisiteEdge] = Field(default_factory=list)
