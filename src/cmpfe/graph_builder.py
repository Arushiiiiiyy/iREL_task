from __future__ import annotations

from pathlib import Path

import networkx as nx

from .io_utils import dump_json
from .models import VideoKnowledgeGraph


def export_graph(video_graph: VideoKnowledgeGraph, out_dir: Path) -> None:
    graph = nx.DiGraph()
    for concept in video_graph.concept_nodes:
        graph.add_node(concept)
    for edge in video_graph.prerequisite_edges:
        graph.add_edge(edge.prereq, edge.target, reason=edge.reason, confidence=edge.confidence)

    data = {
        "meta": {
            "video_id": video_graph.video_id,
            "title": video_graph.title,
            "url": video_graph.source_url,
            "domain": video_graph.domain,
            "languages_detected": video_graph.languages_detected,
        },
        "nodes": [{"id": n} for n in graph.nodes],
        "edges": [
            {
                "source": u,
                "target": v,
                "reason": d.get("reason", ""),
                "confidence": d.get("confidence", 0.5),
            }
            for u, v, d in graph.edges(data=True)
        ],
        "lexical_normalization_map": video_graph.lexical_normalization_map,
        "concept_mentions": [m.model_dump() for m in video_graph.concept_mentions],
    }
    dump_json(out_dir / f"{video_graph.video_id}.knowledge_graph.json", data)
