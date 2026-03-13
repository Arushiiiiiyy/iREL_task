from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .asr import transcribe_audio
from .concept_extractor import extract_concepts
from .graph_builder import export_graph
from .io_utils import ensure_dir, load_yaml
from .models import TranscriptSegment, VideoItem, VideoKnowledgeGraph
from .preprocess import dominant_languages, normalize_segments
from .prereq_mapper import map_prerequisites
from .video_ingest import download_audio


def _load_transcript(path: Path) -> List[TranscriptSegment]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [TranscriptSegment(**item) for item in data["segments"]]


def run_pipeline(
    config_path: Path,
    data_root: Path,
    output_root: Path,
    whisper_model: str = "small",
    skip_download: bool = False,
    ssl_ca_file: Path | None = None,
    insecure_model_download: bool = False,
) -> None:
    cfg = load_yaml(config_path)
    videos = [VideoItem(**v) for v in cfg["videos"]]

    for video in videos:
        video_dir = data_root / video.id
        ensure_dir(video_dir)

        transcript_path = video_dir / "transcript.json"
        if not transcript_path.exists():
            if skip_download:
                raise FileNotFoundError(f"Missing transcript: {transcript_path}")
            audio_path = download_audio(video.url, video_dir)
            transcribe_audio(
                audio_path,
                transcript_path,
                model_size=whisper_model,
                ca_bundle_path=ssl_ca_file,
                allow_insecure_download=insecure_model_download,
            )

        segments = _load_transcript(transcript_path)
        segments = normalize_segments(segments)

        concepts, mentions, norm_map = extract_concepts(segments)
        edges = map_prerequisites(segments, concepts)

        video_graph = VideoKnowledgeGraph(
            video_id=video.id,
            title=video.title,
            source_url=video.url,
            domain=video.domain,
            languages_detected=dominant_languages(segments),
            concept_nodes=concepts,
            lexical_normalization_map=norm_map,
            concept_mentions=mentions,
            prerequisite_edges=edges,
        )

        export_graph(video_graph, output_root)
