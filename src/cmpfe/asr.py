from __future__ import annotations

from pathlib import Path
from typing import List

from faster_whisper import WhisperModel

from .io_utils import dump_json
from .models import TranscriptSegment


def transcribe_audio(
    audio_path: Path,
    transcript_json_path: Path,
    model_size: str = "small",
    compute_type: str = "int8",
) -> List[TranscriptSegment]:
    model = WhisperModel(model_size, compute_type=compute_type)
    segments, info = model.transcribe(str(audio_path), beam_size=5)

    out: List[TranscriptSegment] = []
    for seg in segments:
        out.append(
            TranscriptSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=seg.text.strip(),
                language=info.language,
            )
        )

    dump_json(transcript_json_path, {"segments": [s.model_dump() for s in out]})
    return out
