from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from .io_utils import ensure_dir
from .models import TranscriptSegment


def download_audio(url: str, out_dir: Path) -> Path:
    """Download best available audio using yt-dlp.

    We intentionally avoid `--extract-audio` so ffmpeg is not required at
    download time. This improves portability on macOS/Windows setups where
    ffmpeg is not installed yet.
    """
    ensure_dir(out_dir)
    output_tmpl = out_dir / "audio.%(ext)s"
    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio",
        "--no-playlist",
        "-o",
        str(output_tmpl),
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "yt-dlp failed to download audio. If YouTube extraction warns about "
            "missing JS runtimes, install Node.js or Deno and retry."
        ) from exc

    audio_files = sorted(
        p for p in out_dir.glob("audio.*") if p.suffix.lower() not in {".part", ".ytdl"}
    )
    if not audio_files:
        raise FileNotFoundError(f"No audio file created in {out_dir}")
    return audio_files[0]


def read_segments_json(path: Path) -> List[TranscriptSegment]:
    import json

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw["segments"] if isinstance(raw, dict) and "segments" in raw else raw
    return [TranscriptSegment(**item) for item in items]
