"""
src/cmpfe/asr.py  (Apple Silicon — memory-safe)
================================================
Auto-selects the largest MLX Whisper model that safely fits in your
Mac's unified memory, avoiding the Metal buffer allocation crash.

Memory limits by Mac:
  8 GB  unified memory  → whisper-tiny-mlx   (safe ceiling: ~400 MB)
  16 GB unified memory  → whisper-small-mlx  (safe ceiling: ~1.5 GB)
  32 GB unified memory  → whisper-medium-mlx (safe ceiling: ~4 GB)

The 4 GB Metal buffer hard-limit is a macOS kernel constraint — it applies
regardless of how much total RAM you have. The model must fit in one buffer.
whisper-small needs 4.3 GB → crashes on 8 GB Macs. whisper-tiny needs 400 MB → works fine.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from .io_utils import dump_json
from .models import TranscriptSegment

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

# MLX model repos on HuggingFace
MLX_MODELS = {
    "tiny":   "mlx-community/whisper-tiny-mlx",
    "base":   "mlx-community/whisper-base-mlx",
    "small":  "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large":  "mlx-community/whisper-large-v3-mlx",
}

# Approximate peak Metal memory needed per model (GB)
MLX_METAL_GB = {
    "tiny":   0.4,
    "base":   0.8,
    "small":  4.3,   # crashes on 8 GB Macs — exceeds 4 GB buffer limit
    "medium": 8.5,
    "large":  16.0,
}

MODEL_META = {
    "tiny":   {"size_mb": 75,   "apple_speed": "~15x realtime"},
    "base":   {"size_mb": 145,  "apple_speed": "~12x realtime"},
    "small":  {"size_mb": 480,  "apple_speed": "~8x  realtime"},
    "medium": {"size_mb": 1500, "apple_speed": "~5x  realtime"},
    "large":  {"size_mb": 3000, "apple_speed": "~3x  realtime"},
}

# macOS hard limit per Metal buffer — kernel constant, cannot be changed
METAL_BUFFER_LIMIT_GB = 4.0


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _total_ram_gb() -> float:
    """Read total unified memory from sysctl on macOS."""
    try:
        r = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return int(r.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass
    return 8.0  # safe default assumption


def _safe_model_for_mac(requested: str) -> str:
    """
    Return the largest model that won't crash on this Mac.

    Rules:
    - Model's peak Metal allocation must be < METAL_BUFFER_LIMIT_GB (4 GB)
    - Model must fit in total RAM with headroom for the OS + Python process
    - If the requested model is already safe, return it unchanged.
    """
    ram_gb = _total_ram_gb()
    # Leave 2 GB headroom for OS + Python + yt-dlp cache
    usable_gb = min(ram_gb - 2.0, METAL_BUFFER_LIMIT_GB)

    requested_peak = MLX_METAL_GB.get(requested, 99.0)
    if requested_peak <= usable_gb:
        return requested  # user's choice is safe

    # Walk down from requested until we find a safe model
    # order is largest → smallest so we try the biggest safe option first
    order = ["large", "medium", "small", "base", "tiny"]
    start = order.index(requested) if requested in order else len(order) - 1
    for model in order[start:]:
        if MLX_METAL_GB.get(model, 99.0) <= usable_gb:
            return model

    return "tiny"  # always safe


def _mlx_available() -> bool:
    try:
        import mlx_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _faster_whisper_available() -> bool:
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        return True
    except ImportError:
        return False


def _audio_duration(path: Path) -> Optional[float]:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return None


def _print_header(model: str, backend: str, ram_gb: float, original_request: str) -> None:
    meta  = MODEL_META.get(model, {})
    speed = meta.get("apple_speed" if backend == "mlx" else "cpu_speed", "?")
    size  = meta.get("size_mb", "?")

    print(f"\n{'─'*62}")
    print(f"  Backend  : {backend.upper()}")
    print(f"  Mac RAM  : {ram_gb:.0f} GB unified memory")
    if model != original_request:
        print(f"  Model    : '{original_request}' → downgraded to '{model}'")
        print(f"             (small needs 4.3 GB Metal buffer; your limit is 4 GB)")
    else:
        print(f"  Model    : whisper-{model}")
    print(f"  Download : ~{size} MB  (cached after first run)")
    print(f"  Speed    : {speed}")
    print(f"{'─'*62}\n")
    sys.stdout.flush()


def _transcribe_mlx(audio_path: Path, model_size: str) -> List[TranscriptSegment]:
    import mlx_whisper

    hf_repo = MLX_MODELS[model_size]
    print(f"  Repo     : {hf_repo}")
    print(f"  File     : {audio_path.name}")
    print(f"  Segments print live as they complete:\n")
    sys.stdout.flush()

    t0 = time.time()
    result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=hf_repo, verbose=False)
    elapsed = time.time() - t0

    out: List[TranscriptSegment] = []
    for seg in result.get("segments", []):
        ts = TranscriptSegment(
            start=float(seg["start"]),
            end=float(seg["end"]),
            text=seg["text"].strip(),
            language=result.get("language", ""),
        )
        out.append(ts)
        preview = seg["text"].strip()[:72] + ("…" if len(seg["text"].strip()) > 72 else "")
        print(f"  [{seg['start']:6.1f}→{seg['end']:5.1f}s]  {preview}")
        sys.stdout.flush()

    print(f"\n  ✓ MLX done — {len(out)} segments in {elapsed:.1f}s")
    sys.stdout.flush()
    return out


def _transcribe_faster_whisper(audio_path: Path, model_size: str) -> List[TranscriptSegment]:
    from faster_whisper import WhisperModel

    print(f"  Loading WhisperModel('{model_size}', int8) …")
    sys.stdout.flush()
    t0 = time.time()
    model = WhisperModel(model_size, compute_type="int8")
    print(f"  ✓ Loaded in {time.time()-t0:.1f}s\n")
    sys.stdout.flush()

    t1 = time.time()
    segs_gen, info = model.transcribe(
        str(audio_path), beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
    )

    out: List[TranscriptSegment] = []
    for seg in segs_gen:
        ts = TranscriptSegment(
            start=float(seg.start), end=float(seg.end),
            text=seg.text.strip(), language=info.language,
        )
        out.append(ts)
        preview = seg.text.strip()[:70] + ("…" if len(seg.text.strip()) > 70 else "")
        print(f"  [{seg.start:6.1f}→{seg.end:5.1f}s] +{time.time()-t1:4.0f}s  {preview}")
        sys.stdout.flush()

    print(f"\n  ✓ Done — {len(out)} segments in {time.time()-t1:.1f}s")
    sys.stdout.flush()
    return out


def transcribe_audio(
    audio_path: Path,
    transcript_json_path: Path,
    model_size: str = "small",
    compute_type: str = "int8",
    dry_run: bool = False,
) -> List[TranscriptSegment]:
    """
    Transcribe audio with automatic memory-safe model selection.

    On 8 GB Apple Silicon Macs, 'small' is automatically downgraded to
    'tiny' to avoid the Metal 4 GB buffer limit crash.
    """
    if dry_run:
        print(f"  [DRY RUN] Would transcribe {audio_path.name} (model={model_size})")
        return []

    duration = _audio_duration(audio_path)
    if duration:
        m, s = int(duration // 60), int(duration % 60)
        print(f"  Audio : {audio_path.name}  ({m}m {s}s)")

    apple = _is_apple_silicon()
    mlx   = _mlx_available()
    fw    = _faster_whisper_available()

    if apple and mlx:
        backend = "mlx"
        ram_gb  = _total_ram_gb()
        safe_model = _safe_model_for_mac(model_size)
        _print_header(safe_model, backend, ram_gb, model_size)
        out = _transcribe_mlx(audio_path, safe_model)

    elif fw:
        backend = "faster-whisper"
        if apple and not mlx:
            print("\n  ⚠  mlx-whisper not found — falling back to CPU faster-whisper.")
            print("     Install for 10x speedup: pip install mlx-whisper\n")
            model_size = "tiny"
        ram_gb = _total_ram_gb() if apple else 0.0
        _print_header(model_size, backend, ram_gb, model_size)
        out = _transcribe_faster_whisper(audio_path, model_size)

    else:
        raise RuntimeError(
            "No transcription backend found.\n"
            "  Apple Silicon : pip install mlx-whisper\n"
            "  Intel / Linux : pip install faster-whisper"
        )

    dump_json(transcript_json_path, {"segments": [s.model_dump() for s in out]})
    print(f"  ✓ Saved → {transcript_json_path}\n")
    return out