"""
src/cmpfe/asr.py
================
Uses openai-whisper with automatic Metal (MPS) acceleration on Apple Silicon.
Respects model size — no buffer overflow crashes.

Why not mlx-whisper?
  mlx-whisper has a bug where it ignores path_or_hf_repo and always loads
  whisper-small internally, causing a 4.3 GB Metal allocation crash on 8 GB Macs.

openai-whisper on Apple Silicon:
  - Uses Metal via PyTorch MPS backend automatically
  - Respects model size — tiny = 75 MB, base = 145 MB
  - ~5-8x realtime on M-series with base model
  - No 4 GB buffer limit issue
"""
from __future__ import annotations

import contextlib
import os
import platform
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator, List, Optional

from .io_utils import dump_json
from .models import TranscriptSegment

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

MODEL_META = {
    "tiny":   {"size_mb": 75,   "speed": "~15x realtime on Apple Silicon"},
    "base":   {"size_mb": 145,  "speed": "~8x  realtime on Apple Silicon"},
    "small":  {"size_mb": 480,  "speed": "~4x  realtime on Apple Silicon"},
    "medium": {"size_mb": 1500, "speed": "~2x  realtime on Apple Silicon"},
}

# Peak VRAM/unified memory needed — hard Metal buffer limit is 4 GB
MLX_METAL_GB = {
    "tiny": 0.4, "base": 0.8, "small": 4.3, "medium": 8.5, "large": 16.0
}
METAL_LIMIT_GB = 4.0


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _total_ram_gb() -> float:
    try:
        r = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return int(r.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass
    return 8.0


def _safe_model(requested: str) -> str:
    """Downgrade model if it would exceed the 4 GB Metal buffer limit."""
    ram_gb   = _total_ram_gb()
    usable   = min(ram_gb - 2.0, METAL_LIMIT_GB)
    if MLX_METAL_GB.get(requested, 99.0) <= usable:
        return requested
    for model in ["small", "base", "tiny"]:
        if MLX_METAL_GB.get(model, 99.0) <= usable:
            return model
    return "tiny"


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


def _require_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            "Missing required media tools: "
            f"{', '.join(missing)}. "
            "Install FFmpeg on macOS with 'brew install ffmpeg'."
        )


def _get_device() -> str:
    """Return 'mps' on Apple Silicon if PyTorch supports it, else 'cpu'."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_ca_bundle_path(ca_bundle_path: Optional[Path]) -> Optional[Path]:
    if ca_bundle_path is not None:
        return ca_bundle_path

    for env_name in ("CMPFE_SSL_CA_FILE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        value = os.getenv(env_name)
        if value:
            return Path(value).expanduser()

    return None


def _build_ssl_context(
    ca_bundle_path: Optional[Path],
    allow_insecure_download: bool,
) -> tuple[Optional[ssl.SSLContext], Optional[str]]:
    if allow_insecure_download:
        return ssl._create_unverified_context(), "TLS verification disabled for Whisper model download"

    resolved_path = _resolve_ca_bundle_path(ca_bundle_path)
    if resolved_path is not None:
        if not resolved_path.exists():
            raise FileNotFoundError(f"CA bundle not found: {resolved_path}")
        return (
            ssl.create_default_context(cafile=str(resolved_path)),
            f"using CA bundle {resolved_path}",
        )

    try:
        import certifi
    except ImportError:
        return None, None

    certifi_path = Path(certifi.where())
    if certifi_path.exists():
        return ssl.create_default_context(cafile=str(certifi_path)), f"using certifi bundle {certifi_path}"

    return None, None


def _is_ssl_verification_error(exc: BaseException) -> bool:
    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, ssl.SSLCertVerificationError):
            return True
        return "CERTIFICATE_VERIFY_FAILED" in str(reason)
    return False


@contextlib.contextmanager
def _patched_urlopen(ssl_context: Optional[ssl.SSLContext]) -> Iterator[None]:
    if ssl_context is None:
        yield
        return

    original_urlopen = urllib.request.urlopen

    def _urlopen_with_context(url, *args, **kwargs):
        kwargs.setdefault("context", ssl_context)
        return original_urlopen(url, *args, **kwargs)

    urllib.request.urlopen = _urlopen_with_context
    try:
        yield
    finally:
        urllib.request.urlopen = original_urlopen


def _load_whisper_model(
    whisper_module,
    model_name: str,
    device: str,
    ca_bundle_path: Optional[Path],
    allow_insecure_download: bool,
):
    if allow_insecure_download:
        ssl_context, ssl_description = _build_ssl_context(ca_bundle_path, allow_insecure_download=True)
        print(f"  Retrying model download with {ssl_description}.")
        sys.stdout.flush()
        with _patched_urlopen(ssl_context):
            return whisper_module.load_model(model_name, device=device)

    try:
        return whisper_module.load_model(model_name, device=device)
    except Exception as exc:
        if not _is_ssl_verification_error(exc):
            raise

        ssl_context, ssl_description = _build_ssl_context(ca_bundle_path, allow_insecure_download=False)
        if ssl_context is None:
            raise RuntimeError(
                "Whisper model download failed TLS verification. "
                "Provide a trusted PEM bundle with --ssl-ca-file or set CMPFE_SSL_CA_FILE. "
                "If you understand the risk, rerun with --insecure-model-download."
            ) from exc

        print(f"  TLS verification failed with the default trust store; retrying with {ssl_description}.")
        sys.stdout.flush()
        with _patched_urlopen(ssl_context):
            return whisper_module.load_model(model_name, device=device)


def _should_retry_on_cpu(exc: BaseException, device: str) -> bool:
    if device != "mps":
        return False

    message = str(exc)
    return (
        isinstance(exc, NotImplementedError)
        and "SparseMPS" in message
        and "aten::_sparse_coo_tensor_with_dims_and_tensors" in message
    )


def transcribe_audio(
    audio_path: Path,
    transcript_json_path: Path,
    model_size: str = "base",
    compute_type: str = "int8",   # unused for openai-whisper, kept for API compat
    dry_run: bool = False,
    ca_bundle_path: Optional[Path] = None,
    allow_insecure_download: Optional[bool] = None,
) -> List[TranscriptSegment]:
    """
    Transcribe audio using openai-whisper.
    On Apple Silicon automatically uses Metal (MPS) via PyTorch.
    """
    if dry_run:
        print(f"  [DRY RUN] Would transcribe {audio_path.name} (model={model_size})")
        return []

    if allow_insecure_download is None:
        allow_insecure_download = _env_flag("CMPFE_INSECURE_MODEL_DOWNLOAD")

    _require_ffmpeg()

    try:
        import whisper
    except ImportError:
        raise ImportError(
            "openai-whisper not installed.\n"
            "Run: pip install openai-whisper"
        )

    # Safety check — downgrade model if needed
    safe = _safe_model(model_size)
    if safe != model_size:
        print(f"\n  ⚠  Model '{model_size}' needs {MLX_METAL_GB.get(model_size,99):.1f} GB "
              f"→ exceeds 4 GB Metal limit.")
        print(f"  ⚠  Auto-downgrading to '{safe}' "
              f"({MLX_METAL_GB.get(safe,0):.1f} GB — safe on your {_total_ram_gb():.0f} GB Mac).\n")

    device = _get_device()
    meta   = MODEL_META.get(safe, {})

    duration = _audio_duration(audio_path)
    dur_str  = ""
    if duration:
        m, s    = int(duration // 60), int(duration % 60)
        dur_str = f"  ({m}m {s}s)"

    print(f"\n{'─'*62}")
    print(f"  Backend  : openai-whisper")
    print(f"  Device   : {device.upper()}  {'(Apple Silicon Metal ✓)' if device == 'mps' else '(CPU)'}")
    print(f"  Model    : whisper-{safe}  (~{meta.get('size_mb','?')} MB)")
    print(f"  Speed    : {meta.get('speed','?')}")
    print(f"  File     : {audio_path.name}{dur_str}")
    print(f"{'─'*62}\n")
    sys.stdout.flush()

    print(f"  Loading model (downloads once, then cached) …")
    sys.stdout.flush()
    t0    = time.time()
    try:
        model = _load_whisper_model(
            whisper_module=whisper,
            model_name=safe,
            device=device,
            ca_bundle_path=ca_bundle_path,
            allow_insecure_download=allow_insecure_download,
        )
    except Exception as exc:
        if not _should_retry_on_cpu(exc, device):
            raise
        print("  MPS backend rejected a sparse Whisper tensor; retrying on CPU.")
        sys.stdout.flush()
        device = "cpu"
        model = _load_whisper_model(
            whisper_module=whisper,
            model_name=safe,
            device=device,
            ca_bundle_path=ca_bundle_path,
            allow_insecure_download=allow_insecure_download,
        )
    print(f"  ✓ Model loaded in {time.time()-t0:.1f}s\n")
    sys.stdout.flush()

    print(f"  Transcribing …")
    sys.stdout.flush()
    t1     = time.time()
    result = model.transcribe(
        str(audio_path),
        verbose=False,      # suppress whisper's own output; we print ours
        fp16=(device != "cpu"),
    )
    elapsed = time.time() - t1

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

    print(f"\n  ✓ Done — {len(out)} segments in {elapsed:.1f}s")
    sys.stdout.flush()

    dump_json(transcript_json_path, {"segments": [s.model_dump() for s in out]})
    print(f"  ✓ Saved → {transcript_json_path}\n")

    return out