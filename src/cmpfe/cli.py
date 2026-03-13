from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _run_environment_checks() -> int:
    required = ["python", "yt-dlp", "ffmpeg", "ffprobe"]
    missing = [tool for tool in required if shutil.which(tool) is None]

    if missing:
        print(f"Missing required tools: {', '.join(missing)}", file=sys.stderr)
        if any(tool in {"ffmpeg", "ffprobe"} for tool in missing):
            print("Install FFmpeg on macOS with: brew install ffmpeg", file=sys.stderr)
        return 1

    print("Environment check passed: python, yt-dlp, ffmpeg, and ffprobe found.")
    print("Tip: If yt-dlp reports missing JS runtime for YouTube, install Node.js or Deno.")
    return 0


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Code-Mixed Pedagogical Flow Extractor")
    parser.add_argument("--config", default="config/videos.yaml", help="Path to videos yaml")
    parser.add_argument("--data-root", default="data", help="Data directory")
    parser.add_argument("--output-root", default="outputs", help="Output directory")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model size")
    parser.add_argument(
        "--ssl-ca-file",
        default=os.getenv("CMPFE_SSL_CA_FILE") or os.getenv("SSL_CERT_FILE"),
        help="Path to a PEM CA bundle for Whisper model downloads",
    )
    parser.add_argument(
        "--insecure-model-download",
        action="store_true",
        default=_env_flag("CMPFE_INSECURE_MODEL_DOWNLOAD"),
        help="Disable TLS verification only for Whisper model downloads",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip yt download + ASR and require existing transcripts",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Validate required local tooling and exit",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.check_env:
        raise SystemExit(_run_environment_checks())

    from .pipeline import run_pipeline

    run_pipeline(
        config_path=Path(args.config),
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        whisper_model=args.whisper_model,
        skip_download=args.skip_download,
        ssl_ca_file=Path(args.ssl_ca_file).expanduser() if args.ssl_ca_file else None,
        insecure_model_download=args.insecure_model_download,
    )


if __name__ == "__main__":
    main()
