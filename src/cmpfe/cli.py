from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Code-Mixed Pedagogical Flow Extractor")
    parser.add_argument("--config", default="config/videos.yaml", help="Path to videos yaml")
    parser.add_argument("--data-root", default="data", help="Data directory")
    parser.add_argument("--output-root", default="outputs", help="Output directory")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model size")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip yt download + ASR and require existing transcripts",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(
        config_path=Path(args.config),
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        whisper_model=args.whisper_model,
        skip_download=args.skip_download,
    )


if __name__ == "__main__":
    main()
