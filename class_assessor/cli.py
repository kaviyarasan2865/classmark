"""Command line entry for one recording or the labelled sample library."""

from __future__ import annotations

import argparse
from pathlib import Path

from class_assessor.check import check_library, format_report
from class_assessor.pipeline import assess_media, assess_transcript


def _library(root: Path) -> None:
    rows = []
    for vtt in sorted(root.rglob("*.vtt")):
        folder = vtt.parent.name.strip() or "?"
        try:
            result = assess_transcript(str(vtt))
        except Exception as exc:  # noqa: BLE001 - one bad file should not stop the table
            print(f"{folder:16} {vtt.name[:40]:40} ERROR {exc}")
            continue
        subs = " ".join(f"{name[:3]}={value:.0f}" for name, value in result.subscores.items())
        teacher = next((speaker.id for speaker in result.speakers if speaker.role == "teacher"), "-")
        students = sum(1 for speaker in result.speakers if speaker.role == "student")
        audience = sum(1 for speaker in result.speakers if speaker.role == "audience")
        print(
            f"{folder:16} {result.grade:10} {result.score:5.1f}  "
            f"stu={students} aud={audience} teacher={teacher[:22]:22} {subs}  {vtt.name[:36]}"
        )
        rows.append((folder, result.grade, result.score))
    print(f"\n{len(rows)} transcripts scored.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess an online class recording.")
    parser.add_argument("path", nargs="?", help="Video, audio, or .vtt transcript")
    parser.add_argument("--vtt", help="Optional Zoom transcript with speaker names")
    parser.add_argument("--model", default="small", help="faster-whisper model, default small")
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--max-minutes", type=float, default=None, help="Debug limit. Default is the full file.")
    parser.add_argument("--frame-interval", type=float, default=15.0)
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--library", help="Score every .vtt under this folder and print a table")
    parser.add_argument(
        "--check",
        help="Score the labelled sample set against feedbacks.pdf and print matches",
    )
    args = parser.parse_args()

    if args.check:
        print(format_report(check_library(Path(args.check))))
        return
    if args.library:
        _library(Path(args.library))
        return
    if not args.path:
        parser.error("Pass a recording, or --library.")

    path = Path(args.path)
    if path.suffix.lower() == ".vtt" and not args.vtt:
        result = assess_transcript(str(path))
    else:
        def progress(fraction: float, message: str) -> None:
            print(f"[{fraction:5.0%}] {message}", flush=True)

        result = assess_media(
            str(path),
            vtt_path=args.vtt,
            model_name=args.model,
            beam_size=args.beam,
            frame_interval=args.frame_interval,
            max_seconds=(args.max_minutes * 60) if args.max_minutes else None,
            skip_visual=args.skip_visual,
            on_progress=progress,
        )
    print()
    print(result.to_markdown())
