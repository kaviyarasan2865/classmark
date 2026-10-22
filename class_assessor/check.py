"""Compare grades with the labelled sample sheet in feedbacks.pdf.

The sheet uses Excellent, Good, Average, and Weak. The challenge grade for Weak
is Poor. "Known quality issues" is not a fifth grade: those classes must be
flagged, and they must not come out Good or Excellent.
"""

from __future__ import annotations

from pathlib import Path

from class_assessor.cues import parse_vtt
from class_assessor.score import Assessment, assess
from class_assessor.speakers import assign_roles

SHEET = {
    "excellent": "Excellent",
    "good": "Good",
    "average": "Average",
    "weak": "Poor",
}


def sheet_grade(folder: str) -> str | None:
    """Expected challenge grade for a sample folder. None means a quality-issue file."""
    key = folder.strip().lower()
    if key in SHEET:
        return SHEET[key]
    if "quality" in key:
        return None
    return None


def score_transcript(vtt_path: str, visual: dict | None = None) -> Assessment:
    """Grade one Zoom transcript. Pass screen ratios when the video has been sampled."""
    turns = parse_vtt(vtt_path)
    if not turns:
        raise RuntimeError(f"No cues in {vtt_path}")
    duration = turns[-1].end
    speakers = assign_roles(turns, duration)
    return assess(
        turns,
        speakers,
        duration=duration,
        visual=visual or {},
        language="en",
        source=Path(vtt_path).name,
    )


def _pair_video(vtt: Path) -> Path | None:
    videos = sorted(vtt.parent.glob("*.mp4"))
    stem = vtt.name.split()[0].lower()
    for video in videos:
        if video.name.split()[0].lower() == stem:
            return video
    return videos[0] if len(videos) == 1 else None


def check_library(root: Path, visuals: dict[str, dict] | None = None) -> list[dict]:
    """Score every labelled transcript under `root`.

    `visuals` maps a batch id such as "B491" to screen ratios. When it is omitted,
    each paired video is sampled.
    """
    rows = []
    for vtt in sorted(root.rglob("*.vtt")):
        folder = vtt.parent.name.strip()
        expected = sheet_grade(folder)
        if expected is None and "quality" not in folder.lower():
            continue
        batch = vtt.name.split()[0]
        visual = (visuals or {}).get(batch)
        if visual is None and visuals is None:
            video = _pair_video(vtt)
            if video is not None:
                from class_assessor.media import analyze_frames, probe

                meta = probe(str(video))
                visual = analyze_frames(str(video), meta["width"], meta["height"], interval=20)
        result = score_transcript(str(vtt), visual)
        flagged = any("quality issue" in item.detail.lower() for item in result.observations)
        if expected is None:
            ok = result.grade in {"Poor", "Average"} and flagged
            expected_label = "Quality issue"
        else:
            ok = result.grade == expected
            expected_label = "Weak" if expected == "Poor" and folder.lower().startswith("weak") else expected
        rows.append(
            {
                "folder": folder,
                "batch": batch,
                "expected": expected_label,
                "grade": result.grade,
                "score": result.score,
                "ok": ok,
                "source": vtt.name,
            }
        )
    return rows


def format_report(rows: list[dict]) -> str:
    lines = ["Feedback sheet check", ""]
    matched = 0
    for row in rows:
        mark = "ok" if row["ok"] else "MISS"
        if row["ok"]:
            matched += 1
        lines.append(
            f"{mark:4} {row['folder'][:16]:16} {row['batch']:6} "
            f"sheet={row['expected']:14} app={row['grade']:10} {row['score']:5.1f}"
        )
    lines.append("")
    lines.append(f"{matched}/{len(rows)} match the feedback sheet.")
    return "\n".join(lines)
