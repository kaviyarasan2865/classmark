"""Run a full recording through audio, screen, speech, roles, and the rubric."""

from __future__ import annotations

import os
import tempfile
from typing import Callable

from class_assessor.asr import transcribe
from class_assessor.cues import parse_vtt
from class_assessor.media import (
    analyze_frames,
    audio_profile,
    extract_wav,
    probe,
    read_wav_int16,
)
from class_assessor.judge import apply_judge
from class_assessor.score import Assessment, assess
from class_assessor.speakers import assign_roles, label_voices

Progress = Callable[[float, str], None]


def _noop(fraction: float, message: str) -> None:
    return None


def assess_media(
    path: str,
    vtt_path: str | None = None,
    model_name: str = "small",
    beam_size: int = 1,
    frame_interval: float = 15.0,
    max_seconds: float | None = None,
    skip_visual: bool = False,
    on_progress: Progress | None = None,
) -> Assessment:
    """Assess one class file from start to finish.

    `max_seconds` exists for a quick local check. The default is the whole file,
    including hour-long and two-hour recordings.
    """
    report = on_progress or _noop
    meta = probe(path)
    if not meta["has_audio"]:
        raise RuntimeError("This file has no audio track, so there is nothing to assess.")
    duration = meta["duration"]
    if max_seconds:
        duration = min(duration, max_seconds)

    report(0.02, "Extracting the full soundtrack")
    with tempfile.TemporaryDirectory(prefix="class-assess-") as temp:
        wav_path = os.path.join(temp, "audio.wav")
        extract_wav(path, wav_path, max_seconds=max_seconds)
        report(0.08, "Measuring silence, level, and dead air")
        samples, sample_rate = read_wav_int16(wav_path)
        audio = audio_profile(samples, sample_rate)
        if max_seconds and meta["duration"] > max_seconds:
            audio["truncated_to"] = max_seconds

        visual: dict = {"frames": 0}
        if meta["has_video"] and not skip_visual:
            report(0.12, "Sampling the screen across the whole class")

            def visual_progress(fraction: float, message: str) -> None:
                report(0.12 + 0.16 * fraction, message)

            visual = analyze_frames(
                path,
                meta["width"],
                meta["height"],
                interval=frame_interval,
                max_seconds=max_seconds,
                on_progress=visual_progress,
            )

        if vtt_path:
            report(0.30, "Reading the Zoom transcript and keeping its speaker names")
            turns = parse_vtt(vtt_path)
            if max_seconds:
                turns = [turn for turn in turns if turn.start < max_seconds]
            language = "en"
        else:
            report(0.30, "Transcribing the full class on this machine")

            def asr_progress(fraction: float, message: str) -> None:
                report(0.30 + 0.50 * fraction, message)

            turns, language = transcribe(
                wav_path,
                duration,
                model_name=model_name,
                beam_size=beam_size,
                on_progress=asr_progress,
            )

        report(0.84, "Separating teacher, students, and other voices")
        turns = label_voices(turns, samples, sample_rate)
        speakers = assign_roles(turns, duration, samples, sample_rate)
        report(0.90, "Measuring the class")
        result = assess(
            turns,
            speakers,
            duration=duration,
            audio=audio,
            visual=visual,
            language=language,
            source=os.path.basename(path),
        )
        report(0.94, "Asking the judge")
        result = apply_judge(result)
        report(1.0, f"{result.grade} ({result.score:.0f}/100)")
        return result


def assess_transcript(
    vtt_path: str,
    duration: float | None = None,
    on_progress: Progress | None = None,
) -> Assessment:
    """Score a Zoom transcript when the video itself is not being decoded."""
    turns = parse_vtt(vtt_path)
    if not turns:
        raise RuntimeError("No cues in this transcript.")
    span = duration or turns[-1].end
    speakers = assign_roles(turns, span)
    result = assess(
        turns,
        speakers,
        duration=span,
        language="en",
        source=os.path.basename(vtt_path),
    )
    if on_progress:
        on_progress(0.9, "Asking the judge")
    return apply_judge(result)
