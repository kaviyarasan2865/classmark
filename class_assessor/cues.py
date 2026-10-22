"""Shared turn representation and Zoom VTT parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass


_TS = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})[.,](?P<ms>\d{3})"
)
_CUE_LINE = re.compile(
    r"(?P<a>\d{2}:\d{2}:\d{2}[.,]\d{3})\s-->\s(?P<b>\d{2}:\d{2}:\d{2}[.,]\d{3})"
)
_WORD = re.compile(r"[A-Za-z']+")


@dataclass
class Turn:
    start: float
    end: float
    speaker: str
    text: str
    avg_logprob: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def parse_timestamp(value: str) -> float:
    match = _TS.search(value)
    if not match:
        raise ValueError(f"Bad timestamp: {value}")
    return (
        int(match.group("h")) * 3600
        + int(match.group("m")) * 60
        + int(match.group("s"))
        + int(match.group("ms")) / 1000
    )


def words(text: str) -> list[str]:
    return _WORD.findall(text or "")


def is_shared_media_name(name: str) -> bool:
    return "audio shared" in (name or "").lower()


def parse_vtt(path: str) -> list[Turn]:
    """Parse a Zoom-style WebVTT file into timed speaker turns."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        raw = handle.read()
    turns: list[Turn] = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        cue_index = next((i for i, line in enumerate(lines) if _CUE_LINE.search(line)), None)
        if cue_index is None:
            continue
        match = _CUE_LINE.search(lines[cue_index])
        if not match:
            continue
        body = " ".join(lines[cue_index + 1 :]).strip()
        if not body:
            continue
        speaker, text = "Unknown", body
        if ":" in body:
            label, rest = body.split(":", 1)
            if 0 < len(label) <= 48 and "\n" not in label:
                speaker, text = label.strip(), rest.strip()
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        turns.append(
            Turn(
                start=parse_timestamp(match.group("a")),
                end=parse_timestamp(match.group("b")),
                speaker=speaker,
                text=text,
            )
        )
    turns.sort(key=lambda turn: (turn.start, turn.end))
    return _merge_spellings(turns)


def _merge_spellings(turns: list[Turn]) -> list[Turn]:
    """Zoom sometimes writes the same child as 'Shrithik' and 'shrithik'."""
    counts: dict[str, dict[str, int]] = {}
    for turn in turns:
        bucket = counts.setdefault(turn.speaker.casefold(), {})
        bucket[turn.speaker] = bucket.get(turn.speaker, 0) + 1
    chosen = {key: max(spellings, key=spellings.get) for key, spellings in counts.items()}
    for turn in turns:
        turn.speaker = chosen[turn.speaker.casefold()]
    return turns


def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
