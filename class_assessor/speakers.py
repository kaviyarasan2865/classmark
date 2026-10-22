"""Separate voices, then decide who is teaching, who is in the class, and who is not.

Acoustic clustering answers "how many voices". Roles are decided afterwards from
behaviour: who asks the questions, who answers, who only shows up to say hello,
and whose pitch sits in a child's range rather than an adult's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from class_assessor.cues import Turn, is_shared_media_name, words

LESSON = re.compile(
    r"\b(king|queen|rook|bishop|knight|pawn|checkmate|check|castle|castling|"
    r"board|square|file|rank|piece|capture|pin|fork|skewer|puzzle|lichess|"
    r"opening|position|zugzwang|stalemate|en passant)\b",
    re.I,
)
LOGISTICS = re.compile(
    r"\b(zoom|wifi|wi-fi|network|mute|unmute|hear me|can't hear|cannot hear|"
    r"laptop|reconnect|you're on mute|louder|not clear)\b",
    re.I,
)
QUESTION_WORD = re.compile(
    r"\b(what|why|how|which|who|where|can you|could you|tell me|your move|show me)\b",
    re.I,
)
PRAISE = re.compile(
    r"\b(good|great|nice|well done|excellent|correct|awesome|brilliant|perfect|"
    r"super|well played)\b",
    re.I,
)
GREETING_PRAISE = re.compile(r"\bgood (evening|morning|night|afternoon|one)\b", re.I)


@dataclass
class Speaker:
    id: str
    role: str
    talk_time: float
    turns: int
    median_turn: float
    f0: float | None
    voicing: float
    question_ratio: float
    lesson_ratio: float
    logistics_ratio: float
    praise_count: int
    coverage: float
    edge_ratio: float
    follow_ratio: float
    reasons: list[str] = field(default_factory=list)
    quotes: list[tuple[float, str]] = field(default_factory=list)

    def label(self) -> str:
        names = {
            "teacher": "Teacher",
            "student": "Student",
            "audience": "Audience",
            "shared_media": "Shared audio",
            "background": "Background",
        }
        return f"{names.get(self.role, self.role)} · {self.id}"


def _praise_count(text: str) -> int:
    cleaned = GREETING_PRAISE.sub(" ", text or "")
    return len(PRAISE.findall(cleaned))


def _is_question(text: str) -> bool:
    if "?" in (text or ""):
        return True
    return QUESTION_WORD.search(text or "") is not None


def _embed(chunk: np.ndarray, sample_rate: int) -> np.ndarray:
    """Pitch and spectrum only. Text is applied later, when roles are chosen."""
    if len(chunk) > sample_rate * 6:
        win = sample_rate
        hop = sample_rate // 2
        best_at = 0
        best_energy = -1.0
        usable = chunk.astype(np.float32)
        for start in range(0, len(usable) - win, hop):
            energy = float(np.dot(usable[start : start + win], usable[start : start + win]))
            if energy > best_energy:
                best_energy = energy
                best_at = start
        chunk = chunk[best_at : best_at + sample_rate * 6]
    audio = chunk.astype(np.float32) / 32768.0
    f0, voiced = _pitch(audio, sample_rate)
    bands, centroid, flatness, rms_db = _spectrum(audio, sample_rate)
    vector = np.zeros(13, dtype=np.float32)
    vector[0] = np.log(f0) if f0 else 0.0
    vector[1] = 1.0 if f0 else 0.0
    vector[2] = voiced
    vector[3] = centroid
    vector[4] = flatness
    vector[5] = rms_db
    vector[6:13] = bands[:7]
    return vector


def _pitch(audio: np.ndarray, sample_rate: int) -> tuple[float | None, float]:
    if len(audio) < int(sample_rate * 0.3):
        return None, 0.0
    win = int(0.04 * sample_rate)
    hop = int(0.02 * sample_rate)
    min_lag = int(sample_rate / 420)
    max_lag = int(sample_rate / 75)
    window = np.hanning(win).astype(np.float32)
    found: list[float] = []
    confs: list[float] = []
    limit = min(len(audio) - win, sample_rate * 5)
    for start in range(0, max(0, limit), hop):
        frame = audio[start : start + win] * window
        energy = float(np.dot(frame, frame))
        if energy < 1e-4:
            continue
        spectrum = np.fft.rfft(frame, n=win * 2)
        corr = np.fft.irfft(spectrum * np.conj(spectrum))[: max_lag + 1]
        corr = corr / (corr[0] + 1e-8)
        segment = corr[min_lag:max_lag]
        if len(segment) == 0:
            continue
        peak = int(np.argmax(segment))
        conf = float(segment[peak])
        if conf < 0.40:
            continue
        found.append(sample_rate / float(min_lag + peak))
        confs.append(conf)
    if len(found) < 3:
        return None, 0.0
    return float(np.median(found)), float(np.mean(confs))


def _spectrum(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float, float, float]:
    bands = np.zeros(7, dtype=np.float32)
    if len(audio) < 400:
        return bands, 0.0, 0.0, -60.0
    n = min(len(audio), sample_rate * 2)
    frame = audio[:n] * np.hanning(n)
    mag = np.abs(np.fft.rfft(frame)).astype(np.float32) + 1e-8
    freqs = np.fft.rfftfreq(n, 1 / sample_rate)
    centroid = float((freqs * mag).sum() / mag.sum()) / (sample_rate / 2)
    flatness = float(np.exp(np.log(mag).mean()) / mag.mean())
    rms_db = float(20 * np.log10(np.sqrt(np.mean(frame**2) + 1e-12)))
    edges = np.geomspace(80, min(sample_rate / 2, 7600), 8)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        selected = mag[(freqs >= lo) & (freqs < hi)]
        bands[i] = np.log(selected.mean() + 1e-8) if len(selected) else 0.0
    return bands, centroid, flatness, rms_db


def _cluster(features: np.ndarray) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    if len(features) < 4:
        return np.zeros(len(features), dtype=int)
    clean = np.nan_to_num(features)
    std = clean.std(axis=0)
    std[std < 1e-6] = 1.0
    scaled = (clean - clean.mean(axis=0)) / std

    def fit(threshold: float | None = None, k: int | None = None) -> np.ndarray:
        kwargs = {"linkage": "average"}
        if k is None:
            model = AgglomerativeClustering(
                n_clusters=None, distance_threshold=threshold, metric="euclidean", **kwargs
            )
        else:
            model = AgglomerativeClustering(n_clusters=k, metric="euclidean", **kwargs)
        return model.fit_predict(scaled)

    labels = fit(threshold=1.45)
    distinct = len(set(labels.tolist()))
    if distinct > 8:
        labels = fit(threshold=1.95)
        distinct = len(set(labels.tolist()))
    if distinct > 8:
        labels = fit(k=min(6, len(scaled)))
        distinct = len(set(labels.tolist()))
    if distinct == 1 and len(scaled) >= 16:
        labels = fit(threshold=1.05)
    return labels


def _merge_short_voices(labels: np.ndarray, features: np.ndarray, durations: list[float]) -> np.ndarray:
    """Fold a voice with only a few seconds into the nearest longer voice.

    A single short line ("very nice") should not become its own person when it
    sits next to a voice that already has a minute of speech.
    """
    labels = labels.copy()
    clean = np.nan_to_num(features)
    std = clean.std(axis=0)
    std[std < 1e-6] = 1.0
    scaled = (clean - clean.mean(axis=0)) / std
    talk: dict[int, float] = {}
    for label, duration in zip(labels, durations):
        talk[int(label)] = talk.get(int(label), 0.0) + float(duration)
    large = [label for label, seconds in talk.items() if seconds >= 12]
    if not large:
        return labels
    centroids = {label: scaled[labels == label].mean(axis=0) for label in large}
    for label, seconds in list(talk.items()):
        if seconds >= 12 or label in centroids and seconds >= 12:
            continue
        if label not in talk:
            continue
        rows = scaled[labels == label]
        if len(rows) == 0:
            continue
        centre = rows.mean(axis=0)
        nearest = min(centroids, key=lambda other: float(np.linalg.norm(centre - centroids[other])))
        labels[labels == label] = nearest
    return labels


def label_voices(
    turns: list[Turn],
    samples: np.ndarray | None = None,
    sample_rate: int = 16000,
) -> list[Turn]:
    """Replace speaker ids using voice clustering when the audio is a single mix.

    Zoom VTT names are left untouched. Clustering runs only for transcript lines
    that do not already name a person.
    """
    if samples is None or not turns:
        return turns
    named = any(
        turn.speaker not in {"Unknown", "unknown"} and not turn.speaker.startswith("spk_")
        for turn in turns
    )
    if named:
        return turns

    usable: list[int] = []
    vectors: list[np.ndarray] = []
    for index, turn in enumerate(turns):
        a = max(0, int(turn.start * sample_rate))
        b = min(len(samples), int(turn.end * sample_rate))
        if b - a < int(0.35 * sample_rate):
            continue
        vectors.append(_embed(samples[a:b], sample_rate))
        usable.append(index)
    if len(vectors) < 2:
        for turn in turns:
            turn.speaker = "spk_1"
        return turns
    matrix = np.vstack(vectors)
    labels = _cluster(matrix)
    labels = _merge_short_voices(labels, matrix, [turns[index].duration for index in usable])
    order: dict[int, str] = {}
    next_id = 1
    for index, label in zip(usable, labels):
        key = int(label)
        if key not in order:
            order[key] = f"spk_{next_id}"
            next_id += 1
        turns[index].speaker = order[key]
    known = [turn for turn in turns if turn.speaker.startswith("spk_")]
    for turn in turns:
        if turn.speaker.startswith("spk_"):
            continue
        nearest = min(
            known,
            key=lambda other: min(abs(other.start - turn.end), abs(turn.start - other.end)),
            default=None,
        )
        turn.speaker = nearest.speaker if nearest else "spk_1"
    return turns


def _quotes_for(own: list[Turn]) -> list[tuple[float, str]]:
    ranked = sorted(own, key=lambda turn: (LESSON.search(turn.text) is None, -len(words(turn.text))))
    picked: list[tuple[float, str]] = []
    for turn in ranked:
        snippet = turn.text.strip()
        if len(snippet) < 8:
            continue
        if any(snippet == prev for _, prev in picked):
            continue
        picked.append((turn.start, snippet[:180]))
        if len(picked) == 2:
            break
    return picked


def assign_roles(
    turns: list[Turn],
    duration: float,
    samples: np.ndarray | None = None,
    sample_rate: int = 16000,
) -> list[Speaker]:
    """Decide teacher, student, audience, or shared audio for each speaker id."""
    if not turns:
        return []
    duration = max(duration, turns[-1].end, 1.0)
    by_id: dict[str, list[Turn]] = {}
    for turn in turns:
        by_id.setdefault(turn.speaker, []).append(turn)

    # A question from someone else, used to see who actually answers.
    question_ends = [turn.end for turn in turns if _is_question(turn.text)]

    profiles: list[Speaker] = []
    for speaker_id, own in by_id.items():
        durations = [turn.duration for turn in own]
        talk = float(sum(durations))
        texts = [turn.text for turn in own]
        joined = " ".join(texts)
        token_count = max(1, len(words(joined)))
        lesson_hits = len(LESSON.findall(joined))
        logistics_hits = len(LOGISTICS.findall(joined))
        questions = sum(1 for text in texts if _is_question(text))
        span_start = min(turn.start for turn in own)
        span_end = max(turn.end for turn in own)
        edge_time = 0.0
        edge = duration * 0.08
        for turn in own:
            if turn.start < edge or turn.end > duration - edge:
                edge_time += turn.duration
        follows = 0
        for turn in own:
            if any(0 <= turn.start - end <= 8 for end in question_ends if abs(end - turn.end) > 0.05):
                # Ignore a speaker "answering" their own question.
                prior = [
                    other
                    for other in turns
                    if other.speaker != speaker_id and 0 <= turn.start - other.end <= 8 and _is_question(other.text)
                ]
                if prior:
                    follows += 1
        profiles.append(
            Speaker(
                id=speaker_id,
                role="",
                talk_time=talk,
                turns=len(own),
                median_turn=float(np.median(durations)) if durations else 0.0,
                f0=None,
                voicing=-1.0,
                question_ratio=questions / max(1, len(own)),
                lesson_ratio=lesson_hits / token_count,
                logistics_ratio=logistics_hits / token_count,
                praise_count=_praise_count(joined),
                coverage=min(1.0, (span_end - span_start) / duration),
                edge_ratio=edge_time / talk if talk else 0.0,
                follow_ratio=follows / max(1, len(own)),
                quotes=_quotes_for(own),
            )
        )
    if samples is not None:
        attach_pitch_from_audio(profiles, turns, samples, sample_rate)
    _choose_roles(profiles, duration)
    return profiles


def attach_pitch_from_audio(
    profiles: list[Speaker],
    turns: list[Turn],
    samples: np.ndarray,
    sample_rate: int,
) -> None:
    by_id: dict[str, list[Turn]] = {}
    for turn in turns:
        by_id.setdefault(turn.speaker, []).append(turn)
    for profile in profiles:
        own = sorted(by_id.get(profile.id, []), key=lambda turn: turn.duration, reverse=True)[:8]
        f0s: list[float] = []
        confs: list[float] = []
        for turn in own:
            if turn.duration < 0.6:
                continue
            a = max(0, int(turn.start * sample_rate))
            b = min(len(samples), int(min(turn.end, turn.start + 5) * sample_rate))
            if b - a < int(0.4 * sample_rate):
                continue
            audio = samples[a:b].astype(np.float32) / 32768.0
            f0, conf = _pitch(audio, sample_rate)
            if f0:
                f0s.append(f0)
                confs.append(conf)
        if f0s:
            profile.f0 = float(np.median(f0s))
            profile.voicing = float(np.mean(confs))


def _choose_roles(profiles: list[Speaker], duration: float) -> None:
    if not profiles:
        return
    total_talk = sum(profile.talk_time for profile in profiles) or 1.0
    for profile in profiles:
        if is_shared_media_name(profile.id):
            profile.role = "shared_media"
            profile.reasons.append(
                "Zoom marks this as audio shared from the screen, so it is a video or board, not a person in the call."
            )

    def teacher_score(profile: Speaker) -> float:
        talk_share = profile.talk_time / total_talk
        questions = min(1.0, profile.question_ratio / 0.34)
        lesson = min(1.0, profile.lesson_ratio / 0.18)
        turns_score = min(1.0, profile.turns / 40)
        return 0.36 * talk_share + 0.30 * questions + 0.16 * lesson + 0.12 * profile.coverage + 0.06 * turns_score

    live = [profile for profile in profiles if profile.role != "shared_media" and profile.talk_time >= 8]
    pool = live or [profile for profile in profiles if profile.role != "shared_media"] or profiles
    teacher = max(pool, key=teacher_score)
    teacher.role = "teacher"
    teacher.reasons.append(
        f"Speaks for {teacher.talk_time / 60:.1f} min ({teacher.talk_time / total_talk:.0%} of speech), "
        f"asks questions in {teacher.question_ratio:.0%} of their turns, and stays present for {teacher.coverage:.0%} of the class."
    )
    if teacher.f0:
        teacher.reasons.append(f"Median pitch is about {teacher.f0:.0f} Hz.")

    others = [profile for profile in profiles if profile.role == ""]
    # A second adult who lectures in long stretches without taking questions is
    # usually a lesson video playing through screen share, mixed into the call.
    for profile in others:
        different_pitch = (
            profile.f0 is not None
            and teacher.f0 is not None
            and abs(profile.f0 - teacher.f0) > 25
        )
        scripted = (
            profile.median_turn >= 7
            and profile.question_ratio < 0.06
            and teacher.question_ratio > profile.question_ratio + 0.08
            and profile.follow_ratio < 0.15
            and (different_pitch or (0 <= profile.voicing < 0.45))
        )
        playback = (
            0 <= profile.voicing < 0.30
            and profile.talk_time > 25
            and profile.follow_ratio < 0.2
        )
        if scripted or playback:
            profile.role = "shared_media"
            profile.reasons.append(
                "Long turns, almost no questions, and little back-and-forth. Treated as played lesson audio rather than the live teacher."
            )

    remaining = [profile for profile in profiles if profile.role == ""]
    only_companion = len(remaining) == 1
    for profile in remaining:
        child = profile.f0 is not None and profile.f0 >= 205
        adultish = profile.f0 is not None and profile.f0 < 190
        if profile.talk_time < 15 and not only_companion:
            profile.role = "audience"
            profile.reasons.append(
                f"Only {profile.talk_time:.0f}s of speech, which is too little to count as someone taking the class."
            )
            continue
        if (
            profile.edge_ratio > 0.55
            and profile.talk_time < 80
            and profile.lesson_ratio < 0.08
            and profile.follow_ratio < 0.25
            and not only_companion
        ):
            profile.role = "audience"
            profile.reasons.append(
                "Most of this speech is at the very start or end (joining or leaving), not during the lesson."
            )
            continue
        if profile.logistics_ratio > 0.015 and profile.lesson_ratio < 0.05 and profile.talk_time < 180 and profile.follow_ratio < 0.35:
            profile.role = "audience"
            profile.reasons.append(
                "The lines are about hearing, mute, or the connection, and they rarely answer a lesson question."
            )
            continue
        if (
            adultish
            and not child
            and profile.talk_time < 150
            and profile.lesson_ratio < 0.08
            and profile.follow_ratio < 0.28
            and not only_companion
        ):
            profile.role = "audience"
            profile.reasons.append(
                f"Adult pitch near {profile.f0:.0f} Hz, little chess language, and few answers. This fits a parent or someone beside the student."
            )
            continue
        profile.role = "student"
        bits = [f"{profile.talk_time / 60:.1f} min across {profile.turns} turns"]
        if child and profile.f0:
            bits.append(f"pitch near {profile.f0:.0f} Hz, in a child's range")
        elif profile.f0:
            bits.append(f"pitch near {profile.f0:.0f} Hz")
        if profile.follow_ratio >= 0.2:
            bits.append(f"{profile.follow_ratio:.0%} of turns come right after a question")
        profile.reasons.append("Treated as a student: " + "; ".join(bits) + ".")

    for profile in profiles:
        if not profile.role:
            profile.role = "background"
            profile.reasons.append("Not enough evidence to place this voice in the class.")
