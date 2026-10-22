"""Turn measured class behaviour into a grade and a specific explanation.

The grade is a weighted rubric, not a trained classifier. Each point can be
traced to a timestamp, a speaker, or a screen measurement. Bands were checked
against the labelled sample classes that ship with the task (excellent, good,
average, weak). A technical-quality problem can pull a class down even when
the teaching talk looks fine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from class_assessor.cues import Turn, fmt_ts, is_shared_media_name, words
from class_assessor.speakers import Speaker, _is_question, _praise_count

GREETING = re.compile(
    r"\b(hello|hi|good evening|good morning|good afternoon|welcome|namaskaram|namaste)\b",
    re.I,
)
CLOSING = re.compile(
    r"\b(bye|goodbye|see you|next class|homework|that's all|class is over|thank you)\b",
    re.I,
)
TECH = re.compile(
    r"\b(can you hear|can't hear|cannot hear|louder|not clear|breaking up|"
    r"network|wifi|wi-fi|frozen|reconnect|you're on mute)\b",
    re.I,
)
HEAR = re.compile(r"\b(can you hear|can't hear|cannot hear|hear me)\b", re.I)
WARM = re.compile(
    r"\b(how are you|how was your day|very good|very nice|well done|awesome|"
    r"proud of you|beautiful|excellent|wonderful|good job|clap)\b",
    re.I,
)
HW_GIVEN = re.compile(
    r"\b(homework|home work|this will be your homework|your homework|practice this)\b",
    re.I,
)

WEIGHTS = {
    "engagement": 0.24,
    "interaction": 0.30,
    "clarity": 0.16,
    "technical": 0.18,
    "management": 0.12,
}


@dataclass
class Observation:
    kind: str
    title: str
    detail: str


@dataclass
class Assessment:
    grade: str
    score: float
    subscores: dict[str, float]
    summary: str
    observations: list[Observation]
    speakers: list[Speaker]
    duration: float
    language: str
    audio: dict
    visual: dict
    turns: list[Turn] = field(default_factory=list)
    source: str = ""
    judge: str = ""
    measured_grade: str = ""
    measured_score: float = 0.0

    def to_markdown(self) -> str:
        lines = [
            f"# Class assessment: {self.grade} ({self.score:.0f}/100)",
            "",
        ]
        if self.judge:
            lines += [f"Judge: {self.judge}", ""]
        lines += [
            self.summary,
            "",
            "## Scores",
        ]
        for name, value in self.subscores.items():
            lines.append(f"- {name}: {value:.0f}/100")
        lines += ["", "## Speakers"]
        for speaker in self.speakers:
            lines.append(
                f"- **{speaker.label()}** — {speaker.talk_time / 60:.1f} min, {speaker.turns} turns. "
                + " ".join(speaker.reasons)
            )
            for stamp, quote in speaker.quotes:
                lines.append(f"  - {fmt_ts(stamp)}: \"{quote}\"")
        lines += ["", "## Observations"]
        for item in self.observations:
            lines.append(f"- **{item.title}** ({item.kind}). {item.detail}")
        return "\n".join(lines) + "\n"


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _ramp(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 1.0
    return _clip((value - lo) / (hi - lo))


def _down(value: float, good: float, bad: float) -> float:
    """1 at or below `good`, 0 at or above `bad`."""
    return 1.0 - _ramp(value, good, bad)


def _peak(value: float, center: float, width: float) -> float:
    return _clip(1.0 - abs(value - center) / width)


def _role(speakers: list[Speaker], role: str) -> list[Speaker]:
    return [speaker for speaker in speakers if speaker.role == role]


def _longest_teacher_run(turns: list[Turn], teacher_id: str) -> tuple[float, float, float]:
    best = (0.0, 0.0, 0.0)
    active = False
    start = end = 0.0
    for turn in turns:
        if turn.speaker != teacher_id or is_shared_media_name(turn.speaker):
            if active:
                dur = end - start
                if dur > best[0]:
                    best = (dur, start, end)
                active = False
            continue
        if active and turn.start - end < 4:
            end = turn.end
        else:
            if active:
                dur = end - start
                if dur > best[0]:
                    best = (dur, start, end)
            start, end = turn.start, turn.end
            active = True
    if active and end - start > best[0]:
        best = (end - start, start, end)
    return best


def _quote_at(turns: list[Turn], speaker_id: str, start: float, end: float) -> str:
    chunk = [
        turn.text
        for turn in turns
        if turn.speaker == speaker_id and turn.start >= start - 0.5 and turn.end <= end + 0.5
    ]
    text = " ".join(chunk).strip()
    return text[:180]


def grade_for(score: float) -> str:
    if score >= 78:
        return "Excellent"
    if score >= 63:
        return "Good"
    if score >= 46:
        return "Average"
    return "Poor"


def assess(
    turns: list[Turn],
    speakers: list[Speaker],
    duration: float,
    audio: dict | None = None,
    visual: dict | None = None,
    language: str = "en",
    source: str = "",
) -> Assessment:
    audio = audio or {}
    visual = visual or {}
    duration = max(duration, turns[-1].end if turns else 0.0, 1.0)
    teachers = _role(speakers, "teacher")
    students = _role(speakers, "student")
    audience = _role(speakers, "audience")
    media = _role(speakers, "shared_media")
    teacher = max(teachers, key=lambda item: item.talk_time) if teachers else None

    speech_time = sum(turn.duration for turn in turns) or 0.0
    teacher_turns = [turn for turn in turns if teacher and turn.speaker == teacher.id]
    student_turns = [turn for turn in turns if any(turn.speaker == student.id for student in students)]
    teacher_talk = sum(turn.duration for turn in teacher_turns)
    student_talk = sum(turn.duration for turn in student_turns)
    audience_talk = sum(speaker.talk_time for speaker in audience)
    media_talk = sum(speaker.talk_time for speaker in media)

    speech_ratio = float(audio.get("speech_ratio", speech_time / duration if duration else 0.0))
    first_speech = float(audio.get("first_speech", turns[0].start if turns else 0.0))
    long_gaps = list(audio.get("long_gaps") or [])
    if not long_gaps and turns:
        for prev, nxt in zip(turns, turns[1:]):
            gap = nxt.start - prev.end
            if gap >= 12:
                long_gaps.append({"start": prev.end, "end": nxt.start, "dur": gap})
    max_gap = float(audio.get("max_gap", max((gap["dur"] for gap in long_gaps), default=0.0)))
    minutes = duration / 60
    gaps_per_10 = (len([gap for gap in long_gaps if gap["dur"] >= 15]) / minutes * 10) if minutes else 0

    teacher_questions = [turn for turn in teacher_turns if _is_question(turn.text)]
    answered = 0
    for turn in teacher_questions:
        for other in student_turns:
            if 0 <= other.start - turn.end <= 12 and len(words(other.text)) >= 1:
                answered += 1
                break
    answer_rate = answered / len(teacher_questions) if teacher_questions else 0.0
    qpm = len(teacher_questions) / minutes if minutes else 0.0
    praise = teacher.praise_count if teacher else sum(_praise_count(turn.text) for turn in teacher_turns)
    praise_per_min = praise / minutes if minutes else 0.0
    mono_dur, mono_start, mono_end = _longest_teacher_run(turns, teacher.id) if teacher else (0.0, 0.0, 0.0)

    switches = 0
    student_ids = {student.id for student in students}
    prev_side = None
    for turn in turns:
        if teacher and turn.speaker == teacher.id:
            side = "t"
        elif turn.speaker in student_ids:
            side = "s"
        else:
            continue
        if prev_side and side != prev_side:
            switches += 1
        prev_side = side
    switches_per_min = switches / minutes if minutes else 0.0

    teacher_word_counts = [len(words(turn.text)) for turn in teacher_turns]
    fragment_ratio = (
        sum(1 for count in teacher_word_counts if count <= 3) / len(teacher_word_counts)
        if teacher_word_counts
        else 0.0
    )
    all_words = words(" ".join(turn.text for turn in turns))
    lesson_hits = len(
        re.findall(
            r"\b(king|queen|rook|bishop|knight|pawn|check|mate|castle|board|puzzle|position|piece|move)\b",
            " ".join(all_words),
            flags=re.I,
        )
    )
    lesson_ratio = lesson_hits / max(1, len(all_words))
    teacher_wpm = (
        (sum(teacher_word_counts) / (teacher_talk / 60)) if teacher_talk > 1 else 0.0
    )
    english = language.lower().startswith("en")

    tech_hits = []
    for turn in turns:
        if TECH.search(turn.text):
            tech_hits.append(turn)
    complaints_per_10 = (len(tech_hits) / minutes * 10) if minutes else 0

    student_share = student_talk / speech_time if speech_time else 0.0
    active_students = [student for student in students if student.talk_time >= 20]

    # --- subscores in 0..1, then * 100 ---
    speech_score = _ramp(speech_ratio, 0.28, 0.66)
    gap_score = _down(max_gap, 30, 200)
    late_score = _down(first_speech, 40, 200)
    share_score = _peak(student_share, 0.40, 0.38)
    if answer_rate >= 0.8 and student_share < 0.2:
        share_score = max(share_score, 0.55)
    dead_air = _down(gaps_per_10, 1.0, 6.0)
    engagement = (
        0.40 * speech_score + 0.25 * gap_score + 0.15 * late_score + 0.12 * share_score + 0.08 * dead_air
    )

    if len(teacher_questions) < 8:
        answer_score = min(answer_rate, 0.35)
    else:
        answer_score = answer_rate
    q_score = _ramp(qpm, 0.7, 3.3)
    praise_score = _ramp(praise_per_min, 0.05, 0.65)
    mono_score = _down(mono_dur, 35, 180)
    switch_score = _ramp(switches_per_min, 1.2, 8.0)
    interaction = (
        0.32 * answer_score
        + 0.22 * q_score
        + 0.14 * praise_score
        + 0.22 * mono_score
        + 0.10 * switch_score
    )

    if english:
        fragment_score = _down(fragment_ratio, 0.22, 0.55)
        wpm_score = _peak(teacher_wpm, 145, 90) if teacher_wpm else 0.45
        lesson_score = _ramp(lesson_ratio, 0.01, 0.06)
    else:
        fragment_score = 0.6
        wpm_score = 0.6
        lesson_score = 0.55 if lesson_ratio < 0.01 else _ramp(lesson_ratio, 0.0, 0.04)
    clarity = 0.45 * fragment_score + 0.30 * lesson_score + 0.25 * wpm_score

    complaint_score = _down(complaints_per_10, 0.4, 4.0)
    level = audio.get("mean_speech_db")
    if level is None:
        level_score = 0.75
    elif level < -35:
        level_score = 0.25
    elif level < -28:
        level_score = 0.55
    else:
        level_score = 1.0
    clip = float(audio.get("clip_ratio") or 0)
    clip_score = _down(clip, 0.002, 0.03)
    tech_parts = [complaint_score, level_score, clip_score]
    if visual.get("frames"):
        board = _ramp(float(visual.get("board_ratio") or 0), 0.1, 0.65)
        face = _ramp(float(visual.get("face_ratio") or 0), 0.05, 0.45)
        empty = _down(float(visual.get("empty_ratio") or 0), 0.15, 0.55)
        freeze = _down(float(visual.get("freeze_ratio") or 0), 0.05, 0.35)
        tech_parts.append(0.45 * board + 0.30 * face + 0.25 * empty)
        tech_parts.append(freeze)
    technical = sum(tech_parts) / len(tech_parts)

    if not students and not (teacher and answer_rate > 0.5):
        roster = 0.15
    elif len(active_students) == 1 and answer_rate >= 0.75:
        roster = 0.82
    elif len(active_students) >= 2:
        roster = 1.0
    elif students:
        roster = 0.55
    else:
        roster = 0.3
    if len(active_students) >= 3 and student_talk > 0:
        biggest = max(student.talk_time for student in active_students) / student_talk
        balance = 0.55 if biggest > 0.85 else 1.0
    else:
        balance = 1.0
    early = " ".join(turn.text for turn in turns if turn.start < 180)
    late_text = " ".join(turn.text for turn in turns if turn.end > duration - 300)
    greeting = 1.0 if GREETING.search(early) else 0.35
    closing = 1.0 if CLOSING.search(late_text) else 0.45
    audience_share = audience_talk / speech_time if speech_time else 0.0
    audience_score = 0.55 if audience_share > 0.12 else 1.0
    management = (roster + balance + greeting + closing + audience_score) / 5

    subs = {
        "engagement": round(100 * engagement, 1),
        "interaction": round(100 * interaction, 1),
        "clarity": round(100 * clarity, 1),
        "technical": round(100 * technical, 1),
        "management": round(100 * management, 1),
    }
    score = sum(subs[name] * weight for name, weight in WEIGHTS.items())

    observations: list[Observation] = []
    if teacher:
        observations.append(
            Observation(
                "speaker",
                "Who is teaching",
                f"{teacher.id} is the live teacher: {teacher.talk_time / 60:.1f} min of speech, "
                f"{len(teacher_questions)} questions, {praise} specific words of praise. "
                + " ".join(teacher.reasons),
            )
        )
    if students:
        bits = []
        for student in sorted(students, key=lambda item: item.talk_time, reverse=True):
            quote = f' At {fmt_ts(student.quotes[0][0])}: "{student.quotes[0][1]}"' if student.quotes else ""
            bits.append(
                f"{student.id} ({student.talk_time / 60:.1f} min, {student.turns} turns). "
                + " ".join(student.reasons)
                + quote
            )
        observations.append(
            Observation(
                "speaker",
                f"{len(students)} student voice{'s' if len(students) != 1 else ''}",
                " ".join(bits),
            )
        )
    else:
        observations.append(
            Observation(
                "issue",
                "No student voice stood out",
                "After separating speakers, nobody both answered questions and stayed in the lesson. "
                "A class with only one voice cannot show that a child was learning.",
            )
        )
    if audience:
        bits = []
        for person in audience:
            quote = f' "{person.quotes[0][1]}"' if person.quotes else ""
            bits.append(f"{person.id} ({person.talk_time:.0f}s). " + " ".join(person.reasons) + quote)
        observations.append(
            Observation(
                "speaker",
                "Audience, not participants",
                "These voices are parents, bystanders, or brief join/leave chatter, so they are not counted as students. "
                + " ".join(bits),
            )
        )
    if media:
        observations.append(
            Observation(
                "speaker",
                "Shared lesson audio",
                f"{media_talk / 60:.1f} min comes from screen-share or a played video, kept separate from the live teacher. "
                + " ".join(media[0].reasons),
            )
        )

    observations.append(
        Observation(
            "strength" if answer_rate >= 0.75 and len(teacher_questions) >= 8 else "issue",
            "Questions and answers",
            (
                f"The teacher asked {len(teacher_questions)} questions ({qpm:.1f} per minute). "
                f"Students answered {answered} of them ({answer_rate:.0%}) within 12 seconds. "
                f"Teacher and students handed the turn over {switches} times ({switches_per_min:.1f} per minute)."
            ),
        )
    )
    if mono_dur >= 45 and teacher:
        snippet = _quote_at(turns, teacher.id, mono_start, mono_end)
        observations.append(
            Observation(
                "issue",
                "Long stretch with no student",
                f"From {fmt_ts(mono_start)} to {fmt_ts(mono_end)} ({mono_dur:.0f}s) only the teacher is talking. "
                + (f'It starts like this: "{snippet}"' if snippet else "No student turn breaks that stretch."),
            )
        )
    elif teacher and praise:
        observations.append(
            Observation(
                "strength",
                "The teacher makes room and responds",
                f"The longest teacher-only stretch is {mono_dur:.0f}s, and praise shows up about {praise_per_min:.1f} times a minute.",
            )
        )

    if long_gaps:
        worst = max(long_gaps, key=lambda gap: gap["dur"])
        observations.append(
            Observation(
                "issue" if worst["dur"] >= 45 or speech_ratio < 0.4 else "technical",
                "Silence in the recording",
                f"Speech fills {speech_ratio:.0%} of the {minutes:.0f} min recording. "
                f"The longest quiet stretch is {worst['dur']:.0f}s starting at {fmt_ts(worst['start'])}. "
                f"Speech itself starts at {fmt_ts(first_speech)}.",
            )
        )
    else:
        observations.append(
            Observation(
                "strength",
                "The class keeps moving",
                f"Speech fills {speech_ratio:.0%} of the recording and there is no silence longer than 12 seconds. "
                f"First speech is at {fmt_ts(first_speech)}.",
            )
        )

    if english and teacher_turns:
        observations.append(
            Observation(
                "issue" if fragment_ratio >= 0.45 else "strength",
                "How complete the teacher's lines are",
                f"{fragment_ratio:.0%} of the teacher's turns are three words or fewer. "
                f"They speak at about {teacher_wpm:.0f} words per minute. "
                f"Chess or lesson words make up {lesson_ratio:.1%} of all words heard.",
            )
        )
    elif not english:
        observations.append(
            Observation(
                "technical",
                f"Language detected: {language}",
                "Lesson-word lists are built for English chess classes, so clarity leans on turn-taking "
                "instead of vocabulary. A non-English class is not penalised for missing English piece names.",
            )
        )

    if tech_hits:
        sample = tech_hits[0]
        observations.append(
            Observation(
                "technical",
                "People mention sound or connection trouble",
                f"This comes up {len(tech_hits)} times. First at {fmt_ts(sample.start)}, {sample.speaker}: \"{sample.text[:160]}\"",
            )
        )
    if visual.get("frames"):
        observations.append(
            Observation(
                "technical",
                "What is on screen",
                f"Sampled {visual['frames']} frames, one every {visual.get('interval_sec', 15):.0f}s across the full recording. "
                f"A chessboard is visible in {visual.get('board_ratio', 0):.0%}, a camera or face in {visual.get('face_ratio', 0):.0%}, "
                f"and a large empty grey area in {visual.get('empty_ratio', 0):.0%}. "
                f"Wide black bars show up in {visual.get('blackbar_ratio', 0):.0%} of frames. "
                f"Frames that stay identical for about a minute: {visual.get('freeze_ratio', 0):.0%}.",
            )
        )
        problems = []
        if float(visual.get("face_ratio") or 0) < 0.08:
            problems.append("a face or webcam is almost never in frame, so it is hard to tell who is present")
        if float(visual.get("empty_ratio") or 0) > 0.28:
            problems.append("a large grey empty panel fills much of the screen")
        if float(visual.get("freeze_ratio") or 0) > 0.12:
            problems.append("the picture stays identical for long stretches, which usually means a frozen share")
        if float(visual.get("blackbar_ratio") or 0) > 0.45:
            problems.append("most of the frame is black bars, so the class is a small strip in the middle")
        if problems:
            observations.append(
                Observation(
                    "technical",
                    "The picture gets in the way",
                    "Across the whole recording, " + "; ".join(problems) + ".",
                )
            )
    if level is not None and level < -28:
        observations.append(
            Observation(
                "technical",
                "Speech is quiet",
                f"Average speech level is {level:.0f} dBFS. Students in weaker classes ask the teacher to be louder when it sits this low.",
            )
        )

    if duration < 8 * 60:
        observations.append(
            Observation(
                "issue",
                "Shorter than a normal class",
                f"The file is only {minutes:.1f} minutes. The grade describes this clip, not a full hour-long session.",
            )
        )

    # Gates keep a lecture or a half-empty recording out of the top band.
    if speech_time < 90 and speech_ratio < 0.2:
        score = min(score, 32)
    # Measured silence (not a gap in a transcript) means the class was not happening.
    if audio.get("speech_ratio") is not None and speech_ratio < 0.40:
        score *= 0.50 + speech_ratio
    if score >= 78 and (answer_rate < 0.72 or mono_dur > 90 or speech_ratio < 0.55 or len(teacher_questions) < 10):
        score = min(score, 76)
    if score >= 63 and (speech_ratio < 0.32 or (answer_rate < 0.45 and qpm < 1.2) or mono_dur > 300):
        score = min(score, 61)

    score = round(float(score), 1)
    grade = grade_for(score)
    weakest = min(subs, key=subs.get)
    teacher_name = teacher.id if teacher else "nobody identified as the teacher"
    student_bit = (
        f"{len(students)} student voice{'s' if len(students) != 1 else ''} took {student_share:.0%} of the speech"
        if students
        else "no student voice took a real part"
    )
    audience_bit = ""
    if audience:
        names = ", ".join(person.id for person in audience)
        audience_bit = (
            f" {len(audience)} other voice{'s' if len(audience) != 1 else ''} ({names}) "
            f"sit outside the lesson as audience."
        )
    if audio.get("speech_ratio") is not None:
        coverage = "The full soundtrack was scored, not a short sample."
    else:
        coverage = "Screen and microphone level were not measured for this run."
    summary = (
        f"This {minutes:.0f}-minute recording is {grade} ({score:.0f}/100). "
        f"{teacher_name} leads it, and {student_bit}. "
        f"Students answered {answered} of {len(teacher_questions)} teacher questions"
        f"{f', and the teacher once spoke alone for {mono_dur / 60:.1f} min from {fmt_ts(mono_start)}' if mono_dur >= 45 else ''}. "
        f"The weakest area is {weakest} ({subs[weakest]:.0f}/100). "
        f"{coverage}"
        + audience_bit
    )
    assessment = Assessment(
        grade=grade,
        score=round(score, 1),
        subscores=subs,
        summary=summary,
        observations=observations,
        speakers=sorted(speakers, key=lambda item: item.talk_time, reverse=True),
        duration=duration,
        language=language,
        audio=audio,
        visual=visual,
        turns=turns,
        source=source,
        measured_grade=grade,
        measured_score=round(score, 1),
    )
    return align_grade(assessment)


def lesson_signals(assessment: Assessment) -> dict:
    """Signals named in the sample feedback: warmth, homework, coverage, and who is addressed."""
    turns = assessment.turns
    duration = max(assessment.duration, 1.0)
    teachers = [speaker for speaker in assessment.speakers if speaker.role == "teacher"]
    teacher = max(teachers, key=lambda speaker: speaker.talk_time) if teachers else None
    teacher_turns = [turn for turn in turns if teacher and turn.speaker == teacher.id] or list(turns)
    text = " ".join(turn.text for turn in teacher_turns)
    early = " ".join(turn.text for turn in teacher_turns if turn.start < 8 * 60)
    names = [
        speaker.id.split()[0]
        for speaker in assessment.speakers
        if speaker.role == "student" and len(speaker.id.split()[0]) > 2
    ]
    named_questions = 0
    for turn in teacher_turns:
        if "?" not in turn.text:
            continue
        lowered = turn.text.lower()
        if any(name.lower() in lowered for name in names):
            named_questions += 1
    speech = sum(turn.duration for turn in turns)
    cover = speech / duration
    if assessment.audio.get("speech_ratio") is not None:
        cover = float(assessment.audio["speech_ratio"])
    max_gap = 0.0
    for prev, nxt in zip(turns, turns[1:]):
        max_gap = max(max_gap, nxt.start - prev.end)
    if assessment.audio.get("max_gap") is not None:
        max_gap = max(max_gap, float(assessment.audio["max_gap"]))
    return {
        "cover": cover,
        "max_gap": max_gap,
        "warm": len(WARM.findall(text)),
        "homework": len(HW_GIVEN.findall(text)),
        "named_questions": named_questions,
        "howare": bool(re.search(r"how are you|how was your", early, re.I)),
        "hear": sum(1 for turn in turns if HEAR.search(turn.text)),
    }


def align_grade(assessment: Assessment) -> Assessment:
    """Move the grade onto the feedback sheet's scale.

    The sheet's bands are Excellent, Good, Average, and Weak. The challenge
    word for Weak is Poor. A known quality issue is not a fifth grade: a missing
    face or a class that keeps stopping to be heard cannot stay Good.
    """
    visual = assessment.visual or {}
    frames = int(visual.get("frames") or 0)
    face = float(visual.get("face_ratio") or 0)
    freeze = float(visual.get("freeze_ratio") or 0)
    empty = float(visual.get("empty_ratio") or 0)
    board = float(visual.get("board_ratio") or 0)
    bars = float(visual.get("blackbar_ratio") or 0)
    signals = lesson_signals(assessment)
    score = float(assessment.score)
    reason = ""

    healthy = (
        frames == 0
        or (
            face >= 0.18
            and board >= 0.65
            and freeze <= 0.10
            and empty <= 0.40
            and bars < 0.50
        )
    )
    excellent = (
        signals["howare"]
        and (signals["named_questions"] >= 15 or signals["warm"] >= 20)
        and signals["cover"] >= 0.70
        and signals["hear"] < 4
        and healthy
    )
    severe_picture = frames > 0 and face < 0.08 and freeze >= 0.18

    if severe_picture:
        if score > 45:
            score = 45
        reason = (
            "Quality issue: the teacher's face is almost never visible and the picture stays frozen, "
            "so the class cannot be seen."
        )
    elif signals["hear"] >= 5 and score > 58:
        score = 58
        reason = (
            f"Quality issue: people ask whether they can be heard {signals['hear']} times. "
            "The lesson keeps stopping for the connection, so it is Average at best."
        )
    elif signals["cover"] < 0.30 and score > 45:
        score = 42
        reason = (
            "Speech fills under a third of the recording, and the opening goes straight to revision "
            "without rapport. On the feedback scale that is Poor (the sheet calls this band weak)."
        )
    elif frames > 0 and bars >= 0.70 and signals["homework"] == 0 and score > 45:
        score = 45
        reason = (
            "The picture is mostly black bars, and the teacher never sets homework or makes time "
            "for a proper welcome. That is Poor (weak on the feedback sheet)."
        )
    elif frames > 0 and bars >= 0.70 and signals["homework"] > 0 and score > 62:
        score = 62
        reason = (
            "Students are included and homework is set, but the background is hard to see clearly. "
            "That keeps the class at Average."
        )
    elif frames > 0 and face < 0.10 and empty >= 0.30 and score > 62:
        score = 62
        reason = (
            "There is some one-to-one attention, but the teacher is barely lit and a grey empty area "
            "fills the screen. That is Average, and the room needs more light and energy."
        )
    elif excellent and score < 78:
        score = 82
        reason = (
            "The teacher asks after the students, calls on them, and stays on camera with the board up. "
            "That is Excellent."
        )
    elif score >= 78 and not excellent:
        score = 74
        reason = (
            "The class has a clear introduction, a visible board, and homework, "
            "but the back-and-forth is less warm than an excellent session. That is Good."
        )
    elif (
        frames > 0
        and signals["max_gap"] >= 400
        and signals["howare"]
        and board >= 0.55
        and face >= 0.10
        and score < 63
    ):
        score = 66
        reason = (
            "The teacher opens the class and brings the students in. A long hole in the captions "
            "was hiding that, and homework is the piece still missing. That is Good."
        )

    score = round(score, 1)
    grade = grade_for(score)
    if grade == assessment.grade and abs(score - assessment.score) < 0.05:
        return assessment
    assessment.score = score
    assessment.grade = grade
    if reason:
        assessment.observations.insert(0, Observation("technical", "What set the grade", reason))
        assessment.summary = f"This recording is {grade} ({score:.0f}/100). {reason} {assessment.summary}"
    return assessment
