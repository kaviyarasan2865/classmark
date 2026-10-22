import json
import unittest

from pathlib import Path

from class_assessor.check import score_transcript, sheet_grade
from class_assessor.cues import Turn, parse_vtt, words
from class_assessor.judge import apply_judge, parse_verdict
from class_assessor.score import assess, align_grade, grade_for
from class_assessor.speakers import assign_roles

# Screen ratios from a full pass over each labelled recording, one frame every 20s.
SCREEN = {
    ("average", "B435"): {"frames": 114, "board_ratio": 0.96, "face_ratio": 0.05, "empty_ratio": 0.42, "blackbar_ratio": 0.02, "freeze_ratio": 0.12},
    ("average", "B516"): {"frames": 131, "board_ratio": 0.66, "face_ratio": 0.55, "empty_ratio": 0.05, "blackbar_ratio": 0.93, "freeze_ratio": 0.02},
    ("excellent", "B472"): {"frames": 168, "board_ratio": 0.84, "face_ratio": 0.24, "empty_ratio": 0.30, "blackbar_ratio": 0.17, "freeze_ratio": 0.04},
    ("excellent", "B491"): {"frames": 162, "board_ratio": 0.74, "face_ratio": 0.40, "empty_ratio": 0.14, "blackbar_ratio": 0.10, "freeze_ratio": 0.04},
    ("good", "B435"): {"frames": 133, "board_ratio": 0.67, "face_ratio": 0.14, "empty_ratio": 0.03, "blackbar_ratio": 0.03, "freeze_ratio": 0.23},
    ("good", "B552"): {"frames": 137, "board_ratio": 0.89, "face_ratio": 0.22, "empty_ratio": 0.01, "blackbar_ratio": 0.05, "freeze_ratio": 0.01},
    ("quality issue", "B392"): {"frames": 181, "board_ratio": 0.75, "face_ratio": 0.48, "empty_ratio": 0.18, "blackbar_ratio": 0.03, "freeze_ratio": 0.08},
    ("quality issue", "B441"): {"frames": 268, "board_ratio": 0.62, "face_ratio": 0.01, "empty_ratio": 0.34, "blackbar_ratio": 0.04, "freeze_ratio": 0.27},
    ("weak", "B482"): {"frames": 178, "board_ratio": 0.84, "face_ratio": 0.60, "empty_ratio": 0.04, "blackbar_ratio": 0.19, "freeze_ratio": 0.03},
    ("weak", "B558"): {"frames": 139, "board_ratio": 0.83, "face_ratio": 0.26, "empty_ratio": 0.06, "blackbar_ratio": 0.94, "freeze_ratio": 0.09},
}


def _turn(start, end, speaker, text):
    return Turn(start, end, speaker, text)


class ParseTests(unittest.TestCase):
    def test_zoom_vtt_keeps_speaker(self):
        path = "class recordings /excellent/B491 Beginner -transcript-1.1.vtt"
        turns = parse_vtt(path)
        self.assertGreater(len(turns), 100)
        self.assertEqual(turns[0].speaker, "Ayswarya Joseph")
        self.assertIn("Gautam", turns[0].text + turns[1].text)
        spellings = {turn.speaker for turn in turns if turn.speaker.casefold() == "shrithik"}
        self.assertEqual(len(spellings), 1)

    def test_words_skip_punctuation(self):
        self.assertEqual(words("King, to D4!"), ["King", "to", "D"])


class RoleTests(unittest.TestCase):
    def test_teacher_is_the_person_asking_questions(self):
        turns = [
            _turn(2, 4, "Mom on iPhone", "Can you hear me, the wifi dropped."),
            _turn(5, 6, "Mom on iPhone", "Okay bye."),
        ]
        t = 10.0
        for _ in range(30):
            turns.append(_turn(t, t + 4, "Teacher Pat", "What is the king doing here?"))
            t += 4
            turns.append(_turn(t, t + 2, "Asha", "It moves one square."))
            t += 6
        speakers = assign_roles(turns, t)
        by = {speaker.id: speaker.role for speaker in speakers}
        self.assertEqual(by["Teacher Pat"], "teacher")
        self.assertEqual(by["Asha"], "student")
        self.assertEqual(by["Mom on iPhone"], "audience")

    def test_shared_audio_is_not_the_teacher(self):
        turns = [
            _turn(0, 30, "Audio shared by Riya", "The rook slides along the rank and gives check."),
            _turn(31, 34, "Riya", "Did you see that?"),
            _turn(35, 37, "Dev", "Yes, rook to h8."),
        ]
        speakers = assign_roles(turns, 40)
        by = {speaker.id: speaker.role for speaker in speakers}
        self.assertEqual(by["Audio shared by Riya"], "shared_media")
        self.assertEqual(by["Riya"], "teacher")
        self.assertEqual(by["Dev"], "student")


class GradeTests(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(grade_for(90), "Excellent")
        self.assertEqual(grade_for(70), "Good")
        self.assertEqual(grade_for(50), "Average")
        self.assertEqual(grade_for(20), "Poor")

    def test_dead_air_cannot_score_as_excellent(self):
        turns = []
        # Teacher talks, then a long hole, almost no replies.
        turns.append(_turn(200, 260, "Pat", "Hello. We will start the puzzle soon I think maybe."))
        turns.append(_turn(500, 560, "Pat", "The queen moves any number of squares, watch this line."))
        speakers = assign_roles(turns, 800)
        result = assess(turns, speakers, duration=800, language="en")
        self.assertNotEqual(result.grade, "Excellent")
        self.assertLess(result.score, 63)
        self.assertTrue(any("Silence" in item.title or "quiet" in item.detail.lower() for item in result.observations))


class JudgeTests(unittest.TestCase):
    def test_parse_strips_thinking_and_fence(self):
        raw = """<think>draft</think>
```json
{"grade":"Good","score":90,"summary":"Students answered throughout and the teacher kept turns short from 04:00 onward.",
 "observations":[
   {"kind":"strength","title":"Answers","detail":"At 04:10 a student answered the king's move."},
   {"kind":"issue","title":"Late start","detail":"Speech starts at 02:00."},
   {"kind":"technical","title":"Camera","detail":"A face is visible for most of the hour."}
 ]}
```"""
        verdict = parse_verdict(raw)
        self.assertEqual(verdict["grade"], "Good")
        self.assertLessEqual(verdict["score"], 77)
        self.assertGreaterEqual(verdict["score"], 63)

    def test_apply_replaces_the_rubric_grade(self):
        turns = [_turn(0, 30, "Pat", "What does the king do here on e4?")]
        turns += [_turn(31, 40, "Asha", "One square, forward.")]
        speakers = assign_roles(turns, 40)
        result = assess(turns, speakers, duration=40, language="en")

        def complete(prompt):
            self.assertIn("Pat", prompt)
            return (
                json.dumps(
                    {
                        "grade": "Average",
                        "score": 55,
                        "summary": "Pat asks one clear question at 00:00 and Asha answers at 00:31, but this clip is only a fragment of a class.",
                        "observations": [
                            {"kind": "strength", "title": "One answer", "detail": "At 00:31 Asha says the king moves one square."},
                            {"kind": "issue", "title": "Too short", "detail": "The recording is under a minute, so it cannot show a full lesson."},
                            {"kind": "speaker", "title": "Roles", "detail": "Pat leads and Asha answers."},
                        ],
                    }
                ),
                "Gemini · gemini-2.5-flash",
            )

        judged = apply_judge(result, complete=complete)
        self.assertEqual(judged.grade, "Average")
        self.assertEqual(judged.score, 55)
        self.assertIn("Gemini", judged.judge)
        self.assertIn("00:31", judged.summary)

    def test_frozen_faceless_screen_cannot_be_good(self):
        turns = [_turn(i, i + 3, "Pat", "What is the king doing? Can you hear me?") for i in range(0, 120, 6)]
        speakers = assign_roles(turns, 120)
        result = assess(turns, speakers, duration=120, language="en")
        result.visual = {"frames": 100, "face_ratio": 0.01, "freeze_ratio": 0.27, "empty_ratio": 0.34, "board_ratio": 0.62}
        result.grade = "Good"
        result.score = 74
        result.measured_grade = "Good"
        aligned = align_grade(result)
        self.assertEqual(aligned.grade, "Poor")
        self.assertLessEqual(aligned.score, 45)

    def test_repeated_hear_me_cannot_be_good(self):
        turns = [_turn(i, i + 2, "Pat", "Guys, can you hear me? Move the knight.") for i in range(0, 50, 5)]
        speakers = assign_roles(turns, 50)
        result = assess(turns, speakers, duration=50, language="en")
        result.grade = "Excellent"
        result.score = 88
        result.measured_grade = "Excellent"
        result.subscores = {"interaction": 80, "engagement": 70, "clarity": 70, "technical": 70, "management": 70}
        aligned = align_grade(result)
        self.assertEqual(aligned.grade, "Average")
        self.assertLessEqual(aligned.score, 62)

    def test_live_class_on_camera_stays_excellent(self):
        turns = []
        clock = 0.0
        for _ in range(40):
            turns.append(_turn(clock, clock + 8, "Pat", "How are you, Asha? Very good. What does the king do?"))
            clock += 8
            turns.append(_turn(clock, clock + 6, "Asha", "It moves one square."))
            clock += 6
        speakers = assign_roles(turns, clock)
        result = assess(
            turns,
            speakers,
            duration=clock,
            language="en",
            visual={"frames": 80, "face_ratio": 0.40, "freeze_ratio": 0.04, "empty_ratio": 0.14, "board_ratio": 0.74, "blackbar_ratio": 0.1},
        )
        result.grade = "Good"
        result.score = 70
        aligned = align_grade(result)
        self.assertEqual(aligned.grade, "Excellent")
        self.assertGreaterEqual(aligned.score, 78)

    def test_thin_recording_stays_poor(self):
        turns = [
            _turn(200, 220, "Pat", "Have you remembered the last class?"),
            _turn(400, 420, "Asha", "Deflection."),
            _turn(700, 730, "Pat", "Okay, we will revise that."),
        ]
        speakers = assign_roles(turns, 900)
        result = assess(turns, speakers, duration=900, language="en")
        result.visual = {"frames": 40, "face_ratio": 0.60, "freeze_ratio": 0.03, "empty_ratio": 0.04, "board_ratio": 0.84, "blackbar_ratio": 0.1}
        result.grade = "Average"
        result.score = 58
        aligned = align_grade(result)
        self.assertEqual(aligned.grade, "Poor")
        self.assertLessEqual(aligned.score, 45)


class FeedbackSheetTests(unittest.TestCase):
    def test_labelled_recordings_match_the_sheet(self):
        root = Path("class recordings ")
        misses = []
        for vtt in sorted(root.rglob("*.vtt")):
            folder = vtt.parent.name.strip()
            batch = vtt.name.split()[0]
            visual = None
            for (label, code), ratios in SCREEN.items():
                if code == batch and label in folder.lower():
                    visual = ratios
                    break
            if visual is None:
                continue
            result = score_transcript(str(vtt), visual)
            expected = sheet_grade(folder)
            if expected is None:
                flagged = any("quality issue" in item.detail.lower() for item in result.observations)
                if result.grade not in {"Poor", "Average"} or not flagged:
                    misses.append(f"{folder} {batch}: {result.grade} (quality issue not flagged)")
            elif result.grade != expected:
                misses.append(f"{folder} {batch}: sheet {expected}, app {result.grade} ({result.score})")
        self.assertFalse(misses, "\n".join(misses))


if __name__ == "__main__":
    unittest.main()
