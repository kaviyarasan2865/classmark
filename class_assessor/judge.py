"""An LLM decides the grade. Measurements stay in the evidence packet.

Order: Gemini, then Ollama cloud, then Groq, then a local Ollama model.
The video never leaves the machine. The judge receives the transcript sample,
speaker roles, and the measured notes.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from class_assessor.cues import fmt_ts
from class_assessor.score import Assessment, Observation, align_grade

GRADES = ("Poor", "Average", "Good", "Excellent")
BANDS = {
    "Poor": (0, 45),
    "Average": (46, 62),
    "Good": (63, 77),
    "Excellent": (78, 100),
}
GEMINI_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-flash-latest",
)
OLLAMA_CLOUD_MODELS = ("gpt-oss:20b", "gemma3:4b", "llama3.2")
GROQ_MODELS = ("llama-3.3-70b-versatile", "llama-3.1-8b-instant")


def _load_dotenv() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    # Streamlit Community Cloud mounts this file from the secrets box. It is not committed.
    if (root / ".streamlit" / "secrets.toml").is_file():
        try:
            import streamlit as st

            for key, value in dict(st.secrets).items():
                if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                    os.environ.setdefault(str(key), str(value))
        except Exception:
            return


def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: float = 90) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code} from {url.split('?')[0]}: {body}") from exc


def _get_json(url: str, timeout: float = 3) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _gemini(prompt: str, key: str) -> tuple[str, str]:
    preferred = os.environ.get("GEMINI_MODEL", "").strip()
    models = [preferred] if preferred else []
    models += [name for name in GEMINI_MODELS if name not in models]
    last_error = "Gemini did not respond"
    for model in models:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={key}"
        )
        try:
            data = _post_json(
                url,
                {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "responseMimeType": "application/json",
                    },
                },
                timeout=90,
            )
        except RuntimeError as exc:
            last_error = str(exc)
            if "HTTP 404" in last_error or "HTTP 400" in last_error:
                continue
            raise
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "\n".join(part.get("text", "") for part in parts if not part.get("thought")).strip()
        if text:
            return text, f"Gemini · {model}"
        last_error = f"Gemini {model} returned an empty answer"
    raise RuntimeError(last_error)


def _ollama_chat(base: str, model: str, prompt: str, headers: dict | None, timeout: float) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0.2},
    }
    try:
        data = _post_json(base.rstrip("/") + "/api/chat", payload, headers, timeout)
    except RuntimeError as exc:
        if "HTTP 400" not in str(exc):
            raise
        payload.pop("think", None)
        data = _post_json(base.rstrip("/") + "/api/chat", payload, headers, timeout)
    message = data.get("message") or {}
    return (message.get("content") or message.get("thinking") or "").strip()


def _ollama_cloud(prompt: str, key: str) -> tuple[str, str]:
    preferred = os.environ.get("OLLAMA_MODEL", "").strip()
    models = [preferred] if preferred else list(OLLAMA_CLOUD_MODELS)
    last_error = "Ollama cloud did not respond"
    headers = {"Authorization": f"Bearer {key}"}
    for model in models:
        try:
            text = _ollama_chat("https://ollama.com", model, prompt, headers, timeout=120)
        except RuntimeError as exc:
            last_error = str(exc)
            if any(code in last_error for code in ("HTTP 404", "HTTP 400", "HTTP 402")):
                continue
            raise
        if text:
            return text, f"Ollama cloud · {model}"
        last_error = f"Ollama cloud {model} returned an empty answer"
    raise RuntimeError(last_error)


def _groq(prompt: str, key: str) -> tuple[str, str]:
    last_error = "Groq did not respond"
    for model in GROQ_MODELS:
        try:
            data = _post_json(
                "https://api.groq.com/openai/v1/chat/completions",
                {
                    "model": model,
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": prompt}],
                },
                headers={"Authorization": f"Bearer {key}"},
                timeout=90,
            )
        except RuntimeError as exc:
            last_error = str(exc)
            if "HTTP 404" in last_error or "HTTP 400" in last_error:
                continue
            raise
        text = data["choices"][0]["message"]["content"].strip()
        if text:
            return text, f"Groq · {model}"
    raise RuntimeError(last_error)


def _local_model() -> str | None:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    try:
        data = _get_json(host + "/api/tags", timeout=2)
    except Exception:
        return None
    names = [item.get("name", "") for item in data.get("models", [])]
    preferred = os.environ.get("OLLAMA_LOCAL_MODEL", "").strip()
    if preferred and preferred in names:
        return preferred
    chat = [name for name in names if name and not any(token in name.lower() for token in ("embed", "bge-", "nomic"))]
    if "qwen3:14b" in chat:
        return "qwen3:14b"
    return chat[0] if chat else None


def _local_ollama(prompt: str) -> tuple[str, str]:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = _local_model()
    if not model:
        raise RuntimeError("No local Ollama chat model is installed.")
    text = _ollama_chat(host, model, prompt, None, timeout=180)
    if not text:
        raise RuntimeError(f"Local Ollama {model} returned an empty answer")
    return text, f"Ollama local · {model}"


def judge_label() -> str:
    """Which judge will run, without calling it."""
    _load_dotenv()
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        return f"Judge: Gemini ({model}), with Ollama as fallback"
    if os.environ.get("OLLAMA_API_KEY"):
        model = os.environ.get("OLLAMA_MODEL", OLLAMA_CLOUD_MODELS[0])
        return f"Judge: Ollama cloud ({model}). Set GEMINI_API_KEY to prefer Gemini."
    if os.environ.get("GROQ_API_KEY"):
        return "Judge: Groq free tier. Set GEMINI_API_KEY to prefer Gemini."
    local = _local_model()
    if local:
        return f"Judge: local Ollama ({local}). Set GEMINI_API_KEY or OLLAMA_API_KEY to use a hosted model first."
    return "No Gemini key and no Ollama model found. The measured rubric will be used."


def _call_chain(prompt: str) -> tuple[str, str]:
    _load_dotenv()
    errors: list[str] = []
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        try:
            return _gemini(prompt, gemini_key)
        except Exception as exc:  # noqa: BLE001 - try the next free judge
            errors.append(str(exc))
    ollama_key = os.environ.get("OLLAMA_API_KEY")
    if ollama_key:
        try:
            return _ollama_cloud(prompt, ollama_key)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        try:
            return _groq(prompt, groq_key)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    if _local_model():
        try:
            return _local_ollama(prompt)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    detail = "; ".join(errors[:3]) or "no Gemini key, Ollama key, Groq key, or local model"
    raise RuntimeError(detail)


def build_evidence(assessment: Assessment, max_turns: int = 140) -> str:
    """A packet the judge can read. Long classes are sampled across the whole timeline."""
    role_of = {speaker.id: speaker.role for speaker in assessment.speakers}
    lines = [
        f"Duration: {assessment.duration / 60:.1f} minutes. Language: {assessment.language}.",
        "These figures were measured from the full recording. They are evidence, not a grade.",
        "Subscores from the measurement pass (0-100): "
        + ", ".join(f"{name} {value:.0f}" for name, value in assessment.subscores.items()),
    ]
    audio = assessment.audio or {}
    if audio:
        lines.append(
            "Audio: "
            f"speech fills {float(audio.get('speech_ratio') or 0):.0%} of the file, "
            f"first speech at {fmt_ts(float(audio.get('first_speech') or 0))}, "
            f"longest silence {float(audio.get('max_gap') or 0):.0f}s, "
            f"level {float(audio.get('mean_speech_db') or 0):.0f} dBFS."
        )
    visual = assessment.visual or {}
    if visual.get("frames"):
        lines.append(
            "Screen, sampled across the whole class: "
            f"board {float(visual.get('board_ratio') or 0):.0%}, "
            f"face/camera {float(visual.get('face_ratio') or 0):.0%}, "
            f"empty area {float(visual.get('empty_ratio') or 0):.0%}, "
            f"black bars {float(visual.get('blackbar_ratio') or 0):.0%}, "
            f"frozen {float(visual.get('freeze_ratio') or 0):.0%}."
        )
    lines.append("Speakers:")
    for speaker in assessment.speakers:
        quote = ""
        if speaker.quotes:
            stamp, text = speaker.quotes[0]
            quote = f' Example {fmt_ts(stamp)}: "{text}"'
        lines.append(
            f"- {speaker.role} {speaker.id}: {speaker.talk_time / 60:.1f} min, {speaker.turns} turns. "
            + " ".join(speaker.reasons)
            + quote
        )
    lines.append("Measured notes:")
    for item in assessment.observations:
        lines.append(f"- {item.title}: {item.detail}")
    turns = assessment.turns
    if len(turns) > max_turns:
        step = len(turns) / max_turns
        picked = [turns[min(len(turns) - 1, int(index * step))] for index in range(max_turns)]
        lines.append(
            f"Transcript sample: {len(picked)} lines taken evenly from {len(turns)} across the full class."
        )
    else:
        picked = turns
        lines.append(f"Transcript: all {len(picked)} lines.")
    for turn in picked:
        role = role_of.get(turn.speaker, "")
        lines.append(f"{fmt_ts(turn.start)} [{role}] {turn.speaker}: {turn.text}")
    return "\n".join(lines)


def _prompt(evidence: str) -> str:
    return (
        "You are the judge of one online class for children, usually chess. "
        "Write the result a reviewer can trust. Use only the evidence below. "
        "Cite timestamps and speaker names that appear in it. Do not invent events, names, or quotes.\n"
        "A teacher is the person leading. Students are the children taking part. "
        "Audience is a parent or someone who only joins, leaves, or talks about the connection, not a student. "
        "Shared audio is a played video, not the live teacher.\n"
        "Use only these four grades. The sample sheet calls its bottom band weak; the required word is Poor.\n"
        "Excellent: students answer through the class, the teacher calls on them by name, asks how they are, "
        "the board and the teacher's camera are visible, and the connection is not the subject of the lesson. "
        "Do not call that class Good.\n"
        "Good: the lesson works and the welcome is clear, but the back-and-forth is less warm, "
        "homework is missing, or fewer students are drawn in.\n"
        "Average: the class happens and someone is included, but energy, clarity, lighting, "
        "or one-to-one attention is uneven. Homework may be set while the picture is hard to see.\n"
        "Poor: long silence, a lecture with almost no replies, no real welcome, the teacher goes "
        "straight to revision, or the camera is off for much of the class. "
        "A frozen or empty screen with the teacher's face missing is Poor even when the transcript sounds busy.\n"
        "If people keep asking 'can you hear me', the grade is Average or Poor, never Good or Excellent. "
        "A played lesson with little live back-and-forth is not Excellent.\n"
        "The summary must name the teacher, say whether students answered, and name the one fact that set the grade "
        "(welcome, homework, light, camera, board, or connection). Two or three sentences. No generic praise.\n"
        "Each observation must include a timestamp from the evidence and, when someone spoke, their name. "
        "Say what happened, not what a good class usually does.\n"
        "Return one JSON object and nothing else, with this shape:\n"
        '{"grade":"Good","score":70,"summary":"two or three specific sentences",'
        '"observations":[{"kind":"strength","title":"short","detail":"what happened, with a timestamp and a name"}]}\n'
        "kind is one of strength, issue, technical, speaker. Write 5 or 6 observations. "
        "score is an integer from 0 to 100 that matches the grade "
        "(Poor under 46, Average 46-62, Good 63-77, Excellent 78 or more).\n\n"
        "EVIDENCE\n"
        f"{evidence}"
    )


def parse_verdict(text: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.S)
    blob = fenced.group(1) if fenced else cleaned
    start, end = blob.find("{"), blob.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("The judge did not return JSON.")
    data = json.loads(blob[start : end + 1])
    grade = str(data.get("grade", "")).strip().title()
    grade = {"Weak": "Poor", "Bad": "Poor", "Ok": "Average", "Okay": "Average", "Great": "Excellent"}.get(grade, grade)
    if grade not in GRADES:
        raise ValueError(f"Unexpected grade: {grade}")
    score = float(data.get("score"))
    low, high = BANDS[grade]
    score = min(high, max(low, score))
    summary = str(data.get("summary") or "").strip()
    if len(summary) < 40:
        raise ValueError("The judge summary was too thin.")
    observations: list[Observation] = []
    for item in data.get("observations") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        detail = str(item.get("detail") or "").strip()
        kind = str(item.get("kind") or "issue").strip().lower()
        if kind not in {"strength", "issue", "technical", "speaker"}:
            kind = "issue"
        if title and detail:
            observations.append(Observation(kind, title[:120], detail[:900]))
    if len(observations) < 3:
        raise ValueError("The judge did not give enough observations.")
    return {"grade": grade, "score": round(score, 1), "summary": summary, "observations": observations}


def apply_judge(assessment: Assessment, complete=None) -> Assessment:
    """Replace the headline grade with the model's decision.

    `complete` is a test hook: it takes the prompt and returns (text, label).
    """
    prompt = _prompt(build_evidence(assessment))
    try:
        text, label = (complete or _call_chain)(prompt)
        verdict = parse_verdict(text)
    except Exception as exc:  # noqa: BLE001 - keep a usable grade if the model fails
        assessment.judge = f"Measured rubric. The judge could not be used ({exc})."
        return assessment
    assessment.grade = verdict["grade"]
    assessment.score = verdict["score"]
    assessment.summary = verdict["summary"]
    assessment.observations = verdict["observations"]
    assessment.judge = label
    return align_grade(assessment)
