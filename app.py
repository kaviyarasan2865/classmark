"""Local web app: upload a class recording and read the grade."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from class_assessor.cues import fmt_ts
from class_assessor.judge import judge_label
from class_assessor.pipeline import assess_media, assess_transcript
from class_assessor.score import Assessment

st.set_page_config(page_title="Class quality check", page_icon="♟", layout="wide")

st.markdown(
    """
    <style>
    .grade {font-size: 1.6rem; font-weight: 700; padding: 0.35rem 0.9rem;
            border-radius: 999px; display: inline-block; margin-right: 0.6rem;}
    .Excellent {background: #e7f6ec; color: #0d6b2c;}
    .Good {background: #e8f1fb; color: #0b4f9c;}
    .Average {background: #fff6e0; color: #8a5a00;}
    .Poor {background: #fdecea; color: #9d1c1c;}
    .tl {display: flex; height: 28px; width: 100%; border-radius: 6px; overflow: hidden;}
    .tl i {flex: 1 1 0; display: block;}
    .note {color: #4a5560; font-size: 0.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

ROLE_COLOR = {
    "teacher": "#1f6feb",
    "student": "#2ea043",
    "audience": "#d4a017",
    "shared_media": "#8250df",
    "background": "#8b949e",
    "none": "#e6e8eb",
}


def _timeline(result: Assessment) -> str:
    buckets = 240
    roles = {speaker.id: speaker.role for speaker in result.speakers}
    painted = ["none"] * buckets
    span = max(result.duration, 1.0)
    for turn in result.turns:
        role = roles.get(turn.speaker, "background")
        start = int(turn.start / span * buckets)
        end = max(start + 1, int(turn.end / span * buckets))
        for index in range(max(0, start), min(buckets, end)):
            painted[index] = role
    cells = "".join(f"<i style='background:{ROLE_COLOR.get(role, '#e6e8eb')}'></i>" for role in painted)
    return f"<div class='tl'>{cells}</div>"


def _save_upload(upload, directory: str) -> str:
    destination = os.path.join(directory, upload.name)
    with open(destination, "wb") as handle:
        handle.write(upload.getbuffer())
    return destination


def _download(url: str, directory: str) -> str:
    import urllib.request

    name = url.rstrip("/").split("/")[-1].split("?")[0] or "class-download"
    if "." not in name:
        name += ".mp4"
    destination = os.path.join(directory, name)
    urllib.request.urlretrieve(url, destination)
    return destination


def _render(result: Assessment) -> None:
    st.markdown(
        f"<span class='grade {result.grade}'>{result.grade}</span> "
        f"<strong>{result.score:.0f} / 100</strong>",
        unsafe_allow_html=True,
    )
    if result.judge:
        st.caption(result.judge)
    st.write(result.summary)
    cols = st.columns(len(result.subscores))
    for column, (name, value) in zip(cols, result.subscores.items()):
        column.metric(name.capitalize(), f"{value:.0f}")

    st.subheader("Who spoke, across the whole class")
    st.markdown(_timeline(result), unsafe_allow_html=True)
    legend = " ".join(
        f"<span style='color:{color}'>■</span> {name.replace('_', ' ')}"
        for name, color in ROLE_COLOR.items()
        if name != "none"
    )
    st.markdown(legend, unsafe_allow_html=True)

    st.subheader("Speakers")
    for speaker in result.speakers:
        with st.expander(f"{speaker.label()}  ·  {speaker.talk_time / 60:.1f} min  ·  {speaker.turns} turns"):
            for reason in speaker.reasons:
                st.write(reason)
            for stamp, quote in speaker.quotes:
                st.caption(f"{fmt_ts(stamp)}  {quote}")

    st.subheader("What the grade is based on")
    for item in result.observations:
        st.markdown(f"**{item.title}**")
        st.write(item.detail)

    with st.expander("Timed transcript"):
        role_of = {speaker.id: speaker.role for speaker in result.speakers}
        lines = [
            f"{fmt_ts(turn.start)}  {role_of.get(turn.speaker, '')}  {turn.speaker}: {turn.text}"
            for turn in result.turns
        ]
        st.text("\n".join(lines))

    st.download_button(
        "Download the written report",
        data=result.to_markdown(),
        file_name=f"{result.grade.lower()}-class-report.md",
        mime="text/markdown",
    )


st.title("Class quality check")
st.write(
    "Upload a class recording, or point at a file already on this computer. "
    "The app listens to the full duration, separates the teacher, the students, and other voices, "
    "and grades the session as Poor, Average, Good, or Excellent."
)
st.caption(
    "The video stays on this machine and speech is transcribed locally. "
    "The judge then reads the transcript and the measurements. "
    "Gemini is used when GEMINI_API_KEY is set. Otherwise the app tries Ollama cloud "
    "(OLLAMA_API_KEY), then Groq (GROQ_API_KEY), then a local Ollama model. "
    "A Zoom transcript can contain children's names, so that text is what leaves the machine. "
    "Do not commit or publish the recordings."
)

with st.sidebar:
    st.header("How far to look")
    model_name = st.selectbox("Speech model", ["small", "base", "medium", "tiny"], index=0)
    beam = st.selectbox("Transcription care", [1, 5], index=0, format_func=lambda value: "Faster" if value == 1 else "More careful")
    frame_interval = st.slider("Screen sample, seconds", 10, 30, 15)
    st.caption("Audio is always the full file. The screen is sampled across the whole timeline, not only the opening.")
    st.caption(judge_label())

with st.form("assess_form"):
    left, right = st.columns(2)
    with left:
        upload = st.file_uploader(
            "Recording",
            type=["mp4", "mov", "mkv", "webm", "m4v", "mp3", "wav", "vtt"],
        )
        link = st.text_input("Or a file path or a direct download link")
    with right:
        vtt_upload = st.file_uploader("Optional Zoom transcript (.vtt)", type=["vtt"])
        st.caption("A Zoom transcript keeps real names. Without one, voices are separated from the mixed audio.")
    submitted = st.form_submit_button("Assess the full recording", type="primary")

if submitted:
    if not upload and not link.strip():
        st.error("Add a recording first.")
    else:
        try:
            with tempfile.TemporaryDirectory(prefix="class-upload-") as temp:
                vtt_path = None
                if vtt_upload is not None:
                    vtt_path = _save_upload(vtt_upload, temp)
                if upload is not None:
                    media_path = _save_upload(upload, temp)
                else:
                    target = link.strip()
                    if target.startswith("http://") or target.startswith("https://"):
                        with st.spinner("Downloading the recording"):
                            media_path = _download(target, temp)
                    else:
                        media_path = str(Path(target).expanduser())
                        if not Path(media_path).is_file():
                            raise FileNotFoundError(media_path)

                if media_path.lower().endswith(".vtt") and vtt_path is None:
                    result = assess_transcript(media_path)
                else:
                    bar = st.progress(0.0, text="Starting")

                    def progress(fraction: float, message: str) -> None:
                        bar.progress(min(1.0, max(0.0, fraction)), text=message)

                    result = assess_media(
                        media_path,
                        vtt_path=vtt_path,
                        model_name=model_name,
                        beam_size=int(beam),
                        frame_interval=float(frame_interval),
                        on_progress=progress,
                    )
                    bar.progress(1.0, text=f"{result.grade} ({result.score:.0f}/100)")
                st.session_state["result"] = result
        except Exception as exc:  # noqa: BLE001 - show the failure in the page
            st.error(str(exc))

if "result" in st.session_state:
    _render(st.session_state["result"])
