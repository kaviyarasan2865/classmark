"""Local transcription. Audio is not sent to a hosted speech API."""

from __future__ import annotations

import os
import wave

import numpy as np

from class_assessor.cues import Turn

CHESS_PROMPT = (
    "Online chess class for children. A teacher and students talk about the king, "
    "queen, rook, bishop, knight, pawn, check, checkmate, castling, and the board."
)

_MODELS: dict[str, object] = {}


def _decode_wav(path, sampling_rate: int = 16000, split_stereo: bool = False):
    """Read the wav we already extracted with ffmpeg.

    faster-whisper would otherwise open the file with PyAV. Recent PyAV builds
    reject the keyword that faster-whisper passes, so we stay on the wav we
    control and hand the model a float array.
    """
    del split_stereo
    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if sample_rate != sampling_rate and len(audio):
        new_len = int(len(audio) * sampling_rate / sample_rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return audio


def transcribe(
    wav_path: str,
    duration: float,
    model_name: str = "small",
    beam_size: int = 1,
    on_progress=None,
) -> tuple[list[Turn], str]:
    import faster_whisper.audio as whisper_audio
    import faster_whisper.transcribe as whisper_transcribe
    from faster_whisper import WhisperModel

    # Bound name inside transcribe.py, not only the audio module.
    whisper_audio.decode_audio = _decode_wav
    whisper_transcribe.decode_audio = _decode_wav
    if model_name not in _MODELS:
        if on_progress:
            on_progress(0.0, f"Loading the '{model_name}' speech model")
        # The Xet download path on Hugging Face often returns 403. The normal file path does not.
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        try:
            _MODELS[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8")
        except Exception as exc:
            if "403" not in str(exc) and "Forbidden" not in str(exc):
                raise
            raise RuntimeError(
                f"Hugging Face refused the download of the '{model_name}' speech model (403 Forbidden). "
                "That model only writes the transcript. It is not the judge. "
                "Press Assess again. Tiny is already saved on this machine, and small was downloaded earlier."
            ) from exc
    model = _MODELS[model_name]
    segments, info = model.transcribe(
        wav_path,
        beam_size=beam_size,
        vad_filter=True,
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
        temperature=0.0,
        initial_prompt=CHESS_PROMPT,
    )
    language = info.language or "unknown"
    turns: list[Turn] = []
    previous = None
    repeat = 0
    for segment in segments:
        text = " ".join((segment.text or "").split())
        if not text:
            continue
        key = text.lower()
        if key == previous:
            repeat += 1
        else:
            previous = key
            repeat = 1
        if repeat > 2:
            continue
        turns.append(
            Turn(
                start=float(segment.start),
                end=float(segment.end),
                speaker="unknown",
                text=text,
                avg_logprob=float(segment.avg_logprob),
            )
        )
        if on_progress and duration > 0:
            on_progress(
                min(0.99, segment.end / duration),
                f"Transcribing {segment.end / 60:.0f} of {duration / 60:.0f} min",
            )
    return turns, language
