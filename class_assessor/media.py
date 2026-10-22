"""Audio extraction, energy-based speech detection, and screen sampling.

Everything stays on this machine. Frames are decoded in a stream and discarded
after the metrics are computed, so class videos are not copied into the repo.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Callable

import numpy as np

Progress = Callable[[float, str], None]


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required and were not found on PATH.")


def probe(path: str) -> dict:
    require_ffmpeg()
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
        ]
    )
    data = json.loads(raw)
    duration = float(data.get("format", {}).get("duration") or 0)
    width = height = 0
    has_audio = has_video = False
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "audio":
            has_audio = True
        if stream.get("codec_type") == "video":
            has_video = True
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
    return {
        "duration": duration,
        "width": width,
        "height": height,
        "has_audio": has_audio,
        "has_video": has_video,
        "size": int(data.get("format", {}).get("size") or 0),
    }


def extract_wav(src: str, dst: str, max_seconds: float | None = None) -> None:
    """Decode the full soundtrack to 16 kHz mono PCM, optionally truncated."""
    require_ffmpeg()
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src]
    if max_seconds:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst]
    subprocess.run(cmd, check=True)


def read_wav_int16(path: str) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise RuntimeError("Expected 16-bit PCM audio.")
    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return samples, sample_rate


def audio_profile(samples: np.ndarray, sample_rate: int) -> dict:
    """Speech, silence, level, and long dead-air gaps across the whole file."""
    if len(samples) == 0 or sample_rate <= 0:
        return {
            "speech_ratio": 0.0,
            "mean_speech_db": -90.0,
            "clip_ratio": 0.0,
            "first_speech": 0.0,
            "last_speech": 0.0,
            "long_gaps": [],
            "max_gap": 0.0,
        }
    win = int(sample_rate * 0.02)
    count = len(samples) // win
    frames = samples[: count * win].reshape(count, win).astype(np.float32) / 32768.0
    rms = np.sqrt((frames * frames).mean(axis=1) + 1e-12)
    db = 20.0 * np.log10(rms)
    floor = float(np.percentile(db, 15))
    speech = (db > max(-42.0, floor + 10.0)) & (db > -48.0)
    # Fill gaps shorter than 0.4s so a breath does not split speech.
    hop = 0.02
    min_gap_frames = int(0.4 / hop)
    mask = speech.copy()
    silent = 0
    start = None
    for i, flag in enumerate(speech):
        if not flag:
            if start is None:
                start = i
            silent += 1
        else:
            if start is not None and silent <= min_gap_frames:
                mask[start:i] = True
            start = None
            silent = 0
    speech_ratio = float(mask.mean()) if len(mask) else 0.0
    speech_db = db[mask]
    mean_db = float(speech_db.mean()) if len(speech_db) else -90.0
    clip_ratio = float(np.mean(np.abs(samples) >= 32760))
    times = np.flatnonzero(mask)
    if len(times) == 0:
        first = last = 0.0
    else:
        first = float(times[0] * hop)
        last = float(times[-1] * hop)
    long_gaps = []
    gap_start = None
    # Leading silence counts, and so does a hole in the middle.
    padded = np.concatenate([mask, [True]])
    for i, flag in enumerate(padded):
        if not flag and gap_start is None:
            gap_start = i
        elif flag and gap_start is not None:
            dur = (i - gap_start) * hop
            if dur >= 12.0:
                long_gaps.append(
                    {"start": gap_start * hop, "end": i * hop, "dur": dur}
                )
            gap_start = None
    max_gap = max((gap["dur"] for gap in long_gaps), default=0.0)
    return {
        "speech_ratio": speech_ratio,
        "mean_speech_db": mean_db,
        "clip_ratio": clip_ratio,
        "first_speech": first,
        "last_speech": last,
        "long_gaps": long_gaps,
        "max_gap": float(max_gap),
    }


def _checker_strength(gray: np.ndarray) -> float:
    """Contrast between a one-square shift and a two-square shift.

    A chessboard alternates, so moving one square changes the colour and moving
    two squares lands on a similar colour. Faces and toolbars do not.
    Tuned on the sample classes: a real board scores about 1.0 or more, a
    camera-only frame stays under 0.7.
    """
    image = gray.astype(np.float32)
    height, width = image.shape
    best = 0.0
    windows = (
        (0, height, 0, width),
        (int(height * 0.08), int(height * 0.92), int(width * 0.04), int(width * 0.72)),
        (int(height * 0.12), int(height * 0.88), int(width * 0.18), int(width * 0.82)),
    )
    for period in range(10, 56, 2):
        for y0, y1, x0, x1 in windows:
            crop = image[y0:y1, x0:x1]
            if crop.shape[0] <= 2 * period + 2 or crop.shape[1] <= 2 * period + 2:
                continue
            differ_x = float(np.abs(crop[:, :-period] - crop[:, period:]).mean())
            same_x = float(np.abs(crop[:, : -2 * period] - crop[:, 2 * period :]).mean())
            differ_y = float(np.abs(crop[:-period, :] - crop[period:, :]).mean())
            same_y = float(np.abs(crop[: -2 * period, :] - crop[2 * period :, :]).mean())
            if min(differ_x, differ_y) < 12:
                continue
            score = min(differ_x / (same_x + 4.0), differ_y / (same_y + 4.0))
            if score > best:
                best = score
    return best


def _camera_visible(gray: np.ndarray, rgb: np.ndarray, cascade) -> bool:
    """A large face, or a small webcam bubble in a top corner."""
    import cv2

    del rgb  # colour is unused; detection is on the grayscale frame
    height, width = gray.shape
    equalized = cv2_equalize(gray)
    faces = cascade.detectMultiScale(
        equalized, scaleFactor=1.12, minNeighbors=4, minSize=(18, 18)
    )
    for (x, y, fw, fh) in faces:
        cx = x + fw / 2
        area = (fw * fh) / (width * height)
        on_edge = cx < width * 0.32 or cx > width * 0.68 or y < height * 0.25
        if area > 0.03 or on_edge:
            return True
    corners = (
        gray[: int(height * 0.36), int(width * 0.70) :],
        gray[: int(height * 0.36), : int(width * 0.30)],
    )
    for crop in corners:
        if crop.size < 80:
            continue
        enlarged = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        found = cascade.detectMultiScale(
            cv2.equalizeHist(enlarged), scaleFactor=1.1, minNeighbors=3, minSize=(22, 22)
        )
        if len(found):
            return True
    return False


def cv2_equalize(gray: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.equalizeHist(gray)


def _empty_fraction(gray: np.ndarray) -> float:
    block = 24
    height, width = gray.shape
    total = 0
    empty = 0
    for y in range(0, height - block + 1, block):
        for x in range(0, width - block + 1, block):
            patch = gray[y : y + block, x : x + block]
            total += 1
            if float(patch.std()) < 10 and 25 <= float(patch.mean()) <= 95:
                empty += 1
    return empty / max(1, total)


def _blackbar_fraction(gray: np.ndarray) -> float:
    dark_cols = float((gray.mean(axis=0) < 16).mean())
    dark_rows = float((gray.mean(axis=1) < 16).mean())
    return max(dark_cols, dark_rows)


def analyze_frames(
    path: str,
    width: int,
    height: int,
    interval: float = 15.0,
    max_seconds: float | None = None,
    on_progress: Progress | None = None,
) -> dict:
    """Sample frames across the entire duration. One frame every `interval` seconds."""
    import cv2

    if width <= 0 or height <= 0:
        return {"frames": 0}
    out_w = 320
    out_h = max(2, int(round(height * out_w / width / 2.0) * 2))
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if max_seconds:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += [
        "-vf",
        f"fps=1/{interval},scale={out_w}:{out_h}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = out_w * out_h * 3
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    counts = {"board": 0, "face": 0, "empty": 0, "blackbar": 0, "freeze": 0}
    seen = 0
    prev = None
    freeze_run = 0
    assert proc.stdout is not None
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        rgb = np.frombuffer(buf, dtype=np.uint8).reshape(out_h, out_w, 3).copy()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        seen += 1
        if _checker_strength(gray) >= 0.85:
            counts["board"] += 1
        if _camera_visible(gray, rgb, cascade):
            counts["face"] += 1
        if _empty_fraction(gray) >= 0.34:
            counts["empty"] += 1
        if _blackbar_fraction(gray) >= 0.18:
            counts["blackbar"] += 1
        if prev is not None:
            delta = float(np.mean(np.abs(gray.astype(np.int16) - prev)))
            if delta < 1.1:
                freeze_run += 1
                if freeze_run >= 3:
                    counts["freeze"] += 1
            else:
                freeze_run = 0
        prev = gray
        if on_progress and seen % 20 == 0:
            on_progress(min(0.99, seen / 400), f"Looking at the screen ({seen} frames)")
    proc.wait()
    if seen == 0:
        return {"frames": 0, "interval_sec": interval}
    return {
        "frames": seen,
        "interval_sec": interval,
        "board_ratio": counts["board"] / seen,
        "face_ratio": counts["face"] / seen,
        "empty_ratio": counts["empty"] / seen,
        "blackbar_ratio": counts["blackbar"] / seen,
        "freeze_ratio": counts["freeze"] / seen,
    }
