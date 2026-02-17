#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[3]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import shutil
import subprocess
import re
import urllib.request
import uuid
import threading
from typing import Optional, List, Dict

import whisper
import spacy
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

app = FastAPI()

nlp = None
_whisper_model = None
TRANSCRIBE_LOCK = threading.Lock()


@app.on_event("startup")
def _startup():
    global nlp
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        raise RuntimeError(f"spaCy model load failed: {e}")


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def ensure_tools():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise HTTPException(status_code=500, detail="ffmpeg or ffprobe missing")


def download_file(url: str, dest_path: str):
    if not url:
        raise HTTPException(status_code=400, detail="video_url missing")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        with urllib.request.urlopen(url) as resp:
            status = getattr(resp, "status", 200)
            if status != 200:
                raise HTTPException(status_code=400, detail=f"download failed http {status}")
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"download failed: {e}")


def get_duration_or_zero(path: str) -> float:
    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ],
        capture_output=True, text=True
    )
    out = (r.stdout or "").strip()
    try:
        return float(out)
    except Exception:
        return 0.0


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def normalize_word(w: str) -> str:
    w = (w or "").strip().lower()
    w = w.replace("’", "'")
    w = re.sub(r"[^a-z']", "", w)
    return w


def strip_ass_tags(s: str) -> str:
    return re.sub(r"\{\\[^}]*\}", "", s or "").strip()


def ass_hex_color(rrggbb: str) -> str:
    s = (rrggbb or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join([c * 2 for c in s])
    if len(s) != 6 or any(c not in "0123456789abcdefABCDEF" for c in s):
        raise HTTPException(status_code=400, detail="sub_primary_hex must be like FFFF00")
    r = s[0:2]
    g = s[2:4]
    b = s[4:6]
    return f"&H{b}{g}{r}"


def build_sub_style(primary_hex: str) -> str:
    # Keep list fixed in the middle area for ALL events:
    # - Alignment=5 anchors center
    # - MarginV gives a stable y offset from true center
    # Smaller than before.
    return (
        f"Fontname=Arial,"
        f"Fontsize=16,"
        f"PrimaryColour={ass_hex_color(primary_hex)},"
        f"OutlineColour={ass_hex_color('000000')},"
        f"BorderStyle=1,"
        f"Outline=3,"
        f"Shadow=0,"
        f"Bold=1,"
        f"Alignment=5,"
        f"MarginL=0,"
        f"MarginR=0,"
        f"MarginV=130"
    )


def esc_ff_filter(s: str) -> str:
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def clamp_time(t: float, lo: float, hi: float) -> float:
    if t < lo:
        return lo
    if t > hi:
        return hi
    return t


def build_progress_list(slots: int, filled: Dict[int, str]) -> str:
    # Always exactly N lines so block height stays constant.
    lines = []
    for i in range(1, slots + 1):
        if i in filled and filled[i]:
            lines.append(f"{i}. {filled[i]}")
        else:
            lines.append(f"{i}.")
    return "\n".join(lines)


class ProcessReq(BaseModel):
    video_url: str = Field(..., description="Direct public mp4 url")
    slots: int = 5
    target_fps: int = 30
    sub_primary_hex: str = "FFFF00"
    logo_enabled: bool = True
    logo_url: Optional[str] = None
    output_prefix: str = "ReelFive_"


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/process")
def process(req: ProcessReq):
    ensure_tools()

    if nlp is None:
        raise HTTPException(status_code=500, detail="spaCy not loaded")

    slots = int(req.slots)
    if slots < 1:
        raise HTTPException(status_code=400, detail="slots must be >= 1")
    if slots > 10:
        raise HTTPException(status_code=400, detail="slots too high, max 10")

    work = f"/tmp/work_{uuid.uuid4().hex}"
    os.makedirs(work, exist_ok=True)

    in_path = os.path.join(work, "input.mp4")
    download_file(req.video_url, in_path)
    if not os.path.exists(in_path):
        raise HTTPException(status_code=400, detail="video download failed")

    duration = get_duration_or_zero(in_path)
    if duration <= 0:
        raise HTTPException(status_code=400, detail="ffprobe duration failed")

    model = get_whisper_model()

    with TRANSCRIBE_LOCK:
        try:
            result = model.transcribe(in_path, word_timestamps=True, fp16=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"whisper transcribe failed: {e}")

    words: List[Dict] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            token = (w.get("word") or "").strip()
            if token:
                words.append({"word": token, "start": float(w.get("start", 0) or 0)})

    if len(words) == 0:
        raise HTTPException(status_code=500, detail="no transcript words")

    docs = list(nlp.pipe([w["word"] for w in words]))
    for i, doc in enumerate(docs):
        words[i]["pos"] = doc[0].pos_ if len(doc) else "X"

    preferred = [w for w in words if w["pos"] in ("NOUN", "PROPN")]
    seg_len = duration / max(slots, 1)

    overlay = []
    for i in range(slots):
        seg_start = i * seg_len
        seg_end = seg_start + seg_len
        seg_words = [w for w in preferred if seg_start <= w["start"] < seg_end]
        chosen = seg_words[0] if len(seg_words) > 0 else words[min(i, len(words) - 1)]

        clean = normalize_word(chosen["word"]).upper()
        final_word = clean if clean else (chosen["word"] or "").strip().upper()
        final_word = strip_ass_tags(final_word)

        t = float(chosen["start"])
        t = clamp_time(t, 0.0, max(0.0, duration - 0.01))

        overlay.append({"slot": i + 1, "word": final_word, "time": t})

    overlay.sort(key=lambda x: x["time"])

    # Build progressive list subtitles. Always same position, same line count.
    srt_path = os.path.join(work, "subtitles.srt")
    min_chunk = 0.60

    filled: Dict[int, str] = {}
    events = []

    # Start with blanks from 0 to first timestamp
    first_time = overlay[0]["time"]
    if first_time < min_chunk:
        first_time = min_chunk
    events.append({"start": 0.0, "end": first_time, "text": build_progress_list(slots, filled)})

    # Then update list at each word time
    for i, item in enumerate(overlay):
        filled[item["slot"]] = item["word"]

        start = item["time"]
        if i < len(overlay) - 1:
            end = overlay[i + 1]["time"]
        else:
            end = duration

        if end - start < min_chunk:
            end = min(duration, start + min_chunk)

        events.append({"start": start, "end": end, "text": build_progress_list(slots, filled)})

    with open(srt_path, "w", encoding="utf-8") as f:
        idx = 1
        for ev in events:
            start = clamp_time(ev["start"], 0.0, duration)
            end = clamp_time(ev["end"], 0.0, duration)
            if end <= start:
                continue
            text = strip_ass_tags(ev["text"])
            f.write(f"{idx}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{text}\n\n")
            idx += 1

    sub_style = build_sub_style(req.sub_primary_hex)

    # Logo back on again.
    # If logo_url not provided, try local /app/logo.png
    use_logo = False
    logo_path = os.path.join(work, "logo.png")

    if req.logo_enabled:
        if req.logo_url:
            download_file(req.logo_url, logo_path)
            use_logo = os.path.exists(logo_path)
        else:
            local_logo = "/app/logo.png"
            if os.path.exists(local_logo):
                shutil.copy(local_logo, logo_path)
                use_logo = os.path.exists(logo_path)

    out_name = f"{req.output_prefix}{uuid.uuid4().hex}.mp4"
    out_path = os.path.join(work, out_name)

    srt_f = esc_ff_filter(srt_path)
    logo_f = esc_ff_filter(logo_path)

    # Subtitle filter (always same style, same placement)
    sub_filter = f"subtitles='{srt_f}':force_style='{sub_style}'"

    if use_logo:
        # Keep logo top-right with padding
        vf = (
            f"[0:v]fps={int(req.target_fps)},{sub_filter}[v];"
            f"movie='{logo_f}',scale=160:-1[logo];"
            f"[v][logo]overlay=W-w-30:30:format=auto"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", in_path,
            "-filter_complex", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-r", str(int(req.target_fps)),
            "-c:a", "aac", "-b:a", "128k",
            out_path
        ]
    else:
        vf = f"fps={int(req.target_fps)},{sub_filter}"
        cmd = [
            "ffmpeg", "-y",
            "-i", in_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-r", str(int(req.target_fps)),
            "-c:a", "aac", "-b:a", "128k",
            out_path
        ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "")[-2000:]
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {tail}")

    if not os.path.exists(out_path):
        raise HTTPException(status_code=500, detail="output mp4 missing")

    cleanup = BackgroundTask(shutil.rmtree, work, ignore_errors=True)
    return FileResponse(out_path, media_type="video/mp4", filename=out_name, background=cleanup)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




