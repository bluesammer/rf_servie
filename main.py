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
from pathlib import Path
from typing import Optional

import whisper
import spacy
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI()

# ---------- LOAD MODELS ON STARTUP ----------

try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise RuntimeError(f"spaCy model load failed: {e}")

# Whisper loads on first request, keeps in memory after
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model

# ---------- HELPERS ----------

def download_file(url: str, dest_path: str):
    if not url:
        raise HTTPException(status_code=400, detail="video_url missing")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        with urllib.request.urlopen(url) as resp:
            if getattr(resp, "status", 200) != 200:
                raise HTTPException(status_code=400, detail=f"download failed http {resp.status}")
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"download failed: {e}")

def get_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    out = (r.stdout or "").strip()
    if not out:
        raise HTTPException(status_code=500, detail="ffprobe failed")
    return float(out)

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
    return (
        f"Fontsize=10,"
        f"PrimaryColour={ass_hex_color(primary_hex)},"
        f"OutlineColour={ass_hex_color('000000')},"
        f"Outline=3,"
        f"Alignment=7,"
        f"MarginL=20,"
        f"MarginV=50"
    )

def ensure_tools():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise HTTPException(status_code=500, detail="ffmpeg or ffprobe missing")

# ---------- API ----------

class ProcessReq(BaseModel):
    video_url: str = Field(..., description="Direct public mp4 url")
    slots: int = 5
    target_fps: int = 30
    sub_primary_hex: str = "FFFF00"

    logo_enabled: bool = False
    logo_url: Optional[str] = None

    output_prefix: str = "ReelFive_"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/process")
def process(req: ProcessReq):
    ensure_tools()

    work = "/tmp/work"
    os.makedirs(work, exist_ok=True)

    in_name = "input.mp4"
    in_path = os.path.join(work, in_name)

    download_file(req.video_url, in_path)

    if not os.path.exists(in_path):
        raise HTTPException(status_code=400, detail="video download failed")

    duration = get_duration(in_path)

    model = get_whisper_model()
    result = model.transcribe(in_path, word_timestamps=True, fp16=False)

    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            token = (w.get("word") or "").strip()
            if token:
                words.append({"word": token, "start": float(w.get("start", 0) or 0)})

    if not words:
        raise HTTPException(status_code=500, detail="no transcript words")

    docs = list(nlp.pipe([w["word"] for w in words]))
    for i, doc in enumerate(docs):
        words[i]["pos"] = doc[0].pos_ if len(doc) else "X"

    preferred = [w for w in words if w["pos"] in ("NOUN", "PROPN")]
    seg_len = duration / max(req.slots, 1)

    overlay = []
    for i in range(req.slots):
        seg_start = i * seg_len
        seg_end = seg_start + seg_len
        seg_words = [w for w in preferred if seg_start <= w["start"] < seg_end]
        chosen = seg_words[0] if seg_words else words[min(i, len(words) - 1)]

        clean = normalize_word(chosen["word"]).upper()
        final_word = clean if clean else (chosen["word"] or "").strip().upper()

        overlay.append({"slot": i + 1, "word": final_word, "time": float(chosen["start"])})

    srt_path = os.path.join(work, "subtitles.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        idx = 1
        for item in overlay:
            f.write(f"{idx}\n")
            f.write(f"{format_time(item['time'])} --> {format_time(duration)}\n")
            f.write(f"{item['slot']}. {item['word']}\n\n")
            idx += 1

    sub_style = build_sub_style(req.sub_primary_hex)

    use_logo = False
    logo_path = os.path.join(work, "logo.png")
    if req.logo_enabled and req.logo_url:
        download_file(req.logo_url, logo_path)
        use_logo = os.path.exists(logo_path)

    out_name = f"{req.output_prefix}{Path(in_name).stem}.mp4"
    out_path = os.path.join(work, out_name)

    if use_logo:
        vf = (
            f"[0:v]fps={req.target_fps},subtitles='{srt_path}':force_style='{sub_style}'[v];"
            f"movie='{logo_path}',scale=200:-1[logo];"
            f"[v][logo]overlay=30:H-h-30:format=auto"
        )
    else:
        vf = f"fps={req.target_fps},subtitles='{srt_path}':force_style='{sub_style}'"

    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-filter_complex", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-r", str(req.target_fps),
        "-c:a", "aac", "-b:a", "128k",
        out_path
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "")[-2000:]
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {tail}")

    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename=out_name
    )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




