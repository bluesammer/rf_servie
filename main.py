#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[3]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import subprocess
import json
import shutil
import re
import whisper
import spacy
from pathlib import Path
import urllib.request

# ---------- ENV HELPERS ----------

def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if isinstance(v, str) and v.strip() else default

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = v.strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return default

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    s = v.strip()
    if not s:
        return default
    try:
        return int(s)
    except:
        return default

# ---------- ENV SWITCHES ----------

RUN_ENV = env_str("RUN_ENV", "railway" if os.getenv("RAILWAY_ENVIRONMENT") else "local")
BASE_DIR = "/tmp" if RUN_ENV == "railway" else os.getcwd()

VIDEO_SOURCE = env_str("VIDEO_SOURCE", "url").lower()  # local | url
VIDEO_NAME = env_str("VIDEO_NAME", "tiktest4.mp4")
VIDEO_URL = env_str("VIDEO_URL", "")

LOGO_ENABLED = env_bool("LOGO_ENABLED", False)
LOGO_SOURCE = env_str("LOGO_SOURCE", "local").lower()  # local | url
LOGO_NAME = env_str("LOGO_NAME", "logo.png")
LOGO_URL = env_str("LOGO_URL", "")

STOPWORDS_NAME = env_str("STOPWORDS_NAME", "stopwords.json")

SLOTS = env_int("SLOTS", 5)
TARGET_FPS = env_int("TARGET_FPS", 30)
SUB_PRIMARY_HEX = env_str("SUB_PRIMARY_HEX", "FFFF00")

OUTPUT_PREFIX = env_str("OUTPUT_PREFIX", "ReelFive_")

# ---------- IDLE GUARD ----------
# Railway runs main.py on container start.
# If you do not provide VIDEO_URL, exit cleanly so it does not restart-loop.

if VIDEO_SOURCE == "url" and not VIDEO_URL:
    print("No VIDEO_URL provided. Service idle.")
    raise SystemExit(0)

# ---------- PATHS ----------

video_path = os.path.join(BASE_DIR, VIDEO_NAME)
logo_path = os.path.join(BASE_DIR, LOGO_NAME)
stopwords_json_path = os.path.join(BASE_DIR, STOPWORDS_NAME)

out_name = f"{OUTPUT_PREFIX}{Path(VIDEO_NAME).stem}.mp4"
srt_name = "subtitles.srt"

output_path = os.path.join(BASE_DIR, out_name)
srt_path = os.path.join(BASE_DIR, srt_name)

# ---------- DOWNLOAD HELPERS ----------

def download_file(url: str, dest_path: str):
    if not url:
        raise SystemExit("Missing URL")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print("Downloading:", url)
    try:
        with urllib.request.urlopen(url) as resp:
            if resp.status != 200:
                raise SystemExit(f"Download failed. HTTP {resp.status}")
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)
    except Exception as e:
        raise SystemExit(f"Download failed: {e}")
    print("Saved:", dest_path, "bytes:", os.path.getsize(dest_path))

# ---------- ENSURE INPUT VIDEO EXISTS ----------

if VIDEO_SOURCE == "url":
    download_file(VIDEO_URL, video_path)

if not os.path.exists(video_path):
    raise SystemExit(f"Video not found: {video_path}")

# ---------- ENSURE LOGO EXISTS IF NEEDED ----------

if LOGO_ENABLED and LOGO_SOURCE == "url":
    if not LOGO_URL:
        print("Logo enabled but LOGO_URL missing, continuing without logo")
    else:
        download_file(LOGO_URL, logo_path)

# ---------- CHECK FFMPEG ----------

if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
    raise SystemExit("FFmpeg/FFprobe not installed")

# ---------- LOAD SPACY ----------
# Do not download here. Dockerfile installs the model.

try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print("Failed to load spaCy model en_core_web_sm.")
    print("Fix: install the model in Dockerfile using the .whl URL.")
    print("Error:", str(e))
    raise SystemExit(1)

# ---------- STYLE ----------

def ass_hex_color(rrggbb: str) -> str:
    s = (rrggbb or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join([c * 2 for c in s])
    if len(s) != 6 or any(c not in "0123456789abcdefABCDEF" for c in s):
        raise SystemExit("SUB_PRIMARY_HEX must be like FFFF00")
    r = s[0:2]
    g = s[2:4]
    b = s[4:6]
    return f"&H{b}{g}{r}"

def build_sub_style():
    return (
        f"Fontsize=10,"
        f"PrimaryColour={ass_hex_color(SUB_PRIMARY_HEX)},"
        f"OutlineColour={ass_hex_color('000000')},"
        f"Outline=3,"
        f"Alignment=7,"
        f"MarginL=20,"
        f"MarginV=50"
    )

SUB_STYLE = build_sub_style()

# ---------- HELPERS ----------

def get_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    out = (r.stdout or "").strip()
    if not out:
        raise SystemExit("ffprobe failed to read duration")
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

# ---------- LOGS ----------

print("Run env:", RUN_ENV)
print("Base dir:", BASE_DIR)
print("Video source:", VIDEO_SOURCE)
print("Video path:", video_path)
print("Logo enabled:", LOGO_ENABLED)
print("Output:", output_path)

# ---------- DURATION ----------

duration = get_duration(video_path)
print("Duration seconds:", round(duration, 2))

# ---------- TRANSCRIBE ----------

print("Loading Whisper model: base")
model = whisper.load_model("base")
result = model.transcribe(video_path, word_timestamps=True, fp16=False)

words = []
for seg in result.get("segments", []):
    for w in seg.get("words", []):
        token = (w.get("word") or "").strip()
        if token:
            words.append({
                "word": token,
                "start": float(w.get("start", 0) or 0)
            })

if not words:
    raise SystemExit("No transcript words extracted")

print("Transcript words:", len(words))

# ---------- POS TAG + PICK WORDS ----------

docs = list(nlp.pipe([w["word"] for w in words]))
for i, doc in enumerate(docs):
    words[i]["pos"] = doc[0].pos_ if len(doc) else "X"

preferred = [w for w in words if w["pos"] in ("NOUN", "PROPN")]

seg_len = duration / max(SLOTS, 1)
overlay = []

for i in range(SLOTS):
    seg_start = i * seg_len
    seg_end = seg_start + seg_len
    seg_words = [w for w in preferred if seg_start <= w["start"] < seg_end]

    if seg_words:
        chosen = seg_words[0]
    else:
        chosen = words[min(i, len(words) - 1)]

    clean = normalize_word(chosen["word"]).upper()
    final_word = clean if clean else (chosen["word"] or "").strip().upper()

    overlay.append({
        "slot": i + 1,
        "word": final_word,
        "time": float(chosen["start"])
    })

print("Overlay plan:")
for item in overlay:
    print(item["slot"], item["word"], "at", round(item["time"], 2))

# ---------- WRITE SRT ----------

with open(srt_path, "w", encoding="utf-8") as f:
    idx = 1
    for item in overlay:
        f.write(f"{idx}\n")
        f.write(f"{format_time(item['time'])} --> {format_time(duration)}\n")
        f.write(f"{item['slot']}. {item['word']}\n\n")
        idx += 1

print("Wrote SRT:", srt_path)

# ---------- FFMPEG BURN ----------

use_logo = LOGO_ENABLED and os.path.exists(logo_path)
if LOGO_ENABLED and not os.path.exists(logo_path):
    print("Logo enabled but logo file missing, continuing without logo")

if use_logo:
    vf = (
        f"[0:v]fps={TARGET_FPS},subtitles='{srt_path}':force_style='{SUB_STYLE}'[v];"
        f"movie='{logo_path}',scale=200:-1[logo];"
        f"[v][logo]overlay=30:H-h-30:format=auto"
    )
else:
    vf = f"fps={TARGET_FPS},subtitles='{srt_path}':force_style='{SUB_STYLE}'"

cmd = [
    "ffmpeg", "-y",
    "-i", video_path,
    "-filter_complex", vf,
    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    "-r", str(TARGET_FPS),
    "-c:a", "aac", "-b:a", "128k",
    output_path
]

print("Running ffmpeg...")
p = subprocess.run(cmd, capture_output=True, text=True)

if p.returncode != 0:
    print("FFmpeg failed. Last stderr:")
    print((p.stderr or "")[-3000:])
    raise SystemExit(1)

print("Done")
print("Output path:", output_path)
print("Output exists:", os.path.exists(output_path))
if os.path.exists(output_path):
    mb = os.path.getsize(output_path) / (1024 * 1024)
    print("Output size MB:", round(mb, 2))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




