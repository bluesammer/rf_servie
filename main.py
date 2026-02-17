#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[3]:


# --- SETUP ---

#http://localhost:8889/notebooks/Video_add_nouns_12345/BEST_NOUN_5_V9_MASTER_WOrksLogo_WOW.ipynb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only

import subprocess, json, shutil, re
import whisper
import spacy

# ---------- ENV SWITCHES ----------
RUN_ENV = "local"       # "local" or "railway"
LOGO_ENABLED = False     # True or False

# Auto-detect Railway
if os.getenv("RAILWAY_ENVIRONMENT"):
    RUN_ENV = "railway"

# ---------- CONFIG ----------
SLOTS = 5
TARGET_FPS = 30  # helps with VFR mobile clips

# Easy color control (normal hex like FFFF00). Change this later any time.
SUB_PRIMARY_HEX = "FFFF00"   # yellow
# SUB_PRIMARY_HEX = "FFFFFF" # white
# SUB_PRIMARY_HEX = "00FFFF" # cyan

# Output filenames
VIDEO_NAME = "tiktest4.mp4"
LOGO_NAME = "logo.png"
OUT_NAME = "out_reel_final77.mp4"
SRT_NAME = "subtitles.srt"
STOPWORDS_NAME = "stopwords.json"

# ---------- PATH SETUP ----------
if RUN_ENV == "railway":
    BASE_DIR = "/tmp"   # writable on Railway
else:
    BASE_DIR = os.getcwd()

video_path = os.path.join(BASE_DIR, VIDEO_NAME)
logo_path = os.path.join(BASE_DIR, LOGO_NAME)         # optional
output_path = os.path.join(BASE_DIR, OUT_NAME)
srt_path = os.path.join(BASE_DIR, SRT_NAME)
stopwords_json_path = os.path.join(BASE_DIR, STOPWORDS_NAME)

# ---------- 0. CHECK FFMPEG ----------
if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
    raise SystemExit("FFmpeg/FFprobe not found. Install from https://ffmpeg.org/download.html")

# ---------- 1. LOAD SPACY MODEL ----------
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ---------- STYLE HELPERS ----------
def ass_hex_color(rrggbb: str) -> str:
    s = (rrggbb or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join([c * 2 for c in s])
    if len(s) != 6 or any(c not in "0123456789abcdefABCDEF" for c in s):
        raise ValueError("Color must be hex like FFFF00 or #FFFF00")
    r = s[0:2]
    g = s[2:4]
    b = s[4:6]
    return f"&H{b}{g}{r}"  # ASS expects BBGGRR

def build_sub_style(
    fontsize: int = 10,
    primary_hex: str = "FFFF00",
    outline_hex: str = "000000",
    outline: int = 3,
    alignment: int = 7,
    margin_l: int = 20,
    margin_v: int = 50
) -> str:
    return (
        f"Fontsize={fontsize},"
        f"PrimaryColour={ass_hex_color(primary_hex)},"
        f"OutlineColour={ass_hex_color(outline_hex)},"
        f"Outline={outline},"
        f"Alignment={alignment},"
        f"MarginL={margin_l},"
        f"MarginV={margin_v}"
    )

SUB_STYLE = build_sub_style(
    fontsize=10,
    primary_hex=SUB_PRIMARY_HEX,
    outline_hex="000000",
    outline=3,
    alignment=7,
    margin_l=20,
    margin_v=50
)

# ---------- HELPERS ----------
def get_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    out = (r.stdout or "").strip()
    if not out:
        raise RuntimeError("ffprobe failed to read duration")
    return float(out)

def get_rotation(path: str) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream_tags=rotate",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    raw = (r.stdout or "").strip()
    if not raw:
        return 0
    try:
        rot = int(raw) % 360
        if rot in (0, 90, 180, 270):
            return rot
    except:
        pass
    return 0

def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    if millis == 1000:
        secs += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def escape_ffmpeg_path(p: str) -> str:
    return p.replace("\\", "/").replace(":", r"\:")

def load_stopwords(json_path: str):
    if not os.path.exists(json_path):
        print(f"⚠️ stopwords.json not found at {json_path}. Continuing without stopwords.")
        return set(), []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stopwords = set(w.strip().lower() for w in data.get("stopwords", []) if isinstance(w, str) and w.strip())
    patterns = []
    for p in data.get("stop_patterns", []):
        if isinstance(p, str) and p.strip():
            patterns.append(re.compile(p.strip(), re.IGNORECASE))
    return stopwords, patterns

def normalize_word(w: str) -> str:
    w = (w or "").strip().lower()
    w = w.replace("’", "'")
    w = re.sub(r"[^a-z']", "", w)
    return w

def is_blocked(word: str, stopwords: set, patterns: list) -> bool:
    nw = normalize_word(word)
    if not nw:
        return True
    if nw in stopwords:
        return True
    for pat in patterns:
        if pat.match(nw):
            return True
    return False

# ---------- BASIC FILE CHECKS ----------
if not os.path.exists(video_path):
    raise SystemExit(f"Video not found: {video_path}")

print("🧭 Run env:", RUN_ENV)
print("🎞️ Video path:", video_path)
print("🖼️ Logo path:", logo_path)
print("🧾 Stopwords path:", stopwords_json_path)

# ---------- 2. GET VIDEO DURATION ----------
duration = get_duration(video_path)
rotation = get_rotation(video_path)
print("🎬 Video duration:", round(duration, 2), "seconds")
print("📱 Rotation metadata:", rotation)

# ---------- 3. TRANSCRIBE AUDIO ----------
def transcribe_video(video: str):
    print("🎤 Transcribing on CPU...")
    model = whisper.load_model("base")
    result = model.transcribe(video, word_timestamps=True, fp16=False)

    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            token = (w.get("word") or "").strip()
            if token:
                words.append({
                    "word": token,
                    "start": float(w.get("start", 0) or 0),
                    "end": float(w.get("end", 0) or 0)
                })

    if not words:
        raise RuntimeError("No words extracted from Whisper output")

    print(f"✅ Transcript words: {len(words)}")
    return words

words = transcribe_video(video_path)

# ---------- 4. POS TAG + PICK OVERLAY WORDS ----------
stopwords_set, stop_patterns = load_stopwords(stopwords_json_path)

def add_pos_tags(words_list):
    docs = list(nlp.pipe([w["word"] for w in words_list]))
    for i, doc in enumerate(docs):
        if len(doc) >= 1:
            words_list[i]["pos"] = doc[0].pos_
        else:
            words_list[i]["pos"] = "X"

def pick_overlay_words(words_list, total_dur: float, slots: int):
    add_pos_tags(words_list)

    preferred = [
        w for w in words_list
        if w.get("pos") in ("NOUN", "PROPN") and not is_blocked(w["word"], stopwords_set, stop_patterns)
    ]

    seg_len = total_dur / slots
    overlay = []

    for i in range(slots):
        seg_start = i * seg_len
        seg_end = seg_start + seg_len

        seg_pref = [w for w in preferred if seg_start <= w["start"] < seg_end]
        if seg_pref:
            chosen = seg_pref[0]
        else:
            seg_any = [
                w for w in words_list
                if seg_start <= w["start"] < seg_end and not is_blocked(w["word"], stopwords_set, stop_patterns)
            ]
            chosen = seg_any[0] if seg_any else words_list[-1]

        clean = normalize_word(chosen["word"]).upper()
        final_word = clean if clean else chosen["word"].upper()

        overlay.append({
            "slot": i + 1,
            "word": final_word,
            "time": round(float(chosen["start"]), 2)
        })

    return overlay

overlay_plan = pick_overlay_words(words, duration, SLOTS)

print("\n📋 OVERLAY PLAN - Words will appear at these times:")
for item in overlay_plan:
    print(f"   Slot {item['slot']}: '{item['word']}' appears at {item['time']}s")
print()

# ---------- 5. CREATE SRT SUBTITLE FILE ----------
with open(srt_path, "w", encoding="utf-8") as f:
    subtitle_num = 1
    for item in overlay_plan:
        slot = item["slot"]
        word = item["word"]
        start_time = float(item["time"])
        end_time = float(duration)

        f.write(f"{subtitle_num}\n")
        f.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
        f.write(f"{slot}. {word}\n\n")
        subtitle_num += 1

print(f"📝 Created subtitle file: {srt_path}")

# ---------- 6. BURN SUBTITLES + LOGO INTO VIDEO ----------
print("\n🚀 Burning subtitles and logo into video...")

use_logo = LOGO_ENABLED and os.path.exists(logo_path)
if use_logo:
    print("✅ Logo enabled and found")
elif LOGO_ENABLED and not os.path.exists(logo_path):
    print("⚠️ Logo enabled but file missing, proceeding without logo")
else:
    print("ℹ️ Logo disabled")

srt_path_escaped = escape_ffmpeg_path(srt_path)
logo_path_escaped = escape_ffmpeg_path(logo_path)

rotate_filter = ""
if rotation == 90:
    rotate_filter = "transpose=1,"
elif rotation == 270:
    rotate_filter = "transpose=2,"
elif rotation == 180:
    rotate_filter = "hflip,vflip,"

fps_filter = f"fps={TARGET_FPS},"

if use_logo:
    vf = (
        f"[0:v]{rotate_filter}{fps_filter}"
        f"subtitles='{srt_path_escaped}':force_style='{SUB_STYLE}'[v];"
        f"movie='{logo_path_escaped}',scale=200:-1[logo];"
        f"[v][logo]overlay=30:H-h-30:format=auto"
    )
else:
    vf = (
        f"{rotate_filter}{fps_filter}"
        f"subtitles='{srt_path_escaped}':force_style='{SUB_STYLE}'"
    )

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

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print("\n❌ FFmpeg error (last 3000 chars):")
    print(result.stderr[-3000:])
    raise SystemExit(1)

print("\n✅ Video created successfully!")
print(f"\n📁 Output: {output_path}")
print(f"   File exists: {os.path.exists(output_path)}")
if os.path.exists(output_path):
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   File size: {size_mb:.2f} MB")

print("\n💡 Words will appear at these times and STAY on screen:")
for item in overlay_plan:
    print(f"   {item['time']}s: {item['slot']}. {item['word']}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




