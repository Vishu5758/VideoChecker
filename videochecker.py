import streamlit as st
import os
import pandas as pd
import datetime
import hashlib
import tempfile
from io import BytesIO
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import subprocess
import json

# ----------------------------- SETUP -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MIN_DURATION_DEFAULT = 20  # seconds
MAX_DURATION_DEFAULT = 30  # seconds
SUPPORTED_FORMATS = ["mp4", "mov", "avi", "mkv", "webm"]
MAX_FILE_SIZE_MB = 500
ALLOWED_RESOLUTIONS = [(1280, 720), (1920, 1080), (3840, 2160)]

def is_ffmpeg_available():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except FileNotFoundError:
        return False

def get_ffmpeg_info(path):
    """Run ffprobe, return dict with duration, width, height, framerate, bitrate, has_audio."""
    if not is_ffmpeg_available():
        return {}
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        logger.error(f"ffprobe error: {res.stderr}")
        return {}
    data = json.loads(res.stdout)
    out = {}
    fmt = data.get("format", {})
    if "duration" in fmt:
        out["duration"] = float(fmt["duration"])
    if "bit_rate" in fmt:
        out["bitrate"] = f"{int(fmt['bit_rate'])//1000} Kbps"
    streams = data.get("streams", [])
    for s in streams:
        if s.get("codec_type") == "video":
            out["width"] = s.get("width")
            out["height"] = s.get("height")
            if "r_frame_rate" in s:
                num, den = map(int, s["r_frame_rate"].split("/"))
                out["framerate"] = round(num/den, 2) if den else "Unknown"
        elif s.get("codec_type") == "audio":
            out["has_audio"] = True
    return out

def get_file_info(path):
    """Gather file-size and then overlay any ffprobe info we can."""
    size_mb = os.path.getsize(path) / (1024*1024)
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    info = {
        "size_mb": round(size_mb, 2),
        "format": ext,
        "width": "Unknown",
        "height": "Unknown",
        "duration": None,
        "framerate": "Unknown",
        "has_audio": False,
        "bitrate": "Unknown"
    }
    ff = get_ffmpeg_info(path)
    for k in ("width","height","duration","framerate","bitrate","has_audio"):
        if k in ff:
            info[k] = ff[k]
    return info

def extract_metadata(path):
    try:
        ctime = datetime.datetime.fromtimestamp(os.path.getctime(path))
        return {
            "creation_date": ctime.strftime("%Y-%m-%d %H:%M:%S"),
            "software": "Unknown",
            "author": "Unknown",
            "raw_metadata": {"filename": os.path.basename(path)}
        }
    except Exception as e:
        logger.error(f"metadata error: {e}")
        return {"creation_date": "Unknown", "software":"Unknown", "author":"Unknown", "raw_metadata":{}}

def extract_thumbnail(path):
    """Grab a single frame at halfway; return bytes or None."""
    if not is_ffmpeg_available():
        return None
    info = get_file_info(path)
    if not info.get("duration"):
        return None
    midpoint = info["duration"] / 2
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        thumb_path = tmp.name
    cmd = [
        "ffmpeg",
        "-ss", str(midpoint),
        "-i", path,
        "-vframes", "1",
        "-q:v", "2",
        thumb_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.path.exists(thumb_path):
        with open(thumb_path, "rb") as f:
            data = f.read()
        os.unlink(thumb_path)
        return data
    return None

def check_ai_indicators(path, md, vp):
    indicators = []
    score = 0
    fname = md["raw_metadata"].get("filename","").lower()
    kws = ["dall-e","midjourney","stable diffusion","runway","synthesia","deepfake","neural","ai","generated","openai"]
    if any(k in fname for k in kws):
        indicators.append(f"AI keyword: {fname}")
        score += 30
    w,h = vp.get("width"), vp.get("height")
    try:
        if isinstance(w,int) and w==h and (w&(w-1)==0):
            indicators.append(f"Power-of-2 res: {w}×{h}")
            score += 20
        if isinstance(w,int) and isinstance(h,int) and (w,h) not in ALLOWED_RESOLUTIONS:
            indicators.append(f"Unusual res: {w}×{h}")
            score += 15
    except:
        pass
    size,dur = vp["size_mb"], vp["duration"] or 0
    if dur>0 and size/dur < 0.1:
        indicators.append("Very low size/duration")
        score += 10
    score = min(score,100)
    if score>70: lik="High"
    elif score>30: lik="Medium"
    else: lik="Low"
    return {"ai_indicators": indicators, "ai_score": score, "ai_likelihood": lik}

def analyze_video(path, name):
    info = get_file_info(path)
    md   = extract_metadata(path)
    ai   = check_ai_indicators(path, md, info)
    dur  = info["duration"] or 0
    has_audio = info["has_audio"]
    duration_status = "PASS" if MIN_DURATION_DEFAULT <= dur <= MAX_DURATION_DEFAULT else "FAIL"
    audio_status    = "FAIL" if has_audio else "PASS"
    res = f"{info['width']}×{info['height']}"
    resolution_check = "SD"
    try:
        if int(info["width"])>=1280: resolution_check="HD+"
    except: pass
    return {
        "Filename": name,
        "Duration (s)": round(dur,2),
        "Duration Status": duration_status,
        "Audio Present": "Yes" if has_audio else "No",
        "Audio Status": audio_status,
        "Resolution": res,
        "Resolution Check": resolution_check,
        "Codec": info["format"],
        "Framerate": info["framerate"],
        "File Size (MB)": info["size_mb"],
        "Bitrate": info["bitrate"],
        "Creation Date": md["creation_date"],
        "AI Likelihood": ai["ai_likelihood"],
        "File Hash": hashlib.md5(open(path,"rb").read(1024*1024)).hexdigest()[:10]+"...",
        "Software": md["software"],
        "_ai_indicators": ai["ai_indicators"],
        "_metadata": md,
        "_temp_path": path
    }

def plot_ai_likelihood(results):
    vals = [r["AI Likelihood"] for r in results]
    counts = pd.Series(vals).value_counts().reindex(["Low","Medium","High"],fill_value=0)
    plt.figure(figsize=(6,4))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title("AI Likelihood Distribution")
    plt.xlabel("Likelihood")
    plt.ylabel("Count")
    buf = BytesIO()
    plt.savefig(buf,format="png",bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf.getvalue()

# ----------------------------- STREAMLIT APP -----------------------------
st.set_page_config(page_title="Video Prompt Validator", layout="wide")
ffmpeg_ok = is_ffmpeg_available()

st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<style>
.stButton>button { background:#2563eb; color:#fff; border-radius:.375rem; padding:.5rem 1rem }
.stButton>button:hover { background:#1e40af }
.sidebar .stButton>button { width:100% }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="text-3xl font-bold mb-4">🎥 Video Prompt Validation Tool</h1>', unsafe_allow_html=True)
if not ffmpeg_ok:
    st.warning("⚠️ FFmpeg/ffprobe not found; feature set will be limited.", icon="⚠️")

# Session state
if "results" not in st.session_state:
    st.session_state.results = []
    st.session_state.temp_files = []
    st.session_state.selected_video = None
    st.session_state.processing = False

# Sidebar controls
with st.sidebar:
    min_dur = st.slider("Min Duration (s)", 1, 60, MIN_DURATION_DEFAULT, key="min_dur")
    max_dur = st.slider("Max Duration (s)", min_dur, 180, MAX_DURATION_DEFAULT, key="max_dur")
    check_ai = st.checkbox("Check AI Indicators", True)
    gen_report = st.checkbox("Generate Report", True)
    if st.button("Clear All"):
        for f in st.session_state.temp_files:
            try: os.unlink(f)
            except: pass
        st.session_state.results.clear()
        st.session_state.temp_files.clear()
        st.session_state.selected_video = None
        st.experimental_rerun()

col1, col2 = st.columns([3,1])

with col1:
    st.markdown("## Upload Videos")
    files = st.file_uploader("Select files", type=SUPPORTED_FORMATS, accept_multiple_files=True)
    if files:
        if any(f.size > MAX_FILE_SIZE_MB*1024*1024 for f in files):
            st.error(f"One or more exceed {MAX_FILE_SIZE_MB}MB")
        elif st.button("Analyze Videos", disabled=st.session_state.processing):
            st.session_state.processing = True
            prog = st.progress(0)
            new = []
            for i, f in enumerate(files):
                with tempfile.NamedTemporaryFile(delete=False, suffix="."+f.name.split(".")[-1]) as tmp:
                    tmp.write(f.getvalue())
                    path = tmp.name
                st.session_state.temp_files.append(path)
                new.append(analyze_video(path, f.name))
                prog.progress((i+1)/len(files))
            st.session_state.results.extend(new)
            if not st.session_state.selected_video:
                st.session_state.selected_video = new[0]["Filename"]
            st.session_state.processing = False
            st.experimental_rerun()

    if st.session_state.results:
        cols = ["Filename","Duration (s)","Duration Status","Audio Present","Audio Status","Resolution","Framerate","File Size (MB)","AI Likelihood"]
        df = pd.DataFrame([{c:r.get(c) for c in cols} for r in st.session_state.results])
        def hl(r):
            bg=""
            if r["Duration Status"]=="FAIL": bg="#fee2e2"
            elif r["Audio Status"]=="FAIL": bg="#fef3c7"
            elif r["AI Likelihood"]=="High": bg="#ffedd5"
            return [bg]*len(r)
        styled = df.style.apply(hl,axis=1)
        st.markdown("## Results")
        st.dataframe(styled, use_container_width=True, height=350)

        st.markdown("## Summary")
        c1,c2,c3 = st.columns(3)
        c1.metric("Valid Dur", f"{sum(r['Duration Status']=='PASS' for r in st.session_state.results)}/{len(st.session_state.results)}")
        c2.metric("With Audio", f"{sum(r['Audio Status']=='FAIL' for r in st.session_state.results)}/{len(st.session_state.results)}")
        c3.metric("Likely AI", f"{sum(r['AI Likelihood']=='High' for r in st.session_state.results)}/{len(st.session_state.results)}")

        if check_ai:
            img = plot_ai_likelihood(st.session_state.results)
            if img: st.image(img, use_column_width=True)

        if gen_report:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = df.fillna("Unknown").replace([np.inf,-np.inf],"Unknown")
            csv = out.to_csv(index=False).encode()
            js  = json.dumps(out.to_dict("records"), default=str)
            r1,r2 = st.columns(2)
            with r1:
                st.download_button("Download CSV", data=csv, file_name=f"report_{ts}.csv", mime="text/csv")
            with r2:
                st.download_button("Download JSON", data=js, file_name=f"report_{ts}.json", mime="application/json")

with col2:
    st.markdown("## Video Details")
    if st.session_state.results:
        names = [r["Filename"] for r in st.session_state.results]
        sel = st.selectbox("Select Video", names, index=names.index(st.session_state.selected_video))
        st.session_state.selected_video = sel
        vd = next(r for r in st.session_state.results if r["Filename"]==sel)
        tp = vd["_temp_path"]
        if tp and os.path.exists(tp) and ffmpeg_ok:
            thumb = extract_thumbnail(tp)
            if thumb: st.image(thumb, width=250)
        for k in ("Duration (s)","Resolution","File Size (MB)","Codec","Framerate","Creation Date"):
            st.markdown(f"**{k}:** {vd.get(k)}")
        if check_ai and vd["_ai_indicators"]:
            st.markdown("### 🤖 AI Indicators")
            for ind in vd["_ai_indicators"]:
                st.markdown(f"- {ind}")

st.markdown("---")
st.markdown(f"**Criteria:** Duration {MIN_DURATION_DEFAULT}–{MAX_DURATION_DEFAULT}s · No audio · HD+ res")
