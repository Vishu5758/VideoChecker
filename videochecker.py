import streamlit as st
import os
import pandas as pd
import datetime
import hashlib
import tempfile
import base64
from io import BytesIO
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import subprocess
import json
import re
import cv2

# ----------------------------- SETUP -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MIN_DURATION_DEFAULT = 20  # seconds
MAX_DURATION_DEFAULT = 30  # seconds
SUPPORTED_FORMATS = ["mp4", "mov", "avi", "mkv", "webm"]
MAX_FILE_SIZE_MB = 500
ALLOWED_RESOLUTIONS = [(1280, 720), (1920, 1080), (3840, 2160)]

def is_ffmpeg_available():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def get_ffmpeg_info(file_path):
    if not is_ffmpeg_available():
        return {}
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode != 0:
            logger.error(f"FFprobe failed: {result.stderr}")
            return {}
        data = json.loads(result.stdout)
        fmt = data.get('format', {})
        streams = data.get('streams', [])
        info = {}
        # format-level
        if 'duration' in fmt:
            info['duration'] = float(fmt['duration'])
        if 'bit_rate' in fmt:
            info['bitrate'] = f"{int(fmt['bit_rate'])//1000} Kbps"
        # streams
        for stmp in streams:
            if stmp.get('codec_type') == 'video':
                info['width'] = stmp.get('width')
                info['height'] = stmp.get('height')
                if 'r_frame_rate' in stmp:
                    num, den = map(int, stmp['r_frame_rate'].split('/'))
                    info['framerate'] = round(num/den, 2) if den else 'Unknown'
            if stmp.get('codec_type') == 'audio':
                info['has_audio'] = True
        return info
    except Exception as e:
        logger.error(f"Error getting FFmpeg info: {e}")
        return {}

def get_file_info(file_path):
    """Get file info; try FFmpeg, fallback to OpenCV."""
    size_mb = os.path.getsize(file_path) / (1024*1024)
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    info = {
        'size_mb': round(size_mb, 2),
        'format': ext,
        'width': 'Unknown',
        'height': 'Unknown',
        'duration': None,
        'framerate': 'Unknown',
        'has_audio': False,
        'bitrate': 'Unknown'
    }

    ff = get_ffmpeg_info(file_path)
    if ff:
        info.update({
            'width': ff.get('width', info['width']),
            'height': ff.get('height', info['height']),
            'duration': ff.get('duration', info['duration']),
            'framerate': ff.get('framerate', info['framerate']),
            'bitrate': ff.get('bitrate', info['bitrate']),
            'has_audio': ff.get('has_audio', False)
        })
        return info

    # Fallback: OpenCV
    try:
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if w > 0 and h > 0:
                info['width'] = int(w)
                info['height'] = int(h)
            if fps > 0:
                info['framerate'] = round(fps, 2)
            if fps > 0 and frames > 0:
                info['duration'] = round(frames / fps, 2)
        cap.release()
    except Exception as e:
        logger.error(f"OpenCV fallback failed: {e}")
    return info

def check_duration(file_path):
    return get_file_info(file_path).get('duration')

def get_video_properties(file_path):
    fi = get_file_info(file_path)
    return {
        'width': fi['width'],
        'height': fi['height'],
        'codec': fi['format'],
        'framerate': fi['framerate'],
        'size_mb': fi['size_mb'],
        'bitrate': fi['bitrate']
    }

def has_audio_stream(file_path):
    fi = get_file_info(file_path)
    return fi['has_audio'], {}

def extract_metadata(file_path):
    try:
        ctime = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
        return {
            'creation_date': ctime.strftime('%Y-%m-%d %H:%M:%S'),
            'software': 'Unknown',
            'author': 'Unknown',
            'raw_metadata': {'filename': os.path.basename(file_path)}
        }
    except Exception as e:
        logger.error(f"Metadata error: {e}")
        return {'creation_date': 'Unknown', 'software': 'Unknown', 'author': 'Unknown', 'raw_metadata': {}}

def extract_thumbnail(file_path):
    if not is_ffmpeg_available():
        return None
    try:
        duration = check_duration(file_path)
        if not duration:
            return None
        mid = duration / 2
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            thumb_path = tmp.name
        cmd = [
            'ffmpeg',
            '-ss', str(mid),
            '-i', file_path,
            '-vframes', '1',
            '-q:v', '2',
            thumb_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if os.path.exists(thumb_path):
            with open(thumb_path, 'rb') as f:
                data = f.read()
            os.unlink(thumb_path)
            return data
    except Exception as e:
        logger.error(f"Thumbnail error: {e}")
    return None

def check_ai_indicators(file_path, metadata, video_props):
    ai_inds = []
    score = 0
    fname = metadata['raw_metadata'].get('filename','').lower()
    kws = ['dall-e','midjourney','stable diffusion','runway','synthesia','deepfake','neural','ai','generated','openai']
    if any(k in fname for k in kws):
        ai_inds.append(f"AI keyword: {fname}")
        score += 30
    w,h = video_props.get('width'), video_props.get('height')
    try:
        if isinstance(w,int) and w==h and (w&(w-1)==0):
            ai_inds.append(f"Power-of-2 resolution: {w}x{h}")
            score += 20
        if isinstance(w,int) and isinstance(h,int) and (w,h) not in ALLOWED_RESOLUTIONS:
            ai_inds.append(f"Unusual resolution: {w}x{h}")
            score += 15
    except:
        pass
    size = video_props.get('size_mb',0)
    dur = check_duration(file_path) or 0
    try:
        if dur>0 and size/dur<0.1:
            ai_inds.append("Very low size/duration")
            score += 10
    except:
        pass
    score = min(score,100)
    likelihood = "High" if score>70 else "Medium" if score>30 else "Low"
    return {'ai_indicators': ai_inds, 'ai_score': score, 'ai_likelihood': likelihood}

def analyze_video(video_path, filename):
    try:
        dur = check_duration(video_path) or 0
        has_audio, audio_props = has_audio_stream(video_path)
        vp = get_video_properties(video_path)
        md = extract_metadata(video_path)
        ai = check_ai_indicators(video_path, md, vp)
        file_hash = hashlib.md5(open(video_path,'rb').read(1024*1024)).hexdigest()[:10] + "..."
        duration_status = "PASS" if MIN_DURATION_DEFAULT <= dur <= MAX_DURATION_DEFAULT else "FAIL"
        audio_status = "FAIL" if has_audio else "PASS"
        res = f"{vp['width']}x{vp['height']}"
        resolution_check = "SD"
        try:
            if int(vp['width']) >= 1280:
                resolution_check = "HD+"
        except:
            pass
        size_mb = vp['size_mb']
        size_status = "Large" if size_mb>25 else "Medium" if size_mb>5 else "Small"
        return {
            "Filename": filename,
            "Duration (s)": round(dur,2),
            "Duration Status": duration_status,
            "Audio Present": "Yes" if has_audio else "No",
            "Audio Status": audio_status,
            "Resolution": res,
            "Resolution Check": resolution_check,
            "Codec": vp['codec'],
            "Framerate": vp['framerate'],
            "File Size (MB)": size_mb,
            "Bitrate": vp['bitrate'],
            "Creation Date": md['creation_date'],
            "AI Likelihood": ai['ai_likelihood'],
            "File Hash": file_hash,
            "Software": md['software'],
            "_audio_props": audio_props,
            "_ai_indicators": ai['ai_indicators'],
            "_metadata": md,
            "_temp_path": video_path
        }
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        return {
            "Filename": filename,
            "Duration (s)": "Error",
            "Duration Status": "FAIL",
            "Audio Present": "Unknown",
            "Audio Status": "Unknown",
            "Resolution": "Error",
            "Resolution Check": "Error",
            "Codec": "Error",
            "Framerate": "Error",
            "File Size (MB)": "Error",
            "Bitrate": "Error",
            "Creation Date": "Unknown",
            "AI Likelihood": "Unknown",
            "File Hash": "Error",
            "Software": "Unknown",
            "_audio_props": {},
            "_ai_indicators": [f"Error: {e}"],
            "_metadata": {"raw_metadata": {}},
            "_temp_path": video_path
        }

def plot_ai_likelihood(results):
    try:
        lik = [r["AI Likelihood"] for r in results]
        counts = pd.Series(lik).value_counts().reindex(["Low","Medium","High"],fill_value=0)
        plt.figure(figsize=(6,4))
        sns.barplot(x=counts.index, y=counts.values)
        plt.title("AI Likelihood Distribution")
        plt.xlabel("Likelihood")
        plt.ylabel("Count")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close()
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Plot error: {e}")
        return None

# ----------------------------- STREAMLIT APP -----------------------------
st.set_page_config(page_title="Video Prompt Validator", layout="wide")

ffmpeg_available = is_ffmpeg_available()

st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<style>
.stButton>button { background-color: #2563eb; color: white; border-radius: .375rem; padding: .5rem 1rem }
.stButton>button:hover { background-color: #1e40af }
.sidebar .stButton>button { width: 100% }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="text-3xl font-bold mb-4">🎥 Video Prompt Validation Tool</h1>', unsafe_allow_html=True)
if not ffmpeg_available:
    st.warning("⚠️ FFmpeg not installed; using fallback methods.", icon="⚠️")

if 'results' not in st.session_state:
    st.session_state.results = []
    st.session_state.temp_files = []
    st.session_state.selected_video = None
    st.session_state.processing = False

with st.sidebar:
    min_duration = st.slider("Min Duration (s)", 1, 60, MIN_DURATION_DEFAULT, key="min_duration")
    max_duration = st.slider("Max Duration (s)", min_duration, 180, MAX_DURATION_DEFAULT, key="max_duration")
    check_ai = st.checkbox("Check AI Indicators", value=True)
    extract_full_metadata = st.checkbox("Full Metadata", value=False)
    generate_report = st.checkbox("Generate Report", value=True)
    if st.button("Clear All"):
        for f in st.session_state.temp_files:
            try: os.unlink(f)
            except: pass
        st.session_state.results.clear()
        st.session_state.temp_files.clear()
        st.session_state.selected_video = None
        st.experimental_rerun()
    if st.button("Clean Temp Files"):
        count = 0
        for f in st.session_state.temp_files:
            if os.path.exists(f):
                os.unlink(f)
                count += 1
        st.session_state.temp_files = []
        st.success(f"Removed {count} temp files")

col1, col2 = st.columns([3,1])

with col1:
    st.markdown('## Upload Videos')
    uploaded = st.file_uploader(
        "Select files", type=SUPPORTED_FORMATS, accept_multiple_files=True, key="uploader"
    )
    if uploaded:
        if any(f.size > MAX_FILE_SIZE_MB*1024*1024 for f in uploaded):
            st.error(f"Files exceed {MAX_FILE_SIZE_MB}MB limit")
        else:
            if st.button("Analyze Videos", disabled=st.session_state.processing):
                st.session_state.processing = True
                progress = st.progress(0)
                new_res = []
                with st.spinner(f"Analyzing {len(uploaded)} videos..."):
                    for i, file in enumerate(uploaded):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix="."+file.name.split('.')[-1]) as tmp:
                                tmp.write(file.getvalue())
                                path = tmp.name
                            st.session_state.temp_files.append(path)
                            with st.spinner(f"Analyzing {file.name}..."):
                                new_res.append(analyze_video(path, file.name))
                        except Exception as e:
                            logger.error(f"Error: {e}")
                            new_res.append(analyze_video(path, file.name))
                        progress.progress((i+1)/len(uploaded))
                st.session_state.results.extend(new_res)
                if not st.session_state.selected_video and new_res:
                    st.session_state.selected_video = new_res[0]["Filename"]
                st.session_state.processing = False
                st.experimental_rerun()

    if st.session_state.results:
        display_cols = ["Filename","Duration (s)","Duration Status","Audio Present","Audio Status","Resolution","Framerate","File Size (MB)","AI Likelihood"]
        df = pd.DataFrame([{c: r.get(c,"") for c in display_cols} for r in st.session_state.results])
        def hl(r):
            color=''
            if r["Duration Status"]=="FAIL": color="#fee2e2"
            elif r["Audio Status"]=="FAIL": color="#fef3c7"
            elif r["AI Likelihood"]=="High": color="#ffedd5"
            return [color]*len(r)
        styled = df.style.apply(hl,axis=1)
        st.markdown('## Analysis Results')
        st.dataframe(styled, use_container_width=True, height=400)

        st.markdown('## Summary Statistics')
        c1, c2, c3 = st.columns(3)
        c1.metric("Valid Duration", f"{sum(r['Duration Status']=='PASS' for r in st.session_state.results)}/{len(st.session_state.results)}")
        c2.metric("With Audio", f"{sum(r['Audio Status']=='FAIL' for r in st.session_state.results)}/{len(st.session_state.results)}")
        c3.metric("Likely AI", f"{sum(r['AI Likelihood']=='High' for r in st.session_state.results)}/{len(st.session_state.results)}")

        if check_ai:
            img = plot_ai_likelihood(st.session_state.results)
            if img: st.image(img, use_column_width=True)

        if generate_report:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = df.fillna("Unknown").replace([np.inf,-np.inf],"Unknown")
            csv = out.to_csv(index=False).encode()
            js = json.dumps(out.to_dict(orient="records"), default=str)
            r1, r2 = st.columns(2)
            with r1:
                st.download_button("Download CSV", data=csv, file_name=f"video_{ts}.csv", mime="text/csv")
            with r2:
                st.download_button("Download JSON", data=js, file_name=f"video_{ts}.json", mime="application/json")

with col2:
    st.markdown('## Video Details')
    if st.session_state.results:
        names = [r["Filename"] for r in st.session_state.results]
        sel = st.selectbox("Select Video", names, index=names.index(st.session_state.selected_video) if st.session_state.selected_video in names else 0)
        st.session_state.selected_video = sel
        vd = next(r for r in st.session_state.results if r["Filename"]==sel)
        tp = vd.get("_temp_path")
        if tp and os.path.exists(tp) and ffmpeg_available:
            thumb = extract_thumbnail(tp)
            if thumb: st.image(thumb, caption="Thumbnail", width=250)
        st.markdown(f"**Duration:** {vd['Duration (s)']} s")
        st.markdown(f"**Resolution:** {vd['Resolution']}")
        st.markdown(f"**Size:** {vd['File Size (MB)']} MB")
        st.markdown(f"**Codec:** {vd['Codec']}")
        st.markdown(f"**Framerate:** {vd['Framerate']}")
        st.markdown(f"**Created:** {vd['Creation Date']}")
        if check_ai and vd["_ai_indicators"]:
            st.markdown("### 🤖 AI Indicators")
            for ind in vd["_ai_indicators"]:
                st.markdown(f"- {ind}")
        else:
            st.markdown("**AI Indicators:** None")
        if extract_full_metadata:
            with st.expander("Full Metadata"):
                st.json(vd["_metadata"]["raw_metadata"])

st.markdown("---")
st.markdown(f"""
**Criteria:**  
- Duration: {min_duration}–{max_duration} s  
- Audio: Should be absent  
- Resolution: HD+ (≥1280×720)  
""")
