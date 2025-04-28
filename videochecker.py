import streamlit as st
import os
import subprocess
import json
import pandas as pd
import datetime
import hashlib
import tempfile
from io import BytesIO
import plotly.express as px
from PIL import Image

# ----------------------------- SETTINGS -----------------------------
MIN_DURATION = 20  # seconds
MAX_DURATION = 30  # seconds

# ----------------------------- FUNCTIONS -----------------------------
def install_ffmpeg():
    """Install FFmpeg in the cloud environment (apt.txt is preferred)."""
    try:
        st.info("Installing FFmpeg... this may take a minute.")
        # run both commands in one shell call
        result = subprocess.run(
            "apt-get update -qq && apt-get install -y ffmpeg",
            shell=True, capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception as e:
        st.error(f"Failed to install FFmpeg: {e}")
        return False

def check_ffmpeg_installed():
    """Check if ffmpeg is installed and accessible."""
    try:
        return subprocess.run(["ffmpeg", "-version"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0
    except FileNotFoundError:
        return False

def check_ffprobe_installed():
    """Check if ffprobe is installed and accessible."""
    try:
        return subprocess.run(["ffprobe", "-version"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0
    except FileNotFoundError:
        return False

def check_duration(video_path):
    """Get video duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", video_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            st.error(f"FFprobe error: {res.stderr}")
            return None
        data = json.loads(res.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return None

def get_video_properties(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name,r_frame_rate,bit_rate",
        "-show_entries", "format=size,bit_rate",
        "-of", "json", video_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(res.stderr)
        data = json.loads(res.stdout)
        s = data.get("streams", [{}])[0]
        # width/height
        try: width = int(s.get("width", 0))
        except: width = "Unknown"
        try: height = int(s.get("height", 0))
        except: height = "Unknown"
        codec = s.get("codec_name", "Unknown")
        # framerate
        rf = s.get("r_frame_rate", "0/0")
        if "/" in rf:
            num, den = map(int, rf.split("/"))
            fr = round(num/den,2) if den else "Unknown"
        else:
            fr = float(rf) if rf else "Unknown"
        # size
        fmt = data.get("format", {})
        try:
            sz = int(fmt.get("size", 0))
            size_mb = round(sz/(1024*1024),2)
        except:
            size_mb = "Unknown"
        # bitrate
        br = fmt.get("bit_rate", "")
        try:
            bitrate = f"{round(int(br)/1000,2)} Kbps"
        except:
            bitrate = "Unknown"
        return {
            "width": width, "height": height,
            "codec": codec, "framerate": fr,
            "size_mb": size_mb, "bitrate": bitrate
        }
    except Exception:
        return {"width":"Error","height":"Error","codec":"Error",
                "framerate":"Error","size_mb":"Error","bitrate":"Error"}

def has_audio_stream(video_path):
    cmd = [
        "ffprobe","-v","error",
        "-select_streams","a",
        "-show_entries","stream=codec_name,bit_rate,sample_rate,channels",
        "-of","json", video_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            return False, {}
        streams = json.loads(res.stdout).get("streams", [])
        if not streams:
            return False, {}
        st0 = streams[0]
        # codec
        codec = st0.get("codec_name","Unknown")
        # bitrate
        br = st0.get("bit_rate","")
        try:
            bitrate = f"{round(int(br)/1000,2)} Kbps"
        except:
            bitrate = "Unknown"
        return True, {
            "codec": codec,
            "bitrate": bitrate,
            "sample_rate": st0.get("sample_rate","Unknown"),
            "channels": st0.get("channels","Unknown")
        }
    except Exception:
        return False, {}

def extract_metadata(video_path):
    cmd = ["ffprobe","-v","error","-show_entries","format_tags","-of","json", video_path]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            return {"raw_metadata":{}, "error":res.stderr}
        tags = json.loads(res.stdout).get("format",{}).get("tags",{})
        return {
            "creation_date": tags.get("creation_time"),
            "software": tags.get("encoder") or tags.get("software"),
            "author": tags.get("artist") or tags.get("author"),
            "raw_metadata": tags
        }
    except Exception:
        return {"raw_metadata":{}, "error":"Metadata parse error"}

def extract_thumbnail(video_path):
    dur = check_duration(video_path)
    if not dur:
        return None
    t = dur/2
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    cmd = ["ffmpeg","-ss",str(t),"-i",video_path,"-vframes","1","-q:v","2",tmp.name]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        return None
    img = Image.open(tmp.name)
    buf = BytesIO(); img.save(buf,"JPEG"); buf.seek(0)
    os.unlink(tmp.name)
    return buf.read()

def check_ai_indicators(video_path, metadata):
    kws = ['dall-e','midjourney','stable diffusion','runway','synthesia','deepfake','neural','ai generated','openai']
    raw = metadata.get("raw_metadata",{})
    indic,score = [],0
    sw = (metadata.get("software") or "").lower()
    for kw in kws:
        if kw in sw:
            indic.append(f"AI software: {sw}"); score+=30; break
    for v in raw.values():
        vs = str(v).lower()
        for kw in kws:
            if kw in vs:
                indic.append(f"AI metadata: {v}"); score+=20; break
    # codec/bitrate heuristics
    sp = get_video_properties(video_path)
    br = sp.get("bitrate","").split()[0]
    if br.isdigit() and int(br)%1000==0:
        indic.append("Round bitrate"); score+=15
    w,h = sp.get("width"), sp.get("height")
    if w==h and isinstance(w,int) and (w & (w-1)==0):
        indic.append(f"Square {w}"); score+=20
    return {
        "ai_indicators": indic,
        "ai_score": min(score,100),
        "ai_likelihood": "High" if score>70 else "Medium" if score>30 else "Low"
    }

def analyze_video(video_path, filename):
    duration = check_duration(video_path)
    has_audio,audio_props = has_audio_stream(video_path)
    props    = get_video_properties(video_path)
    metadata = extract_metadata(video_path)
    ai_check = check_ai_indicators(video_path, metadata)
    # hash
    try:
        h = hashlib.md5(open(video_path,"rb").read(1024*1024)).hexdigest()
    except:
        h = "Error"
    # statuses
    dur_stat = "PASS" if duration and MIN_DURATION<=duration<=MAX_DURATION else "FAIL"
    aud_stat = "FAIL" if has_audio else "PASS"
    # resolution check
    res_check = "HD+" if isinstance(props["width"],int) and props["width"]>=1280 else "SD"
    # file size
    try:
        fs = os.path.getsize(video_path)/(1024*1024)
    except:
        fs = 0
    fs_stat = "Large" if fs>50 else "Medium" if fs>10 else "Small"
    # format creation date
    cd = metadata.get("creation_date") or "Unknown"
    for fmt in ['%Y:%m:%d %H:%M:%S','%Y-%m-%dT%H:%M:%S.%fZ','%Y-%m-%d %H:%M:%S']:
        try:
            dt = datetime.datetime.strptime(cd,fmt)
            cd = dt.strftime("%Y-%m-%d %H:%M:%S")
            break
        except:
            pass

    return {
        "Filename": filename,
        "Duration (s)": round(duration,2) if duration else "Error",
        "Duration Status": dur_stat,
        "Audio Present": "Yes" if has_audio else "No",
        "Audio Status": aud_stat,
        "Resolution": f"{props['width']}x{props['height']}",
        "Resolution Check": res_check,
        "Codec": props["codec"],
        "Framerate": props["framerate"],
        "File Size (MB)": props["size_mb"],
        "Bitrate": props["bitrate"],
        "Creation Date": cd,
        "AI Likelihood": ai_check["ai_likelihood"],
        "File Hash": h[:10]+"..." if h!="Error" else "Error",
        "Software": metadata.get("software","Unknown"),
        "_audio_props": audio_props,
        "_ai_indicators": ai_check["ai_indicators"],
        "_metadata": metadata,
        "_temp_path": video_path
    }

# ----------------------------- STREAMLIT APP -----------------------------
st.set_page_config(page_title="Video Prompt Validator", layout="wide")
st.title("🎥 Cloud Video Prompt Validation Tool")

# Check ffmpeg/ffprobe
if not check_ffmpeg_installed() or not check_ffprobe_installed():
    st.error("⚠️ ffmpeg/ffprobe not found. Please add `ffmpeg` to apt.txt and redeploy.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Settings")
    min_duration = st.slider("Min Duration (s)",1,60,MIN_DURATION)
    max_duration = st.slider("Max Duration (s)",min_duration,180,MAX_DURATION)
    check_ai = st.checkbox("Check AI indicators",True)
    extract_full_metadata = st.checkbox("Extract full metadata",False)
    generate_report = st.checkbox("Generate CSV report",False)
    if st.button("Clear Results"):
        st.session_state.clear()

# Main
col1, col2 = st.columns([3,1])
with col1:
    uploads = st.file_uploader("Upload Video Files", type=["mp4","mov","avi","mkv"], accept_multiple_files=True)
    if uploads and st.button("Analyze Uploaded Videos"):
        st.session_state.results = []
        st.session_state.temp = []
        for u in uploads:
            # write temp
            ext = os.path.splitext(u.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tmp.write(u.read()); tmp.close()
            st.session_state.temp.append(tmp.name)
            # analyze
            try:
                res = analyze_video(tmp.name,u.name)
            except Exception as e:
                res = {"Filename":u.name, "Duration Status":"FAIL", "Audio Status":"Error", "AI Likelihood":"Error"}
            st.session_state.results.append(res)

    if st.session_state.results:
        df = pd.DataFrame(st.session_state.results)
        # display table
        def hl(r):
            return ["background-color:#ffe5e5" if r["Duration Status"]=="FAIL" or r["Audio Status"]=="FAIL" else "" for _ in r]
        st.dataframe(df.style.apply(hl,axis=1),height=400,use_container_width=True)
        # metrics
        st.subheader("Summary")
        total=len(df)
        p_dur=(df["Duration Status"]=="PASS").sum()
        p_aud=(df["Audio Status"]=="PASS").sum()
        avg = df["Duration (s)"].replace("Error",pd.NA).astype(float).mean()
        avg_str=f"{avg:.2f}s" if pd.notna(avg) else "N/A"
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total",total); c2.metric("Dur Pass",p_dur)
        c3.metric("Aud Pass",p_aud); c4.metric("Avg Dur",avg_str)
        # charts
        st.subheader("Pass/Fail Charts")
        fig1=px.pie(df,names="Duration Status",title="Duration")
        fig2=px.pie(df,names="Audio Status",title="Audio")
        c1,c2=st.columns(2)
        c1.plotly_chart(fig1,use_container_width=True)
        c2.plotly_chart(fig2,use_container_width=True)
        # download
        if generate_report:
            csv=df.to_csv(index=False).encode()
            st.download_button("Download CSV",csv,"report.csv","text/csv")

with col2:
    st.subheader("Video Details")
    if st.session_state.results:
        names=[r["Filename"] for r in st.session_state.results]
        sel=st.selectbox("Select Video",names)
        info=next(r for r in st.session_state.results if r["Filename"]==sel)
        # thumbnail
        try:
            img_data = extract_thumbnail(info["_temp_path"])
            if img_data:
                st.image(img_data, width=250)
        except: pass
        # basic info
        st.write(f"**Duration:** {info.get('Duration (s)','Error')}s")
        st.write(f"**Resolution:** {info.get('Resolution','')}")
        st.write(f"**Created:** {info.get('Creation Date','')}")
        # AI Indicators
        if info["_ai_indicators"]:
            st.markdown("**AI Indicators:**")
            for i in info["_ai_indicators"]:
                st.write(f"- {i}")
        # audio props
        if info["Audio Present"]=="Yes":
            ap=info["_audio_props"]
            st.markdown("**Audio Props:**")
            for k,v in ap.items(): st.write(f"**{k}:** {v}")

# Cleanup temps on exit
import atexit
def cleanup():
    for f in st.session_state.get("temp",[]):
        try: os.unlink(f)
        except: pass
atexit.register(cleanup)
