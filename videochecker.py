import streamlit as st
import subprocess
import json
import pandas as pd
import tempfile
import os
import hashlib
import datetime
from io import BytesIO
from PIL import Image
import plotly.express as px

# ─── Configuration ─────────────────────────────────────────────────────────────
MIN_DURATION = 20  # seconds
MAX_DURATION = 30  # seconds
ALLOWED_EXTS = ('.mp4', '.mov', '.avi', '.mkv')
FFPROBE = "ffprobe"
FFMPEG  = "ffmpeg"

# ─── Helpers ───────────────────────────────────────────────────────────────────
def run_ffprobe(args):
    cmd = [FFPROBE, "-v", "error", "-of", "json"] + args
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(res.stdout) if res.stdout else {}

def get_video_info(path):
    info = run_ffprobe(["-show_entries", "format=duration", "-show_entries", "stream=codec_type", path])
    dur = float(info.get("format",{}).get("duration", 0)) if info.get("format") else None
    has_audio = any(s.get("codec_type")=="audio" for s in info.get("streams", []))
    return dur, has_audio

def get_stream_props(path):
    info = run_ffprobe([
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name,r_frame_rate,bit_rate",
        "-of", "json", path
    ])
    s = info.get("streams",[{}])[0]
    w,h = s.get("width"), s.get("height")
    codec = s.get("codec_name")
    fr = s.get("r_frame_rate","0/0")
    framerate = (lambda n,d: n/d if d else 0)(*map(int,fr.split("/"))) if "/" in fr else float(fr)
    br = s.get("bit_rate")
    mb = round(int(dr:=s.get("bit_rate",0)) /1e6,2) if dr and dr.isdigit() else None
    return {
        "width": w, "height": h, "codec": codec,
        "framerate": round(framerate,2), "bitrate": f"{round(int(br)/1000,2)} Kbps" if br else None
    }

def extract_metadata(path):
    info = run_ffprobe(["-show_entries","format_tags","-of","json",path])
    tags = info.get("format",{}).get("tags",{})
    return {
        "software": tags.get("encoder") or tags.get("software"),
        "creation_time": tags.get("creation_time"),
        "raw": tags
    }

def extract_thumbnail(path):
    # grab middle frame
    dur,_ = get_video_info(path)
    if not dur: return None
    t = dur/2
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    subprocess.run([FFMPEG,"-ss",str(t),"-i",path,"-vframes","1","-q:v","2",tmp.name],
                   stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    img = Image.open(tmp.name)
    buf = BytesIO(); img.save(buf,format="JPEG"); buf.seek(0)
    os.unlink(tmp.name)
    return buf

def check_ai(path, meta):
    keywords = ['dall-e','midjourney','stable diffusion','runway','synthesia','deepfake','neural','ai generated','openai']
    score=0; inds=[]
    sw = (meta.get("software") or "").lower()
    for kw in keywords:
        if kw in sw:
            inds.append(f"AI software: {sw}"); score+=30; break
    for v in meta["raw"].values():
        if any(kw in str(v).lower() for kw in keywords):
            inds.append(f"AI tag: {v}"); score+=20; break
    # codec/bitrate heuristics
    sp = get_stream_props(path)
    br = sp.get("bitrate","").split()[0]
    if br.isdigit() and int(br)%1000==0:
        inds.append("Round bitrate"); score+=15
    if sp.get("width")==sp.get("height") and sp.get("width") and (sp["width"] & (sp["width"]-1)==0):
        inds.append(f"Square {sp['width']}"); score+=20
    return {
        "indicators": inds,
        "score": min(score,100),
        "likelihood": "High" if score>70 else "Medium" if score>30 else "Low"
    }

def analyze_file(uploaded):
    # write temp
    ext=os.path.splitext(uploaded.name)[1]
    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=ext)
    tmp.write(uploaded.read()); tmp.close()
    path=tmp.name

    dur,aud = get_video_info(path)
    props   = get_stream_props(path)
    meta    = extract_metadata(path)
    ai      = check_ai(path,meta)
    thumb   = extract_thumbnail(path)
    # hash first MB
    with open(path,"rb") as f: h=hashlib.md5(f.read(1024*1024)).hexdigest()
    os.unlink(path)

    return dict(
        Filename=uploaded.name,
        Duration=round(dur,2) if dur else None,
        Dur_Status="PASS" if dur and MIN_DURATION<=dur<=MAX_DURATION else "FAIL",
        Audio="Yes" if aud else "No",
        Aud_Status="FAIL" if aud else "PASS",
        Resolution=f"{props['width']}x{props['height']}",
        Codec=props['codec'],
        Framerate=props['framerate'],
        Bitrate=props['bitrate'],
        Software=meta["software"] or "Unknown",
        Created=meta["creation_time"] or "Unknown",
        AI=ai["likelihood"],
        Indicators="; ".join(ai["indicators"]),
        Hash=h[:10]+"..."
    ), thumb

# ─── App Layout ────────────────────────────────────────────────────────────────
st.set_page_config("🎥 Advanced Video Validator",layout="wide")
st.title("🎥 Advanced Video Prompt Validator")

# ensure ffmpeg/ffprobe present (Cloud installs via apt.txt)
for tool in (FFPROBE,FFMPEG):
    if subprocess.run([tool,"-version"],stdout=subprocess.PIPE,stderr=subprocess.PIPE).returncode!=0:
        st.error(f"`{tool}` not found! Did you add it to apt.txt?")
        st.stop()

upl=st.file_uploader("Upload videos",type=[e.strip('.') for e in ALLOWED_EXTS],accept_multiple_files=True)
if not upl:
    st.info("Please upload mp4/mov/avi/mkv files to begin.")
    st.stop()

if st.button("Analyze"):
    records=[]; thumbs={}
    with st.spinner("Analyzing..."):
        for u in upl:
            rec,tn=analyze_file(u)
            records.append(rec)
            if tn: thumbs[rec["Filename"]]=tn

    df=pd.DataFrame(records)
    # Summary
    st.header("📊 Summary")
    total=len(df)
    pass_dur=int(df.Dur_Status.eq("PASS").sum())
    fail_aud=int(df.Aud_Status.eq("FAIL").sum())
    high_ai=int(df.AI.eq("High").sum())
    avg=df.Duration.dropna().mean()
    cols=st.columns(4)
    cols[0].metric("Total",total)
    cols[1].metric("Dur Pass",pass_dur)
    cols[2].metric("Aud Fail",fail_aud)
    cols[3].metric("Avg Dur",f"{avg:.2f}s" if pd.notna(avg) else "N/A")

    # Charts
    st.subheader("📈 Pass/Fail Charts")
    fig1=px.pie(df,names="Dur_Status",title="Duration",color_discrete_sequence=px.colors.sequential.Blues)
    fig2=px.pie(df,names="Aud_Status",title="Audio",color_discrete_sequence=px.colors.sequential.Blues)
    c1,c2=st.columns(2)
    c1.plotly_chart(fig1,use_container_width=True)
    c2.plotly_chart(fig2,use_container_width=True)

    # Table
    st.subheader("📋 Results")
    def hl(r):
        return ["background-color:#ffe5e5" if (r.Dur_Status=="FAIL" or r.Aud_Status=="FAIL") else "" for _ in r]
    st.dataframe(df.style.apply(hl,axis=1),use_container_width=True)

    # Download
    csv=df.to_csv(index=False).encode()
    st.download_button("Download CSV",csv,"report.csv","text/csv")

    # Details pane
    st.subheader("🔍 Details")
    sel=st.selectbox("Select file",df.Filename.tolist())
    info=next(r for r in records if r["Filename"]==sel)
    if sel in thumbs:
        st.image(Image.open(thumbs[sel]),caption="Thumbnail",use_column_width=False)
    st.json({k:info[k] for k in info if k not in ("Indicators",)},expanded=False)
    if info["Indicators"]:
        st.markdown("**AI Indicators:**")
        for i in info["Indicators"].split("; "):
            st.write(f"- {i}")
