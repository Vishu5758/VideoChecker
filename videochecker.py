# VideoChecker.py

import streamlit as st
import subprocess
import json
import pandas as pd
import tempfile
import os
import plotly.express as px

# ─── Configuration ─────────────────────────────────────────────────────────────
MIN_DURATION = 20  # seconds
MAX_DURATION = 30  # seconds
ALLOWED_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv')

FFPROBE = "ffprobe"  # Use the system‐bundled ffprobe on Streamlit Cloud

# ─── Helper Functions ──────────────────────────────────────────────────────────
def get_video_info(video_path):
    try:
        cmd = [
            FFPROBE,
            "-v", "error",
            "-show_entries", "format=duration",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "json",
            video_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        info = json.loads(res.stdout)
        dur = float(info.get("format", {}).get("duration", 0))
        has_audio = len(info.get("streams", [])) > 0
        return dur, has_audio
    except Exception:
        return None, False

def analyze_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    duration, has_audio = get_video_info(tmp_path)
    os.remove(tmp_path)

    dur_status = "PASS" if (duration is not None and MIN_DURATION <= duration <= MAX_DURATION) else "FAIL"
    aud_status = "FAIL" if has_audio else "PASS"

    return {
        "Filename": uploaded_file.name,
        "Duration (s)": round(duration, 2) if duration is not None else None,
        "Duration Status": dur_status,
        "Audio Present": "Yes" if has_audio else "No",
        "Audio Status": aud_status
    }

# ─── Streamlit App Layout ──────────────────────────────────────────────────────
st.set_page_config(page_title="🎥 Video Prompt Validator", layout="wide")

st.title("🎥 Video Prompt Validator")
st.markdown("A **professional** dashboard to validate video prompts (20–30 s & no extra audio).")

st.header("📂 Upload Video Files")
uploaded_files = st.file_uploader(
    "Drag & drop or select your videos",
    type=[ext.strip('.') for ext in ALLOWED_EXTENSIONS],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded.")
    if st.button("🔍 Analyze Videos"):
        records = [analyze_uploaded_file(f) for f in uploaded_files]
        df = pd.DataFrame(records)

        st.header("📊 Summary")
        total = len(df)
        passed_dur = int(df['Duration Status'].eq('PASS').sum())
        passed_aud = int(df['Audio Status'].eq('PASS').sum())
        valid_durs = df['Duration (s)'].dropna()
        avg_dur = valid_durs.mean() if not valid_durs.empty else None
        avg_str = f"{avg_dur:.2f}s" if avg_dur is not None else "N/A"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Videos", total)
        c2.metric("Passed Duration", passed_dur)
        c3.metric("Passed Audio", passed_aud)
        c4.metric("Average Duration", avg_str)

        st.subheader("📈 Pass/Fail Distribution")
        dur_fig = px.pie(
            names=["Pass", "Fail"],
            values=[passed_dur, total - passed_dur],
            title="Duration Check",
            color_discrete_sequence=px.colors.sequential.Blues
        )
        aud_fig = px.pie(
            names=["Pass", "Fail"],
            values=[passed_aud, total - passed_aud],
            title="Audio Check",
            color_discrete_sequence=px.colors.sequential.Blues
        )
        pc1, pc2 = st.columns(2)
        with pc1: st.plotly_chart(dur_fig, use_container_width=True)
        with pc2: st.plotly_chart(aud_fig, use_container_width=True)

        st.header("📋 Detailed Results")
        def highlight_fail(r):
            return [
                "background-color: #ffe5e5" if (r["Duration Status"]=="FAIL" or r["Audio Status"]=="FAIL") else ""
                for _ in r
            ]
        st.dataframe(df.style.apply(highlight_fail, axis=1), use_container_width=True)

        st.header("📥 Download Full Report")
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="video_validation_report.csv",
            mime="text/csv"
        )
else:
    st.info("📂 Please upload one or more video files to begin.")
