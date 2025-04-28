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
from moviepy import VideoFileClip

# ----------------------------- SETUP -----------------------------
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MIN_DURATION_DEFAULT = 20  # seconds
MAX_DURATION_DEFAULT = 30  # seconds
SUPPORTED_FORMATS = ["mp4", "mov", "avi", "mkv", "webm"]
MAX_FILE_SIZE_MB = 50  # Maximum file size in MB
ALLOWED_RESOLUTIONS = [(1280, 720), (1920, 1080), (3840, 2160)]  # HD, Full HD, 4K

# ----------------------------- FUNCTIONS -----------------------------
def check_duration(video_path):
    """Get video duration using moviepy."""
    try:
        with VideoFileClip(video_path) as clip:
            duration = clip.duration
        return duration if duration > 0 else None
    except Exception as e:
        logger.error(f"Error getting duration for {video_path}: {str(e)}")
        st.error(f"Error getting duration: {str(e)}")
        return None

def get_video_properties(video_path):
    """Get video properties using moviepy."""
    try:
        with VideoFileClip(video_path) as clip:
            width, height = clip.size
            framerate = clip.fps
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            # Estimate bitrate (approximation: file size / duration)
            duration = clip.duration
            bitrate = (size_mb * 8 * 1024) / duration if duration else 'Unknown'  # Kbps
            codec = 'Unknown'  # moviepy doesn't reliably provide codec
        return {
            'width': width,
            'height': height,
            'codec': codec,
            'framerate': round(framerate, 2) if framerate else 'Unknown',
            'size_mb': round(size_mb, 2),
            'bitrate': f"{round(bitrate, 2)} Kbps" if isinstance(bitrate, (int, float)) else 'Unknown'
        }
    except Exception as e:
        logger.error(f"Error getting video properties: {str(e)}")
        return {
            'width': 'Error',
            'height': 'Error',
            'codec': 'Error',
            'framerate': 'Error',
            'size_mb': 'Error',
            'bitrate': 'Error'
        }

def has_audio_stream(video_path):
    """Check for audio stream using moviepy."""
    try:
        with VideoFileClip(video_path) as clip:
            audio = clip.audio is not None
            audio_props = {}
            if audio:
                audio_props = {
                    'codec': 'Unknown',  # moviepy doesn't provide audio codec
                    'bitrate': 'Unknown',
                    'sample_rate': clip.audio.fps if clip.audio else 'Unknown',
                    'channels': clip.audio.nchannels if clip.audio else 'Unknown'
                }
        return audio, audio_props
    except Exception as e:
        logger.error(f"Error checking audio stream: {str(e)}")
        return False, {}

def extract_metadata(video_path):
    """Extract basic metadata (file-based)."""
    try:
        creation_time = datetime.datetime.fromtimestamp(os.path.getctime(video_path)).strftime('%Y-%m-%d %H:%M:%S')
        return {
            'creation_date': creation_time,
            'software': 'Unknown',
            'author': 'Unknown',
            'raw_metadata': {'filename': os.path.basename(video_path)}
        }
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        return {'raw_metadata': {}, 'error': str(e), 'creation_date': 'Unknown', 'software': 'Unknown', 'author': 'Unknown'}

def extract_thumbnail(video_path):
    """Extract a thumbnail using moviepy."""
    try:
        with VideoFileClip(video_path) as clip:
            duration = clip.duration
            if not duration:
                return None
            frame = clip.get_frame(duration / 2)  # Middle frame
            img = BytesIO()
            plt.imsave(img, frame, format='jpg')
            img.seek(0)
            return img.read()
    except Exception as e:
        logger.error(f"Error extracting thumbnail: {str(e)}")
        return None

def check_ai_indicators(video_path, metadata, video_props):
    """Check for AI-generated indicators using heuristics."""
    ai_indicators = []
    ai_score = 0
    
    # Check filename and metadata for AI keywords
    filename = metadata.get('raw_metadata', {}).get('filename', '').lower()
    ai_keywords = ['dall-e', 'midjourney', 'stable diffusion', 'runway', 'synthesia', 
                   'deepfake', 'neural', 'ai', 'generated', 'openai']
    if any(keyword in filename for keyword in ai_keywords):
        ai_indicators.append(f"AI keyword in filename: {filename}")
        ai_score += 30
    
    # Check resolution for AI-typical patterns (e.g., power-of-2 sizes)
    width = video_props.get('width', 0)
    height = video_props.get('height', 0)
    if isinstance(width, int) and isinstance(height, int):
        if width == height and width > 0 and (width & (width - 1) == 0):  # Power of 2
            ai_indicators.append(f"AI-typical resolution: {width}x{height}")
            ai_score += 20
        if (width, height) not in ALLOWED_RESOLUTIONS:
            ai_indicators.append(f"Unusual resolution: {width}x{height}")
            ai_score += 15
    
    # Check file size anomalies (e.g., suspiciously small for duration)
    size_mb = video_props.get('size_mb', 0)
    duration = check_duration(video_path)
    if duration and size_mb and isinstance(size_mb, (int, float)):
        if size_mb / duration < 0.1:  # Very low MB per second
            ai_indicators.append("Suspiciously low file size for duration")
            ai_score += 10
    
    ai_score = min(ai_score, 100)
    likelihood = "High" if ai_score > 70 else "Medium" if ai_score > 30 else "Low"
    return {'ai_indicators': ai_indicators, 'ai_score': ai_score, 'ai_likelihood': likelihood}

def analyze_video(video_path, filename):
    """Analyze a video file and return its properties."""
    try:
        duration = check_duration(video_path)
        has_audio, audio_props = has_audio_stream(video_path)
        video_props = get_video_properties(video_path)
        metadata = extract_metadata(video_path)
        ai_check = check_ai_indicators(video_path, metadata, video_props)
        
        file_hash = hashlib.md5(open(video_path, 'rb').read(1024*1024)).hexdigest()[:10] + "..."
        duration_status = "PASS" if duration and MIN_DURATION_DEFAULT <= duration <= MAX_DURATION_DEFAULT else "FAIL"
        audio_status = "FAIL" if has_audio else "PASS"
        
        resolution = f"{video_props['width']}x{video_props['height']}"
        resolution_check = "HD+" if isinstance(video_props['width'], int) and video_props['width'] >= 1280 else "SD"
        
        file_size_mb = video_props['size_mb'] if video_props['size_mb'] != 'Error' else 0
        file_size_status = "Large" if file_size_mb > 25 else "Medium" if file_size_mb > 5 else "Small"
        
        return {
            "Filename": filename,
            "Duration (s)": round(duration, 2) if duration else "Error",
            "Duration Status": duration_status,
            "Audio Present": "Yes" if has_audio else "No",
            "Audio Status": audio_status,
            "Resolution": resolution,
            "Resolution Check": resolution_check,
            "Codec": video_props['codec'],
            "Framerate": video_props['framerate'],
            "File Size (MB)": file_size_mb,
            "Bitrate": video_props['bitrate'],
            "Creation Date": metadata['creation_date'],
            "AI Likelihood": ai_check['ai_likelihood'],
            "File Hash": file_hash,
            "Software": metadata['software'],
            "_audio_props": audio_props,
            "_ai_indicators": ai_check['ai_indicators'],
            "_metadata": metadata,
            "_temp_path": video_path
        }
    except Exception as e:
        logger.error(f"Error analyzing video {filename}: {str(e)}")
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
            "_ai_indicators": [f"Analysis failed: {str(e)}"],
            "_metadata": {"raw_metadata": {}},
            "_temp_path": video_path
        }

def plot_ai_likelihood(results):
    """Generate a bar chart for AI likelihood distribution."""
    try:
        likelihoods = [r["AI Likelihood"] for r in results]
        counts = pd.Series(likelihoods).value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
        plt.figure(figsize=(6, 4))
        sns.barplot(x=counts.index, y=counts.values, palette="viridis")
        plt.title("AI Likelihood Distribution")
        plt.xlabel("Likelihood")
        plt.ylabel("Number of Videos")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Error generating AI likelihood plot: {str(e)}")
        return None

# ----------------------------- STREAMLIT APP -----------------------------
st.set_page_config(page_title="Video Prompt Validator", layout="wide", initial_sidebar_state="expanded")

# Inject Tailwind CSS for enhanced styling
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .stButton>button {
            background-color: #2563eb;
            color: white;
            border-radius: 0.375rem;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #1e40af;
        }
        .stDataFrame {
            border-radius: 0.375rem;
            overflow: hidden;
        }
        .sidebar .stButton>button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="text-3xl font-bold text-gray-800 mb-4">🎥 Video Prompt Validation Tool</h1>', unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []
if 'selected_video' not in st.session_state:
    st.session_state.selected_video = None
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'has_write_access' not in st.session_state:
    st.session_state.has_write_access = True

# Sidebar
with st.sidebar:
    st.markdown('<h2 class="text-xl font-semibold text-gray-700">Settings</h2>', unsafe_allow_html=True)
    min_duration = st.slider("Minimum Duration (seconds)", 1, 60, MIN_DURATION_DEFAULT, key="min_duration")
    max_duration = st.slider("Maximum Duration (seconds)", min_duration, 180, MAX_DURATION_DEFAULT, key="max_duration")
    
    st.markdown('<h3 class="text-lg font-medium text-gray-600 mt-4">Validation Options</h3>', unsafe_allow_html=True)
    check_ai = st.checkbox("Check for AI-generated indicators", value=True, key="check_ai")
    extract_full_metadata = st.checkbox("Extract full metadata", value=False, key="full_metadata")
    generate_report = st.checkbox("Generate downloadable report", value=True, key="generate_report")
    
    if st.button("Clear All Results", key="clear_results"):
        for temp_file in st.session_state.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        st.session_state.results = []
        st.session_state.selected_video = None
        st.session_state.temp_files = []
        st.rerun()
    
    if st.button("Clean Temporary Files", key="clean_temp"):
        removed = sum(1 for temp_file in st.session_state.temp_files if os.path.exists(temp_file) and not os.unlink(temp_file))
        st.session_state.temp_files = []
        st.success(f"Cleaned up {removed} temporary files.")

# Main layout
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<h2 class="text-xl font-semibold text-gray-700 mb-2">Upload Videos</h2>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Select video files",
        type=SUPPORTED_FORMATS,
        accept_multiple_files=True,
        key="video_uploader",
        help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}. Max size: {MAX_FILE_SIZE_MB}MB per file."
    )
    
    if uploaded_files:
        if any(f.size > MAX_FILE_SIZE_MB * 1024 * 1024 for f in uploaded_files):
            st.error(f"Some files exceed the {MAX_FILE_SIZE_MB}MB size limit.")
        else:
            analyze_btn = st.button(
                "Analyze Videos",
                key="analyze_btn",
                disabled=st.session_state.processing,
                help="Click to analyze uploaded videos."
            )
            
            if analyze_btn:
                st.session_state.processing = True
                progress_bar = st.progress(0)
                new_results = []
                
                with st.spinner(f"Analyzing {len(uploaded_files)} videos..."):
                    for i, uploaded_file in enumerate(uploaded_files):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                                temp_file.write(uploaded_file.getvalue())
                                temp_path = temp_file.name
                            st.session_state.temp_files.append(temp_path)
                            
                            with st.status(f"Analyzing {uploaded_file.name}..."):
                                result = analyze_video(temp_path, uploaded_file.name)
                                new_results.append(result)
                        except Exception as e:
                            logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                            new_results.append({
                                "Filename": uploaded_file.name,
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
                                "_ai_indicators": [f"Analysis failed: {str(e)}"],
                                "_metadata": {"raw_metadata": {}},
                                "_temp_path": temp_path
                            })
                        progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.session_state.results.extend(new_results)
                if not st.session_state.selected_video and new_results:
                    st.session_state.selected_video = new_results[0]["Filename"]
                st.session_state.processing = False
                st.rerun()
    
    if st.session_state.results:
        display_columns = [
            "Filename", "Duration (s)", "Duration Status", "Audio Present",
            "Audio Status", "Resolution", "Framerate", "File Size (MB)", "AI Likelihood"
        ]
        df = pd.DataFrame([{col: r.get(col, "Error") for col in display_columns} for r in st.session_state.results])
        
        def highlight_row(row):
            color = ''
            if row['Duration Status'] == 'FAIL':
                color = 'background-color: #fee2e2'
            elif row['Audio Status'] == 'FAIL':
                color = 'background-color: #fef3c7'
            elif row['AI Likelihood'] == 'High':
                color = 'background-color: #ffedd5'
            return [color] * len(row)
        
        styled_df = df.style.apply(highlight_row, axis=1).format(precision=2)
        st.markdown('<h2 class="text-xl font-semibold text-gray-700 mb-2">Analysis Results</h2>', unsafe_allow_html=True)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        st.markdown('<h2 class="text-xl font-semibold text-gray-700 mt-6 mb-2">Summary Statistics</h2>', unsafe_allow_html=True)
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            pass_count = sum(1 for r in st.session_state.results if r["Duration Status"] == "PASS")
            st.metric("Valid Duration", f"{pass_count}/{len(st.session_state.results)}")
        with col_stats2:
            audio_fail = sum(1 for r in st.session_state.results if r["Audio Status"] == "FAIL")
            st.metric("With Audio", f"{audio_fail}/{len(st.session_state.results)}")
        with col_stats3:
            ai_high = sum(1 for r in st.session_state.results if r["AI Likelihood"] == "High")
            st.metric("Likely AI-Generated", f"{ai_high}/{len(st.session_state.results)}")
        
        # AI Likelihood Visualization
        plot_data = plot_ai_likelihood(st.session_state.results)
        if plot_data:
            st.markdown('<h3 class="text-lg font-medium text-gray-600 mt-4">AI Likelihood Distribution</h3>', unsafe_allow_html=True)
            st.image(plot_data, use_column_width=True)
        
        # Downloadable Reports
        if generate_report:
            st.markdown('<h3 class="text-lg font-medium text-gray-600 mt-4">Download Reports</h3>', unsafe_allow_html=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            csv = df.to_csv(index=False).encode('utf-8')
            json_data = df.to_json(orient='records', lines=True)
            col_report1, col_report2 = st.columns(2)
            with col_report1:
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"video_analysis_{timestamp}.csv",
                    mime="text/csv",
                    key="download_csv"
                )
            with col_report2:
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"video_analysis_{timestamp}.json",
                    mime="application/json",
                    key="download_json"
                )

with col2:
    st.markdown('<h2 class="text-xl font-semibold text-gray-700 mb-2">Video Details</h2>', unsafe_allow_html=True)
    if st.session_state.results:
        filenames = [r["Filename"] for r in st.session_state.results]
        selected_video = st.selectbox(
            "Select Video",
            filenames,
            index=filenames.index(st.session_state.selected_video) if st.session_state.selected_video in filenames else 0,
            key="video_select"
        )
        st.session_state.selected_video = selected_video
        video_data = next((r for r in st.session_state.results if r["Filename"] == selected_video), None)
        
        if video_data:
            temp_path = video_data.get("_temp_path")
            if temp_path and os.path.exists(temp_path):
                with st.spinner("Loading thumbnail..."):
                    thumb_data = extract_thumbnail(temp_path)
                    if thumb_data:
                        st.image(thumb_data, caption="Video Thumbnail", width=250)
                    else:
                        st.warning("Could not load thumbnail.")
            else:
                st.warning("Video file not found.")
            
            st.markdown('<div class="bg-gray-50 p-4 rounded-lg shadow-sm">', unsafe_allow_html=True)
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown(f"**Duration:** {video_data['Duration (s)']} seconds")
                st.markdown(f"**Resolution:** {video_data['Resolution']}")
                st.markdown(f"**File Size:** {video_data['File Size (MB)']} MB")
            with col_info2:
                st.markdown(f"**Codec:** {video_data['Codec']}")
                st.markdown(f"**Framerate:** {video_data['Framerate']}")
                st.markdown(f"**Created:** {video_data['Creation Date']}")
            
            if video_data.get("_ai_indicators"):
                st.markdown('<h3 class="text-lg font-medium text-gray-600 mt-4">🤖 AI Indicators</h3>', unsafe_allow_html=True)
                for indicator in video_data["_ai_indicators"]:
                    st.markdown(f"- {indicator}")
            else:
                st.markdown("**AI Indicators:** None detected")
            
            if video_data["Audio Present"] == "Yes" and video_data["_audio_props"]:
                st.markdown('<h3 class="text-lg font-medium text-gray-600 mt-4">🔊 Audio Properties</h3>', unsafe_allow_html=True)
                audio = video_data["_audio_props"]
                st.markdown(f"**Codec:** {audio.get('codec', 'Unknown')}")
                st.markdown(f"**Bitrate:** {audio.get('bitrate', 'Unknown')}")
                st.markdown(f"**Sample Rate:** {audio.get('sample_rate', 'Unknown')} Hz")
                st.markdown(f"**Channels:** {audio.get('channels', 'Unknown')}")
            
            if extract_full_metadata and video_data["_metadata"].get("raw_metadata"):
                with st.expander("Full Metadata"):
                    st.json(video_data["_metadata"]["raw_metadata"])
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No videos analyzed yet. Upload videos to begin.")

# Footer
st.markdown("---")
st.markdown(f"""
    **Validation Criteria:**
    - Duration: {min_duration} - {max_duration} seconds
    - Audio: Should be absent (FAIL if present)
    - Resolution: HD+ (1280p or higher recommended)
    - AI Likelihood: Based on filename and resolution analysis
""")

with st.expander("About This Tool"):
    st.markdown("""
        ### Cloud Video Prompt Validation Tool
        Analyze video files for compliance with prompt requirements without external dependencies.
        
        **Features:**
        - Validates duration, audio presence, and resolution.
        - Detects potential AI-generated content using heuristics.
        - Generates detailed reports and visualizations.
        
        **Usage:**
        1. Upload videos (MP4, MOV, AVI, MKV, WebM, max {MAX_FILE_SIZE_MB}MB).
        2. Configure settings in the sidebar.
        3. Click "Analyze Videos" to process.
        4. View results and download reports.
    """)

# Cleanup temporary files on session end
def cleanup():
    for temp_file in st.session_state.temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except:
            pass
    logger.info("Cleaned up temporary files.")

import atexit
atexit.register(cleanup)

# Check write access
if 'has_write_access' not in st.session_state:
    try:
        with tempfile.TemporaryDirectory():
            st.session_state.has_write_access = True
    except Exception as e:
        st.session_state.has_write_access = False
        st.error(f"Warning: Limited file system access. Error: {str(e)}")

