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

# ----------------------------- SETUP -----------------------------
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MIN_DURATION_DEFAULT = 20  # seconds
MAX_DURATION_DEFAULT = 30  # seconds
SUPPORTED_FORMATS = ["mp4", "mov", "avi", "mkv", "webm"]
MAX_FILE_SIZE_MB = 500  # Maximum file size in MB
ALLOWED_RESOLUTIONS = [(1280, 720), (1920, 1080), (3840, 2160)]  # HD, Full HD, 4K

# Check if FFmpeg is available
def is_ffmpeg_available():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# Alternative to moviepy using Python only
def get_file_info(file_path):
    """Get basic file information without external dependencies"""
    file_size = os.path.getsize(file_path)
    size_mb = file_size / (1024 * 1024)
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # These are dummy values; we'll try to replace them with FFmpeg data if available
    return {
        'size_mb': round(size_mb, 2),
        'format': file_ext.lstrip('.'),
        'width': 'Unknown',
        'height': 'Unknown',
        'duration': None,
        'framerate': 'Unknown',
        'has_audio': False,
        'bitrate': 'Unknown'
    }

# Use FFmpeg to extract info if available
def get_ffmpeg_info(file_path):
    """Extract video information using FFmpeg if available"""
    if not is_ffmpeg_available():
        return get_file_info(file_path)
    
    try:
        # Run FFprobe to get video information
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
            return get_file_info(file_path)
        
        data = json.loads(result.stdout)
        
        # Initialize with defaults
        info = get_file_info(file_path)
        
        # Get format information
        if 'format' in data:
            format_data = data['format']
            if 'duration' in format_data:
                info['duration'] = float(format_data['duration'])
            if 'bit_rate' in format_data:
                info['bitrate'] = f"{int(format_data['bit_rate']) // 1000} Kbps"
        
        # Get video stream information
        video_stream = None
        has_audio = False
        
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video' and not video_stream:
                video_stream = stream
            elif stream.get('codec_type') == 'audio':
                has_audio = True
        
        if video_stream:
            info['width'] = video_stream.get('width', 'Unknown')
            info['height'] = video_stream.get('height', 'Unknown')
            
            # Get framerate
            if 'r_frame_rate' in video_stream:
                try:
                    num, den = map(int, video_stream['r_frame_rate'].split('/'))
                    info['framerate'] = round(num / den, 2) if den else 'Unknown'
                except (ValueError, ZeroDivisionError):
                    info['framerate'] = 'Unknown'
        
        info['has_audio'] = has_audio
        return info
        
    except Exception as e:
        logger.error(f"Error getting FFmpeg info: {str(e)}")
        return get_file_info(file_path)

def check_duration(file_path):
    """Get video duration using FFmpeg if available."""
    info = get_ffmpeg_info(file_path)
    return info['duration']

def get_video_properties(file_path):
    """Get video properties using FFmpeg."""
    info = get_ffmpeg_info(file_path)
    return {
        'width': info['width'],
        'height': info['height'],
        'codec': info.get('format', 'Unknown'),
        'framerate': info['framerate'],
        'size_mb': info['size_mb'],
        'bitrate': info['bitrate']
    }

def has_audio_stream(file_path):
    """Check for audio stream."""
    info = get_ffmpeg_info(file_path)
    has_audio = info['has_audio']
    
    audio_props = {
        'codec': 'Unknown',
        'bitrate': 'Unknown',
        'sample_rate': 'Unknown',
        'channels': 'Unknown'
    }
    
    return has_audio, audio_props

def extract_metadata(file_path):
    """Extract basic metadata (file-based)."""
    try:
        creation_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
        return {
            'creation_date': creation_time,
            'software': 'Unknown',
            'author': 'Unknown',
            'raw_metadata': {'filename': os.path.basename(file_path)}
        }
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        return {'raw_metadata': {}, 'error': str(e), 'creation_date': 'Unknown', 'software': 'Unknown', 'author': 'Unknown'}

def extract_thumbnail(file_path):
    """Extract a thumbnail using FFmpeg if available."""
    if not is_ffmpeg_available():
        return None
    
    try:
        # Create a temporary file for the thumbnail
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_thumbnail = temp_file.name
        
        # Run FFmpeg to extract a frame from the middle of the video
        duration = check_duration(file_path)
        if not duration:
            return None
            
        middle_time = duration / 2
        cmd = [
            'ffmpeg',
            '-ss', str(middle_time),
            '-i', file_path,
            '-vframes', '1',
            '-q:v', '2',
            temp_thumbnail
        ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        # Read the thumbnail file
        if os.path.exists(temp_thumbnail) and os.path.getsize(temp_thumbnail) > 0:
            with open(temp_thumbnail, 'rb') as f:
                data = f.read()
            
            # Clean up temporary file
            try:
                os.unlink(temp_thumbnail)
            except:
                pass
                
            return data
        
        return None
    except Exception as e:
        logger.error(f"Error extracting thumbnail: {str(e)}")
        return None

def check_ai_indicators(file_path, metadata, video_props):
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
    width = video_props.get('width', 'Unknown')
    height = video_props.get('height', 'Unknown')
    
    # Convert to int if possible
    try:
        width = int(width)
        height = int(height)
        
        if width == height and width > 0 and (width & (width - 1) == 0):  # Power of 2
            ai_indicators.append(f"AI-typical resolution: {width}x{height}")
            ai_score += 20
        if (width, height) not in ALLOWED_RESOLUTIONS:
            ai_indicators.append(f"Unusual resolution: {width}x{height}")
            ai_score += 15
    except (ValueError, TypeError):
        pass
    
    # Check file size anomalies (e.g., suspiciously small for duration)
    size_mb = video_props.get('size_mb', 0)
    duration = check_duration(file_path)
    
    try:
        size_mb = float(size_mb)
        duration = float(duration) if duration else 0
        
        if duration > 0 and size_mb / duration < 0.1:  # Very low MB per second
            ai_indicators.append("Suspiciously low file size for duration")
            ai_score += 10
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    
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
        
        # Determine resolution check
        resolution_check = "SD"
        try:
            width = int(video_props['width'])
            if width >= 1280:
                resolution_check = "HD+"
        except (ValueError, TypeError):
            pass
        
        file_size_mb = video_props['size_mb']
        file_size_status = "Large" if file_size_mb > 25 else "Medium" if file_size_mb > 5 else "Small"
        
        return {
            "Filename": filename,
            "Duration (s)": round(duration, 2) if duration else "Unknown",
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
        buf.seek(0)  # Reset buffer position to beginning
        plt.close()
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Error generating AI likelihood plot: {str(e)}")
        return None

# ----------------------------- STREAMLIT APP -----------------------------
st.set_page_config(page_title="Video Prompt Validator", layout="wide", initial_sidebar_state="expanded")

# Display FFmpeg status
ffmpeg_available = is_ffmpeg_available()

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

# FFmpeg status indicator
if not ffmpeg_available:
    st.warning("⚠️ FFmpeg is not available. Some features like duration detection and thumbnails will be limited.", icon="⚠️")

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
            except Exception as e:
                logger.error(f"Error removing temp file: {str(e)}")
        st.session_state.results = []
        st.session_state.selected_video = None
        st.session_state.temp_files = []
        st.rerun()
    
    if st.button("Clean Temporary Files", key="clean_temp"):
        removed = 0
        for temp_file in st.session_state.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    removed += 1
            except Exception as e:
                logger.error(f"Error removing temp file: {str(e)}")
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
        
        styled_df = df.style.apply(highlight_row, axis=1)
        
        # Handle DataFrame formatting more carefully to avoid TypeError
        number_columns = ["File Size (MB)"]
        for col in number_columns:
            try:
                styled_df = styled_df.format({col: '{:.2f}'}, na_rep="N/A")
            except:
                pass
        
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
        if check_ai:  # Only generate plot if AI checking is enabled
            plot_data = plot_ai_likelihood(st.session_state.results)
            if plot_data:
                st.markdown('<h3 class="text-lg font-medium text-gray-600 mt-4">AI Likelihood Distribution</h3>', unsafe_allow_html=True)
                st.image(plot_data, use_column_width=True)
        
        # Downloadable Reports
        if generate_report:
            st.markdown('<h3 class="text-lg font-medium text-gray-600 mt-4">Download Reports</h3>', unsafe_allow_html=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Ensure we only include serializable data
            export_df = df.fillna("Unknown").replace([np.inf, -np.inf], "Unknown")
            csv = export_df.to_csv(index=False).encode('utf-8')
            
            # Use a simpler approach for JSON to avoid serialization issues
            json_data = export_df.to_dict(orient='records')
            json_str = json.dumps(json_data, default=str)
            
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
                    data=json_str,
                    file_name=f"video_analysis_{timestamp}.json",
                    mime="application/json",
                    key="download_json"
                )

with col2:
    st.markdown('<h2 class="text-xl font-semibold text-gray-700 mb-2">Video Details</h2>', unsafe_allow_html=True)
    if st.session_state.results:
        filenames = [r["Filename"] for r in st.session_state.results]
        
        # Add safeguard for selected_video
        if st.session_state.selected_video is None and filenames:
            st.session_state.selected_video = filenames[0]
        
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
            if temp_path and os.path.exists(temp_path) and ffmpeg_available:
                try:
                    with st.spinner("Loading thumbnail..."):
                        thumb_data = extract_thumbnail(temp_path)
                        if thumb_data:
                            st.image(thumb_data, caption="Video Thumbnail", width=250)
                        else:
                            st.warning("Could not load thumbnail.")
                except Exception as e:
                    logger.error(f"Error displaying thumbnail: {str(e)}")
                    st.warning("Error loading thumbnail.")
            elif not ffmpeg_available:
                st.info("Thumbnails require FFmpeg to be installed.")
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
            
            if check_ai and video_data.get("_ai_indicators"):
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
    st.markdown(f"""
        ### Cloud Video Prompt Validation Tool
        Analyze video files for compliance with prompt requirements.
        
        **Features:**
        - Validates duration, audio presence, and resolution.
        - Detects potential AI-generated content using heuristics.
        - Generates detailed reports and visualizations.
        
        **Usage:**
        1. Upload videos (MP4, MOV, AVI, MKV, WebM, max {MAX_FILE_SIZE_MB}MB).
        2. Configure settings in the sidebar.
        3. Click "Analyze Videos" to process.
        4. View results and download reports.
        
        **Technical Notes:**
        {"- Using FFmpeg for enhanced video analysis." if ffmpeg_available else "- FFmpeg not available. Using fallback methods."}
    """)

# Cleanup temporary files on session end
def cleanup():
    for temp_file in st.session_state.temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except Exception as e:
            logger.error(f"Error cleaning up temp file: {str(e)}")
    logger.info("Cleaned up temporary files.")

import atexit
atexit.register(cleanup)

# Check write access
if st.session_state.has_write_access is not True:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "test.txt"), "w") as f:
                f.write("test")
            st.session_state.has_write_access = True
    except Exception as e:
        st.session_state.has_write_access = False
        st.error(f"Warning: Limited file system access. Error: {str(e)}")
