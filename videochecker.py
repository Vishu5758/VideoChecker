import streamlit as st
import os
import subprocess
import json
import pandas as pd
import datetime
import hashlib
import tempfile
import base64
from io import BytesIO
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

# ----------------------------- SETUP -----------------------------
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MIN_DURATION_DEFAULT = 20  # seconds
MAX_DURATION_DEFAULT = 30  # seconds
SUPPORTED_FORMATS = ["mp4", "mov", "avi", "mkv", "webm"]
MAX_FILE_SIZE_MB = 200  # Maximum file size in MB

# ----------------------------- FUNCTIONS -----------------------------
def install_ffmpeg():
    """Install FFmpeg in the cloud environment."""
    try:
        st.info("Installing FFmpeg... This may take a minute.")
        result = subprocess.run(
            ["apt-get", "update", "-qq", "&&", "apt-get", "install", "-y", "ffmpeg"],
            shell=True, capture_output=True, text=True, timeout=300
        )
        logger.info("FFmpeg installation completed.")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg installation timed out.")
        st.error("FFmpeg installation timed out. Please try again or contact support.")
        return False
    except Exception as e:
        logger.error(f"Failed to install FFmpeg: {str(e)}")
        st.error(f"Failed to install FFmpeg: {str(e)}")
        return False

def check_ffmpeg_installed():
    """Check if FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        return result.returncode == 0
    except Exception:
        logger.warning("FFmpeg not detected.")
        return False

def check_ffprobe_installed():
    """Check if ffprobe is installed and accessible."""
    try:
        result = subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        return result.returncode == 0
    except Exception:
        logger.warning("ffprobe not detected.")
        return False

def check_duration(video_path):
    """Get video duration using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if result.returncode != 0:
            logger.error(f"ffprobe error: {result.stderr}")
            st.error(f"ffprobe error: {result.stderr}")
            return None
        data = json.loads(result.stdout)
        duration = float(data.get('format', {}).get('duration', 0))
        return duration if duration > 0 else None
    except json.JSONDecodeError:
        logger.error("Error parsing JSON output from ffprobe.")
        st.error("Error parsing ffprobe output.")
        return None
    except Exception as e:
        logger.error(f"Error getting duration: {str(e)}")
        st.error(f"Error getting duration: {str(e)}")
        return None

def get_video_properties(video_path):
    """Get video properties using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=width,height,codec_name,r_frame_rate,bit_rate",
        "-show_entries", "format=duration,size,bit_rate",
        "-select_streams", "v:0",
        "-of", "json",
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if result.returncode != 0:
            logger.error(f"ffprobe error for properties: {result.stderr}")
            return {
                'width': 'Error',
                'height': 'Error',
                'codec': 'Error',
                'framerate': 'Error',
                'size_mb': 'Error',
                'bitrate': 'Error'
            }
        data = json.loads(result.stdout)
        stream_info = data.get('streams', [{}])[0]
        format_info = data.get('format', {})

        width = int(stream_info.get('width', 0)) or 'Unknown'
        height = int(stream_info.get('height', 0)) or 'Unknown'
        codec = stream_info.get('codec_name', 'Unknown')
        
        r_frame_rate = stream_info.get('r_frame_rate', '0/0')
        try:
            num, den = map(int, r_frame_rate.split('/'))
            framerate = round(num / den, 2) if den != 0 else 'Unknown'
        except (ValueError, ZeroDivisionError):
            framerate = 'Unknown'

        size_mb = round(int(format_info.get('size', 0)) / (1024 * 1024), 2) if format_info.get('size') else 'Unknown'
        bitrate = format_info.get('bit_rate')
        bitrate = f"{round(int(bitrate) / 1000, 2)} Kbps" if bitrate and bitrate.isdigit() else 'Unknown'

        return {
            'width': width,
            'height': height,
            'codec': codec,
            'framerate': framerate,
            'size_mb': size_mb,
            'bitrate': bitrate
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
    """Check for audio stream in video."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type,codec_name,bit_rate,sample_rate,channels",
        "-of", "json",
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if result.returncode != 0:
            return False, {}
        audio_info = json.loads(result.stdout)
        streams = audio_info.get('streams', [])
        if not streams:
            return False, {}
        stream = streams[0]
        audio_bitrate = stream.get('bit_rate')
        audio_bitrate = f"{round(int(audio_bitrate) / 1000, 2)} Kbps" if audio_bitrate and audio_bitrate.isdigit() else 'Unknown'
        return True, {
            'codec': stream.get('codec_name', 'Unknown'),
            'bitrate': audio_bitrate,
            'sample_rate': stream.get('sample_rate', 'Unknown'),
            'channels': stream.get('channels', 'Unknown')
        }
    except Exception as e:
        logger.error(f"Error checking audio stream: {str(e)}")
        return False, {}

def extract_metadata(video_path):
    """Extract video metadata."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-of", "json",
        "-show_entries", "format_tags",
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if result.returncode != 0:
            return {'raw_metadata': {}, 'error': result.stderr}
        metadata = json.loads(result.stdout)
        tags = metadata.get('format', {}).get('tags', {})
        creation_date = tags.get('creation_time')
        if creation_date:
            try:
                dt = datetime.datetime.strptime(creation_date, '%Y-%m-%dT%H:%M:%S.%fZ')
                creation_date = dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass
        return {
            'creation_date': creation_date or 'Unknown',
            'software': tags.get('encoder', tags.get('software', 'Unknown')),
            'author': tags.get('artist', tags.get('author', 'Unknown')),
            'raw_metadata': tags
        }
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        return {'raw_metadata': {}, 'error': str(e)}

def extract_thumbnail(video_path):
    """Extract a thumbnail from the video."""
    try:
        duration = check_duration(video_path)
        if not duration:
            return None
        time_position = duration / 2
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_filename = temp_file.name
        cmd = [
            "ffmpeg",
            "-ss", str(time_position),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            temp_filename
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        if result.returncode != 0:
            logger.error(f"FFmpeg thumbnail extraction failed: {result.stderr.decode()}")
            return None
        with open(temp_filename, "rb") as img_file:
            img_data = img_file.read()
        os.unlink(temp_filename)
        return img_data
    except Exception as e:
        logger.error(f"Error extracting thumbnail: {str(e)}")
        if 'temp_filename' in locals():
            os.unlink(temp_filename)
        return None

def check_ai_indicators(video_path, metadata):
    """Check for AI-generated indicators."""
    ai_indicators = []
    ai_score = 0
    software = metadata.get('software', '').lower() if metadata.get('software') else ''
    raw_metadata = metadata.get('raw_metadata', {})
    
    ai_keywords = ['dall-e', 'midjourney', 'stable diffusion', 'runway', 'synthesia', 
                   'deepfake', 'neural', 'ai generated', 'openai', 'generated']
    
    if software and any(keyword in software for keyword in ai_keywords):
        ai_indicators.append(f"AI software detected: {software}")
        ai_score += 30
    
    for key, value in raw_metadata.items():
        if isinstance(value, str) and any(keyword in value.lower() for keyword in ai_keywords):
            ai_indicators.append(f"AI keyword in metadata: {key}={value}")
            ai_score += 20
    
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,bit_rate,width,height",
        "-of", "json",
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            stream = data.get('streams', [{}])[0]
            bit_rate = stream.get('bit_rate', '')
            width = int(stream.get('width', 0)) or 0
            height = int(stream.get('height', 0)) or 0
            if bit_rate and bit_rate.isdigit() and int(bit_rate) % 1000000 == 0:
                ai_indicators.append(f"Suspiciously round bitrate: {int(bit_rate)/1000000}M")
                ai_score += 15
            if width == height and width > 0 and (width & (width - 1) == 0):
                ai_indicators.append(f"AI-typical resolution: {width}x{height}")
                ai_score += 20
    except Exception:
        logger.warning("Failed to check AI indicators via ffprobe.")

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
        ai_check = check_ai_indicators(video_path, metadata)
        
        file_hash = hashlib.md5(open(video_path, 'rb').read(1024*1024)).hexdigest()[:10] + "..."
        duration_status = "PASS" if duration and MIN_DURATION_DEFAULT <= duration <= MAX_DURATION_DEFAULT else "FAIL"
        audio_status = "FAIL" if has_audio else "PASS"
        
        resolution = f"{video_props['width']}x{video_props['height']}"
        resolution_check = "HD+" if isinstance(video_props['width'], int) and video_props['width'] >= 1280 else "SD"
        
        file_size_mb = video_props['size_mb'] if video_props['size_mb'] != 'Error' else 0
        file_size_status = "Large" if file_size_mb > 50 else "Medium" if file_size_mb > 10 else "Small"
        
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

# Check FFmpeg installation
ffmpeg_installed = check_ffmpeg_installed()
ffprobe_installed = check_ffprobe_installed()

if not (ffmpeg_installed and ffprobe_installed):
    st.warning("FFmpeg not detected. Attempting to install...")
    if install_ffmpeg():
        st.success("FFmpeg installed successfully!")
        ffmpeg_installed = ffprobe_installed = True
    else:
        st.error("Failed to install FFmpeg. Please contact your administrator.")
        st.stop()

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
    - AI Likelihood: Based on metadata and codec analysis
""")

with st.expander("About This Tool"):
    st.markdown("""
        ### Cloud Video Prompt Validation Tool
        Analyze video files for compliance with prompt requirements.
        
        **Features:**
        - Validates duration, audio presence, and resolution.
        - Detects potential AI-generated content.
        - Generates detailed reports and visualizations.
        
        **Usage:**
        1. Upload videos (MP4, MOV, AVI, MKV, WebM).
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
