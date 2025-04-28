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

# ----------------------------- SETTINGS -----------------------------
MIN_DURATION = 20  # seconds
MAX_DURATION = 30  # seconds

# ----------------------------- FUNCTIONS -----------------------------
def install_ffmpeg():
    """Install FFmpeg in the cloud environment"""
    try:
        # Try installing ffmpeg via apt-get (works on Debian/Ubuntu-based systems)
        st.info("Installing FFmpeg... this may take a minute.")
        result = subprocess.run(
            ["apt-get", "update", "-qq", "&&", "apt-get", "install", "-y", "ffmpeg"],
            shell=True, capture_output=True, text=True
        )
        return True
    except Exception as e:
        st.error(f"Failed to install FFmpeg: {str(e)}")
        return False

def check_ffmpeg_installed():
    """Check if ffmpeg is installed and accessible"""
    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except:
        return False

def check_ffprobe_installed():
    """Check if ffprobe is installed and accessible"""
    try:
        result = subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except:
        return False

def check_duration(video_path):
    """Get video duration using ffprobe"""
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "json", 
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            st.error(f"FFprobe error: {result.stderr}")
            return None
            
        data = json.loads(result.stdout)
        if 'format' in data and 'duration' in data['format']:
            duration = float(data['format']['duration'])
            return duration
        else:
            st.warning("Duration information not found in video")
            return None
    except json.JSONDecodeError:
        st.error(f"Error parsing JSON output from ffprobe")
        return None
    except Exception as e:
        st.error(f"Error getting duration: {str(e)}")
        return None

def safe_convert_to_int(value, default='Unknown'):
    """Safely convert value to int"""
    if value == 'Unknown' or value == 'Error':
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def get_video_properties(video_path):
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
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return {
                'width': 'Error',
                'height': 'Error',
                'codec': 'Error',
                'framerate': 'Error',
                'size_mb': 'Error',
                'bitrate': 'Error'
            }
            
        data = json.loads(result.stdout)
        
        # Extract stream info
        stream_info = data.get('streams', [{}])[0]
        
        # Try to convert width and height to integers
        try:
            width = int(stream_info.get('width', 0))
        except (ValueError, TypeError):
            width = 'Unknown'
            
        try:
            height = int(stream_info.get('height', 0))
        except (ValueError, TypeError):
            height = 'Unknown'
            
        codec = stream_info.get('codec_name', 'Unknown')
        
        # Extract framerate
        r_frame_rate = stream_info.get('r_frame_rate', '0/0')
        try:
            if '/' in r_frame_rate:
                num, den = map(int, r_frame_rate.split('/'))
                framerate = round(num / den, 2) if den != 0 else 0
            else:
                framerate = float(r_frame_rate)
        except (ValueError, ZeroDivisionError):
            framerate = 'Unknown'
        
        # Extract format info
        format_info = data.get('format', {})
        try:
            size_bytes = int(format_info.get('size', 0))
            size_mb = round(size_bytes / (1024 * 1024), 2)
        except (ValueError, TypeError):
            size_mb = 'Unknown'
            
        bitrate = format_info.get('bit_rate', 'Unknown')
        if bitrate != 'Unknown' and bitrate:
            try:
                bitrate = f"{round(int(bitrate) / 1000, 2)} Kbps"
            except (ValueError, TypeError):
                bitrate = 'Unknown'
            
        return {
            'width': width,
            'height': height,
            'codec': codec,
            'framerate': framerate,
            'size_mb': size_mb,
            'bitrate': bitrate
        }
    except json.JSONDecodeError:
        st.error(f"Error parsing JSON from ffprobe")
        return {
            'width': 'Error',
            'height': 'Error',
            'codec': 'Error',
            'framerate': 'Error',
            'size_mb': 'Error',
            'bitrate': 'Error'
        }
    except Exception as e:
        st.error(f"Error getting video properties: {str(e)}")
        return {
            'width': 'Error',
            'height': 'Error',
            'codec': 'Error',
            'framerate': 'Error',
            'size_mb': 'Error',
            'bitrate': 'Error'
        }

def has_audio_stream(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type,codec_name,bit_rate,sample_rate,channels",
        "-of", "json",
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return False, {}
            
        audio_info = json.loads(result.stdout)
        streams = audio_info.get('streams', [])
        
        if len(streams) > 0:
            stream = streams[0]
            audio_codec = stream.get('codec_name', 'Unknown')
            audio_bitrate = stream.get('bit_rate', 'Unknown')
            
            if audio_bitrate != 'Unknown' and audio_bitrate:
                try:
                    audio_bitrate = f"{round(int(audio_bitrate) / 1000, 2)} Kbps"
                except (ValueError, TypeError):
                    audio_bitrate = 'Unknown'
                    
            audio_sample_rate = stream.get('sample_rate', 'Unknown')
            audio_channels = stream.get('channels', 'Unknown')
            
            return True, {
                'codec': audio_codec,
                'bitrate': audio_bitrate,
                'sample_rate': audio_sample_rate,
                'channels': audio_channels
            }
        return False, {}
    except json.JSONDecodeError:
        st.error(f"Error parsing JSON from ffprobe")
        return False, {}
    except Exception as e:
        st.error(f"Error checking audio stream: {str(e)}")
        return False, {}

def extract_metadata(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-of", "json",
        "-show_entries", "format_tags",
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return {'raw_metadata': {}, 'error': result.stderr}
            
        metadata = json.loads(result.stdout)
        tags = metadata.get('format', {}).get('tags', {})
        
        # Check for creation date
        creation_date = tags.get('creation_time', None)
        software = tags.get('encoder', tags.get('software', None))
        author = tags.get('artist', tags.get('author', None))
        
        return {
            'creation_date': creation_date,
            'software': software,
            'author': author,
            'raw_metadata': tags
        }
    except json.JSONDecodeError:
        st.error(f"Error parsing JSON metadata")
        return {'raw_metadata': {}, 'error': 'JSON parse error'}
    except Exception as e:
        st.error(f"Error extracting metadata: {str(e)}")
        return {'raw_metadata': {}, 'error': str(e)}

def extract_thumbnail(video_path):
    """Extract thumbnail using ffmpeg"""
    try:
        # Create a temporary file for the thumbnail
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        # Get video duration first to calculate middle point
        duration = check_duration(video_path)
        if not duration:
            return None
            
        # Extract frame from middle of video
        time_position = duration / 2
        
        cmd = [
            "ffmpeg",
            "-ss", str(time_position),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            temp_filename
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            st.error(f"Error running ffmpeg to extract thumbnail: {result.stderr}")
            return None
        
        # Read the image and convert to base64
        with open(temp_filename, "rb") as img_file:
            img_data = img_file.read()
            
        # Clean up temp file
        try:
            os.unlink(temp_filename)
        except Exception:
            pass
        
        return img_data
    except Exception as e:
        st.error(f"Error extracting thumbnail: {str(e)}")
        if 'temp_filename' in locals():
            try:
                os.unlink(temp_filename)
            except:
                pass
        return None

def check_ai_indicators(video_path, metadata):
    """Check for potential AI generation indicators"""
    ai_indicators = []
    ai_score = 0
    
    # Check metadata for AI-related software
    software = metadata.get('software', '').lower() if metadata.get('software') else ''
    raw_metadata = metadata.get('raw_metadata', {})
    
    ai_software_keywords = ['dall-e', 'midjourney', 'stable diffusion', 'runway', 'synthesia', 
                           'deepfake', 'neural', 'ai generated', 'openai', 'generated']
    
    # Check software field
    if software:
        for keyword in ai_software_keywords:
            if keyword in software:
                ai_indicators.append(f"AI software detected: {software}")
                ai_score += 30
                break
    
    # Check all metadata fields for AI keywords
    for key, value in raw_metadata.items():
        if isinstance(value, str):
            value_lower = value.lower()
            for keyword in ai_software_keywords:
                if keyword in value_lower:
                    ai_indicators.append(f"AI keyword in metadata: {key}={value}")
                    ai_score += 20
                    break
    
    # Check for unusual codec combinations or perfect bitrates
    # (common in AI-generated content)
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,bit_rate,width,height",
        "-of", "json",
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get('streams', [])
            
            if streams:
                stream = streams[0]
                codec = stream.get('codec_name', '').lower()
                bit_rate = stream.get('bit_rate', '')
                
                try:
                    width = int(stream.get('width', 0))
                    height = int(stream.get('height', 0))
                except (ValueError, TypeError):
                    width = height = 0
                
                # Check for perfectly round bitrates (common in AI generation)
                if bit_rate and isinstance(bit_rate, str) and bit_rate.isdigit():
                    try:
                        br = int(bit_rate)
                        if br % 1000000 == 0 and br > 0:
                            ai_indicators.append(f"Suspiciously round bitrate: {br/1000000}M")
                            ai_score += 15
                    except (ValueError, TypeError):
                        pass
                
                # Check for unusual resolution/codec combinations
                if codec == 'h264' and width > 0 and height > 0:
                    # If it's exactly 512x512, 1024x1024, etc. (common in AI)
                    if width == height and width > 0 and (width & (width - 1) == 0):  # Power of 2
                        ai_indicators.append(f"AI-typical resolution: {width}x{height}")
                        ai_score += 20
    except json.JSONDecodeError:
        pass
    except Exception:
        pass
                
    # Normalize score to 0-100
    ai_score = min(ai_score, 100)
    
    return {
        'ai_indicators': ai_indicators,
        'ai_score': ai_score,
        'ai_likelihood': "High" if ai_score > 70 else "Medium" if ai_score > 30 else "Low"
    }

def analyze_video(video_path, filename):
    """Analyze a video file and return its properties"""
    # Basic checks
    duration = check_duration(video_path)
    has_audio, audio_props = has_audio_stream(video_path)
    
    # Get more detailed properties
    video_props = get_video_properties(video_path)
    
    # Get metadata
    metadata = extract_metadata(video_path)
    
    # Check for AI indicators
    ai_check = check_ai_indicators(video_path, metadata)
    
    # Calculate file hash for identity verification
    try:
        file_hash = hashlib.md5(open(video_path, 'rb').read(1024*1024)).hexdigest()  # First MB only
    except Exception as e:
        file_hash = "Error"
    
    # Status checks
    duration_status = "PASS" if (duration and MIN_DURATION <= duration <= MAX_DURATION) else "FAIL"
    audio_status = "FAIL" if has_audio else "PASS"  # FAIL if audio present
    
    # Fix the resolution check to handle non-numeric values
    width = video_props['width']
    if isinstance(width, (int, float)) and width >= 1280:
        resolution_check = "HD+"
    else:
        resolution_check = "SD"
    
    try:
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        file_size_status = "Large" if file_size > 50 else "Medium" if file_size > 10 else "Small"
    except Exception as e:
        file_size = 0
        file_size_status = "Error"
    
    # Format creation date nicely if available
    creation_date = metadata.get('creation_date', 'Unknown')
    if creation_date and creation_date != 'Unknown':
        try:
            # Try various date formats
            for fmt in ['%Y:%m:%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d %H:%M:%S']:
                try:
                    dt = datetime.datetime.strptime(creation_date, fmt)
                    creation_date = dt.strftime('%Y-%m-%d %H:%M:%S')
                    break
                except ValueError:
                    continue
        except Exception:
            pass
    
    return {
        "Filename": filename,
        "Duration (s)": round(duration, 2) if duration else "Error",
        "Duration Status": duration_status,
        "Audio Present": "Yes" if has_audio else "No",
        "Audio Status": audio_status,
        "Resolution": f"{video_props['width']}x{video_props['height']}",
        "Resolution Check": resolution_check,
        "Codec": video_props['codec'],
        "Framerate": video_props['framerate'],
        "File Size (MB)": video_props['size_mb'],
        "Bitrate": video_props['bitrate'],
        "Creation Date": creation_date,
        "AI Likelihood": ai_check['ai_likelihood'],
        "File Hash": file_hash[:10] + "..." if file_hash != "Error" else "Error",
        "Software": metadata.get('software', 'Unknown'),
        "_audio_props": audio_props,
        "_ai_indicators": ai_check['ai_indicators'],
        "_metadata": metadata,
        "_temp_path": video_path  # For thumbnail extraction
    }

# ----------------------------- STREAMLIT APP -----------------------------
st.set_page_config(page_title="Video Prompt Validator", layout="wide")
st.title("🎥 Cloud Video Prompt Validation Tool")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []
if 'selected_video' not in st.session_state:
    st.session_state.selected_video = None
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Auto-install FFmpeg if needed
ffmpeg_installed = check_ffmpeg_installed()
ffprobe_installed = check_ffprobe_installed()

if not ffmpeg_installed or not ffprobe_installed:
    st.warning("FFmpeg not detected. Attempting to install...")
    success = install_ffmpeg()
    if success:
        st.success("FFmpeg installed successfully!")
        ffmpeg_installed = True
        ffprobe_installed = True
    else:
        st.error("Failed to install FFmpeg. Please contact your administrator.")

# Only show the app if FFmpeg is available
if ffmpeg_installed and ffprobe_installed:
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        min_duration = st.slider("Minimum Duration (seconds):", 1, 60, MIN_DURATION)
        max_duration = st.slider("Maximum Duration (seconds):", min_duration, 180, MAX_DURATION)
        
        st.subheader("Validation Options")
        check_ai = st.checkbox("Check for AI-generated indicators", True)
        extract_full_metadata = st.checkbox("Extract full metadata", False)
        generate_report = st.checkbox("Generate CSV report", False)
        
        # Clear results
        if st.button("Clear All Results"):
            # Clean up any temporary files
            for temp_file in st.session_state.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
            st.session_state.results = []
            st.session_state.selected_video = None
            st.session_state.temp_files = []
            st.experimental_rerun()

    # Main area
    col1, col2 = st.columns([3, 1])
    with col1:
        # File uploader for videos
        uploaded_files = st.file_uploader("Upload Video Files", 
                                        type=["mp4", "mov", "avi", "mkv", "webm"], 
                                        accept_multiple_files=True)
        
        if uploaded_files:
            analyze_btn = st.button("Analyze Uploaded Videos", key="analyze_btn", 
                                   disabled=st.session_state.processing)
            
            if analyze_btn:
                st.session_state.processing = True
                progress_bar = st.progress(0)
                
                with st.spinner(f"Analyzing {len(uploaded_files)} videos..."):
                    new_results = []
                    
                    # Process each uploaded file
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Create a temporary file for FFmpeg to work with
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                                temp_file.write(uploaded_file.getvalue())
                                temp_path = temp_file.name
                            
                            # Keep track of temp files to delete later
                            st.session_state.temp_files.append(temp_path)
                            
                            # Analyze the video
                            with st.status(f"Analyzing {uploaded_file.name}..."):
                                try:
                                    result = analyze_video(temp_path, uploaded_file.name)
                                    new_results.append(result)
                                except Exception as e:
                                    st.error(f"Error analyzing {uploaded_file.name}: {str(e)}")
                                    # Add a placeholder result for failed files
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
                                        "_ai_indicators": ["Analysis failed: " + str(e)],
                                        "_metadata": {"raw_metadata": {}},
                                        "_temp_path": temp_path
                                    })
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        
                        # Update progress
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                    
                    # Add new results to existing ones
                    st.session_state.results.extend(new_results)
                    
                    if not st.session_state.selected_video and new_results:
                        st.session_state.selected_video = new_results[0]["Filename"]
                
                st.session_state.processing = False
                st.experimental_rerun()  # Refresh to update UI

        # Display results if we have any
        if st.session_state.results:
            # Create DataFrame for display
            display_columns = ["Filename", "Duration (s)", "Duration Status", 
                              "Audio Present", "Audio Status", "Resolution", 
                              "Framerate", "File Size (MB)", "AI Likelihood"]
            
            # Create a dataframe with safe error handling
            data_for_df = []
            for result in st.session_state.results:
                row = {}
                for col in display_columns:
                    row[col] = result.get(col, "Error")
                data_for_df.append(row)
            
            df = pd.DataFrame(data_for_df)
            
            # Highlight rows with issues
            def highlight_row(row):
                color = ''
                if row.get('Duration Status') == 'FAIL':
                    color = 'background-color: #ffcccc'
                if row.get('Audio Status') == 'FAIL':
                    color = 'background-color: #ffffcc'
                if row.get('AI Likelihood') == 'High':
                    color = 'background-color: #ffdddd'
                return [color] * len(row)
            
            try:
                styled_df = df.style.apply(highlight_row, axis=1)
                st.dataframe(styled_df, use_container_width=True, height=400)
            except Exception as e:
                st.error(f"Error displaying dataframe: {str(e)}")
                st.dataframe(df, use_container_width=True, height=400)
            
            # Summary statistics
            st.subheader("Summary")
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            
            with col_stats1:
                pass_count = sum([1 for r in st.session_state.results if r.get('Duration Status') == 'PASS'])
                st.metric("Valid Duration", f"{pass_count}/{len(st.session_state.results)}")
                
            with col_stats2:
                audio_fail = sum([1 for r in st.session_state.results if r.get('Audio Status') == 'FAIL'])
                st.metric("With Audio", f"{audio_fail}/{len(st.session_state.results)}")
                
            with col_stats3:
                ai_high = sum([1 for r in st.session_state.results if r.get('AI Likelihood') == 'High'])
                st.metric("Likely AI Generated", f"{ai_high}/{len(st.session_state.results)}")
            
            # Save report if requested
            if generate_report:
                try:
                    # Create CSV in memory
                    csv = df.to_csv(index=False).encode('utf-8')
                    
                    # Create download button
                    st.download_button(
                        label="Download CSV Report",
                        data=csv,
                        file_name=f"video_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                
        else:
            st.info("Upload videos and click 'Analyze Uploaded Videos' to begin.")

    with col2:
        st.subheader("Video Details")
        
        if st.session_state.results:
            # Create a selectbox with all video filenames
            filenames = [r["Filename"] for r in st.session_state.results]
            
            try:
                selected_index = filenames.index(st.session_state.selected_video) if st.session_state.selected_video in filenames else 0
            except ValueError:
                selected_index = 0
                
            selected_video = st.selectbox("Select Video", filenames, index=selected_index)
            st.session_state.selected_video = selected_video
            
            # Find the selected video data
            video_data = next((r for r in st.session_state.results if r["Filename"] == selected_video), None)
            
            if video_data:
                # Try to extract thumbnail if possible
                try:
                    temp_path = video_data.get("_temp_path")
                    if temp_path and os.path.exists(temp_path):
                        with st.spinner("Loading thumbnail..."):
                            thumb_data = extract_thumbnail(temp_path)
                            if thumb_data:
                                st.image(thumb_data, caption=f"Thumbnail", width=250)
                            else:
                                st.warning("Could not load thumbnail")
                    else:
                        st.warning("Video file not found")
                except Exception as e:
                    st.warning(f"Could not load thumbnail: {str(e)}")
                
                # Display basic info in a more structured way
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.write(f"**Duration:** {video_data['Duration (s)']} seconds")
                    st.write(f"**Resolution:** {video_data['Resolution']}")
                    st.write(f"**File Size:** {video_data['File Size (MB)']} MB")
                
                with info_col2:
                    st.write(f"**Codec:** {video_data['Codec']}")
                    st.write(f"**Framerate:** {video_data['Framerate']}")
                    st.write(f"**Created:** {video_data['Creation Date']}")
                
                # AI indicators
                if video_data.get("_ai_indicators", []):
                    st.subheader("🤖 AI Indicators")
                    for indicator in video_data["_ai_indicators"]:
                        st.write(f"- {indicator}")
                else:
                    st.write("**AI Indicators:** None detected")
                
                # Audio details
                if video_data.get("Audio Present") == "Yes" and video_data.get("_audio_props", {}):
                    st.subheader("🔊 Audio Properties")
                    audio_props = video_data["_audio_props"]
                    st.write(f"**Codec:** {audio_props.get('codec', 'Unknown')}")
                    st.write(f"**Bitrate:** {audio_props.get('bitrate', 'Unknown')}")
                    st.write(f"**Sample Rate:** {audio_props.get('sample_rate', 'Unknown')} Hz")
                    st.write(f"**Channels:** {audio_props.get('channels', 'Unknown')}")
                
                # Detailed metadata (collapsible)
                if extract_full_metadata and video_data.get("_metadata", {}).get("raw_metadata"):
                    with st.expander("Full Metadata"):
                        try:
                            st.json(video_data["_metadata"]["raw_metadata"])
                        except Exception as e:
                            st.error(f"Error displaying metadata: {str(e)}")
                            st.write(str(video_data["_metadata"]["raw_metadata"]))
        else:
            st.info("No videos analyzed yet")

    # Add helpful information at the bottom
    st.markdown("---")
    st.write("""
    **Notes:**
    - Duration Status: FAIL if outside the specified range ({} - {} seconds)
    - Audio Status: FAIL if audio is present (for voiceless video prompts)
    - AI Likelihood: Based on metadata and content analysis
    """.format(min_duration, max_duration))

    # Display instructions for cloud usage
    with st.expander("About This Tool"):
        st.markdown("""
        ### Cloud Video Prompt Validation Tool
        
        This tool helps validate video prompts by analyzing:
        - Duration requirements
        - Audio presence/absence
        - Video quality and properties
        - Potential AI-generated content markers
        - Metadata and technical specifications

        **Usage Instructions:**
        1. Upload your video files using the uploader
        2. Click "Analyze Uploaded Videos"
        3. View results in the table and detailed view
        4. Download a CSV report if needed
        
        **Validation Criteria:**
        - Video duration should be between the specified min/max seconds
        - Videos should not contain audio (for silent prompts)
        - Videos should meet quality requirements for resolution and bitrate
        """)
else:
    st.error("This app requires FFmpeg to function. Please contact the administrator.")

# Clean up temp files on session end
def cleanup():
    for temp_file in st.session_state.temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except:
            pass

# Register the cleanup function to run at app shutdown
try:
    import atexit
    atexit.register(cleanup)
except:
    pass

# Add error handling for file operations
def safe_file_operations():
    # Attempt to create a temporary directory to ensure write permissions
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            st.session_state.temp_dir = temp_dir
            st.session_state.has_write_access = True
    except Exception as e:
        st.session_state.has_write_access = False
        st.error(f"Warning: Limited file system access. Some features may not work properly. Error: {str(e)}")

# Run this on startup
if 'has_write_access' not in st.session_state:
    safe_file_operations()

# Add cache cleanup function
def clear_temp_files():
    if st.session_state.temp_files:
        removed = 0
        for temp_file in st.session_state.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    removed += 1
            except Exception:
                pass
        
        if removed > 0:
            st.success(f"Cleaned up {removed} temporary files")
        
        st.session_state.temp_files = []

# Add this button to the sidebar
with st.sidebar:
    if st.button("Clean Temporary Files"):
        clear_temp_files()
        
# Function to handle file upload more robustly
def process_uploaded_file(uploaded_file):
    """Process an uploaded file with robust error handling"""
    try:
        # Create a temp file with appropriate extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
            
        # Verify the file exists and is accessible
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return None, f"Failed to write temporary file for {uploaded_file.name}"
            
        # Verify it's a valid video file
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "json", temp_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            return None, f"Not a valid video file: {uploaded_file.name}"
            
        return temp_path, None
    except Exception as e:
        return None, f"Error processing {uploaded_file.name}: {str(e)}"
