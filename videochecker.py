import streamlit as st
import os
import subprocess
import json
import pandas as pd
import datetime
from moviepy.editor import VideoFileClip
import hashlib
from PIL import Image
import io
import numpy as np
import re

# ----------------------------- SETTINGS -----------------------------
VIDEO_FOLDER = r"D:\VideoPrompts"  # <-- Change this to your videos folder
MIN_DURATION = 20  # seconds
MAX_DURATION = 30  # seconds

# ----------------------------- FUNCTIONS -----------------------------
def get_video_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]

def check_duration(video_path):
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
        return duration
    except Exception as e:
        return None

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
        data = json.loads(result.stdout)
        
        # Extract stream info
        stream_info = data.get('streams', [{}])[0]
        width = stream_info.get('width', 'Unknown')
        height = stream_info.get('height', 'Unknown')
        codec = stream_info.get('codec_name', 'Unknown')
        
        # Extract framerate
        r_frame_rate = stream_info.get('r_frame_rate', '0/0')
        if '/' in r_frame_rate:
            num, den = map(int, r_frame_rate.split('/'))
            framerate = round(num / den, 2) if den != 0 else 0
        else:
            framerate = float(r_frame_rate)
        
        # Extract format info
        format_info = data.get('format', {})
        size_bytes = int(format_info.get('size', 0))
        size_mb = round(size_bytes / (1024 * 1024), 2)
        bitrate = format_info.get('bit_rate', 'Unknown')
        if bitrate != 'Unknown':
            bitrate = f"{round(int(bitrate) / 1000, 2)} Kbps"
            
        return {
            'width': width,
            'height': height,
            'codec': codec,
            'framerate': framerate,
            'size_mb': size_mb,
            'bitrate': bitrate
        }
    except Exception as e:
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
        audio_info = json.loads(result.stdout)
        streams = audio_info.get('streams', [])
        
        if len(streams) > 0:
            stream = streams[0]
            audio_codec = stream.get('codec_name', 'Unknown')
            audio_bitrate = stream.get('bit_rate', 'Unknown')
            if audio_bitrate != 'Unknown':
                audio_bitrate = f"{round(int(audio_bitrate) / 1000, 2)} Kbps"
            audio_sample_rate = stream.get('sample_rate', 'Unknown')
            audio_channels = stream.get('channels', 'Unknown')
            
            return True, {
                'codec': audio_codec,
                'bitrate': audio_bitrate,
                'sample_rate': audio_sample_rate,
                'channels': audio_channels
            }
        return False, {}
    except Exception as e:
        return False, {}

def extract_metadata(video_path):
    cmd = [
        "exiftool",
        "-json",
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            # If exiftool is not installed, use ffprobe instead
            cmd = [
                "ffprobe",
                "-v", "error",
                "-of", "json",
                "-show_entries", "format_tags",
                video_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
        else:
            metadata = json.loads(result.stdout)
            if metadata and len(metadata) > 0:
                md = metadata[0]
                return {
                    'creation_date': md.get('CreateDate', None),
                    'software': md.get('Software', None),
                    'author': md.get('Author', None),
                    'raw_metadata': md
                }
            return {'raw_metadata': {}}
    except Exception as e:
        return {'raw_metadata': {}, 'error': str(e)}

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
    
    # Try to extract a frame and analyze it (simplified)
    try:
        clip = VideoFileClip(video_path)
        if clip.duration > 0:
            # Extract middle frame
            frame = clip.get_frame(clip.duration / 2)
            clip.close()
            
            # Convert to PIL Image for analysis
            img = Image.fromarray((frame * 255).astype('uint8'))
            
            # Check for excessive smoothness (potential AI indicator)
            img_array = np.array(img)
            edges = np.std(img_array)
            
            if edges < 20:  # Very low edge detail
                ai_indicators.append("Low detail variation (smooth textures)")
                ai_score += 15
            
            # Hash the middle frame
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            
            # Here you could compare against known AI model frame signatures
            # (simplified example)
            if img_hash.startswith('a') and img_hash.endswith('f'):
                ai_indicators.append("Frame signature matches known AI pattern")
                ai_score += 10
                
    except Exception as e:
        pass
    
    # Normalize score to 0-100
    ai_score = min(ai_score, 100)
    
    return {
        'ai_indicators': ai_indicators,
        'ai_score': ai_score,
        'ai_likelihood': "High" if ai_score > 70 else "Medium" if ai_score > 30 else "Low"
    }

def analyze_video(video_path):
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
    file_hash = hashlib.md5(open(video_path, 'rb').read(1024*1024)).hexdigest()  # First MB only
    
    # Status checks
    duration_status = "PASS" if (duration and MIN_DURATION <= duration <= MAX_DURATION) else "FAIL"
    audio_status = "FAIL" if has_audio else "PASS"
    resolution_check = "HD+" if video_props['width'] and video_props['width'] >= 1280 else "SD"
    
    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
    file_size_status = "Large" if file_size > 50 else "Medium" if file_size > 10 else "Small"
    
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
        except:
            pass
    
    return {
        "Filename": os.path.basename(video_path),
        "Duration (s)": round(duration, 2) if duration else "Error",
        "Duration Status": duration_status,
        "Audio Present": "Yes" if has_audio else "No",
        "Audio Status": audio_status,
        "Resolution": f"{video_props['width']}x{video_props['height']}",
        "Codec": video_props['codec'],
        "Framerate": video_props['framerate'],
        "File Size (MB)": video_props['size_mb'],
        "Bitrate": video_props['bitrate'],
        "Creation Date": creation_date,
        "AI Likelihood": ai_check['ai_likelihood'],
        "File Hash": file_hash[:10] + "...",  # Truncated for display
        "Software": metadata.get('software', 'Unknown'),
        "Full Path": video_path,
        # Store additional details for the detailed view
        "_audio_props": audio_props,
        "_ai_indicators": ai_check['ai_indicators'],
        "_metadata": metadata
    }

# ----------------------------- STREAMLIT APP -----------------------------
st.set_page_config(page_title="Video Prompt Validator", layout="wide")
st.title("🎥 Advanced Video Prompt Validation Tool")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    video_folder = st.text_input("Video Folder Path:", VIDEO_FOLDER)
    min_duration = st.slider("Minimum Duration (seconds):", 1, 60, MIN_DURATION)
    max_duration = st.slider("Maximum Duration (seconds):", min_duration, 180, MAX_DURATION)
    
    st.subheader("Validation Options")
    check_ai = st.checkbox("Check for AI-generated indicators", True)
    extract_full_metadata = st.checkbox("Extract full metadata", False)
    generate_report = st.checkbox("Generate CSV report", False)

# Main area
col1, col2 = st.columns([3, 1])
with col1:
    if st.button("Analyze Videos", key="analyze_btn"):
        if not os.path.isdir(video_folder):
            st.error("Folder not found!")
        else:
            video_files = get_video_files(video_folder)
            if not video_files:
                st.warning("No video files found in the folder.")
            else:
                with st.spinner(f"Analyzing {len(video_files)} videos..."):
                    results = []
                    for video in video_files:
                        result = analyze_video(video)
                        results.append(result)
                    
                    # Create DataFrame for display
                    display_columns = ["Filename", "Duration (s)", "Duration Status", 
                                      "Audio Present", "Audio Status", "Resolution", 
                                      "Framerate", "File Size (MB)", "AI Likelihood"]
                    
                    df = pd.DataFrame(results)
                    display_df = df[display_columns].copy()
                    
                    # Highlight rows with issues
                    def highlight_status(s):
                        styles = [''] * len(s)
                        if s['Duration Status'] == 'FAIL':
                            styles = ['background-color: #ffcccc'] * len(s)
                        if s['Audio Status'] == 'FAIL':
                            styles = ['background-color: #ffffcc'] * len(s)
                        if s['AI Likelihood'] == 'High':
                            styles = ['background-color: #ffdddd'] * len(s)
                        return styles
                    
                    st.dataframe(display_df.style.apply(highlight_status, axis=1), 
                                use_container_width=True, height=400)
                    
                    # Summary statistics
                    st.subheader("Summary")
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    
                    with col_stats1:
                        pass_count = sum(df['Duration Status'] == 'PASS')
                        st.metric("Valid Duration", f"{pass_count}/{len(df)}")
                        
                    with col_stats2:
                        audio_fail = sum(df['Audio Status'] == 'FAIL')
                        st.metric("With Audio", f"{audio_fail}/{len(df)}")
                        
                    with col_stats3:
                        ai_high = sum(df['AI Likelihood'] == 'High')
                        st.metric("Likely AI Generated", f"{ai_high}/{len(df)}")
                    
                    # Save report if requested
                    if generate_report:
                        report_path = os.path.join(video_folder, f"video_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                        df.to_csv(report_path, index=False)
                        st.success(f"Report saved to: {report_path}")
                    
                    # Store results for detailed view
                    st.session_state.results = results
                    st.session_state.selected_video = None if not results else results[0]["Filename"]

with col2:
    st.subheader("Video Details")
    
    if 'results' in st.session_state and st.session_state.results:
        # Create a selectbox with all video filenames
        filenames = [r["Filename"] for r in st.session_state.results]
        selected_video = st.selectbox("Select Video", filenames, 
                                    index=filenames.index(st.session_state.selected_video) if st.session_state.selected_video in filenames else 0)
        st.session_state.selected_video = selected_video
        
        # Find the selected video data
        video_data = next((r for r in st.session_state.results if r["Filename"] == selected_video), None)
        
        if video_data:
            # Display thumbnail if possible
            try:
                full_path = video_data["Full Path"]
                clip = VideoFileClip(full_path)
                # Get frame from 1 second in or middle if shorter
                frame_time = min(1.0, clip.duration / 2)
                frame = clip.get_frame(frame_time)
                clip.close()
                
                # Convert to PIL Image and then to bytes for st.image
                img = Image.fromarray((frame * 255).astype('uint8'))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                
                st.image(buf.getvalue(), caption=f"{selected_video} (Thumbnail)", width=250)
            except Exception as e:
                st.warning("Could not load thumbnail")
            
            # Display basic info
            st.write(f"**Duration:** {video_data['Duration (s)']} seconds")
            st.write(f"**Resolution:** {video_data['Resolution']}")
            st.write(f"**Codec:** {video_data['Codec']}")
            st.write(f"**Created:** {video_data['Creation Date']}")
            
            # AI indicators
            if video_data["_ai_indicators"]:
                st.subheader("🤖 AI Indicators")
                for indicator in video_data["_ai_indicators"]:
                    st.write(f"- {indicator}")
            
            # Audio details
            if video_data["Audio Present"] == "Yes" and video_data["_audio_props"]:
                st.subheader("🔊 Audio Properties")
                audio_props = video_data["_audio_props"]
                st.write(f"**Codec:** {audio_props.get('codec', 'Unknown')}")
                st.write(f"**Bitrate:** {audio_props.get('bitrate', 'Unknown')}")
                st.write(f"**Sample Rate:** {audio_props.get('sample_rate', 'Unknown')} Hz")
                st.write(f"**Channels:** {audio_props.get('channels', 'Unknown')}")
            
            # Detailed metadata (collapsible)
            if extract_full_metadata and video_data["_metadata"].get("raw_metadata"):
                with st.expander("Full Metadata"):
                    st.json(video_data["_metadata"]["raw_metadata"])
    else:
        st.info("Run analysis to see video details")

# Add helpful information at the bottom
st.markdown("---")
st.write("""
**Notes:**
- Duration Status: FAIL if outside the specified range
- Audio Status: FAIL if audio is present (for voiceless video prompts)
- AI Likelihood: Based on metadata and content analysis
""")
