import os
import uuid
import logging
import tempfile
import requests
import numpy as np
import threading
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont, ImageFilter
# Patch for Pillow 10+ compatibility with MoviePy 1.0.3
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS
from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeVideoClip, 
    ImageClip, concatenate_videoclips
)
from moviepy.audio.fx.audio_loop import audio_loop
from moviepy.video.fx.crop import crop
from moviepy.video.fx.resize import resize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
OUTPUT_WIDTH = 720
OUTPUT_HEIGHT = 1280
FPS = 30
TEMP_DIR = '/tmp'
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",  # macOS fallback
    "C:\\Windows\\Fonts\\Arial.ttf",       # Windows fallback
]

def get_font(font_size, bold=True):
    """Attempt to load a bold font, fall back to default"""
    font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for font_path in FONT_PATHS:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, font_size)
        except:
            continue
    # Fallback to default font (will be small but better than nothing)
    logger.warning("No suitable font found, using default font")
    return ImageFont.load_default()

def make_gradient_background(width, height, color_top=(10,10,10), color_bottom=(26,26,46)):
    """Create a vertical gradient background image"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        ratio = y / height
        r = int(color_top[0] * (1 - ratio) + color_bottom[0] * ratio)
        g = int(color_top[1] * (1 - ratio) + color_bottom[1] * ratio)
        b = int(color_top[2] * (1 - ratio) + color_bottom[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    return img

def make_text_clip(text, width, height, font_size, duration, bg_image=None):
    """Create a clip with centered text on optional background"""
    if bg_image is None:
        bg_image = make_gradient_background(width, height)
    else:
        bg_image = bg_image.copy()
    
    draw = ImageDraw.Draw(bg_image)
    font = get_font(font_size, bold=True)
    
    # Word wrap
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        test_line = ' '.join(current_line)
        bbox = draw.textbbox((0,0), test_line, font=font)
        if bbox[2] > width * 0.85 and len(current_line) > 1:
            current_line.pop()
            lines.append(' '.join(current_line))
            current_line = [word]
    lines.append(' '.join(current_line))
    
    # Calculate total height and draw centered
    line_height = font_size + 10
    total_height = len(lines) * line_height
    y = (height - total_height) // 2
    
    for line in lines:
        bbox = draw.textbbox((0,0), line, font=font)
        x = (width - (bbox[2] - bbox[0])) // 2
        # Add text shadow
        draw.text((x+3, y+3), line, fill='#555555', font=font)
        draw.text((x, y), line, fill='white', font=font)
        y += line_height
    
    img_clip = ImageClip(np.array(bg_image), duration=duration)
    return img_clip

def make_subtitle_bar(text, width, bar_height=120):
    """Create a semi-transparent bar with centered text for subtitles"""
    img = Image.new('RGBA', (width, bar_height), (0, 0, 0, 180))
    draw = ImageDraw.Draw(img)
    font = get_font(55, bold=True)
    
    # Word wrap for subtitle (max 4 words per chunk as requested)
    words = text.split()
    if len(words) > 4:
        # Take first 4 words or split intelligently
        text = ' '.join(words[:4]) + ('...' if len(words) > 4 else '')
    
    bbox = draw.textbbox((0,0), text, font=font)
    x = (width - (bbox[2] - bbox[0])) // 2
    y = (bar_height - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, fill='white', font=font)
    
    return np.array(img)

def crop_to_portrait(clip, target_w=OUTPUT_WIDTH, target_h=OUTPUT_HEIGHT):
    """Crop video to 9:16 aspect ratio, centered"""
    src_w, src_h = clip.size
    src_aspect = src_w / src_h
    target_aspect = target_w / target_h
    
    if src_aspect > target_aspect:
        # Crop width
        new_w = int(src_h * target_aspect)
        x_center = src_w // 2
        clip = crop(clip, x_center - new_w//2, 0, x_center + new_w//2, src_h)
    else:
        # Crop height
        new_h = int(src_w / target_aspect)
        y_center = src_h // 2
        clip = crop(clip, 0, y_center - new_h//2, src_w, y_center + new_h//2)
    
    return clip.resize((target_w, target_h))

def apply_ken_burns(clip, duration, zoom_factor=0.08):
    """Apply subtle zoom effect (Ken Burns)"""
    def make_zoom(t):
        return 1 + (zoom_factor * (t / duration))
    return clip.resize(lambda t: make_zoom(t)).resize((OUTPUT_WIDTH, OUTPUT_HEIGHT))

def blur_image_clip(image_clip, radius=20):
    """Apply Gaussian blur to an ImageClip using PIL"""
    # Get the frame as PIL Image
    frame = image_clip.get_frame(0)
    pil_img = Image.fromarray(frame.astype('uint8'))
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return ImageClip(np.array(blurred), duration=image_clip.duration)

@app.route('/', methods=['GET', 'HEAD'])
def root():
    return jsonify({"status": "axis-renderer alive"})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "axis-renderer alive"})

@app.route('/video/<filename>', methods=['GET'])
def serve_video(filename):
    """Serve rendered video files"""
    return send_from_directory(TEMP_DIR, filename, as_attachment=False)

@app.route('/render', methods=['POST'])
def render_video():
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data"}), 400
    job_id = data.get('job_id', f'render_{int(datetime.now().timestamp())}')
    thread = threading.Thread(target=process_render_job, args=(data, job_id))
    thread.daemon = True
    thread.start()
    return jsonify({"status": "accepted", "job_id": job_id})

def process_render_job(data, job_id):
    """Background processing of render job"""
    try:
        video_url = data.get('video_url')
        audio_url = data.get('audio_url')
        scenes = data.get('scenes', [])
        title = data.get('title', 'output')
        
        if not video_url or not audio_url or not scenes:
            logger.error("Missing required fields in background job")
            # Fire failure callback if provided
            callback_url = data.get('callback_url')
            if callback_url and job_id:
                try:
                    requests.get(f"{callback_url}&status=failed", timeout=5)
                except Exception:
                    pass
            return
        
        logger.info(f"Starting render for video: {title}, job: {job_id}")
        
        # Download main video
        video_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        logger.info(f"Downloading video from {video_url}")
        video_resp = requests.get(video_url, stream=True)
        video_resp.raise_for_status()
        for chunk in video_resp.iter_content(chunk_size=8192):
            video_temp.write(chunk)
        video_temp.close()
        
        # Download audio
        audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        logger.info(f"Downloading audio from {audio_url}")
        audio_resp = requests.get(audio_url, stream=True)
        audio_resp.raise_for_status()
        for chunk in audio_resp.iter_content(chunk_size=8192):
            audio_temp.write(chunk)
        audio_temp.close()
        
        # Load video and audio clips
        main_video = VideoFileClip(video_temp.name)
        main_audio = AudioFileClip(audio_temp.name)
        
        # Track current position in source video for screen recording scenes
        current_video_time = 0.0
        scene_clips = []
        
        for idx, scene in enumerate(scenes):
            scene_type = scene.get('type')
            duration = float(scene.get('duration', 3.0))
            logger.info(f"Processing scene {idx+1}: {scene_type} (duration: {duration}s)")
            
            if scene_type == 'text_only':
                text = scene.get('text', '')
                clip = make_text_clip(text, OUTPUT_WIDTH, OUTPUT_HEIGHT, 80, duration)
                scene_clips.append(clip)
                
            elif scene_type == 'screen_recording':
                # Extract segment from main video
                if current_video_time >= main_video.duration:
                    # If beyond video length, use last frame as still
                    frame_clip = main_video.to_ImageClip(t=main_video.duration - 0.1)
                    frame_clip = frame_clip.set_duration(duration)
                    clip = frame_clip
                else:
                    end_time = min(current_video_time + duration, main_video.duration)
                    clip = main_video.subclip(current_video_time, end_time)
                    if end_time - current_video_time < duration:
                        # Pad with last frame if needed
                        last_frame = main_video.to_ImageClip(t=end_time - 0.1)
                        last_frame = last_frame.set_duration(duration - (end_time - current_video_time))
                        clip = concatenate_videoclips([clip, last_frame])
                    current_video_time = end_time
                
                # Crop to portrait
                clip = crop_to_portrait(clip)
                
                # Apply zoom if requested
                if scene.get('zoom', False):
                    clip = apply_ken_burns(clip, clip.duration)
                
                # Add subtitle if present
                if scene.get('subtitle'):
                    subtitle_text = scene['subtitle']
                    subtitle_arr = make_subtitle_bar(subtitle_text, OUTPUT_WIDTH)
                    subtitle_bar = ImageClip(subtitle_arr, duration=duration)
                    subtitle_bar = subtitle_bar.set_position(('center', OUTPUT_HEIGHT - 150))
                    clip = CompositeVideoClip([clip, subtitle_bar], size=(OUTPUT_WIDTH, OUTPUT_HEIGHT))
                
                scene_clips.append(clip)
                
            elif scene_type == 'stat_overlay':
                # Take a frame from screen recording (current video time position)
                frame_time = min(current_video_time - 0.1, main_video.duration - 0.1) if current_video_time > 0 else 0
                if frame_time < 0:
                    frame_time = 0
                frame_clip = main_video.to_ImageClip(t=frame_time)
                frame_clip = frame_clip.set_duration(duration)
                frame_clip = crop_to_portrait(frame_clip)
                
                # Apply blur
                blurred_clip = blur_image_clip(frame_clip, radius=20)
                
                # Add overlay text
                text = scene.get('text', '')
                overlay_clip = make_text_clip(text, OUTPUT_WIDTH, OUTPUT_HEIGHT, 90, duration, 
                                             bg_image=Image.new('RGB', (OUTPUT_WIDTH, OUTPUT_HEIGHT), (0,0,0)))
                # Make overlay semi-transparent
                overlay_clip = overlay_clip.set_opacity(0.9)
                
                clip = CompositeVideoClip([blurred_clip, overlay_clip], size=(OUTPUT_WIDTH, OUTPUT_HEIGHT))
                scene_clips.append(clip)
                
            elif scene_type == 'cta':
                # Dark gradient background
                bg = make_gradient_background(OUTPUT_WIDTH, OUTPUT_HEIGHT, (10,10,10), (26,26,46))
                bg_clip = ImageClip(np.array(bg), duration=duration)
                
                # Main text
                main_text = scene.get('text', '')
                text_clip = make_text_clip(main_text, OUTPUT_WIDTH, OUTPUT_HEIGHT, 100, duration, bg_image=bg)
                
                # Subtitle text if present
                if scene.get('subtitle'):
                    subtitle = scene['subtitle']
                    subtitle_clip = make_text_clip(subtitle, OUTPUT_WIDTH, OUTPUT_HEIGHT, 50, duration, bg_image=bg)
                    # Position subtitle below main text
                    subtitle_clip = subtitle_clip.set_position(('center', OUTPUT_HEIGHT - 300))
                    clip = CompositeVideoClip([text_clip, subtitle_clip], size=(OUTPUT_WIDTH, OUTPUT_HEIGHT))
                else:
                    clip = text_clip
                
                # Apply fade in
                clip = clip.crossfadein(0.5)
                scene_clips.append(clip)
            
            else:
                logger.warning(f"Unknown scene type: {scene_type}, skipping")
        
        # Concatenate all scene clips
        logger.info("Concatenating scenes...")
        final_video = concatenate_videoclips(scene_clips, method="compose")
        
        # Set FPS
        final_video = final_video.set_fps(FPS)
        
        # Handle audio
        logger.info("Processing audio track...")
        video_duration = final_video.duration
        
        if main_audio.duration < video_duration:
            final_audio = audio_loop(main_audio, duration=video_duration)
        else:
            final_audio = main_audio.subclip(0, video_duration)
        
        final_video = final_video.set_audio(final_audio)
        
        # Generate output filename
        output_filename = f"output_{uuid.uuid4().hex}_{int(datetime.now().timestamp())}.mp4"
        output_path = os.path.join(TEMP_DIR, output_filename)
        
        # Write video
        logger.info(f"Writing video to {output_path}")
        final_video.write_videofile(
            output_path,
            fps=FPS,
            codec='libx264',
            audio_codec='aac',
            threads=4,
            preset='medium',
            verbose=False,
            logger=None
        )
        
        # Clean up clips
        final_video.close()
        main_video.close()
        main_audio.close()
        for clip in scene_clips:
            if hasattr(clip, 'close'):
                clip.close()
        
        # Clean up temp files
        os.unlink(video_temp.name)
        os.unlink(audio_temp.name)
        
        # Generate public URL
        public_domain = os.environ.get('RAILWAY_PUBLIC_DOMAIN', 'localhost:5000')
        public_url = f"https://{public_domain}/video/{output_filename}"
        
        logger.info(f"Render complete: {public_url}")
        
        # Fire callback to AXIS if provided
        callback_url = data.get('callback_url')
        if callback_url and job_id:
            try:
                import urllib.parse
                encoded_url = urllib.parse.quote(public_url, safe='')
                full_callback = f"{callback_url}&video_url={encoded_url}&status=done"
                requests.get(full_callback, timeout=10)
                logger.info(f"Callback fired: {full_callback}")
            except Exception as cb_err:
                logger.error(f"Callback failed: {cb_err}")
    
    except Exception as e:
        logger.error(f"Background render error for job {job_id}: {str(e)}", exc_info=True)
        # Fire failure callback if we have job info
        try:
            callback_url = data.get('callback_url') if 'data' in locals() else None
            if callback_url and job_id:
                requests.get(f"{callback_url}&status=failed", timeout=5)
        except Exception:
            pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
