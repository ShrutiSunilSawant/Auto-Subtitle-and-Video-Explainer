import os
import cv2
import torch
import whisper
import librosa
import numpy as np
import moviepy.editor as mp

from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from flask import Flask, request, render_template, Response, url_for, stream_with_context
from transformers import BlipProcessor, BlipForConditionalGeneration
from pydub import AudioSegment
import pysrt
import tempfile
from datetime import timedelta
import shutil

# Initialize Flask app
app = Flask(__name__)



# Load Whisper and BLIP models
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base").to(device)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Convert timedelta to SubRip time format (used for SRT files)
def timedelta_to_subrip_time(td):
    total_seconds = td.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((total_seconds % 1) * 1000)
    return pysrt.SubRipTime(int(hours), int(minutes), int(seconds), milliseconds)

# Step 1: Extract audio from video
def extract_audio_from_video(video_path, audio_path):
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path)

# Step 2: Convert extracted audio to mono WAV format
def convert_to_wav_mono(audio_file):
    sound = AudioSegment.from_file(audio_file)
    sound = sound.set_channels(1).set_frame_rate(16000)
    wav_file = audio_file.replace(".wav", "_mono.wav")
    sound.export(wav_file, format="wav")
    return wav_file

# Step 3: Analyze silence using amplitude and Whisper transcription
def analyze_silence(audio_file, threshold_db=-30):
    y, sr = librosa.load(audio_file, sr=16000)
    intervals = librosa.effects.split(y, top_db=abs(threshold_db))
    
    # Convert intervals to seconds
    silence_intervals = []
    last_end = 0
    for start, end in intervals:
        start_sec, end_sec = start / sr, end / sr
        if start_sec - last_end > 5.0:  # 5-second gap threshold for silence
            silence_intervals.append((last_end, start_sec))
        last_end = end_sec
    
    return silence_intervals

# Step 4: Generate captions for multiple frames in silent intervals using BLIP
def generate_caption_for_silent_interval(video_path, start_time, end_time):
    captions = []
    video_clip = VideoFileClip(video_path).subclip(start_time, end_time)
    
    # Sample frames every 1 second
    for frame in video_clip.iter_frames(fps=1):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        inputs = blip_processor(images=frame_rgb, return_tensors="pt").to(device)
        caption_ids = blip_model.generate(**inputs)
        caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
        captions.append(caption)
    
    # Return the most common caption or first if captions are unique
    return max(set(captions), key=captions.count)

# Generate SRT file with captions
def generate_srt_with_captions(transcription, silence_intervals, captions, output_srt):
    subs = pysrt.SubRipFile()
    index = 1

    # Add Whisper transcriptions
    for text, start, end in transcription:
        subs.append(pysrt.SubRipItem(
            index,
            timedelta_to_subrip_time(timedelta(seconds=start)),
            timedelta_to_subrip_time(timedelta(seconds=end)),
            text
        ))
        index += 1

    # Add captions for silent intervals
    for (start_silence, end_silence), caption in zip(silence_intervals, captions):
        subs.append(pysrt.SubRipItem(
            index,
            timedelta_to_subrip_time(timedelta(seconds=start_silence)),
            timedelta_to_subrip_time(timedelta(seconds=end_silence)),
            f"[{caption}]"
        ))
        index += 1

    subs.save(output_srt, encoding='utf-8')

# Overlay subtitles on video
def overlay_subtitles(video_path, srt_path, output_video_path, subtitle_size):
    video = VideoFileClip(video_path)
    subs = pysrt.open(srt_path)

    font_size = 36 if subtitle_size == "normal" else 54

    def generate_subtitle_clip(txt):
        return TextClip(txt, font='Arial', fontsize=font_size, color='white', 
                        stroke_color='white', stroke_width=0.5, method='caption', 
                        size=(video.size[0], None))

    subtitles = []
    for sub in subs:
        start, end = sub.start.ordinal / 1000, sub.end.ordinal / 1000
        subtitle_clip = generate_subtitle_clip(sub.text).set_position(("center", "bottom")).set_opacity(0.8)
        subtitles.append(subtitle_clip.set_start(start).set_end(end))

    result = CompositeVideoClip([video] + subtitles)
    result.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

# Flask route to handle file uploads and process videos
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        video_file = request.files['file']
        subtitle_language = request.form.get('subtitle-language', 'en')
        subtitle_size = request.form.get('subtitle-size', 'normal')

        if video_file:
            def generate():
                yield 'Starting process...\n'

                # Step 1: Save uploaded video file
                temp_video_path = os.path.join(tempfile.gettempdir(), video_file.filename)
                video_file.save(temp_video_path)
                yield 'Video uploaded successfully.\n'

                # Step 2: Extract audio from video
                temp_audio_path = temp_video_path.replace(".mp4", ".wav")
                extract_audio_from_video(temp_video_path, temp_audio_path)
                yield 'Extracting audio from video completed.\n'

                # Step 3: Convert audio to mono WAV format
                converted_audio_file = convert_to_wav_mono(temp_audio_path)
                yield 'Converting audio to mono completed.\n'

                # Step 4: Analyze silence using amplitude and Whisper transcription
                silence_intervals = analyze_silence(converted_audio_file)
                yield 'Silence detection completed.\n'

                # Step 5: Generate captions for silent intervals using BLIP
                captions = []
                for start, end in silence_intervals:
                    caption = generate_caption_for_silent_interval(temp_video_path, start, end)
                    captions.append(caption)
                    yield f'Caption generated for silence interval {start} to {end} seconds.\n'

                # Step 6: Generate SRT with both spoken words and visual captions
                transcription = whisper_model.transcribe(converted_audio_file)['segments']
                transcription_timed = [(seg['text'], seg['start'], seg['end']) for seg in transcription]
                output_srt = temp_video_path.replace(".mp4", ".srt")
                generate_srt_with_captions(transcription_timed, silence_intervals, captions, output_srt)
                yield 'Generating SRT file completed.\n'

                # Step 7: Overlay subtitles on video (if needed)
                output_video_path = temp_video_path.replace(".mp4", "_with_subtitles.mp4")
                overlay_subtitles(temp_video_path, output_srt, output_video_path, subtitle_size)
                yield 'Overlaying subtitles on video completed.\n'

                # Provide download link to the final video
                static_video_path = os.path.join('static', os.path.basename(output_video_path))
                shutil.move(output_video_path, static_video_path)
                video_url = url_for('static', filename=os.path.basename(static_video_path))
                yield f'Process completed. <a href="{video_url}">Download the video</a>.\n'

            return Response(stream_with_context(generate()), content_type='text/plain')

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True, port=5006)
