import os
import json
import re
import glob
from yt_dlp import YoutubeDL
from moviepy.editor import AudioFileClip  # ✅ FIXED: Use AudioFileClip
from faster_whisper import WhisperModel

def download_and_clean_transcript(
    url,
    preferred_lang='en',
    output_dir='yt_video',
    status_callback=None,
    progress_callback=None
):
    def set_status(message, pct=None):
        print(f"[STATUS] {message}")
        if status_callback:
            status_callback(message)
        if pct is not None and progress_callback:
            progress_callback(pct)

    os.makedirs(output_dir, exist_ok=True)

    # === Step 1: Try to download subtitles ===
    set_status("Fetching video info...", 0.02)
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en-GB'],
        'subtitlesformat': 'json3',
        'quiet': True,
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept-Language': 'en-US,en;q=0.9'
        }
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info.get('id')
        title = info.get('title')
        uploader = info.get('uploader')
        set_status(f"▶ Video: {title} ({uploader})", 0.05)

        try:
            ydl.download([url])
        except Exception as e:
            print("[ERROR] Subtitle download failed:", e)

    # === Check if any English subtitles downloaded ===
    set_status("Checking for subtitles...", 0.10)
    subtitle_files = glob.glob(f'{output_dir}/{video_id}.en*.json3')
    subtitle_path = subtitle_files[0] if subtitle_files else None

    if subtitle_path and os.path.exists(subtitle_path):
        with open(subtitle_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        transcript = []
        for event in data.get("events", []):
            if "segs" in event and "tStartMs" in event:
                text = ''.join(seg["utf8"] for seg in event["segs"]).strip()
                if text:
                    start_time = round(event["tStartMs"] / 1000, 2)
                    cleaned = re.sub(r'\s+', ' ', text.replace("\n", " "))
                    transcript.append({"start": start_time, "text": cleaned})

        cleaned_path = f"{output_dir}/{video_id}_cleaned.json"
        with open(cleaned_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

        set_status(f"✅ Auto/Manual transcript saved: {cleaned_path}", 1.0)
        return transcript, video_id, f"Auto/Manual transcript saved: {cleaned_path}"

    # === No subtitles: fallback to Whisper ===
    set_status("⚠️ No subtitles found. Transcribing audio with Whisper...", 0.15)

    audio_path = os.path.join(output_dir, f"{video_id}_audio.wav")

    # === Download only audio ===
    ydl_opts_audio = {
    'format': 'bestaudio[ext=m4a]/bestaudio/best',
    'outtmpl': os.path.join(output_dir, f"{video_id}.m4a"),
    'quiet': True,
    'nocheckcertificate': True,
    'geo_bypass': True,
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'http_headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': '*/*',
        'Connection': 'keep-alive'
    }
}

    info = ydl.extract_info(url, download=False)
    for f in info['formats']:
        print(f"{f['format_id']:>10} | {f.get('ext', ''):>4} | {f.get('acodec', ''):>6} | {f.get('vcodec', ''):>6}")

    with YoutubeDL(ydl_opts_audio) as ydl_audio:
        ydl_audio.download([url])
    audio_input_path = os.path.join(output_dir, f"{video_id}.m4a")
    set_status("Downloaded audio.", 0.25)

    # === Convert to WAV using AudioFileClip ===
    set_status("Converting audio...", 0.30)
    audio_clip = AudioFileClip(audio_input_path)  # ✅ FIXED
    audio_clip.write_audiofile(audio_path, fps=16000, logger=None)
    audio_duration = audio_clip.duration  # ✅ Use this since no video duration available
    audio_clip.close()

    # === Load Whisper model ===
    set_status("Loading Whisper model...", 0.35)
    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    # === Transcribe ===
    set_status("Transcribing with Whisper... please wait", 0.40)
    segments_gen, _ = model.transcribe(
        audio_path,
        beam_size=1,
        condition_on_previous_text=False,
        word_timestamps=False
    )

    transcript = []
    last_pct = -1

    for segment in segments_gen:
        start = round(segment.start, 2)
        text = re.sub(r'\s+', ' ', segment.text.strip())
        if text:
            transcript.append({"start": start, "text": text})

        progress = min(segment.end / audio_duration, 1.0)
        pct = int(progress * 100)
        if pct > last_pct:
            msg = f"Transcribing completed {pct}%"
            print(f"[Transcribe] {msg} (t={segment.end:.2f}s)")
            if status_callback:
                status_callback(msg)
            if progress_callback:
                progress_callback(progress)
            last_pct = pct

    cleaned_path = f"{output_dir}/{video_id}_transcribed.json"
    with open(cleaned_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    set_status(f"✅ Transcription complete and saved: {cleaned_path}", 1.0)
    return transcript, video_id, f"Transcription complete and saved: {cleaned_path}"