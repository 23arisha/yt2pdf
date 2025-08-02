# yt2pdf_app.py

import os
import re
import json
import uuid
import shutil
import streamlit as st
from fpdf import FPDF
from pytubefix import YouTube
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from download_clean import download_and_clean_transcript
from sentence_transformers import SentenceTransformer, util
import cv2
from yt_dlp import YoutubeDL
from io import BytesIO
import torch
from more_itertools import chunked
from prompts import (
    LINE_BY_LINE_PROMPT,
    STRUCTURED_SUMMARY_PROMPT,
    CODE_FOCUSED_PROMPT,
    INSTRUCTOR_NOTES_PROMPT,
)

# === Cleaning Utilities ===
def clean_text(text):
    text = re.sub(
        r"(Here is a clear, step[\- ]?by[\- ]?step educational tutorial.*?:|"
        r"This tutorial will.*?:|"
        r"In this tutorial.*?:|"
        r"Let me guide you.*?:|"
        r"Step[\- ]?by[\- ]?step tutorial.*?:)",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r'[-*•–—#]', '', text)
    return text.strip()


# === PDF Creation Functions ===
def txt_to_pdf(input_txt_path, output_pdf_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    with open(input_txt_path, "r", encoding="utf-8") as file:
        for line in file:
            line = clean_text(line.strip())
            if line:
                pdf.multi_cell(0, 10, line)
    pdf.output(output_pdf_path)


def save_visual_pdf(tutorial_data, output_path, update_progress=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    total = len(tutorial_data)
    for i, section in enumerate(tutorial_data):
        text = section["tutorial_paragraph"].strip()
        pdf.multi_cell(0, 10, text)
        pdf.ln(3)

        img_path = section.get("screenshot")
        if img_path and os.path.exists(img_path):
            try:
                pdf.image(img_path, w=80, h=60)
                pdf.ln(5)
            except Exception as e:
                print(f"⚠️ Failed to load image {img_path}: {e}")
        pdf.ln(5)

        if update_progress:
            update_progress((i + 1) / total)

    pdf.output(output_path)


def download_and_cleanup(file_path, label, download_name):
    with open(file_path, "rb") as f:
        data = BytesIO(f.read())
    if st.download_button(f"⬇️ Download {label}", data, file_name=download_name):
        shutil.rmtree(session_dir, ignore_errors=True)
        st.success("✅ Download complete. Temporary files cleaned up.")


# === Tutorial Generation ===
def generate_chunk_tutorial(chunk, groq_key):
    llm = ChatGroq(groq_api_key=groq_key, model_name="llama3-70b-8192")
    if tutorial_style == "Line-by-Line":
        full_prompt = LINE_BY_LINE_PROMPT
    elif tutorial_style == "Structured Summary":
        full_prompt = STRUCTURED_SUMMARY_PROMPT
    elif tutorial_style == "Code-Focused":
        full_prompt = CODE_FOCUSED_PROMPT
    elif tutorial_style == "Instructor Notes":
        full_prompt = INSTRUCTOR_NOTES_PROMPT
    else:
        full_prompt = LINE_BY_LINE_PROMPT

    full_prompt = full_prompt.strip() + "\n\n" + chunk.strip()
    result = llm.invoke(full_prompt)
    cleaned = clean_text(result.content.strip())

    lines = []
    for line in cleaned.splitlines():
        if re.match(r"^(Tutorial|Step \d+):", line.strip(), re.IGNORECASE):
            lines.append("## " + line.strip())
        else:
            lines.append(line)
    return "\n".join(lines).strip()


# === Streamlit UI ===
st.set_page_config(page_title="YouTube to PDF", layout="centered")
st.title("📹 YouTube → 📘 Smart Tutorial Generator")

# === Session Setup ===
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id
session_dir = os.path.join("yt_video", session_id)
os.makedirs(session_dir, exist_ok=True)

# === Paths ===
tutorial_txt_path = os.path.join(session_dir, "tutorial.txt")
transcript_path = os.path.join(session_dir, "transcript_cleaned.json")
mapping_path = os.path.join(session_dir, "matched_mapping.json")
screenshots_path = os.path.join(session_dir, "with_screenshots.json")
video_path = os.path.join(session_dir, "high_quality.mp4")
screenshot_dir = os.path.join(session_dir, "screenshots")

# === Inputs ===
url = st.text_input("📎 Paste YouTube URL")
groq_key = st.secrets.get("GROQ_API_KEY")

# === Style Selection ===
tutorial_style = st.selectbox(
    "🧠 Choose Tutorial Style",
    ["Line-by-Line", "Structured Summary", "Code-Focused", "Instructor Notes"]
)

# === Generate Tutorial Button ===
if st.button("Generate Tutorial"):
    if not url or not groq_key:
        st.error("❌ Missing YouTube URL or API Key")
        st.stop()

    status = st.empty()
    progress = st.progress(0)

    transcript, video_id, msg = download_and_clean_transcript(
        url, output_dir=session_dir,
        status_callback=lambda s: status.info(s),
        progress_callback=lambda p: progress.progress(p)
    )
    status.success(msg)
    progress.empty()

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2)

    text = " ".join([line["text"] for line in transcript])
    chunks = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300).split_text(text)

    st.info("✍️ Generating tutorial...")
    tutorial_sections = []
    bar = st.progress(0)

    for i, chunk in enumerate(chunks):
        section = generate_chunk_tutorial(chunk, groq_key)
        tutorial_sections.append(section)
        bar.progress((i + 1) / len(chunks))

    full_tutorial = "\n\n".join(tutorial_sections)
    with open(tutorial_txt_path, "w", encoding="utf-8") as f:
        f.write(full_tutorial)

    st.session_state["transcript"] = transcript
    st.session_state["full_tutorial_text"] = full_tutorial
    bar.empty()
    st.success("✅ Tutorial ready. Choose export option below.")

from more_itertools import chunked

# === PDF Export Options ===
if "full_tutorial_text" in st.session_state:
    st.subheader("📥 Export Tutorial to PDF")
    st.text_area("📖 Preview", st.session_state["full_tutorial_text"][:2000], height=300)

    choice = st.radio(
        "Include Screenshots?",
        options=["Select", "No", "Yes (Visual)"],
        index=0
    )

    if choice == "No":
        text_pdf_path = os.path.join(session_dir, "text_only.pdf")

        # Check if tutorial text exists before generating PDF
        if os.path.exists(tutorial_txt_path):
            txt_to_pdf(tutorial_txt_path, text_pdf_path)

            # Check if the PDF was successfully created
            if os.path.exists(text_pdf_path):
                download_and_cleanup(text_pdf_path, "Text Tutorial", "text_only_tutorial.pdf")
            else:
                st.write("⚠️ PDF could not be generated. Please try again.")
        else:
            st.write("⚠️ Tutorial text not found. Please run the tutorial generation step first.")


    elif choice == "Yes (Visual)":
        if os.path.exists(screenshots_path):
            st.info("✅ Using cached screenshots.")
            with open(screenshots_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        else:
            # === Download Video ===
            st.warning("⬇️ Downloading HD video for screenshots...")
            bar = st.progress(0)

            def progress_hook(d):
                if d['status'] == 'downloading':
                    pct = d.get('_percent_str', '0').replace('%', '')
                    try:
                        bar.progress(min(int(float(pct)) / 100, 1.0))
                    except:
                        pass

            ydl_opts = {
                'format': 'bestvideo[vcodec^=avc1][height<=1080]+bestaudio/best',
                'outtmpl': os.path.join(session_dir, 'high_quality.%(ext)s'),
                'merge_output_format': 'mp4',
                'progress_hooks': [progress_hook],
                'quiet': True,
            }

            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            bar.empty()
            st.success("🎥 Video downloaded.")

        # === Semantic Mapping ===
        st.info("🔗 Mapping tutorial to transcript")

        @st.cache_resource
        def load_model():
            return SentenceTransformer("all-MiniLM-L6-v2")

        model = load_model()
        transcript = st.session_state["transcript"]
        window_size = 5

        # Step 1: Precompute sliding windows
        transcript_windows = []
        window_texts = []
        for start_idx in range(0, len(transcript) - window_size + 1):
            window = transcript[start_idx: start_idx + window_size]
            window_text = " ".join([w["text"] for w in window])
            transcript_windows.append(window)
            window_texts.append(window_text)

        # Step 2: Encode all windows once (batch)
        window_embeddings = model.encode(window_texts, convert_to_tensor=True)
        if len(window_embeddings) == 0:
            st.error("❌ Not enough transcript to perform semantic mapping.")
            st.stop()


        # Step 3: Load and clean tutorial paragraphs
        with open(tutorial_txt_path, "r", encoding="utf-8") as f:
            paras = [clean_text(p.strip()) for p in f.read().split("\n\n") if p.strip()]

        # Step 4: Map paragraphs to transcript in batches
        mapping = []
        batch_size = 15
        total = len(paras)
        done = 0
        progress_bar = st.progress(0)
        batch_label = st.empty()
        para_batches = list(chunked(paras, batch_size))

        for batch_index, batch in enumerate(para_batches):
            batch_label.text(f"Processing tutorial of batch {batch_index + 1}/{len(para_batches)}")
            valid_paras = [(i + done, p) for i, p in enumerate(batch) if p and len(p) >= 50]
            if not valid_paras:
                done += len(batch)
                continue

            batch_texts = [p for _, p in valid_paras]
            para_embs = model.encode(batch_texts, convert_to_tensor=True)

            for original_i, para_emb in zip([i for i, _ in valid_paras], para_embs):
                sims = util.cos_sim(para_emb, window_embeddings)[0]
                top_idx = torch.argmax(sims).item()
                best_score = sims[top_idx].item()
                best_window = transcript_windows[top_idx]

                mapping.append({
                    "paragraph_index": original_i,
                    "tutorial_paragraph": paras[original_i],
                    "matched_transcript": [
                        {
                            "index": top_idx + j,
                            "start": best_window[j]["start"],
                            "text": best_window[j]["text"],
                            "score": best_score,
                        }
                        for j in range(window_size)
                    ]
                })

                print(f"\n=== Paragraph {original_i} ===")
                print("Matched to transcript:")
                for j, t in enumerate(best_window):
                    print(f"[{top_idx + j} | {t['start']:.2f}s] -> {t['text']}")

                progress_bar.progress(min((original_i + 1) / total, 1.0))


            done += len(batch)

        progress_bar.empty()

        # === Screenshot Capture ===
        st.info("Capturing relevant from video...")
        # === Screenshot Capture ===
        os.makedirs(screenshot_dir, exist_ok=True)
        # Validate video file
        if not os.path.exists(video_path):
            st.error(f"❌ Video not found at {video_path}")
            st.stop()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Failed to open the video file.")
            st.stop()
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps else 0
        print(f"🎥 Video duration: {duration:.2f}s | Total frames: {int(frame_count)} | FPS: {fps}")

        bar = st.progress(0)
        for i, item in enumerate(mapping):
            matched = item["matched_transcript"]
            times = [m["start"] for m in matched]
            scores = [m["score"] for m in matched]

            if len(set(scores)) == 1 and len(times) >= 2:
                midpoint_time = round(sum(times[-2:]) / 2, 2)
            else:
                top_two = sorted(matched, key=lambda x: x["score"], reverse=True)[:2]
                midpoint_time = round(sum([t["start"] for t in top_two]) / 2, 2) if top_two else 0.0

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(midpoint_time * fps))
            success, frame = cap.read()
            img_path = os.path.join(screenshot_dir, f"img_{i}.jpg")
            if success and frame is not None:
                cv2.imwrite(img_path, frame)
                item["screenshot"] = img_path
                print(f"[{i}] ✅ Captured frame at {midpoint_time}s -> Saved to {img_path}")
            else:
                print(f"[{i}] ❌ Failed to capture frame at {midpoint_time}s")

            bar.progress((i + 1) / len(mapping))

        cap.release()
        bar.empty()

        with open(screenshots_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)

        # === Visual PDF ===
        st.info("📄 Generating PDF with screenshots...")
        visual_pdf_path = os.path.join(session_dir, "visual_tutorial.pdf")
        pdf_bar = st.progress(0)
        save_visual_pdf(mapping, visual_pdf_path, update_progress=pdf_bar.progress)
        pdf_bar.empty()
        if os.path.exists(visual_pdf_path):
            download_and_cleanup(visual_pdf_path, "Visual Tutorial", "visual_tutorial.pdf")
        else:
            st.write("⚠️ Visual PDF could not be created. Please try again.")
