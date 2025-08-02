# 📹 YouTube → 📘 Smart Tutorial Generator

Turn any YouTube video into a structured, AI-generated PDF tutorial — optionally enhanced with **screenshots**, **semantic mapping**, and **code explanations**.

---

## Overview

This app allows educators, students, and developers to **convert YouTube videos into clean, readable tutorials** in PDF format. It uses **LLMs (Groq + LLaMA 3)** to understand the video content and generate explanations, notes, and even code-focused insights. Optionally, it extracts **relevant frames** from the video to create a **visual tutorial**.

---

## Tech Stack

| Component        | Technology |
|------------------|------------|
| UI / Frontend    | Streamlit  |
| Transcription    | yt-dlp, faster-whisper |
| Summarization    | LangChain + Groq (LLaMA 3) |
| Embeddings       | sentence-transformers (MiniLM) |
| Screenshots      | OpenCV     |
| PDF Export       | FPDF       |

---

## Supported Tutorial Styles

| Style              | Description |
|--------------------|-------------|
| **Line-by-Line**   | Detailed breakdown of each line in the transcript. |
| **Structured Summary** | Sections with clean summaries and key concepts. |
| **Code-Focused**   | Emphasis on code, logic, and technical explanations. |
| **Instructor Notes** | Notes-style writeups with tips and clarifications. |

---

## Features

- 🔍 Automatic or fallback Whisper transcription.
- 🧠 AI-generated summaries using LLaMA 3 (via Groq).
- 🪄 Four tutorial generation styles.
- 🖼️ Screenshot extraction and paragraph-to-video semantic mapping.
- 📄 PDF export options (text-only or visual).
- 📥 Download-ready output with cleaned-up session.

---

## 🔄 
Follow this step-by-step process:

- **Step 1**: Paste a YouTube URL into the app.

- **Step 2**: The app tries to download subtitles:
  - If unavailable, it falls back to audio transcription using **faster-whisper**.

- **Step 3**: The transcript is chunked and sent to **Groq’s LLaMA 3 model** via **LangChain**.

- **Step 4**: AI generates structured tutorial sections based on the selected style.

- **Step 5**: If screenshots are enabled:
  - The full video is downloaded in HD.
  - Tutorial text is mapped semantically to transcript windows.
  - Best-matching timestamps are identified and relevant frames extracted using **OpenCV**.

- **Step 6**: Generate a PDF:
  - ✅ **Text-only** or
  - ✅ **Visual (includes screenshots)**.

- **Step 7**: PDF is offered for download.

- **Step 8**: Temporary session files are deleted after export.
