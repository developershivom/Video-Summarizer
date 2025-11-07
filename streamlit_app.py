#!/usr/bin/env python3
"""
Streamlit app: Fetch YouTube transcript, reconstruct a correct, non-repeating,
time-ordered transcript, AND provide a summary.
"""
from __future__ import annotations
import re
import os
import glob
import tempfile
from typing import List, Optional, Tuple, Dict, Any
import difflib
import streamlit as st
import textwrap

# --- Import transformers for summarization ---
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pipeline = None
    TRANSFORMERS_AVAILABLE = False
    st.warning("`transformers` or `torch` not found. Summarization will be disabled. Install with: `pip install transformers torch`")

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from xml.etree.ElementTree import ParseError

# Optional - only used as final fallback to get VTT files
try:
    import yt_dlp  # type: ignore
    YTDLP_AVAILABLE = True
except Exception:
    yt_dlp = None
    YTDLP_AVAILABLE = False

st.set_page_config(page_title="YouTube → Transcript & Summary", layout="centered")
st.title("YouTube → Transcript & Summary 🚀")

# -----------------------
# --- Summarization Helpers ---
# -----------------------

@st.cache_resource
def load_summarizer():
    """
    Loads the summarization pipeline from transformers.
    This is cached to prevent reloading on every run.
    """
    if not TRANSFORMERS_AVAILABLE:
        st.error("Summarization failed: `transformers` library not found. Please install it: `pip install transformers torch`")
        return None
    
    st.info("Loading summarization model (first time only)...")
    try:
        # Using a default, smaller model for quicker loading
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
        st.success("Summarization model loaded!")
        return summarizer
    except Exception as e:
        st.error(f"Error loading summarization model: {e}")
        return None

def summarize_text(text_to_summarize: str, summarizer) -> str:
    """
    Summarizes the text by splitting it into chunks that fit the model's limit.
    """
    if not summarizer:
        return "Summarization is unavailable. (Check installation)"
        
    # Split text into words
    words = text_to_summarize.split()
    
    # Model's max token limit is 1024.
    # We'll use 500 words as a safe chunk size (~700-800 tokens)
    chunk_size = 500
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    # Summarize each chunk
    summaries = []
    
    # Only show this if chunking is necessary
    if len(chunks) > 1:
        st.info(f"Transcript is long. Summarizing in {len(chunks)} chunk(s)...")
    
    try:
        for chunk in chunks:
            # We set min_length and max_length for each chunk's summary
            summary = summarizer(chunk, max_length=120, min_length=25, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        
        # Combine the summaries of all chunks
        return " ".join(summaries)
    except Exception as e:
        st.warning(f"Summarization failed during processing: {e}")
        return "Error during summary generation."

# -----------------------
# Helpers: fetch transcript segments
# -----------------------
def extract_video_id(url: str) -> Optional[str]:
    if not url or not isinstance(url, str):
        return None
    patterns = [
        r"v=([0-9A-Za-z_-]{11})",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
        r"embed/([0-9A-Za-z_-]{11})",
        r"\/([0-9A-Za-z_-]{11})$",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

def fetch_segments_via_api(video_id: str, lang: str = "en") -> Optional[Tuple[List[Dict[str, Any]], str]]:
    """
    Try get_transcript -> returns list of dicts with keys 'text','start','duration'.
    If fails, try list_transcripts() with manual/generate/find; returns same shape.
    """
    try:
        items = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
        if items:
            return items, "get_transcript()"
    except TranscriptsDisabled:
        raise Exception("Transcripts are disabled for this video.")
    except VideoUnavailable:
        raise Exception("This video is unavailable or private.")
    except NoTranscriptFound:
        pass
    except ParseError:
        pass
    except Exception:
        pass

    # Try list_transcripts fallback
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            t = transcripts.find_transcript([lang])
            items = t.fetch()
            if items:
                return items, "list_transcripts().find_transcript()"
        except NoTranscriptFound:
            try:
                tg = transcripts.find_generated_transcript([lang])
                items = tg.fetch()
                if items:
                    return items, "list_transcripts().find_generated_transcript()"
            except NoTranscriptFound:
                for candidate in transcripts:
                    try:
                        items = candidate.fetch()
                        if items:
                            return items, f"list_transcripts() fallback ({candidate.language_code})"
                    except Exception:
                        continue
    except ParseError:
        pass
    except Exception:
        pass

    return None

# -----------------------
# Helpers: parse VTT (yt-dlp fallback)
# -----------------------
def parse_vtt_to_segments(vtt_text: str) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    v = vtt_text.replace('\r\n', '\n').replace('\r', '\n')
    blocks = re.split(r'\n\s*\n', v)
    time_re = re.compile(r'(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?\s*-->\s*(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?')
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        ts_line_idx = None
        for i, ln in enumerate(lines):
            if '-->' in ln:
                ts_line_idx = i
                break
        if ts_line_idx is None:
            continue
        ts_line = lines[ts_line_idx]
        m = time_re.search(ts_line)
        if not m:
            continue
        def to_secs(h, m_, s, ms):
            h = int(h); m_ = int(m_); s = int(s); ms = int(ms) if ms else 0
            return h*3600 + m_*60 + s + ms/1000.0
        start = to_secs(m.group(1), m.group(2), m.group(3), m.group(4))
        end = to_secs(m.group(5), m.group(6), m.group(7), m.group(8))
        text_lines = lines[ts_line_idx+1:]
        text = " ".join([re.sub(r'<[^>]+>', ' ', ln).strip() for ln in text_lines if ln.strip()])
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            segments.append({'start': start, 'end': end, 'text': text})
    return segments

def download_vtt_with_yt_dlp(url: str, lang: str = "en") -> Optional[Tuple[List[Dict[str, Any]], str]]:
    if not YTDLP_AVAILABLE:
        return None
    temp_dir = tempfile.mkdtemp(prefix="ytdlp_subs_")
    opts = {
        "skip_download": True,
        "writeautomaticsub": True,
        "writesubtitles": False,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
        "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        vtt_files = glob.glob(os.path.join(temp_dir, "*.vtt"))
        if not vtt_files:
            return None
        chosen = None
        for f in vtt_files:
            if re.search(r"[._-]en(\.|$)", os.path.basename(f), re.IGNORECASE):
                chosen = f
                break
        if not chosen:
            chosen = vtt_files[0]
        with open(chosen, 'r', encoding='utf-8', errors='ignore') as fh:
            content = fh.read()
        segments = parse_vtt_to_segments(content)
        api_like = []
        for seg in segments:
            api_like.append({'text': seg['text'], 'start': seg['start'], 'duration': max(0.0, seg['end'] - seg['start'])})
        return api_like, f"yt-dlp({os.path.basename(chosen)})"
    except Exception:
        return None

# -----------------------
# Helpers: deduplicate & assemble
# -----------------------
def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r'(?i)^kind:\s*captions\s*language:\s*\w+\s*', '', s).strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def are_near_duplicates(a: str, b: str, threshold: float = 0.86) -> bool:
    if not a or not b:
        return False
    a_n = re.sub(r'\s+', ' ', a.strip().lower())
    b_n = re.sub(r'\s+', ' ', b.strip().lower())
    if a_n in b_n or b_n in a_n:
        return True
    ratio = difflib.SequenceMatcher(None, a_n, b_n).ratio()
    return ratio >= threshold

def assemble_from_segments(segments: List[Dict[str, Any]]) -> str:
    if not segments:
        return ""
    for i, seg in enumerate(segments):
        seg["text"] = normalize_text(seg.get("text", ""))
        if "start" not in seg or seg["start"] is None:
            seg["start"] = float(i)

    segments_sorted = sorted(segments, key=lambda s: float(s.get("start", 0.0)))

    kept: List[str] = []
    seen_texts: set[str] = set()

    for seg in segments_sorted:
        text = seg.get("text", "").strip()
        if not text:
            continue
        norm = re.sub(r"\s+", " ", text.lower()).strip()
        if any(are_near_duplicates(norm, t, threshold=0.88) for t in seen_texts):
            continue
        if kept and are_near_duplicates(kept[-1], text, threshold=0.88):
            continue
        seen_texts.add(norm)
        kept.append(text)

    merged = []
    for t in kept:
        if not t:
            continue
        if not re.search(r"[.!?]$", t):
            if len(t.split()) > 8:
                t += "."
        merged.append(t)

    final = " ".join(merged)
    final = re.sub(r"\s+([,.:;!?])", r"\1", final)
    final = re.sub(r"\s+", " ", final).strip()
    return final

# -----------------------
# UI + orchestration
# -----------------------

# --- Load the summarizer model once on startup ---
summarizer = load_summarizer()

st.markdown(
    "- Paste a YouTube URL (public video with captions) and the app will attempt to fetch the captions\n"
    "- The app reconstructs text in time order and removes repeated/overlapping fragments\n"
    "- It will also generate a summary of the final, clean transcript"
)

col1, col2 = st.columns([1, 2])
with col1:
    yt_url = st.text_input("YouTube URL")
with col2:
    lang = st.text_input("Preferred caption language (ISO code)", value="en")

if yt_url:
    vid = extract_video_id(yt_url)
    if vid:
        st.image(f"https://img.youtube.com/vi/{vid}/0.jpg", width=320)
    else:
        st.info("Could not extract video id; paste full URL (watch?v=...).")

if st.button("Fetch Transcript & Summary"): 
    if not yt_url or not extract_video_id(yt_url):
        st.error("Please enter a valid YouTube URL.")
    else:
        with st.spinner("Fetching transcript segments..."):
            video_id = extract_video_id(yt_url)
            segments_result = None
            method_desc = ""
            try:
                api_res = fetch_segments_via_api(video_id, lang=lang)
                if api_res:
                    items, method_desc = api_res
                    segments_result = [{'text': it.get('text',''), 'start': float(it.get('start',0.0)), 'duration': float(it.get('duration',0.0))} for it in items]
            except Exception as e:
                st.warning(f"API fetch raised exception: {e}")

            if not segments_result:
                if YTDLP_AVAILABLE:
                    st.info("API failed, trying yt-dlp fallback...")
                    ytdlp_res = download_vtt_with_yt_dlp(yt_url, lang=lang)
                    if ytdlp_res:
                        segments_result, method_desc = ytdlp_res
                else:
                    st.info("yt-dlp not installed; install with `pip install yt-dlp` to enable final fallback.")

            if not segments_result:
                st.error("Could not fetch any transcript segments. The video may have no captions or captions are restricted.")
            else:
                st.success(f"Fetched segments via: {method_desc}")
                
                # Assemble and dedupe in time order
                assembled = assemble_from_segments(segments_result)
                
                st.subheader("Accurate, Deduplicated Transcript")
                st.text_area("Full Transcript", value=assembled, height=300) 

                # Download
                vid_id = extract_video_id(yt_url) or "transcript"
                st.download_button("Download transcript (.txt)", data=assembled, file_name=f"{vid_id}_transcript.txt", mime="text/plain")

                # --- Generate Summary ---
                st.markdown("---")
                if not assembled.strip():
                     st.warning("Transcript is empty, cannot generate summary.")
                else:
                    with st.spinner("Generating summary..."):
                        final_summary = summarize_text(assembled, summarizer)
                        st.subheader("Summary")
                        # Use st.success for the final summary output
                        st.success(final_summary)

st.markdown(
    "Notes:\n"
    "- This tool tries to preserve the exact caption text while fixing ordering and duplicate fragments.\n"
    "- Auto-generated captions can still contain recognition errors — this script does not try to 'correct' content, only to remove repetitions and reorder.\n"
)
