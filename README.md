# YouTube → Transcript & Summary 🚀

This Streamlit app fetches YouTube video transcripts (captions), reconstructs them into a **clean, non-repeating, time-ordered transcript**, and optionally provides a **summary** using a Transformer-based summarization model.

---

## 🚀 Features

- Fetches transcripts directly using the **YouTube Transcript API**
- Automatically removes **duplicate or overlapping text**
- Keeps captions in proper **time order**
- Falls back to `yt-dlp` if API fetching fails
- Generates **summaries** using Hugging Face Transformers (`distilbart-cnn-6-6`)
- One-click transcript **download** in `.txt` format
- Works directly in your browser via **Streamlit UI**

---

## 🧠 Tech Stack

- **Python 3.9+**
- **Streamlit** for UI
- **YouTubeTranscriptApi** for captions
- **Transformers + Torch** for summarization
- **yt-dlp** as fallback transcript retriever

---

## ⚙️ Installation

```bash
# 1. Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, create one with the following:

```
streamlit
youtube-transcript-api
transformers
torch
yt-dlp
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 🧩 Usage

1. Paste a valid YouTube video URL  
2. Optionally specify a **caption language** (default: `"en"`)  
3. Click **“Fetch Transcript & Summary”**  
4. View or download the full transcript  
5. Read the automatically generated summary (if available)

---

## 🧱 Project Structure

```
📂 youtube-transcript-summary/
│
├── video.py            # Main Streamlit application
├── README.md           # This file
└── requirements.txt    # Python dependencies
```

---

## ⚠️ Notes

- Works only for **public videos** with available captions
- Auto-generated captions may contain recognition errors
- Summarization can take time on large transcripts (model runs locally)

---

## 🧠 Credits

- [Streamlit](https://streamlit.io)
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
