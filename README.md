# 🎯 InterviewIQ — AI Interview Assistant

A full-featured AI-powered interview coaching tool written entirely in Python.
Includes real-time NLP analysis, speech recognition, and optional Claude AI feedback.

---

## ✨ Features

| Feature | Technology |
|---|---|
| Flask web server | Python + Flask |
| NLP Engine | Pure Python (no ML libraries needed) |
| Speech Recognition | Browser Web Speech API (JS) |
| AI Coaching | Anthropic Claude API (optional) |
| Confidence scoring | Lexicon-based analysis |
| STAR method detection | Pattern matching |
| Filler word detection | Tokenization + lexicon |
| Keyword analysis | Category-specific keyword banks |
| Quantifier detection | Regex pattern matching |
| CLI tool | Colorama terminal UI |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install flask requests colorama
```

### 2. Set your Anthropic API key (optional — for AI coaching tips)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Run the web app

```bash
python app.py
```

Open: **http://localhost:5000**

---

## 🖥️ CLI Mode (no browser needed)

### Interactive CLI:
```bash
python cli_analyzer.py
```

### Demo mode (pre-loaded sample answer):
```bash
python cli_analyzer.py --demo
```

---

## 📁 Project Structure

```
interview_assistant/
├── app.py              ← Flask web server + API routes
├── nlp_engine.py       ← Pure Python NLP analysis engine
├── cli_analyzer.py     ← Terminal CLI tool
├── requirements.txt    ← Python dependencies
└── templates/
    └── index.html      ← Full web UI (HTML/CSS/JS)
```

---

## 🔬 NLP Analysis Modules

The `nlp_engine.py` module runs these analyses locally (no API calls):

1. **Confidence Scoring** — Detects positive/negative language, passive voice, first-person active use
2. **Clarity Scoring** — Flesch-Kincaid grade level, sentence length, vocabulary richness
3. **Relevance Scoring** — Category-specific keyword matching (behavioral/technical/situational/common)
4. **Depth Scoring** — STAR method detection, quantifier extraction, power verb usage
5. **Filler Word Detection** — Identifies "um", "like", "basically" etc. with counts
6. **Readability Analysis** — Grade level, type-token ratio, average sentence length

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/api/questions` | All questions by category |
| GET | `/api/sample-answer/<cat>` | Sample answer for category |
| POST | `/api/analyze` | Analyze a response |
| GET | `/api/history/<session_id>` | Session history |
| DELETE | `/api/history/<session_id>` | Clear session history |

### POST /api/analyze

```json
{
  "response":   "Your interview answer text...",
  "question":   "Tell me about yourself.",
  "category":   "common",
  "session_id": "sess_abc123",
  "use_ai":     true
}
```

---

## 🎤 Speech Recognition

- Click the **microphone button** in the web UI
- Speak your answer — text appears in real time
- Click again to stop
- Requires Chrome, Edge, or Safari with microphone permission
- Falls back gracefully if not supported

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | _(empty)_ | Enables Claude AI coaching feedback |
| `PORT` | `5000` | Server port |
| `DEBUG` | `false` | Flask debug mode |
| `FLASK_SECRET_KEY` | auto | Session secret key |
