# Self-Healing RAG Frontend Guide

## Quick Start

### Step 1: Ensure API is Running
```bash
cd /Users/yashbhatia/Desktop/self_healing_rag
source .venv/bin/activate
python -m backend.main
```
The API will start on `http://localhost:8000`

### Step 2: Open Frontend

**Option A: Direct file (no server needed)**
```bash
open frontend/index.html
```

**Option B: Python HTTP Server (recommended)**
```bash
python -m http.server 3000
# Then visit http://localhost:3000/frontend/index.html
```

**Option C: Live Server (VS Code)**
- Install "Live Server" extension
- Right-click `frontend/index.html` → "Open with Live Server"

## Features

### 🎯 Query Interface
- Type a question in the textarea
- Click "Ask" or press Ctrl+Enter (or Cmd+Enter on Mac)
- Watch the self-healing RAG in action!

### 📊 Results Display
- **Answer**: Large, readable answer with gradient background
- **Retries**: Badge showing how many times the answer was refined
  - Green (0) = No retries needed ✓
  - Yellow (1) = One retry 
  - Red (2+) = Multiple retries
- **Groundedness**: Shows if answer is based on source documents
- **Self-healed Banner**: Appears when answer was automatically improved
- **Sources**: Expandable list of retrieved documents used

### ➕ Document Ingestion
1. Click "➕ Add Documents" at the bottom
2. Paste your documents (separate with blank lines for chunks)
3. Click "Ingest"
4. See success toast and document count updates

### 🔗 Status Pill
- Shows connection status (green=online, red=offline)
- Displays number of documents in the system
- Automatically checks on page load

## API Configuration

To change the API endpoint, edit line 404 in `frontend/index.html`:
```javascript
const API_BASE = "http://localhost:8000";  // Change this URL
```

## Example Queries

Try these questions with the sample documents:
- "How does the Python GIL work?"
- "What is gradient descent?"
- "How does DNS work?"
- "What is Docker?"
- "How does HTTPS handshake work?"

## Troubleshooting

### "API is offline" message?
- Make sure `python -m backend.main` is running
- Check that it's running on http://localhost:8000
- Verify no firewall is blocking port 8000

### Documents not showing up?
- Run the ingest script: `python -m backend.ingest`
- Or use the frontend's "Add Documents" button
- Check the status pill for document count

### Query returns empty answer?
- Add more documents using the ingest feature
- Try a simpler question
- Check the critique text for clues

## Architecture

```
Frontend (HTML/CSS/JS)
        ↓ async fetch
API Gateway (FastAPI)
        ↓
LangGraph Pipeline
        ├→ Retriever (ChromaDB)
        ├→ Generator (LLM: Groq/OpenAI)
        └→ Critic (LLM: Groq/OpenAI)
                ↓
        Return refined answer
```

## Performance Notes

- First query takes 2-5 seconds (LLM latency)
- Subsequent queries may retry (0-2s each)
- Max 3 retries before graceful fallback
- Retrievals are very fast (<100ms)

## Browser Compatibility

- ✅ Chrome/Edge 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Mobile browsers (responsive design)

## Keyboard Shortcuts

- **Ctrl+Enter** (Windows/Linux): Submit query
- **Cmd+Enter** (Mac): Submit query
- Both trigger the "Ask" button programmatically

## Dark Theme

The frontend uses a dark theme inspired by ChatGPT/Perplexity. All colors are defined as CSS variables in the `<style>` section - easy to customize!

## Next Steps

1. ✅ Add your Groq/OpenAI API keys to `.env`
2. ✅ Test with sample documents
3. ✅ Add your own documents
4. ✅ Deploy to production (Docker or cloud)
5. ✅ Monitor and refine prompts

Enjoy your Self-Healing RAG! 🧠✨
