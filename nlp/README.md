# DepViz: Dependency Parser Visualizer

DepViz is a full-stack NLP app that parses English sentences and visualizes dependency relationships as interactive arcs.

You can run parsing with either:
- spaCy (true dependency parse)
- NLTK (lightweight heuristic approximation)

The app is designed for quick parser comparison, token-level inspection, and clear visual exploration of sentence structure.

## Features

- Parse any sentence with spaCy or NLTK
- Compare spaCy and NLTK outputs side-by-side
- Interactive dependency arc graph with hover details
- Token table with id, lemma, POS, head, and dependency label
- Example sentence chips for fast demos
- Backend model preloading on startup for smoother first parse

## Tech Stack

- Backend: FastAPI, Pydantic, spaCy, NLTK, Uvicorn
- Frontend: React, TypeScript, Vite

## Project Structure

```
nlp/
	backend/
		app/
			main.py        # FastAPI app, routes, startup preload
			parsers.py     # spaCy + NLTK parser adapters
			schemas.py     # request/response models
		requirements.txt
	frontend/
		src/
			App.tsx
			components/
				DependencyArcView.tsx
			hooks/
				useParser.ts
```

## API Contract

### Health Check

- Method: `GET`
- Path: `/health`
- Response:

```json
{ "status": "ok" }
```

### Parse Sentence

- Method: `POST`
- Path: `/parse`
- Request body:

```json
{
	"text": "The quick brown fox jumps over the lazy dog.",
	"parser": "spacy"
}
```

- `parser` can be `spacy` or `nltk`.
- For NLTK mode, the API includes a `note` about heuristic parsing.

- Response shape:

```json
{
	"parser": "spacy",
	"tokens": [
		{
			"id": 0,
			"text": "The",
			"lemma": "the",
			"pos": "DET",
			"head_id": 3,
			"dep_label": "det"
		}
	],
	"root_id": 4,
	"note": null
}
```

## Local Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm

### 1) Start Backend

From the `nlp` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\backend\requirements.txt
python -m uvicorn app.main:app --reload --app-dir .\backend --host 127.0.0.1 --port 8000
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2) Start Frontend

In a new terminal:

```powershell
cd .\frontend
npm install
npm run dev
```

### 3) Open the App

- Frontend: http://127.0.0.1:5173
- Backend health: http://127.0.0.1:8000/health

## Notes

- First startup may take longer because spaCy model and NLTK resources are auto-downloaded if missing.
- NLTK mode intentionally uses heuristic dependencies and is not linguistically equivalent to spaCy's parser.
- For quick command-only startup instructions, see `run.md`.

## Troubleshooting

- Empty input returns HTTP 400 from `/parse`.
- If backend is running but UI cannot parse, verify frontend points to the backend host/port used by Uvicorn.
- If Python package installs fail, upgrade pip first: `python -m pip install --upgrade pip`.

## Roadmap Ideas

- Add export of parse output as JSON/PNG
- Add sentence history and saved examples
- Add model/language selection
- Add parser accuracy comparison metrics
