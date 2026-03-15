# Dependency Parser Visualization

A React + FastAPI NLP project that parses sentences and visualizes dependency relationships.

## Features

- Parse sentences with **spaCy** (true dependency parse)
- Parse sentences with **NLTK** (lightweight heuristic dependency approximation)
- Interactive dependency arc visualization in React
- Token detail table with lemma, POS, head, and dependency label
- ReactBits-inspired visual style with animated hero and polished controls

## Project Structure

- `frontend/` - React (Vite + TypeScript) UI
- `backend/` - FastAPI parser API

## Prerequisites

- Node.js 18+
- Python 3.10+

## Backend Setup

```bash
cd backend
python -m venv ../.venv
../.venv/Scripts/python -m pip install -r requirements.txt
../.venv/Scripts/python -m spacy download en_core_web_sm
../.venv/Scripts/python -m uvicorn app.main:app --reload
```

Backend runs on `http://127.0.0.1:8000`.

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://127.0.0.1:5173`.

## API

### `POST /parse`

Request body:

```json
{
  "text": "The quick brown fox jumps over the lazy dog.",
  "parser": "spacy"
}
```

`parser` can be `spacy` or `nltk`.

### `GET /health`

Returns service health status.

## Notes

- `spaCy` mode returns real dependency labels from model output.
- `NLTK` mode is a heuristic approximation intended for educational comparison.
