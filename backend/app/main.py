from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .parsers import parse_with_nltk, parse_with_spacy, preload_models
from .schemas import ParseRequest, ParseResponse


app = FastAPI(title="Dependency Parser API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    preload_models()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/parse", response_model=ParseResponse)
def parse(req: ParseRequest) -> ParseResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if req.parser == "spacy":
        tokens, root_id = parse_with_spacy(text)
        return ParseResponse(parser="spacy", tokens=tokens, root_id=root_id)

    tokens, root_id = parse_with_nltk(text)
    return ParseResponse(
        parser="nltk",
        tokens=tokens,
        root_id=root_id,
        note="NLTK mode uses a lightweight heuristic dependency approximation.",
    )
