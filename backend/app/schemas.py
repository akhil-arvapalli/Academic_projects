from typing import List, Literal, Optional

from pydantic import BaseModel, Field


ParserName = Literal["spacy", "nltk"]


class ParseRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    parser: ParserName = "spacy"


class TokenOut(BaseModel):
    id: int
    text: str
    lemma: str
    pos: str
    head_id: int
    dep_label: str


class ParseResponse(BaseModel):
    parser: ParserName
    tokens: List[TokenOut]
    root_id: int
    note: Optional[str] = None
