import re
from functools import lru_cache
from typing import List, Tuple

import nltk
import spacy

from .schemas import TokenOut


def _ensure_nltk_data() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, resource in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


@lru_cache(maxsize=1)
def get_spacy_model() -> spacy.language.Language:
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        from spacy.cli import download

        download(model_name)
        return spacy.load(model_name)


def parse_with_spacy(text: str) -> Tuple[List[TokenOut], int]:
    nlp = get_spacy_model()
    doc = nlp(text)
    tokens: List[TokenOut] = []
    root_id = 0

    for token in doc:
        if token.dep_ == "ROOT":
            root_id = token.i
        tokens.append(
            TokenOut(
                id=token.i,
                text=token.text,
                lemma=token.lemma_,
                pos=token.pos_,
                head_id=token.head.i,
                dep_label=token.dep_,
            )
        )

    if tokens and root_id >= len(tokens):
        root_id = 0
    return tokens, root_id


def _nearest_index(
    pairs: List[Tuple[int, str, str]], current_idx: int, tag_prefixes: Tuple[str, ...], direction: str
) -> int | None:
    if direction == "right":
        iterator = range(current_idx + 1, len(pairs))
    else:
        iterator = range(current_idx - 1, -1, -1)

    for idx in iterator:
        _, _, pos = pairs[idx]
        if pos.startswith(tag_prefixes):
            return idx
    return None


def _guess_root(pairs: List[Tuple[int, str, str]]) -> int:
    for idx, _, pos in pairs:
        if pos.startswith("VB"):
            return idx
    for idx, _, pos in pairs:
        if pos.startswith("NN"):
            return idx
    return 0


def parse_with_nltk(text: str) -> Tuple[List[TokenOut], int]:
    _ensure_nltk_data()

    raw_tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(raw_tokens)
    pairs = [(i, tok, pos) for i, (tok, pos) in enumerate(tagged)]

    if not pairs:
        return [], 0

    root_id = _guess_root(pairs)
    outputs: List[TokenOut] = []

    for idx, tok, pos in pairs:
        if idx == root_id:
            outputs.append(
                TokenOut(
                    id=idx,
                    text=tok,
                    lemma=tok.lower(),
                    pos=pos,
                    head_id=idx,
                    dep_label="ROOT",
                )
            )
            continue

        dep_label = "dep"
        head_id = root_id

        if pos.startswith("DT") or pos.startswith("PRP$"):
            noun_idx = _nearest_index(pairs, idx, ("NN",), "right")
            if noun_idx is not None:
                head_id = noun_idx
            dep_label = "det"
        elif pos.startswith("JJ"):
            noun_idx = _nearest_index(pairs, idx, ("NN",), "right")
            if noun_idx is not None:
                head_id = noun_idx
            dep_label = "amod"
        elif pos.startswith("RB"):
            verb_idx = _nearest_index(pairs, idx, ("VB",), "right")
            if verb_idx is None:
                verb_idx = _nearest_index(pairs, idx, ("VB",), "left")
            if verb_idx is not None:
                head_id = verb_idx
            dep_label = "advmod"
        elif pos.startswith("IN"):
            prev_head = _nearest_index(pairs, idx, ("VB", "NN"), "left")
            if prev_head is not None:
                head_id = prev_head
            dep_label = "prep"
        elif pos.startswith("NN") or pos.startswith("PRP"):
            if idx < root_id:
                dep_label = "nsubj"
                head_id = root_id
            else:
                dep_label = "obj"
                head_id = root_id
        elif re.match(r"^[\.,;:!?]$", tok):
            dep_label = "punct"
            head_id = root_id

        outputs.append(
            TokenOut(
                id=idx,
                text=tok,
                lemma=tok.lower(),
                pos=pos,
                head_id=head_id,
                dep_label=dep_label,
            )
        )

    return outputs, root_id


def preload_models() -> None:
    _ensure_nltk_data()
    get_spacy_model()
