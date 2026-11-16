"""Keyword-set retrieval over snippet titles and descriptions."""
from functools import lru_cache
import re

from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()
_wordnet_pos = {"J": "a", "V": "v", "N": "n", "R": "r"}


@lru_cache(maxsize=16384)
def normalize_terms(text):
    """Lowercase, tokenize and lemmatize English terms without expansion."""
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    if not tokens:
        return frozenset()
    try:
        return frozenset(
            _lemmatizer.lemmatize(token, _wordnet_pos.get(tag[0], "n"))
            for token, tag in pos_tag(tokens, lang="eng")
        )
    except LookupError as exc:
        raise RuntimeError(
            "Missing NLTK language resources. Using the project's Python, run: "
            "python -m nltk.downloader wordnet averaged_perceptron_tagger_eng"
        ) from exc


def keyword_terms(keywords):
    if isinstance(keywords, str):
        keywords = [keywords]
    return frozenset(
        term for keyword in (keywords or [])
        for term in normalize_terms(str(keyword))
    )


def example_terms(example):
    return normalize_terms(str(example.get("example_title") or "")) | normalize_terms(
        str(example.get("description") or "")
    )


def calculate_example_score(keywords, example):
    """s(q,e) = |K_q intersect K_e|; each shared lemma counts once."""
    return len(keyword_terms(keywords) & example_terms(example))


def find_top_examples(keywords, all_examples, top_k=10):
    """Return at most ten positive-score snippets; ties retain input order."""
    query = keyword_terms(keywords)
    if not query:
        return []
    scored_examples = []
    for example in all_examples:
        score = len(query & example_terms(example))
        if score > 0:
            result = dict(example)
            result["score"] = score
            scored_examples.append(result)
    scored_examples.sort(key=lambda example: example["score"], reverse=True)
    return scored_examples[:max(0, min(top_k, 10))]
