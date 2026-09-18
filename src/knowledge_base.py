"""Pure-Python TF-IDF search over the Avis knowledge base (30 articles).

No external vector DB or embeddings API needed — the corpus is small enough
for in-memory cosine similarity over word-frequency vectors.
"""
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Load articles
# ---------------------------------------------------------------------------

_KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge-base" / "articles.json"

with open(_KB_PATH) as f:
    _ARTICLES: list[dict] = json.load(f)

# Authority boost: official-policy gets a slight relevance boost
_AUTHORITY_BOOST = {"official-policy": 1.15, "help-center": 1.0, "legacy": 0.9}


# ---------------------------------------------------------------------------
# TF-IDF index
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_index(articles: list[dict]) -> tuple[list[Counter], dict[str, float]]:
    """Build TF vectors per document and IDF weights across the corpus."""
    doc_tfs = []
    doc_count = len(articles)
    df: Counter = Counter()  # how many docs contain each term

    for article in articles:
        tokens = _tokenize(article["title"] + " " + article["body"])
        tf = Counter(tokens)
        doc_tfs.append(tf)
        for term in set(tokens):
            df[term] += 1

    idf = {term: math.log(doc_count / count) for term, count in df.items()}
    return doc_tfs, idf


_DOC_TFS, _IDF = _build_index(_ARTICLES)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _cosine_similarity(query_vec: dict[str, float], doc_vec: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors (dicts)."""
    common_terms = set(query_vec) & set(doc_vec)
    if not common_terms:
        return 0.0
    dot = sum(query_vec[t] * doc_vec[t] for t in common_terms)
    mag_q = math.sqrt(sum(v * v for v in query_vec.values()))
    mag_d = math.sqrt(sum(v * v for v in doc_vec.values()))
    if mag_q == 0 or mag_d == 0:
        return 0.0
    return dot / (mag_q * mag_d)


def search(query: str, top_k: int = 3) -> list[dict]:
    """Search the knowledge base and return the top-k most relevant articles.

    Returns a list of dicts with keys: id, title, category, body, authority, score.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    query_tf = Counter(query_tokens)
    query_vec = {term: tf * _IDF.get(term, 0) for term, tf in query_tf.items()}

    scored = []
    for i, article in enumerate(_ARTICLES):
        doc_tf = _DOC_TFS[i]
        doc_vec = {term: tf * _IDF.get(term, 0) for term, tf in doc_tf.items()}
        sim = _cosine_similarity(query_vec, doc_vec)
        # Apply authority boost
        boost = _AUTHORITY_BOOST.get(article.get("authority", ""), 1.0)
        scored.append((sim * boost, i))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, idx in scored[:top_k]:
        if score <= 0:
            break
        article = _ARTICLES[idx]
        results.append({
            "id": article["id"],
            "title": article["title"],
            "category": article["category"],
            "body": article["body"],
            "authority": article["authority"],
            "score": round(score, 4),
        })

    return results


if __name__ == "__main__":
    for r in search("How do I extend my rental?"):
        print(f"  [{r['score']:.3f}] {r['title']} ({r['authority']})")
        print(f"         {r['body'][:80]}...")
        print()
