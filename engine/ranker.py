import math #needed for log function
from engine.indexer import InvertedIndex #my indexer.py file has the InvertedIndex class, which we need to use here
from engine.tokenizer import tokenize #converts text into tokens (words) and removes stopwords, punctuation, and stems the words

def compute_tfidf(token: str, doc_id: int, index: InvertedIndex) -> float:
    postings = index.get_postings(token)
    
    if doc_id not in postings:
        return 0.0
    
    # TF: how often does this token appear in this doc?
    term_count = len(postings[doc_id])
    doc_length = index.doc_lengths[doc_id]
    tf = term_count / doc_length if doc_length > 0 else 0.0
    
    # IDF: how rare is this token across all documents?
    # +1 inside log avoids division by zero
    docs_with_term = len(postings)
    idf = math.log((index.num_docs + 1) / (docs_with_term + 1))
    
    return tf * idf


def score_tfidf(query: str, index: InvertedIndex) -> list[tuple[int, float]]:
    """Return list of (doc_id, score) sorted by score descending."""
    tokens = tokenize(query)
    scores = {}
    
    for token in tokens:
        postings = index.get_postings(token)
        for doc_id in postings:
            score = compute_tfidf(token, doc_id, index)
            # accumulate scores across all query terms
            scores[doc_id] = scores.get(doc_id, 0.0) + score
    
    # sort by score, highest first
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)