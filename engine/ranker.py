import math                                        # for log()
from engine.indexer import InvertedIndex          # inverted index class
from engine.tokenizer import tokenize             # converts text to tokens

def compute_tfidf(token: str, doc_id: int, index: InvertedIndex) -> float:
    postings = index.get_postings(token)          # docs containing token
    
    if doc_id not in postings:                    # token not in this doc
        return 0.0
    
    # TF: how often does this token appear in this doc?
    term_count = len(postings[doc_id])            # number of occurrences
    doc_length = index.doc_lengths[doc_id]        # total words in doc
    tf = term_count / doc_length if doc_length > 0 else 0.0
    # normalized frequency
    
    # IDF: how rare is this token across all documents?
    docs_with_term = len(postings)                # docs containing token
    idf = math.log((index.num_docs + 1) / (docs_with_term + 1))
    # rarer word = higher score
    
    return tf * idf                               # final TF-IDF score


def score_tfidf(query: str, index: InvertedIndex) -> list[tuple[int, float]]:
    """Return list of (doc_id, score) sorted by score descending."""
    tokens = tokenize(query)                      # split query into words
    scores = {}                                   # {doc_id: total_score}
    
    for token in tokens:                          # process each query word
        postings = index.get_postings(token)      # matching documents
        
        for doc_id in postings:                   # each matching document
            score = compute_tfidf(token, doc_id, index)
            # score this token in this document
            
            # accumulate scores across all query terms
            scores[doc_id] = scores.get(doc_id, 0.0) + score
            # add to previous score
    
    # sort by score, highest first
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # x[1] = score, reverse gives highest first