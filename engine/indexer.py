from collections import defaultdict #whenever someone asks for a missing key, automatically create a new entry for it (new list)
from engine.tokenizer import tokenize

class InvertedIndex: #class for storing search index
    def __init__(self): #constructor for the InvertedIndex class
        # index structure:
        # { token: { doc_id: [positions] } }
        self.index = defaultdict(lambda: defaultdict(list)) #outer defaultdict creates a new entry for a missing token, inner defaultdict creates a new entry for a missing doc_id, and the list stores the positions of the token in the document
        
        # track how many tokens are in each document
        # we need this later for BM25 (longer docs get penalised)
        self.doc_lengths = {}
        
        # total number of documents indexed
        self.num_docs = 0

    def add_document(self, doc_id: int, text: str):
        tokens = tokenize(text)
        
        # store how many tokens this document has
        self.doc_lengths[doc_id] = len(tokens)
        
        # record the position of every token in this document
        for position, token in enumerate(tokens): #enumerate gives both index and value
            self.index[token][doc_id].append(position) #we are storing the position of the token in the document, so we can later use it for phrase queries (ex: "the cat" should return documents where "the" and "cat" are next to each other)
        
        self.num_docs += 1 #increment as we keep adding documents

    def get_postings(self, token: str) -> dict:
        """Return all doc_ids and positions for a token."""
        return self.index.get(token, {}) #get() because it returns an empty dict if the token is not in the index, instead of raising a KeyError

    def __len__(self):
        return self.num_docs