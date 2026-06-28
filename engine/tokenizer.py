import re #removes punctuation
#"Bring me the dictionary of English stopwords."
from nltk.corpus import stopwords # imports stopwords like "the", "a", "an", etc. (does not use it yet)
from nltk.stem import PorterStemmer #reduces words to their root forms ex: running become run

#Give me the list of English stopwords... we are doing this so that we acn later check if a word is a stopword or not. If it is, we will remove it from the list of tokens.
STOPWORDS = set(stopwords.words('english')) #loads the list of stopwords from NLTK
stemmer = PorterStemmer() #creates streamer object ex: running → run

def tokenize(text: str) -> list[str]: #input: string, output: list of strings
    # Step 1: lowercase everything
    text = text.lower()
    
    # Step 2: replace punctuation with spaces
    # "dog,cat" → "dog cat" not "dogcat"
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Step 3: split whwerever there is whitespace
    tokens = text.split()
    
    # Step 4: remove stopwords and stem each word
    tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in STOPWORDS and len(token) > 1
    ]
    
    return tokens