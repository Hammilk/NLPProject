import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

def get_wordnet_synsets(word):
    return wn.synsets(word)

def calculate_similarity(doc1, doc2):
    #converst strings into lists of substrings with a lowercase conversion
    words_doc1 = word_tokenize(doc1.lower())
    words_doc2 = word_tokenize(doc2.lower())
    
    #get unique words
    unique_words_doc1 = set(words_doc1)
    unique_words_doc2 = set(words_doc2)

    total_similarity 

