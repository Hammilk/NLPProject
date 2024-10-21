import pandas as pd
import torch
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import numpy as np

class MyCorpus:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data.iloc[:, 1].notna() & ~self.data.iloc[:, 1].str.contains("PROPER_NAME")]

    def __iter__(self):
        for doc in self.data.iloc[:, 1]:
            cleaned_doc = remove_stopwords(doc)
            tokens = simple_preprocess(cleaned_doc)
            yield tokens

#    def __len__(self):
#        return len(self.data)
"""
    def __iter__(self):
        chunksize = 500
        for chunk in pd.read_csv(self.csv_file, chunksize=chunksize):
            for doc in chunk.iloc[:, 1]:
                if 'PROPER_NAME' in doc:
                    continue
                clean_doc = remove_stopwords(doc)
                tokens = simple_preprocess(clean_doc)
                yield tokens
"""

#Load file
file = '~/Dev/pythonProjects/train.csv'
corpus_stream = MyCorpus(file)

#Load Model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


#Function to encode sentences
def encode_sentences(sentences):
    with torch.no_grad():
        #Tokenize and encode sentences
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


# Function to encode sentences in batches
def encode_sentences_in_batches(sentences, batch_size):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings)
    return torch.cat(embeddings)


documents = []

for doc in corpus_stream:
    documents.append(" ".join(doc))

batch_size = 100

#embeddings = encode_sentences(documents)
embeddings = encode_sentences_in_batches(documents, batch_size)

similarity_matrix = cosine_similarity(embeddings.numpy())

print("Cosine Similarity Matrix: ")
print(similarity_matrix)

# Find the indices of the top 5 most similar document pairs
num_top_pairs = 5  # Number of top pairs you want to retrieve
most_similar_indices = np.unravel_index(np.argsort(similarity_matrix, axis=None)[-num_top_pairs:], similarity_matrix.shape)

# Print the top 5 most similar document pairs and their similarity scores
for i in range(num_top_pairs):
    doc1_index = most_similar_indices[0][i]
    doc2_index = most_similar_indices[1][i]
    similarity_score = similarity_matrix[doc1_index, doc2_index]
    print(f"Document {doc1_index} <-> Document {doc2_index} with similarity {similarity_score:.2f}")







