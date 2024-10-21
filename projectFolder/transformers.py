import pandas as pd
import torch
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import numpy as np
import time
import textwrap

start_time = time.time()

class MyCorpus:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data.iloc[:, 1].notna() & ~self.data.iloc[:, 1].str.contains("PROPER_NAME")]
        self.essay_ids = self.data['essay_id'].tolist()

    def __iter__(self):
        for doc in self.data.iloc[:, 1]:
            cleaned_doc = remove_stopwords(doc)
            tokens = simple_preprocess(cleaned_doc)
            yield tokens
#Load file

load_start_time = time.time()

file = '~/Dev/pythonProjects/train.csv'
corpus_stream = MyCorpus(file)

#Load Model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

load_end_time = time.time()

# Function to encode sentences in batches
def encode_sentences_in_batches(sentences, batch_size):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings)
    return torch.cat(embeddings) if embeddings else torch.empty(0, model.config.hidden_size)



documents = [" ".join(doc) for doc in corpus_stream]


batch_size = 64

embedding_start_time = time.time()

#embeddings = encode_sentences(documents)
embeddings = encode_sentences_in_batches(documents, batch_size)

cosine_similarity_matrix = cosine_similarity(embeddings.cpu().numpy())

del embeddings

embedding_end_time = time.time()


query_start_time = time.time()

#Only need to query top part of matrix since the bottom will be pairwise duplicates
upper_tri_indices = np.triu_indices(cosine_similarity_matrix.shape[0], k=1)
similarity_scores = cosine_similarity_matrix[upper_tri_indices]
top_5_indices = np.argsort(similarity_scores)[-5:][::-1]

#Query for top 5 similar pairs
for idx in top_5_indices:
    i, j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
    essay1_ID = corpus_stream.essay_ids[i]
    essay2_ID = corpus_stream.essay_ids[j]
    print(f"Similarity between Document {essay1_ID} and Document {essay2_ID}: {similarity_scores[idx]:.4f}")
    text_preview_1 = textwrap.shorten(documents[i], width = 70, placeholder="...")
    text_preview_2 = textwrap.shorten(documents[j], width = 70, placeholder="...")
    print(f"Text Preview 1: {text_preview_1}")
    print(f"Text Preview 2: {text_preview_2}")


print("***************************************")
print("***************************************")


end_time = time.time()
total_time = end_time - start_time
total_load_time = load_end_time - load_start_time
total_embedding_time = embedding_end_time - embedding_start_time
total_query_time = end_time - query_start_time

print(f"Total Time: {total_time}, Model Loading Time: {total_load_time}")
print(f"Total Embedding Time: {total_embedding_time}, Total Query Time: {total_query_time}")








