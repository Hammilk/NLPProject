from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import CountVectorizer
from gensim.utils import simple_preprocess
from sentence_transformers import SentenceTransformer

##Process text into a list for count vectorization
df = pd.read_csv('~/Dev/pythonProjects/train.csv')

filtered_df = df[~df['full_text'].str.contains("PROPER_NAME", case=False, na=False)].reset_index()

documents = [filtered_df.loc[index, 'full_text'] for index in filtered_df.index]
model_name = 'paraphrase-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

embeddings = model.encode(documents, show_progress_bar=True)
embeddings = embeddings.astype('float32')


# Step 3: Initialize FAISS Index for Approximate Nearest Neighbor Search
d = embeddings.shape[1]  # Dimensionality of the embeddings
nlist = 130  # Number of clusters (adjust based on your dataset size)

# Create a quantizer
quantizer = faiss.IndexFlatL2(d)  # Use L2 distance for the quantizer

# Create the IVFFlat index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

# Step 4: Train the index with the embeddings
index.train(embeddings)  # Train the index with your embeddings

# Step 5: Add embeddings to the index
index.add(embeddings)  # Add all embeddings to the index

# Step 6: Collect pairs of document similarities
pairs = []  # To store document pairs and their distances

# Define number of nearest neighbors to retrieve
k = 5  # You can adjust this based on your needs

for i in range(len(documents)):
    query_vector = embeddings[i].reshape(1, -1)  # Reshape for the search
    distances, indices = index.search(query_vector, k)  # Search for nearest neighbors

    for j in range(1, k):  # Start from 1 to exclude the document itself
        neighbor_index = indices[0][j]
        if i < neighbor_index:  # Only consider pairs (i, neighbor_index) once
            # Store the distance instead of similarity
            pairs.append((i, neighbor_index, distances[0][j]))  # Append the distance

# Step 7: Sort pairs by distance and select the top 5 closest pairs
top_pairs = sorted(pairs, key=lambda x: x[2])[:5]  # Sort by distance (ascending)

# Step 8: Display the indices of the top 5 document pairings
print("Top 5 Document Pairing Indices with Closest Distances:")
for idx1, idx2, dist in top_pairs:
    print(f"Index 1: {idx1}, Index 2: {idx2}, Distance: {dist:.4f}")
