import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import time
import textwrap

start_time = time.time() 

## Process text into a list for count vectorization
df = pd.read_csv('~/Dev/pythonProjects/train.csv')

# Extract the essay IDs from the original dataframe (assuming the first column is the essay ID)
essay_ids = df.iloc[:, 0]

# Filter out rows containing "PROPER_NAME" without resetting the index
filtered_df = df[~df['full_text'].str.contains("PROPER_NAME", case=False, na=False)]

# Filter the essay IDs to match the filtered_df
filtered_essay_ids = essay_ids[filtered_df.index].tolist()

# Extract documents from filtered_df
documents = filtered_df['full_text'].tolist()
model_name = 'paraphrase-MiniLM-L6-v2'

model_start_time = time.time()

model = SentenceTransformer(model_name)

model_end_time = time.time()
embedding_start_time = model_end_time

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

embedding_end_time = time.time()
query_start_time = time.time()

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

print(pairs)

# Step 7: Sort pairs by distance and select the top 5 closest pairs
top_pairs = sorted(pairs, key=lambda x: x[2])[:5]  # Sort by distance (ascending)

# Step 8: Display the essay IDs of the top 5 document pairings
print("Top 5 Document Pairing Essay IDs with Closest Distances:")
for idx1, idx2, dist in top_pairs:
    essay_id_1 = filtered_essay_ids[idx1]
    essay_id_2 = filtered_essay_ids[idx2]
    
    print(f"Essay ID 1: {essay_id_1}, Essay ID 2: {essay_id_2}, Distance: {dist:.4f}")
    text_preview_1 = textwrap.shorten(documents[idx1], width=70, placeholder="...")
    text_preview_2 = textwrap.shorten(documents[idx2], width=70, placeholder="...")
    print(f"Text Preview 1: {text_preview_1}")
    print(f"Text Preview 2: {text_preview_2}")

query_end_time = time.time()

end_time = time.time()
total_time = end_time - start_time
total_load_time = model_end_time - model_start_time
total_embedding_time = embedding_end_time - embedding_start_time
total_query_time = end_time - query_start_time

print("*************************************")

print(f"Total Time: {total_time}, Model Loading Time: {total_load_time}")
print(f"Total Embedding Time: {total_embedding_time}, Total Query Time: {total_query_time}")

