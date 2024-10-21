import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap


##Process text into a list for count vectorization
df = pd.read_csv('~/Dev/pythonProjects/train.csv')

#Drop all documents that contains the string "PROPER_NAME"
filtered_df = df[~df['full_text'].str.contains("PROPER_NAME", case=False, na=False)].reset_index()

#Extract full_text column from dataframe
text = [filtered_df.loc[index, 'full_text'] for index in filtered_df.index]

#Set up count vectorizer model
count_vect = CountVectorizer(lowercase=True, stop_words='english')
count_matrix = count_vect.fit_transform(text)

#Apply cosine_similarity to the matrix
cosine_similarity_matrix = cosine_similarity(count_matrix)

#Only need to query top part of matrix since the bottom will be pairwise duplicates
upper_tri_indices = np.triu_indices(cosine_similarity_matrix.shape[0], k=1)
similarity_scores = cosine_similarity_matrix[upper_tri_indices]
top_5_indices = np.argsort(similarity_scores)[-5:][::-1]

#Query for top 5 similar pairs
for idx in top_5_indices:
    i, j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
    essay1_ID = filtered_df.at[i, 'essay_id']
    essay2_ID = filtered_df.at[j, 'essay_id']
    print(f"Similarity between Document {essay1_ID} and Document {essay2_ID}: {similarity_scores[idx]:.4f}")
    text_preview_1 = textwrap.shorten(text[i], width = 70, placeholder="...")
    text_preview_2 = textwrap.shorten(text[j], width = 70, placeholder="...")
    print(f"Text Preview 1 {text_preview_1}")
    print(f"Text Preview 2 {text_preview_2}")


