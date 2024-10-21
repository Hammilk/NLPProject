import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


##Process text into a list for count vectorization
df = pd.read_csv('train.csv')

filtered_df = df[~df['full_text'].str.contains("PROPER_NAME", case=False, na=False)].reset_index()

text = [filtered_df.loc[index, 'full_text'] for index in filtered_df.index]

#Set up similarity
count_vect = CountVectorizer(lowercase=True, stop_words='english')
count_matrix = count_vect.fit_transform(text)

cosine_similarity_matrix = cosine_similarity(count_matrix)

upper_tri_indices = np.triu_indices(cosine_similarity_matrix.shape[0], k=1)
similarity_scores = cosine_similarity_matrix[upper_tri_indices]
top_5_indices = np.argsort(similarity_scores)[-5:][::-1]

for idx in top_5_indices:
    i, j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
    essay1_ID = filtered_df.at[i, 'essay_id']
    essay2_ID = filtered_df.at[j, 'essay_id']
    print(f"Similarity between Document {essay1_ID} and Document {essay2_ID}: {similarity_scores[idx]:.4f}")

