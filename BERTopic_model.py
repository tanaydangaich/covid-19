from db_connection import df_publications
import pandas as pd
from bertopic import BERTopic

# Assuming df_publications is already loaded as a pandas dataframe
# Filter out rows with missing titles or abstracts
df_publications_filtered = df_publications.dropna(subset=['Title', 'Abstract'])

# Combine titles and abstracts
combined_texts = (df_publications_filtered['Title'] + ' ' + df_publications_filtered['Abstract']).tolist()
pub_years = df_publications_filtered['PubYear'].tolist()

# Apply BERTopic
topic_model = BERTopic()
topics, _ = topic_model.fit_transform(combined_texts)

# Method 2 - pytorch
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
topic_model.save("path/to/my/model_dir", serialization="pytorch", save_ctfidf=True, save_embedding_model=embedding_model)

