import os
import pandas as pd
import plotly.express as px

df_keywords = pd.read_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/Cleantech Concepts/df_cpc_subclass_keywords_yake_embeddings_hdbscan_2d_cluster100.json')

# fig = px.scatter(df_keywords, x='keywords_embeddings_bertforpatents_umap_x', y='keywords_embeddings_bertforpatents_umap_y', color='keywords_embeddings_bertforpatents_hdbscan')
fig = px.scatter(df_keywords, 
                 x='keywords_embeddings_climatebert_umap_x', 
                 y='keywords_embeddings_climatebert_umap_y', 
                 color='cpc_subclass',
                 color_discrete_sequence=px.colors.qualitative.Plotly)
fig.show()

