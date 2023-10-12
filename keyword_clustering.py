import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

df = pd.read_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/Cleantech Concepts/df_cpc_subclass_keywords_yake.json')

keywords = df['keywords_yake_desc_5000'].tolist()

# Erase leading space for each keyword
keywords = [[keyword.strip() for keyword in keyword_list] for keyword_list in keywords]

# Only take the first 10 keywords for each row
keywords = [keyword_list[:10] for keyword_list in keywords]

dfs_list = []

for idx, keyword_list in enumerate(keywords):
    # Convert keywords to a TF-IDF matrix
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(keyword_list)

    # Reduce dimensionality to 2D using PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    # Determine the number of clusters
    n_clusters = 8

    # Cluster keywords using k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)

    # Create a DataFrame for this set of keywords
    df_temp = pd.DataFrame(X_pca, columns=['x', 'y'])
    df_temp['cluster'] = kmeans.labels_
    df_temp['keyword'] = keyword_list
    df_temp['group'] = idx  # this column indicates which group of keywords each keyword belongs to

    # Append to list of DataFrames
    dfs_list.append(df_temp)

# Append to result DataFrame
result_df = pd.concat(dfs_list, ignore_index=True)

# Plot clusters using Plotly Express, colored by group
fig = px.scatter(result_df, x='x', y='y', symbol='cluster', text='keyword',
                 title='Keyword Clustering')

# Show plot
fig.show()