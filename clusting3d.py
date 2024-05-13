def clustring3d(X_train,y_train):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import plotly.express as px

    kmeans = KMeans(n_clusters=5)  # Choose the number of clusters
    kmeans.fit(X_train)

    # Get cluster labels and centroids
    cluster_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Apply PCA to reduce the data to 3 dimensions
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X_train)

    # Create a DataFrame for the reduced data
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])

    # Add cluster labels to the PCA DataFrame
    pca_df['Cluster'] = cluster_labels
    pca_df['class']=y_train
    # Plot the clusters in 3D space
    fig = px.scatter_3d(
        pca_df,  # Dataframe containing features and cluster labels (after PCA)
        x='PC1',  # Use principal components for axes
        y='PC2',
        z='PC3',
        color='Cluster',  # Color based on cluster labels
        opacity=0.8,  # Adjust opacity for better visibility (optional)
    )
    fig.update_layout(title='KMeans Clustering with PCA - 3D Scatter Plot')

    fig.show()
