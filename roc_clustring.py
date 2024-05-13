def roc(predictions,y_test):
            from sklearn.metrics import roc_curve, auc
            import matplotlib.pyplot as plt
            from sklearn.preprocessing import label_binarize
            from itertools import cycle

            # Assuming y_true are the true labels and y_score are the predicted probabilities
            for n in  range(len(predictions.columns)):
                        y_test_binarized = label_binarize(y_test, classes=  y_test.unique())
                        y_predictions_binarized = label_binarize(predictions.iloc[:,n], classes=  y_test.unique())
                        n_classes = len(y_test.unique())

                        # Compute ROC curve and ROC area for each class
                        fpr = dict()
                        tpr = dict()
                        roc_auc = dict()
                        for i in range(n_classes):
                            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_predictions_binarized[:, i])
                            roc_auc[i] = auc(fpr[i], tpr[i])

                        # Plot ROC curve for each class
                        plt.figure()
                        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
                        for i, color in zip(range(n_classes), colors):
                            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                                     label='ROC curve of class {0} (AUC = {1:0.2f})'
                                     ''.format(i, roc_auc[i]))

                        plt.plot([0, 1], [0, 1], 'k--', lw=2)
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title(f'Receiver Operating {predictions.columns[n]}')
                        plt.legend(loc="lower right")
                        plt.show()
            plt.plot([0, 1], [0, 1], 'k--', lw=2)

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve for Multiclass')
            plt.legend(loc="lower right")
            plt.show()
def clusters(X_train, X_test, y_train, y_test):
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go  # Import Plotly Graph Objects for creating traces
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=5)  # Choose the number of clusters
        kmeans.fit(X_train)

        # Get cluster labels and centroids
        cluster_labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # Apply PCA to reduce the data to 2 dimensions
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_train)

        # Create a DataFrame for the reduced data
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

        # Add cluster labels to the PCA DataFrame
        pca_df['Cluster'] = cluster_labels

        # Plot using Plotly Express
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title='K-means Clustering', 
                         color_continuous_scale='Viridis')

        # Plot centroids using Plotly Graph Objects
        fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', marker=dict(symbol='x', size=12, color='black'), 
                                 name='Centroids'))

        fig.show()
def cluster3d(X_train, X_test, y_train, y_test):        
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from mpl_toolkits.mplot3d import Axes3D

        # Assuming df is your DataFrame containing the data

        # Preprocess the data if necessary (e.g., scaling)
        # Perform K-means clustering
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

        fig.show(renderer='iframe_connected')
