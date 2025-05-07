# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# -----------------------------------------------
# Scaling for Preprocessing (for K-Means and PCA)
# -----------------------------------------------
def preprocess_features(df, selected_features):
    """Scales and encodes selected features from the dataset."""
    X = df[selected_features].copy()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X

# -----------------------------------------------
# Cluster Accuracy Utility (for K-Means and Hierarchical Clustering)
# -----------------------------------------------
def cluster_accuracy(y_true, y_pred):
    """Hungarian method for best label alignment."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / y_pred.size

# -----------------------------------------------
# Global Color Palette for Label Visualization
# -----------------------------------------------
color_palette = ['navy', 'darkorange', 'green', 'red', 'purple',
                 'brown', 'pink', 'gray', 'olive', 'cyan', 
                 '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
                 '#bcbd22', '#17becf']

# -----------------------------------------------
# Title and Description
# -----------------------------------------------
st.title("Unsupervised Machine Learning Explorer: PCA, K-Means Clustering & Hierarchical Clustering")

st.expander("About This Application", expanded=True).markdown("""
This interactive application demonstrates the differences in performance for your own CSV file using either K-Means Clustering, Principle Component Analysis (PCA), or Hierarchical Clustering. You can:
- **Upload your own data set** to see how how the model performance handles your own data.
- **Select the desired model** to explore how your data set is handled in either PCA, K-Means Clustering, or Hierarchical Clustering.
- **Select the feature columns and the true column** tell the model what features you want it to analyze and what the true column is.
- **Adjust the model settings** with sliders to see how different parameters affect the performance.
- **View the performance metrics** that correspond to the respective model to see how well it is performing.
        
#### How to Use This Application
1. Upload your dataset in CSV format on the slider on the left.
2. Select the feature columns from the dropdowns.
3. If desired, select a true column to determine how well the model is classifying the 4ata.
4. Choose the model you want to use (PCA, K-Means Clustering, or Hierarchical Clustering).
5. Adjust the model settings using the sliders.
6. View the results on the bottom of the page
""")

# -----------------------------------------------
# File Upload
# -----------------------------------------------
st.sidebar.title("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset uploaded successfully!")
else:
    st.sidebar.warning("Please upload a CSV file to continue.")
    st.stop()

# -----------------------------------------------
# Feature Selection
# -----------------------------------------------
st.subheader("Step 1: Select Features and True Labels")
with st.expander("Feature and True Labels Selection Explanation", expanded=True):
    st.markdown("""
#### What are the features and true labels?
- **Features**: The variables used in the model's prediction. These are the inputs to your model. An example would be the features being the **calories, fat content, etc.** of a pizza brand.
- **True Labels**: The actual labels or classes of the data. This is the output of your model. An example would be the **pizza brand** itself.
 #### What should you do next?
1. Select the feature columns of your data set from the dropdown menu. Two or more are needed for PCA or clustering to work correctly.
2. If you have a true label column, select it from the dropdown menu. It is optional but recommended for evaluating clustering performance.
*Note: ID columns are automatically excluded.*""")

exclude_cols = ['id'] if 'id' in df.columns else []
all_columns = [col for col in df.columns if col not in exclude_cols]
feature_cols = st.multiselect("Select Feature Columns", all_columns)

# Optional True Label Selection
label_options = ["None"] + all_columns
label_col = st.selectbox("Select True Label Column (optional)", label_options, index=0)
if label_col == "None":
    label_col = None


# -----------------------------------------------
# Preprocessing Selected Features
# -----------------------------------------------
st.subheader("Step 2: Data Preprocessing")
with st.expander("Data Preprocessing Explanation", expanded=True):
    st.markdown("""
    #### What is data preprocessing?
    Data preprocessing is the process of transforming raw data into a format that is suitable for analysis. This includes:  
    - **Handling missing values**: Removing or imputing missing data.  
    - **Encoding categorical variables**: Converting categorical variables into numerical format.  
    - **Feature scaling**: Normalizing or standardizing features to ensure they are on the same scale.  
    - **Warnings**: If there are too few samples in the dataset (less than 5), the app will warn you that it may not be the most reliable. 

    #### What should you do next?
    1. Check the preview of the processed data to see if the data is ready for modeling.  
    2. If the data is not ready, you may need to make edits in the data set to ensure that the data is ready for modeling.
    """)

# Remove any rows with missing values
df = df.dropna()

# Warns if too few of samples in dataset
if df.shape[0] < 5:
    st.warning("Your dataset has very few samples. Clustering accuracy and PCA projection may not be reliable.")

# Guard clauses for empty or too few features
if not feature_cols:
    st.warning("Please select at least one feature column to continue.")
    st.stop()
if len(feature_cols) < 2:
    st.error("Select at least 2 features for PCA or clustering to work correctly.")
    st.stop()

# Apply preprocessing pipeline
X_scaled, X_encoded = preprocess_features(df, feature_cols)

# Show processed features preview
with st.expander("Preview Processed Data"):
    st.write("Features (X):", pd.DataFrame(X_scaled, columns=X_encoded.columns).head())

# -----------------------------------------------
# Model Selection Sidebar
# -----------------------------------------------
st.sidebar.title("Model Settings")
model_choice = st.sidebar.selectbox("Choose a Model", ["K-Means Clustering", "PCA", "Hierarchical Clustering"])

#-------------------------------------------------
st.subheader("Step 3: Select your Model")
st.markdown(
    """Deciding which model to use is an important step in the machine learning process. 
    ***PCA*** is a dimensionality reduction technique that transforms the data while preserving as much variance as possible.
    ***K-Means Clustering*** is a clustering algorithm that groups data points into clusters based on their similarity. 
    ***Hierarchical Clustering*** is a clustering algorithm that builds a hierarchy of clusters based on the distance between data points.
    Each of them have pros and cons when looking at a data set, so this application is meant to help you determine which one is more accurate and better for your dataset."""
            )

st.expander("PCA Description", expanded=True).markdown("""
Principle Component Analysis (PCA) is a dimensionality reduction technique that transforms the data, where the greatest variance by any projection lies on the first coordinate (the first principal component), the second greatest variance on the second coordinate, and so on. 
**Pros:** It reduces the dimensionality of the data, helps to visualize high-dimensional data, and can reduce noise.
**Cons:** It may lose some information, and the components may not be interpretable. 
*PCA is often used as a preprocessing step before applying other machine learning algorithms (it is used in the K-Means Clustering processing).* 
                                                                        
#### Hyperparameters:
- **Number of Principal Components**: The number of components to keep.
    - A higher number of components may lead to overfitting, while a lower number may not capture enough variance.
""")

st.expander("K-Means Clustering Description", expanded=True).markdown("""
K-Means Clustering is an unsupervised learning algorithm that groups data into similar groups based on the distance between their features. It uses centroids (which is the mean or median of all the points in the cluster).
**Pros:** It is simple to use, works well with large datasets, and is efficient.
**Cons:** It requires the number of clusters to be specified beforehand and is sensitive to outliers.
        
#### Hyperparameters:
- **Number of clusters (k)**: The number of clusters and centroids to form.
    - A higher number of clusters may cause overfitting, but a lower number may not represent the data well. 
- **Initialization Method**: The method used to initialize the centroids.
    - **k-means++** is an enhanced version of the k-means algorithm that helps to select initial cluster centers in a smart way.
         -   This method tends to be more efficient and leads to better clustering results (commonly the default method).
    - **Random** sets the initial cluster centers randomly.
- **Max Iterations**: The maximum number of iterations for the algorithm to converge.
    - A higher number of max iterations may allow the model to converge better.
""")

st.expander("Hierarchical Clustering Description", expanded=True).markdown("""
Hierarchical Clustering is an unsupervised learning algorithm that builds a tree of culsters that starts larger and slowly breaks down into smaller groups at each branching point. This makes a **dendrogram**, which is a tree-like diagram that shows the breakdown of the clusters.
**Pros:** It is easy to visualize and interpret, which is helpful for understanding the data and its outliers, and it does not require the number of clusters to be predetermined for the dendrogram. 
**Cons:** It can be difficult for large datasets, and it is sensitive to outliers. 
        
#### Hyperparameters:
- **Number of Clusters**: The number of clusters to form in PCA.
    - A higher number of clusters may cause overfitting, but a lower number may not represent the data well. 
- **Linkage Method**: The method used to calculate the distance between clusters.
    - **Ward** minimizes the variance within each cluster.
    - **Complete** uses the maximum distance between points in two clusters.
    - **Average** uses the average distance between points in two clusters.
    - **Single** uses the minimum distance between points in two clusters.
""")

# -----------------------------------------------
# K-Means Clustering Section
# -----------------------------------------------
if model_choice == "K-Means Clustering":
    st.subheader("Step 4: K-Means Clustering")

    # Sidebar parameters for KMeans
    n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
    init_method = st.sidebar.selectbox("Initialization Method", ["k-means++", "random"])
    max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 300)

    # Fit KMeans
    model = KMeans(n_clusters=n_clusters, init=init_method, max_iter=max_iter, random_state=42)
    labels = model.fit_predict(X_scaled)

    # PCA projection for 2D visualization
    st.subheader("Cluster Visualization")
    st.markdown("""
    The K-Means Clustering algorithm has been applied to the data set. The clusters are shown in the plot below, where each point represents a sample and the color represents the cluster it belongs to.
    This is a way to visualize the clusters in a 2D space using PCA. 
    """)

    pca_vis = PCA(n_components=2)
    X_vis = pca_vis.fit_transform(X_scaled)

    # Plot combined PCA scatter with conditional labeling
    st.subheader("K-Means Clustering: 2D PCA Projection")
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = color_palette[:len(unique_labels)]

    if label_col is not None:
        df[label_col] = df[label_col].astype("category")
        true_labels = df[label_col].cat.codes
        target_names = df[label_col].cat.categories.tolist()

        for color, cluster_id in zip(colors, unique_labels):
            mask = labels == cluster_id
            label = target_names[cluster_id] if cluster_id < len(target_names) else f"Cluster {cluster_id}"
            ax.scatter(
                X_vis[mask, 0],
                X_vis[mask, 1],
                color=color,
                alpha=0.7,
                edgecolor='k',
                s=60,
                label=label
            )
    else:
        for color, cluster_id in zip(colors, unique_labels):
            mask = labels == cluster_id
            ax.scatter(
                X_vis[mask, 0],
                X_vis[mask, 1],
                color=color,
                alpha=0.7,
                edgecolor='k',
                s=60,
                label=f'Cluster {cluster_id}'
            )

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('K-Means Clustering: 2D PCA Projection')
    ax.legend(loc='best')
    ax.grid(True)
    st.pyplot(fig)
    st.markdown("---")

    # Elbow and Silhouette Analysis
    ks = range(2, 11)
    wcss = []
    silhouette_scores = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        score = silhouette_score(X_scaled, km.labels_)
        silhouette_scores.append(score)

    fig_opt_k, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Elbow Plot
    st.subheader("Elbow Plot")
    st.markdown("""
    The Elbow Method is plotting the Within-Cluster Sum of Squares (WCSS) against the number of clusters. 
    The point where the distortion starts to decrease at a slower rate is considered the optimal number of clusters (also called the elbow).
    This is a way to determine the optimal number of clusters for the K-Means Clustering algorithm.
    """)
    ax1.plot(ks, wcss, marker='o')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)

    # Silhouette Plot
    st.subheader("Silhouette Plot")
    st.markdown("""
    The Silhouette Score measures how well-separated the clusters are. Scores range from:
    - **+1**: highly dense and well-separated clusters
    - **0**: overlapping clusters
    - **−1**: poor clustering
    Use this plot to help choose the best number of clusters for Hierarchical Clustering.
    """)
    ax2.plot(ks, silhouette_scores, marker='o', color='green')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score for Optimal k')
    ax2.grid(True)

    plt.tight_layout()
    st.pyplot(fig_opt_k)

    # Determine the best k based on Silhouette Score
    best_k_kmeans = ks[np.argmax(silhouette_scores)]
    best_score_kmeans = max(silhouette_scores)

    st.markdown(f"**Best k by Silhouette Score:** `{best_k_kmeans}` (score = `{best_score_kmeans:.3f}`)")
    st.markdown("""
        Here is a calculation of the best k based on the silhouette score.
        """)

    # Evaluate clustering results against true labels
    st.markdown("---")
    st.header("Evaluate Clusters Against True Labels")
    st.markdown("""
    If you selected a true label column, the model's clustering performance is compared to the actual classes using two metrics:
    - **Adjusted Rand Index (ARI)**: Measures similarity between cluster assignments and true labels. You want a higher ARI score.
    - **Accuracy Score**: Tells you how well the model is correctly classifying the data. You want a higher accuracy score.
    """)

    if label_col is not None:
        df[label_col] = df[label_col].astype('category')
        true_labels = df[label_col].cat.codes

        # Evaluation metrics
        ari_score = adjusted_rand_score(true_labels, labels)
        acc_score = cluster_accuracy(true_labels, labels)

        st.markdown(f"**Adjusted Rand Index (ARI):** `{ari_score:.3f}`")
        st.markdown(f"**Accuracy Score:** `{acc_score:.3f}`")
    else:
        st.info("No true label selected. Skipping performance evaluation.")

# -----------------------------------------------
# PCA Analysis Section
# -----------------------------------------------
elif model_choice == "PCA":
    st.subheader("Step 4: PCA Analysis")

    # Sidebar parameter for number of components
    n_components = st.sidebar.slider("Number of Principal Components", 2, min(len(X_encoded.columns), 10), 2)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])

    # Determine y and target_names only if label_col is valid
    if label_col is not None:
        df[label_col] = df[label_col].astype('category')
        y = df[label_col].cat.codes
        target_names = df[label_col].cat.categories.tolist()
    else:
        y = None
        target_names = []

    # PCA 2D Scatter Plot Colored by True Labels
    st.subheader("PCA: 2D Projection of Data")
    st.markdown("""Each point represents a sample projected onto the first two principal components (each of the two dimensions).
                If a label column is selected, colors indicate different classes.
                If no label column is selected, all points are shown in gray.
                """)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = color_palette

    if y is not None:
        for color, i, target_name in zip(colors, np.unique(y), target_names):
            ax.scatter(
                X_pca[y == i, 0], X_pca[y == i, 1],
                color=color, alpha=0.7, label=target_name, edgecolor='k', s=60
            )
        ax.legend(loc='best')
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], color='gray', alpha=0.6, s=60)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('PCA: 2D Projection of Data')
    ax.grid(True)
    st.pyplot(fig)
    st.markdown("---")  

    # Explained variance table
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    explained_percent = explained_variance * 100
    cumulative_percent = cumulative_variance * 100
    components = np.arange(1, len(explained_variance) + 1)

    st.subheader("Explained Variance Table")
    st.markdown("""
    The **explained variance** ratio indicates the proportion of variance explained by each principal component.
    The **cumulative explained** variance shows the total variance explained by the first n components.
    """)
    variance_df = pd.DataFrame({
        "Principal Component": [f"PC{i}" for i in components],
        "Explained Variance Ratio": explained_variance,
        "Cumulative Explained Variance": cumulative_variance
    })
    st.dataframe(variance_df)
    st.markdown("---")  

    #Combined Bar and Line Plot
    st.subheader("PCA: Variance Explained Plot")
    st.markdown("""
        The bar plot shows the individual variance explained by each principal component, while the line plot shows the cumulative variance explained.
        The y-axis on the left shows the individual variance explained, while the y-axis on the right shows the cumulative variance explained.
        The x-axis shows the principal components. The **indivudual variance** is the proportion of variance explained by each principal component, and **cumulative variance** is the total variance explained by the first n components. When you adust the number of principal components, you can see what the percentages of variance are in each component to determine how many components you want to keep.
                """)   
    fig, ax1 = plt.subplots(figsize=(8, 6))
    bar_color = 'steelblue'
    ax1.bar(components, explained_percent, color=bar_color, alpha=0.8, label='Individual Variance')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Individual Variance Explained (%)', color=bar_color)
    ax1.tick_params(axis='y', labelcolor=bar_color)
    ax1.set_xticks(components)
    ax1.set_xticklabels([f"PC{i}" for i in components])

    for i, v in enumerate(explained_percent):
        ax1.text(components[i], v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10, color='black')

    ax2 = ax1.twinx()
    line_color = 'crimson'
    ax2.plot(components, cumulative_percent, color=line_color, marker='o', label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance Explained (%)', color=line_color)
    ax2.tick_params(axis='y', labelcolor=line_color)
    ax2.set_ylim(0, 100)
    ax1.grid(False)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.85, 0.5))

    plt.title('PCA: Variance Explained', pad=20)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("---")

    # Biplot: PCA Scores and Feature Loadings
    if n_components >= 2:
        st.subheader("Biplot: PCA Scores and Feature Loadings")
        st.markdown("""
                    A **biplot** shows the projected data points (scores) and the contribution of the original features (loadings). The loading vectors express the
                    direction in which the original features contribute most to the variance in the principal components.
                    """)

        loadings = pca.components_.T
        scaling_factor = 50.0

        fig_biplot, ax = plt.subplots(figsize=(8, 6))
        colors = color_palette

        if y is not None:
            for color, i, target_name in zip(colors, np.unique(y), target_names):
                ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=0.7, label=target_name, edgecolor='k', s=60)
            ax.legend(loc='best')
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=60)

        for i, feature in enumerate(X_encoded.columns):
            ax.arrow(0, 0, scaling_factor * loadings[i, 0], scaling_factor * loadings[i, 1], color='r', width=0.02, head_width=0.1)
            ax.text(scaling_factor * loadings[i, 0] * 1.1, scaling_factor * loadings[i, 1] * 1.1, feature, color='r', ha='center', va='center')

        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Biplot: PCA Scores and Loadings")
        ax.grid(True)
        st.pyplot(fig_biplot)

# -----------------------------------------------
# Hierarchical Clustering Section
# -----------------------------------------------
elif model_choice == "Hierarchical Clustering":
    st.subheader("Step 3: Hierarchical Clustering")

    # Sidebar parameters
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
    linkage_method = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])

    # Fit model
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X_scaled)

    # PCA for visualization
    pca_vis = PCA(n_components=2)
    X_vis = pca_vis.fit_transform(X_scaled)

    # Hierarchical Clustering: 2D PCA Projection 
    st.subheader("Hierarchical Clustering: 2D PCA Projection")
    st.markdown("""
    The Hierarchical Clustering algorithm has been applied to the data set. The clusters are shown in the plot below, where each point represents a sample and the color represents the cluster it belongs to.
    This is a way to visualize the clusters in a 2D space using PCA. 
    """)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        X_vis[:, 0], X_vis[:, 1],
        c=labels,
        cmap='viridis',
        s=60,
        edgecolor='k',
        alpha=0.7
    )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Agglomerative Clustering via PCA")

    # Dynamic legend
    if label_col is not None:
        df[label_col] = df[label_col].astype("category")
        target_names = df[label_col].cat.categories.tolist()
        handles, _ = scatter.legend_elements()
        ax.legend(handles, [f"{name}" for name in target_names[:len(handles)]], title="True Labels")
    else:
        ax.legend(*scatter.legend_elements(), title="Clusters")

    ax.grid(True)
    st.pyplot(fig)
    st.markdown("---")

    # Dendrogram 
    st.subheader("Dendrogram")
    st.markdown("""
    A dendrogram shows how the clusters are formed through branches and sub-branches in a tree-like shape. 
    This is the visualization of the hierarchical clustering algorithm, where the y-axis shows the distance between clusters and the x-axis shows the samples.
    """)

    Z = linkage(X_scaled, method=linkage_method)

    if label_col is not None:
        dendro_labels = df[label_col].astype(str).tolist()
    else:
        dendro_labels = df.index.astype(str).tolist()

    fig_dendro, ax_dendro = plt.subplots(figsize=(12, 6))
    dendrogram(Z, labels=dendro_labels, ax=ax_dendro)
    ax_dendro.set_title("Hierarchical Clustering Dendrogram")
    ax_dendro.set_xlabel("Samples")
    ax_dendro.set_ylabel("Distance")
    st.pyplot(fig_dendro)
    st.markdown("---")
    
    # Silhouette Score Plot for Optimal Cluster Count
    st.subheader("Silhouette Analysis for Optimal Cluster Count")
    st.markdown("""
    The Silhouette Score measures how well-separated the clusters are. Scores range from:
    - **+1**: highly dense and well-separated clusters
    - **0**: overlapping clusters
    - **−1**: poor clustering

    Use this plot to help choose the best number of clusters for Hierarchical Clustering.
    """)

    k_range = range(2, 11)
    sil_scores = []

    for k in k_range:
        hc_model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        k_labels = hc_model.fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, k_labels))

    best_k = k_range[np.argmax(sil_scores)]

    fig_sil, ax_sil = plt.subplots(figsize=(7, 4))
    ax_sil.plot(list(k_range), sil_scores, marker='o')
    ax_sil.set_xticks(list(k_range))
    ax_sil.set_xlabel("Number of Clusters (k)")
    ax_sil.set_ylabel("Average Silhouette Score")
    ax_sil.set_title("Silhouette Analysis for Hierarchical Clustering")
    ax_sil.grid(True, alpha=0.3)
    st.pyplot(fig_sil)

    st.markdown(f"**Best k by Silhouette Score:** `{best_k}` (score = `{max(sil_scores):.3f}`)")

    # Evaluation metrics
    st.markdown("---")
    st.header("Evaluate Clusters Against True Labels")
    st.markdown("""
    If you selected a true label column, the model's clustering performance is compared to the actual classes using two metrics:
    - **Adjusted Rand Index (ARI)**: Measures similarity between cluster assignments and true labels. You want a higher ARI score.
    - **Accuracy Score**: Tells you how well the model is correctly classifying the data. You want a higher accuracy score.
    """)

    if label_col is not None:
        true_labels = df[label_col].cat.codes
        ari_score = adjusted_rand_score(true_labels, labels)
        acc_score = cluster_accuracy(true_labels, labels)

        st.markdown(f"**Adjusted Rand Index (ARI):** `{ari_score:.3f}`")
        st.markdown(f"**Accuracy Score:** `{acc_score:.3f}`")
    else:
        st.warning("No true label column was selected. Metrics are unavailable.")


