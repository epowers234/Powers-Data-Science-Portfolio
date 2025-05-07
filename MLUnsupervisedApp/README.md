# Unsupervised Machine Learning Streamlit App

## Project Overview

Learning how to navigate 3 different unsupervised machine learning models and the differences in hyperparameters can be a daunting task for someone with a new data set. It would be a lot of effort to edit the code each time you want to alter a minor difference, and then generate the corresponding plot. The purpose of this app is to see the differences in either PCA, K-Means Clustering, or a Hierarchical Clusting model for machine learning on your dataset. Then, you can change the parameters and see the effects immediately to determine the best settings for your dataset. As you go through these steps, there are also definitions and guides in the app to help explain what is occurring in the model. 

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation and Setup](#installation-and-setup)
3. [Codes and Resources Used](#codes-and-resources-used)
4. [Data](#data)
5. [Principal Component Analysis Overview](#principal-component-analysis-overview)
6. [K-Means Clustering Overview](#k-means-clustering-overview)
7. [Hierarchical Clustering Overview](#hierarchical-clustering-overview)

## Installation and Setup

To code this, VSCode was used. 
<br /> 

If you want to access the app locally, do these steps on your VSCode terminal of the downloaded unsupervisedstreamlit.py file:
1. Download streamlit in the terminal by entering this code:
```pip install streamlit```
2. Ensure you are in the correct directory as your file in the terminal by using ```ls``` (only on Mac) to view in your current directory, ```cd foldername``` to go deeper into a folder or go directly into a specific spot.
3. Once in the current location run this code in the terminal: ```streamlit run unsupervisedstreamlit.py```
4. Then, a local webpage should pop up and you can work on the app there.
<br />

If you want to access the app through a webpage (does not require any downloading), click on this link: [MLUnsupervised App](https://epowers234-powers-mlunsupervisedappunsupervisedstreamlit-z2hjm5.streamlit.app/)
<br />

From there, you just need a dataset to explore! There is one sample option below! **Download one and upload it to the sidebar in the app**, and from there the app will guide you through adjusting the parameters or model type. 

## Codes and Resources Used

1.   **PCA Background and References**
   - Reference webpage: [PCA Background](https://builtin.com/data-science/step-step-explanation-principal-component-analysis).
   - Reference webpage: [PCA Plot Overview](https://support.minitab.com/en-us/minitab/help-and-how-to/statistical-modeling/multivariate/how-to/principal-components/interpret-the-results/all-statistics-and-graphs/).
   - Aided when writing out the definitions and guides.
2.   **K-Means Clustering Background and References**
   - Reference webpage: [K-Means Guide](https://www.ibm.com/think/topics/k-means-clustering).
   - Reference webpage: [K-Means vs K-Means++](https://www.geeksforgeeks.org/k-means-vs-k-means-clustering-algorithm/).
   - Provided insight into what K-Means is and how K-Means++ differs.
3.   **Hierarchical Clustering Background and References**
   - Reference webpage: [Hierarchical Clustering Guide](https://www.ibm.com/think/topics/hierarchical-clustering).
   - Reference webpage: [Advantages of Hierarchical Clustering](https://www.displayr.com/strengths-weaknesses-hierarchical-clustering/#:~:text=The%20strengths%20of%20hierarchical%20clustering,such%20as%20latent%20class%20analysis.).
   - Provided background and insight on why hierarchical clustering is different. 
4.   **Necessary libraries that need to be imported**
   - Pandas
   - Numpy
   - Matplotlib.pyplot
   - Plotly.express
   - Plotly.graphs_objects
   - Scipy.optimize
   - Scipy.cluster.hierarchy
   - From sklearn:
     - sklearn.preprocessing
     - sklearn.decomposition
     - sklearn.cluster
     - sklearn.metrics
  

## Data

Any data set can be used and uploaded to the sidebar in the app. However, here is a data set to explore that works well for unsupervised machine learning, so you can become comfortable prior to exploring a new dataset. 

**Pizza Brand Sample Data Set**
Here is the example Pizza brand data set: [Pizza.csv](https://github.com/user-attachments/files/20075543/Pizza.csv). Ensure you download the dataset and upload it to the app. The brand would be the true column if you choose, and the other variables are the features. 
This dataset was sourced from [Pizza Data Set Source](https://github.com/f-imp/Principal-Component-Analysis-PCA-over-3-datasets).

## Principal Component Analysis Overview
More background is given on the app, but here is an overview: 
Principal Component Analysis (PCA) is a technique that transforms the data by reducing the dimensions of the data where the greatest variance lies on the first principal component and the second greatest variance on the second component, etc. 

**Hyperparameters**:
- Number of Principal Components
  - The number of components to keep. 
 
**Example of PCA made in the app**:
As an example, I uploaded the example pizza dataset to the app and selected PCA with these parameters:
- Feature columns = Mois, Prot, Fat, Sodium, Carb
- True label column = Brand
- Number of Principal Components = 3

### Results: 
#### **PCA 2D Projection of Data**
<img width="400" alt="Screenshot 2025-05-07 at 9 45 52 AM" src="https://github.com/user-attachments/assets/bda330fe-2d96-45b9-be10-1cc382945dac" />
<br /> Visualizes the 2D components of the data, and since a true column was selected (brand), the dots are 
colored to their corresponding brand. 

#### **Explained Variance Table**
<img width="400" alt="Screenshot 2025-05-07 at 9 47 50 AM" src="https://github.com/user-attachments/assets/fbca3c50-40a7-4674-b90d-3cc6feda0048" />
<br /> This shows where the variance is in each principal component, broken down by both explained variance ratio and cumulative explained variance. 

#### **PCA Variance Explained Plot**
<img width="400" alt="Screenshot 2025-05-07 at 9 49 57 AM" src="https://github.com/user-attachments/assets/4aca7edc-9b96-4af6-9759-2f20b8b3dd0c" />
<br /> A combined bar and line plot where the bar plot shows the individual variance explained by each principal component, and the line plot depicts the cumulative variance explained by the first n components (in this case, 3 components). 

#### **Biplot**
<img width="400" alt="Screenshot 2025-05-07 at 9 52 20 AM" src="https://github.com/user-attachments/assets/08a09631-c69c-42ef-9890-c401ff24b0ed" />
<br /> A Biplot will show the projected data points and the contributions of the original features. The vectors of the features show the direction in which they contribute the most to the variance. 

## K-Means Clustering Overview
More background is given on the app, but here is an overview: 
K-Means Clustering groups data into similar groups based on the distance between their features using centroids. 

**Hyperparameters**:
- Number of clusters (k):
 - The number of clusters and centroids that will form.
- Initialization method: The method used to initialize the centroids. 
 - Either k-means++ (enhanced k-means algorithm) or random (initial cluster centers randomly)
- Max iterations: 
 - Maximum number of iterations for the algorithm to converge 
 
**Example of K-means clustering used in the app**:

As an example, I uploaded the pizza dataset to the app and selected K-means clustering with these parameters:
- Feature columns = Mois, Prot, Fat, Sodium, Carb
- True label column = Brand
- Number of clusters (k) = 5
- Initialization method = k-means++
- Max iterations: 400

### Results: 
#### **2D PCA Projection**
<img width="400" alt="Screenshot 2025-05-07 at 10 00 46 AM" src="https://github.com/user-attachments/assets/1a665bcd-7ffb-4fa6-94ae-ac483e34a688" />
<br /> Since K-Means clustering also uses a bit of PCA, it can be helpful to see the 2D PCA projection here as well. Visualizes the 2D components of the data, and since a true column was selected (brand), the dots are colored to their corresponding brand. 

#### **Elbow Plot**
<img width="300" alt="Screenshot 2025-05-07 at 10 01 50 AM" src="https://github.com/user-attachments/assets/a8b44ae4-e0a5-4f07-8045-d0ff74e1370e" />
<br /> The elbow plot allows us to determine where the optimal number of clusters is. The optimal number is where the WCSS begins to decrease at a slower rate. Here, you could say it would likely be around k=5. 

#### **Silhouette Score Plot**
<img width="300" alt="Screenshot 2025-05-07 at 10 01 55 AM" src="https://github.com/user-attachments/assets/91cea93e-050f-456b-940a-4d4ca631b44f" />
<br /> A silhouette score measures how well the clusters are separated. A number closer to 1 indicates better separated and more dense clusters. Based on the graph and the reported best k by silhouette score that the model produces, the optimal k for this model is 5 with a score of 0.684. 

#### **Evaluation Against True Labels**
<img width="200" alt="Screenshot 2025-05-07 at 10 05 29 AM" src="https://github.com/user-attachments/assets/3b9d10cc-2ca6-4ae3-acae-2a4f5d503fd7" />
<br /> To determine how accurate the model is at predicting the correct classification in the clusters, it is optional to select a true label column. By choosing this, the user gets the calculated ARI score (measures similarity between cluster assignments and true labels) and accuracy score (expresses how well the model correctly classifies the data). A higher number is desired for both. 

## Hierarchical Clustering Overview
More background is given on the app, but here is an overview: 
Hierarchical clustering builds a tree of clusters that slowly breaks down into smaller groups, making a dendrogram. 

**Hyperparameters**:
- Number of clusters:
 - The number of clusters and centroids that will form in PCA.
- Linkage method: The method used to calculate the distance between clusters
 - Ward (minimizes distance), Complete (maximum distance between points), Average (average distance between points), Single (minimum distance between points) 
 
**Example of hierarchical clustering used in the app**:

As an example, I uploaded the pizza dataset to the app and selected hierarchical clustering with these parameters:
- Feature columns = Mois, Prot, Fat, Sodium, Carb
- True label column = Brand
- Number of clusters = 6
- Linkage method = average

### Results: 
#### **2D PCA Projection**
<img width="400" alt="Screenshot 2025-05-07 at 10 12 52 AM" src="https://github.com/user-attachments/assets/c40b6459-597e-4189-8c97-9b789a43987d" />
<br /> Since hierarchical clustering also uses a bit of PCA, it can be helpful to see the 2D PCA projection here as well. Visualizes the 2D components of the data, and since a true column was selected (brand), the dots are colored to their corresponding brand. 

#### **Dendrogram**
<img width="400" alt="Screenshot 2025-05-07 at 10 13 49 AM" src="https://github.com/user-attachments/assets/4d6e9f95-440d-41da-8917-cf52ca411fbd" />
<br /> The dendrogram visualizes how the hierarchical clustering algorithm works in a tree-like structure.

#### **Silhouette Score Plot**
<img width="300" alt="Screenshot 2025-05-07 at 10 14 53 AM" src="https://github.com/user-attachments/assets/75e3232c-d95f-47ed-bcff-4c437757bc5d" />
<br /> A silhouette score measures how well the clusters are separated. A number closer to 1 indicates better separated and more dense clusters. Based on the graph and the reported best k by silhouette score that the model produces, the optimal k for this model is 5 with a score of 0.684. 

#### **Evaluation Against True Labels**
<img width="200" alt="Screenshot 2025-05-07 at 10 15 20 AM" src="https://github.com/user-attachments/assets/52f288f7-3271-496b-a1f5-054e9b43810b" />
<br /> To determine how accurate the model is at predicting the correct classification in the clusters, it is optional to select a true label column. By choosing this, the user gets the calculated ARI score (measures similarity between cluster assignments and true labels) and accuracy score (expresses how well the model correctly classifies the data). A higher number is desired for both. 

