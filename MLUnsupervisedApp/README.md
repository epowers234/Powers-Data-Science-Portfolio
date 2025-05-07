# Unsupervised Machine Learning Streamlit App

## Project Overview

Learning how to navigate 3 different unsupervised machine learning models and the differences in hyperparameters can be a daunting task for someone with a new data set. It would be a lot of effort to edit the code each time you want to alter a minor difference, and then generate the corresponding plot. The purpose of this app is to see the differences in either PCA, K-Means Clustering, or a Hierarchical Clusting model for machine learning on your dataset. Then, you can change the parameters and see the effects immediately to determine the best settings for your dataset. As you go through these steps, there are also definitions and guides in the app to help explain what is occurring in the model. 

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation and Setup](#installation-and-setup)
3. [Codes and Resources Used](#codes-and-resources-used)
4. [Data](#data)
5. [Logistic Regression Overview](#Logistic-Regression-Overview)
6. [Decision Tree Overview](#Decision-Tree-Overview)

## Installation and Setup

To code this, VSCode was used. 
<br /> 

If you want to access the app locally, do these steps on your VSCode terminal of the downloaded unsupervisedstreamlit.py file:
1. Download streamlit in the terminal by entering this code:
```pip install streamlit```
2. Ensure you are in the correct directory as your file in the terminal by using ```ls``` (only on Mac) to view in your current directory, ```cd foldername``` to go deeper into a folder or go directly into a specific spot.
3. Once in the current location run this code in the terminal: ```streamlit run MLstreamlit.py```
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

## Logistic Regression Overview 
More background is given on the app, but here is an overview: 
Logistic regression estimates the probability that an input is within a certain class in a simple manner for binary classification problems. 

**Hyperparameters**:
- Inverse Regularization Strength
 - Controls complexity of model
- Max iterations
 - Number of trainings the model performs 
- Test set size 
 - Percentage of dataset used to train the model 
 
**Example of a logistic regression made in the app**:
As an example, I uploaded the example social networks dataset to the app and selected decision tree with these hyperparameters:
- Target column = Purchased
- Feature columns = Age, Gender
- Inverse Regularization Strength = 4.69
- Max iterations = 185
- Test set size (%) = 30

### Results: 
### **Accuracy Score**
<img width="196" alt="Screenshot 2025-04-14 at 10 27 43 AM" src="https://github.com/user-attachments/assets/ab885ae3-5cdf-4728-96b3-74c23654871e" />
<br /> The model predicts the correct value 88% of the time. 

### **Confusion Matrix**
<img width="300" alt="Screenshot 2025-04-14 at 10 27 50 AM" src="https://github.com/user-attachments/assets/19898a2e-1a59-49fb-ad5a-71f0a3673ffd" />
<br /> This is a way to see where the model is failing or doing well at predicting positive and negative values. 

### **Classification report**
<img width="300" alt="Screenshot 2025-04-14 at 10 27 55 AM" src="https://github.com/user-attachments/assets/79358305-986f-4394-97ea-913ac4a27e19" />
<br /> A classification report provides another way of looking at how well the model is doing at classifying the correct values and in what ways is it failing. You can optimize these values depending on what you want to emphasize. 

### **ROC Curve and AUC Score**
<img width="300" alt="Screenshot 2025-04-14 at 10 28 20 AM" src="https://github.com/user-attachments/assets/fbdb3357-c582-4394-9411-f88f3be8bf56" />
<br /> An ROC curve compares the true positive rate to the false positive rate and shows how well it is doing compared to a random guess. The AUC score is the area under the ROC curve to quantify its performance. 

## Decision Tree Overview 
More background is given on the app, but here is an overview: 
Decision trees are powerful machine learning models that create rules to classify target features in a simplistic manner, but they quickly can become complex and overfit.

**Hyperparameters**:
- Max depth
 - How many levels the tree will go in depth
- Min samples split
 - Minimum number of samples to cause a split to occur
- Criterion
 - Either gini (quantifies likelihood of sample being labeled incorrectly) or entropy (amount of uncertainty in a dataset)
- Test set size 
 - Percentage of dataset used to train the model 
 
**Example of a decision tree made in the app**:

As an example, I uploaded the diabetes dataset to the app and selected decision tree with these hyperparameters:
- Target column = Outcome
- Feature columns = Glucose, insulin, age
- Max depth = 5
- Min samples split = 2
- Criterion = gini
- Test set size (%) = 30

### Results: 
### **Accuracy Score**
<img width="223" alt="Screenshot 2025-04-14 at 10 12 01 AM" src="https://github.com/user-attachments/assets/627fabfc-9301-4c39-9a70-2d740309346c" />
<br /> The model predicts the correct value 73% of the time. 

### **Confusion Matrix**
<img width="300" alt="Screenshot 2025-04-14 at 10 12 07 AM" src="https://github.com/user-attachments/assets/aa55274e-f79e-4441-9353-5772af6085a2" />
<br /> This is a way to see where the model is failing or doing well at predicting positive and negative values. 

### **Classification report**
<img width="300" alt="Screenshot 2025-04-14 at 10 14 03 AM" src="https://github.com/user-attachments/assets/56898c5b-f025-42a8-8228-94921a7afd6b" />
<br /> A classification report provides another way of looking at how well the model is doing at classifying the correct values and in what ways is it failing. You can optimize these values depending on what you want to emphasize. 
