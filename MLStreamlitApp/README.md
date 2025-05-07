# Machine Learning Streamlit App

## Project Overview

Learning how to navigate different machine learning models and the differences in hyperparameters can be a daunting task for someone with a new data set. It would be a lot of effort to edit the code each time you want to alter a minor difference, and then generate the corresponding plot. The purpose of this app is to see the differences in either using a decision tree classifier or a logistic regression classifier model for machine learning on your dataset. Then, you can change the hyperparameters and see the effects immediately to determine the best settings for your dataset. As you go through these steps, there are also definitions and guides in the app to help explain what is occurring in the model. 

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

If you want to access the app locally, do these steps on your VSCode terminal of the downloaded MLstreamlit.py file:
1. Download streamlit in the terminal by entering this code:
```pip install streamlit```
2. Ensure you are in the correct directory as your file in the terminal by using ```ls``` (only on Mac) to view in your current directory, ```cd foldername``` to go deeper into a folder or go directly into a specific spot.
3. Once in the current location run this code in the terminal: ```streamlit run MLstreamlit.py```
4. Then, a local webpage should pop up and you can work on the app there.
<br />

If you want to access the app through a webpage (does not require any downloading), click on this link: [MLStreamlit App](https://mlapp-powers.streamlit.app/)
<br />

From there, you just need a dataset to explore! There are two sample options below! **Download one and upload it to the sidebar in the app**, and from there the app will guide you through adjusting the hyperparameters or model type. 

## Codes and Resources Used

1.   **Logistic Regression Background and References**
   - Reference webpage: [ROC and AUC Background](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc).
   - Reference webpage: [Logistic Regression Overview](https://www.bogotobogo.com/python/scikit-learn/scikit-learn_logistic_regression.php).
   - Aided when writing out the definitions and guides for the logistic regression classifier model.
2.   **Decision Tree Background and References**
   - Reference webpage: [Decision Tree Guide](https://scikit-learn.org/stable/modules/tree.html).
   - Reference webpage: [Gini vs Entropy](https://quantdare.com/decision-trees-gini-vs-entropy/).
   - Provided insight into coding decision trees as well as the background on what the hyperparameters mean for explanations in the app. 
3.   **Necessary libraries that need to be imported**
   - Pandas
   - Numpy
   - Matplotlib.pyplot
   - Streamlit
   - Seaborn
   - From sklearn:
     - sklearn.model_selection
     - sklearn.linear_model
     - sklearn.tree
     - sklearn.metrics
     - sklearn.preprocessing

## Data

Any data set can be used and uploaded to the sidebar in the app. However, here are two data sets to explore that are designed for each type of model, so you can become comfortable prior to exploring a new dataset. 

**Logistic Regression Sample Data Set**
[logistic regression dataset-Social_Network_Ads.csv](https://github.com/user-attachments/files/19736490/logistic.regression.dataset-Social_Network_Ads.csv)

The dataset used in this project was sourced from [Logistic Regression Social Network Ads Dataset](https://github.com/sam16tyagi/Machine-Learning-techniques-in-python/tree/master). Ensure you download the dataset and upload it to the app. To make sure it operates correctly in the logistic regression model, a binary value has to be the target column. To do this initially, make "purchased" be the target column. Otherwise, an error will appear when trying to generate an ROC curve. 

**Decision Tree Sample Data Set**
[diabetes_dataset.csv](https://github.com/user-attachments/files/19736583/diabetes_dataset.csv)

The dataset used in this project was sourced from [Decision Tree Diabetes Sample Data Set](https://github.com/sam16tyagi/Machine-Learning-techniques-in-python/tree/master). Ensure you download the dataset and upload it to the app. An example for a target column would be outcome for this dataset. 

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

### **Decision Tree**
<img width="300" alt="Screenshot 2025-04-14 at 10 18 18 AM" src="https://github.com/user-attachments/assets/40bd1b7a-c512-46a1-99d4-b6a13b064f07" />
<br /> The visualization of the decision tree allows you to physically see how the tree is being made and how complex it is. 
