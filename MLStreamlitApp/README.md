# Machine Learning Streamlit App

## Project Overview

Learning how to navigate different machine learning models and the differences in hyperparameters can be a daunting task for someone with a new data set. It would be a lot of effort to edit the code each time you want to alter a minor difference, and then generate the corresponding plot. The purpose of this app is to see the differences in either using a decision tree classifier or a logistic regression classifier model for machine learning on your dataset. Then, you can change the hyperparameters and see the effects immediately to determine the best settings for your dataset. As you go through these steps, there are also definitions and guides in the app to help explain what is occurring in the model. 

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation and Setup](#installation-and-setup)
3. [Codes and Resources Used](#codes-and-resources-used)
4. [Data](#data)
5. [Data Processing and Analysis](#data-processing-and-analysis)
6. [Results and Visualizations](#results-and-visualizations)
7. [Future Work](#future-work)

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

From there, you just need a dataset to explore! There are two sample options below!

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

The dataset used in this project was sourced from [Logistic Regression Social Network Ads Dataset](https://github.com/sam16tyagi/Machine-Learning-techniques-in-python/tree/master). Ensure you download the dataset and upload it to the app. To make sure it operates correctly in the logistic regression model, a binary value has to be the target column. To do this initially, make default be the target column. Otherwise, an error will appear when trying to generate an ROC curve. 

**Decision Tree Sample Data Set**
[diabetes_dataset.csv](https://github.com/user-attachments/files/19736583/diabetes_dataset.csv)

The dataset used in this project was sourced from [Decision Tree Diabetes Sample Data Set](https://github.com/sam16tyagi/Machine-Learning-techniques-in-python/tree/master). Ensure you download the dataset and upload it to the app. An example for a target column would be outcome for this dataset. 

## Data Processing and Analysis

1.   **Tidying the data**
   - Used the pandas library. 
   - Followed basic tidy data commands such as df.melt, df.dropna(), df.str.split, df.str.replace, and df.rename.
2.   **Generating a stacked barplot**
   - Used From Data to Viz as a starting point.
   - Used numpy, matplotlib.pyplot, matplotlib, and pandas libraries.
   - Created a barplot that separates female and male athletes and depicts the number of medals awarded across the three types: gold, silver, and bronze.  
3.   **Creating circular barplots**
   - Used From Data to Viz as a starting point.
   - Used pandas, matplotlib.pyplot, and numpy libraries.
   - Created two circular barplots, one for male and one for female participants, and presented each event and the number of awarded medals.
4.   **Creating pivot tables**
   - Used pandas library.
   - Made two pivot tables.
   - One counts the total number of medals for both genders in each event.
   - The second shows the distribution of those medals in each event by bronze, gold, and silver.

## Results and Visualizations
### **Stacked barplot**
<img width="341" alt="Screenshot 2025-03-21 at 1 16 17 PM" src="https://github.com/user-attachments/assets/80d54099-2e65-43f8-a3c8-cbd6d38dd23b" />
<br /> Showed that more medals are awarded to male participants than female participants overall. 

### **Circular barplots**
#### Male: 
<img width="502" alt="Screenshot 2025-03-21 at 1 16 28 PM" src="https://github.com/user-attachments/assets/01d6813b-bf9b-4924-8da7-01ece3fe0ca5" />

#### Female: 
<img width="509" alt="Screenshot 2025-03-21 at 1 16 34 PM" src="https://github.com/user-attachments/assets/b38ca0be-1218-49d1-aac5-e3e018501911" />

<br /> These two graphs provide insight individually by demonstrating what events are offered and how many athletes participate by gender. Then, one can compare both of them to see the differences on what is offered by gender in the 2008 Olympics. 

### **Pivot tables**
#### Total medal count for both genders:
<img width="169" alt="Screenshot 2025-03-21 at 1 19 39 PM" src="https://github.com/user-attachments/assets/23605407-579f-41d9-9d5b-ddb775d9d86a" />

#### Total medal count for both genders broken down by medal type: 
<img width="265" alt="Screenshot 2025-03-21 at 1 19 45 PM" src="https://github.com/user-attachments/assets/08c622df-9e2e-4471-a02d-6db03f747d34" />

<br /> These tables are a way to visualize and understand how different events have a different number of awarded medals for the overall games, not differentiated by gender. 

## Future Work

- Explore why certain events have differing amounts of medals awarded.
- Analyze past and future Olympic games to see how these medal counts have changed. 
- Apply Tidy Data principles to more expansive and complex data sets. 
