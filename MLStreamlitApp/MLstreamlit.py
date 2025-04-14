import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------
# Application Information
# -----------------------------------------------
st.title("Exploring the Differences in a Decision Tree Classifier vs a Logistic Regression Classifier")

st.expander("About This Application", expanded=True).markdown("""
This interactive application demonstrates the differences in performance for your own CSV file using either a Decision Tree Classifier or a Logistic Regression Classifier. You can:
- **Upload your own data set** to see how how the model performane handles your own data styles.
- **Select the desired model** to explore how your data set is handled in either a Decision Tree Classifier or in a Logistic Regression Classifier.
- **Select the target and feature columns** to train the model that you are looking for.
- **Adjust the model settings** with sliders to see how different parameters affect the performance.
- **View the performance metrics** including accuracy, a confusion matrix, and a classification report.
        
#### How to Use This Application
1. Upload your dataset in CSV format on the slider on the left.
2. Select the target column and feature columns from the dropdowns.
3. Choose the model you want to use (Decision Tree or Logistic Regression).
4. Adjust the model settings using the sliders.
5. View the results on the bottom of the page
""")

# -----------------------------------------------
# Upload Data
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
# Target and feature selection
# -----------------------------------------------
st.subheader("Step 1: Select Target and Features")

with st.expander("Target and Features Explanation", expanded=True):
    st.markdown("""
#### What are the target and features?
- **Target**: The variable you want to predict. This is the overall outcome of your model. An example would be the target variable being the **cost of a house** based on bedrooms, bathrooms, square footage, etc. 
- **Features**: The variables used to predict the target. These are the inputs to your model. An example would be the features being the **bedrooms, bathrooms, square footage, etc.** of a house.
 #### What should you do next?
1. Select the target column of your data set from the dropdown menu.
2. Select the feature columns of your data set from the dropdown menu.""")

all_columns = df.columns.tolist()
target_col = st.selectbox("Select the Target Column", all_columns)
feature_cols = st.multiselect("Select Feature Columns", [col for col in all_columns if col != target_col])

if not feature_cols:
    st.warning("Please select at least one feature column to continue.")
    st.stop()

# -----------------------------------------------
# Preprocessing data 
# -----------------------------------------------
st.subheader("Step 2: Data Preprocessing")
with st.expander("Data Preprocessing Explanation", expanded=True):
    st.markdown("""
    #### What is data preprocessing?
    Data preprocessing is the process of transforming raw data into a format that is suitable for analysis. This includes:  
    - **Handling missing values**: Removing or imputing missing data.  
    - **Encoding categorical variables**: Converting categorical variables into numerical format.  
    - **Feature scaling**: Normalizing or standardizing features to ensure they are on the same scale.  

    #### What should you do next?
    1. Check the preview of the processed data to see if the data is ready for modeling.  
    2. If the data is not ready, you may need to make edits in the data set to ensure that the data is ready for modeling.
    """)

# Drop rows with missing values
df = df.dropna()

# Select X and y after column selection
X = df[feature_cols]
y = df[target_col]

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_cols:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Preview processed data 
with st.expander("Preview Processed Data"):
    st.write("Features (X):", X.head())
    st.write("Target (y):", y.head())


# -----------------------------------------------
# Model Selection
# -----------------------------------------------
st.sidebar.title("Model Settings")
model_choice = st.sidebar.selectbox("Choose a Model", ["Decision Tree", "Logistic Regression"])

# If logistic regression is selected, scale the data
scaler = None
if model_choice == "Logistic Regression":
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

# Placeholder for the model
model = None

# Model-specific settings
if model_choice == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
    criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion
    )

elif model_choice == "Logistic Regression":
    C = st.sidebar.slider("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0)
    max_iter = st.sidebar.slider("Max Iterations", 100, 500, 100)
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="liblinear",  
    )
#-------------------------------------------------
st.subheader("Step 3: Select your Model")
st.markdown(
    """Deciding which model to use is an important step in the machine learning process. ***Decision trees*** can handle many different types of data (including numerical or categorical) and makes predictions by building 'trees' to guide decision making. 
     ***Logistic regression*** is a model that predicts the likehood of a binary outcome based on predictor variables. Each of them have pros and cons when looking at a data set, so this application is meant to help you determine which one is more accurate and better for your dataset."""
            )
st.expander("Decision Tree Description", expanded=True).markdown("""
Decision Trees predicts the target variable by creating simple if-then-else decision rules to classify the features. 
**Pros:** They are simple and easy to visualize with little data preperation
**Cons:** They can become overly complex quickly and are prone to overfitting.
        
#### Hyperparameters:
- **Max Depth**: The maximum depth of the tree. This controls how many levels or how deep the tree will grow to. 
    - A deeper tree can capture more complex patterns but may also lead to overfitting.
- **Min Samples Split**: The minimum number of samples required to split an internal node. 
    - This prevents the model from creating splits that are too specific if not enough of the training data explains that split.
- **Criterion**: The function to measure the quality of a split.
    - **gini** is for the Gini impurity which quantifies the likelihood of a random sample being incorrectly labeled if it was randomly labeled according to the distribution of labels.
         -   A lower Gini impurity indicates less impurity.
    - **entropy** is the measure of the amount of disorder or uncertainty in the data.
        - A lower entropy indicates less disorder.
- **Test Set Size (%)**: The percentage of the dataset to be used for testing the model.
""")

st.expander("Logistic Regression Description", expanded=True).markdown("""
Logistic regression provides an estimate of the probability that a given input point belongs to a certain class.
**Pros:** It is simple to implement and interpret, and it works well for binary classification problems.
**Cons:** It assumes a linear relationship between the features and cannot be used on continuous data.
                                                                        
#### Hyperparameters:
- **Inverse Regularization Strength (C)**: A hyperparameter that controls the complexity of the model.
    - A smaller value of C leads to a simpler model, while a larger value allows for more complexity.
- **Max Iterations**: The maximum number of trainings the model performs.
    - A higher number of max iterations may allow the model to converge better. 
- **Test Set Size (%)**: The percentage of the dataset to be used for testing the model.
                                                                      
##### Note: You can only select logistic regression if you choose a **binary target variable** (0 or 1). 
""")



# -----------------------------------------------
# Train-Test Split & Model Training
# -----------------------------------------------
st.subheader("Step 4: Model Training and Evaluation")

# Split the data
test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 30, step=5) / 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------------------------
# Model Evaluation 
# -----------------------------------------------
st.markdown(
    """Evaluating the model and exploring the differences in how the two models work is an important step in the machine learning process. **Accuracy, confusion matrix, and a classification report** will be generated for both models.

##### If using decision tree:
A **decision tree** will be visualized, and a expander will appear with the decision tree.

##### If using logistic regression:
An **ROC curve** will be used to evaluate the model performance and a visual will be created in an expander that will appear.""")

# Accuracy
with st.expander("Accuracy", expanded=True):
    accuracy = accuracy_score(y_test, y_pred)
    st.markdown(f"**Accuracy Score:** `{accuracy:.2f}`")
    st.markdown("""**Accuracy Score meaning:** Ratio of correctly predicted values to the total values predicted. A ratio closer to 1 indicates a better model. """)

# Confusion Matrix
with st.expander("Confusion Matrix", expanded=True):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    st.markdown("""
**Confusion Matrix meaning:**  
The columns represent predicted values and the rows represent actual values.  
It shows a visual of where the model is getting confused by showing true negatives, false positives, false negatives, and true positives.  

- **True Positives**: Correctly predicted positive cases.  
- **True Negatives**: Correctly predicted negative cases.  
- **False Positives**: Predicted to be positive but is actually negative.  
- **False Negatives**: Predicted to be negative but is actually positive.
""")


# Classification Report
with st.expander("Classification Report", expanded=True):
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    st.markdown("""
**Classification Report meaning:** A summary of the precision, recall, and F1-score for each class.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
    - How accurate the model is at predicting true positives compared to all predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to the all observations in actual class.
    - How accurate the model is at predicting true positives compared to all actual positives.
- **F1-Score**: The weighted average of Precision and Recall.
    - Provides a balance between precision and recall.
- **Support**: The number of actual occurrences of the class in the specified dataset.
    - The number of data points in each class. 
""")


# -----------------------------------------------
# ROC or Decision Tree Visualization 
# -----------------------------------------------
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import plot_tree

if model_choice == "Logistic Regression" and len(np.unique(y_test)) == 2:
    with st.expander("ROC Curve & AUC Score", expanded=True):
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = roc_auc_score(y_test, y_probs)

        st.markdown(f"**AUC Score:** `{roc_auc:.2f}`")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="darkorange")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        st.markdown("""
**ROC Curve explanation:** Drawn by comparing the true positive rate (TPR) to the false positive rate (FPR) at every threashold. 
- A perfect model would have a TPR of 1.0 and a FPR of 0.0. 

**AUC Score explanation:** The area under the ROC curve. 
- A score of 1.0 indicates a perfect model, while a score of 0.5 indicates a random guess.
""")

elif model_choice == "Decision Tree":
    with st.expander("Decision Tree Visualization", expanded=True):
        fig, ax = plt.subplots(figsize=(16, 8))
        plot_tree(
            model,
            feature_names=X.columns,
            class_names=[str(cls) for cls in np.unique(y)],
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax
        )
        st.pyplot(fig)
        st.markdown("""
**Decision tree description:** Visualization of the decision tree model.
""")
