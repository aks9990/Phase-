# Enhancing Road Safety with AI-Driven Traffic Accident Analysis and Prediction
Step 1: Importing Libraries and Loading Data
# ==== Step 1: Upload CSV in Google Colab ====
from google.colab import files
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

uploaded = files.upload()
filename = next(iter(uploaded))
df = pd.read_csv(io.BytesIO(uploaded[filename]))
Use code with caution
Importing Libraries: This section imports necessary libraries for data manipulation, visualization, and machine learning. Libraries like pandas, numpy, sklearn, matplotlib, and seaborn are essential tools for data science tasks.
Loading Data: This code is specifically designed for Google Colab. It uses the files.upload() function to allow the user to upload a CSV file. The file is then read into a pandas DataFrame called df using pd.read_csv.
Step 2: Data Preprocessing - Encoding Categorical Features
# ==== Step 2: Encode Categorical Features ====
target_column = df.columns[-1]  # Use last column as target
print(f"\nUsing '{target_column}' as the target column")

le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        if df[column].nunique() <= 100:
            df[column] = le.fit_transform(df[column])
        else:
            print(f"⚠️ Skipping encoding for column '{column}' with too many unique values.")
Use code with caution
Identifying the Target Variable: The code identifies the last column of the DataFrame (df) as the target_column, which is the variable the model will try to predict.
Encoding Categorical Features: Machine learning models typically work with numerical data. This section uses LabelEncoder to convert categorical features (columns with text data) into numerical representations. If a categorical column has more than 100 unique values, it skips encoding to avoid creating too many new features.
Step 3: Splitting Data into Training and Testing Sets
# ==== Step 3: Train/Test Split ====
X = df.drop(target_column, axis=1)
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Use code with caution
Creating Feature and Target Variables: X stores the features (all columns except the target column), and y stores the target variable.
Splitting the Data: train_test_split divides the data into training and testing sets. 80% of the data (test_size=0.2) is used for training the model (X_train, y_train), and 20% is reserved for testing its performance (X_test, y_test). random_state=42 ensures consistent splitting for reproducibility.
Step 4: Model Training and Evaluation Function
# ==== Step 4: Model Training & Evaluation Function ====
def train_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        plt.figure(figsize=(8, 5))
        plt.barh(np.array(X.columns)[sorted_idx], feature_importance[sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name} Feature Importance')
        plt.tight_layout()
        plt.show()

    return model, accuracy_score(y_test, y_pred)
Use code with caution
train_evaluate Function: This function takes a machine learning model, training and testing data, and the model's name as input.
Training the Model: It trains the model using the training data (model.fit).
Making Predictions: It uses the trained model to predict on the testing data (model.predict).
Evaluating Performance: It prints a classification report, displays a confusion matrix, and (if applicable) shows a plot of feature importances. It returns the trained model and its accuracy score.
Step 5: Training and Comparing Models
# ==== Step 5: Train and Compare Models ====
dt_model, dt_acc = train_evaluate(DecisionTreeClassifier(random_state=42), X_train, X_test, y_train, y_test, "Decision Tree")
rf_model, rf_acc = train_evaluate(RandomForestClassifier(n_estimators=100, random_state=42), X_train, X_test, y_train, y_test, "Random Forest")
Use code with caution
Training Models: This section trains two models: a DecisionTreeClassifier and a RandomForestClassifier.
Evaluating and Storing Results: It calls the train_evaluate function to train and evaluate each model. The trained models and their accuracy scores are stored in variables (e.g., dt_model, dt_acc).
Step 6: Summary and Conclusions
# ==== Step 6: Summary ====
print("\nDescriptive Conclusions")
print("1. Model Performances:")
print(f"   - Decision Tree Accuracy: {dt_acc:.2%}")
print(f"   - Random Forest Accuracy: {rf_acc:.2%}")

print("\n2. Model Comparison:")
if rf_acc > dt_acc:
    print("   - Random Forest performed better, capturing more complex patterns.")
elif rf_acc < dt_acc:
    print("   - Decision Tree performed better, possibly due to model simplicity or overfitting.")
else:
    print("   - Both models performed equally.")

print("\n3. Key Findings:")
if hasattr(rf_model, 'feature_importances_'):
    most_important_feature = X.columns[np.argmax(rf_model.feature_importances_)]
    print(f"   - Most important feature: **{most_important_feature}**")
else:
    print("   - Feature importance not available.")
Use code with caution
Printing Results: This section prints a summary of the models' performances, including their accuracy scores.
Model Comparison: It compares the accuracy of the two models and provides a basic interpretation of which one performed better.
Key Findings: It identifies and prints the most important feature (if available) based on the Random Forest model's feature importances.
