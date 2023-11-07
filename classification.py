import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Load your dataset into a pandas DataFrame 
data = pd.read_csv('project dataset test-classification.csv')

# Separate features (X) and target variable (y)
X = data.iloc[:, :-1].values  # Assuming the last column is the target variable
y = data.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers with their respective hyperparameters (you can modify these as needed)
classifiers = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    SVC()
]

# Initialize list to store confusion matrices for each classifier
confusion_matrices = []

# Train each classifier on the training data and evaluate on the test data
for classifier in classifiers:
    # Train classifier on the training data
    classifier.fit(X_train, y_train)
    
    # Predict on test set using trained model
    y_pred = classifier.predict(X_test)
    
    # Generate confusion matrix and store it in the list
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)

# Plot confusion matrices for each classifier
fig, axes = plt.subplots(1, len(classifiers), figsize=(15, 5))
for i, (classifier, cm) in enumerate(zip(classifiers, confusion_matrices)):
    axes[i].matshow(cm, cmap='Blues')
    axes[i].set_title(type(classifier).__name__)
    for j in range(cm.shape[0]):
        for k in range(cm.shape[1]):
            axes[i].text(k, j, cm[j, k], ha='center', va='center', color='red')
plt.show()
