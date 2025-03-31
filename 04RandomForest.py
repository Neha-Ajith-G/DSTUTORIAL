import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
# The Iris dataset contains 150 samples from three species of Iris flowers (Setosa, Versicolor, Virginica).
# Each sample has four features: sepal length, sepal width, petal length, and petal width.
# The goal is to classify the species based on these features.
data = load_iris()
X, y = data.data, data.target

# Split the dataset into training and testing sets
# 70% of the data is used for training, and 30% is used for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Parameters for the Random Forest
n_trees = 10  # Number of individual decision trees in the forest
random_state = 42  # Seed for reproducibility

# Train individual Decision Trees
# Each tree is trained on the full training data but with a different random state for diversity.
trees = []
for i in range(n_trees):
    tree = DecisionTreeClassifier(random_state=random_state + i)
    tree.fit(X_train, y_train)
    trees.append(tree)

# Ensemble predictions using averaging
# Each tree makes a prediction on the test data, and the results are collected.
predictions = []
for tree in trees:
    predictions.append(tree.predict(X_test))
predictions = np.array(predictions)

# Average the predictions from all trees and round the result to the nearest integer (for classification)
# This averaging approach can be seen as a form of soft voting.
avg_predictions = np.mean(predictions, axis=0)
final_predictions = np.round(avg_predictions).astype(int)

# Calculate accuracy
# The accuracy score is computed by comparing the final ensemble prediction to the actual labels.
accuracy = accuracy_score(y_test, final_predictions)
print(f"Accuracy of Random Forest (Avg Ensemble): {accuracy * 100:.2f}%")
