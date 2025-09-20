#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Load and preprocess the dataset
data = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls', header=1)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
# Split the training set into training and validation sets
print(0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Helper functions
def gini_index(y):
    counts = np.array(list(Counter(y).values()))
    return 1 - np.sum((counts / np.sum(counts)) ** 2)

def split_data(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

def find_best_split(X, y, n_features):
    best_gini = 1
    best_feature_index, best_threshold = None, None
    
    feature_indices = np.random.choice(X.shape[1], n_features, replace=False)
    for feature_index in feature_indices:
        unique_values = np.unique(X[:, feature_index])
        for threshold in unique_values:
            X_left, y_left, X_right, y_right = split_data(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
                
            gini_left, gini_right = gini_index(y_left), gini_index(y_right)
            weighted_gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)
            
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature_index = feature_index
                best_threshold = threshold
                
    return best_feature_index, best_threshold

# Decision tree functions
class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

def build_decision_tree(X, y, n_features, max_depth):
    if max_depth == 1 or len(set(y)) == 1:
        return TreeNode(value=Counter(y).most_common(1)[0][0])
    
    feature_index, threshold = find_best_split(X, y, n_features)
    X_left, y_left, X_right, y_right = split_data(X, y, feature_index, threshold)
    
    left_node = build_decision_tree(X_left, y_left, n_features, max_depth - 1)
    right_node = build_decision_tree(X_right, y_right, n_features, max_depth - 1)
    
    return TreeNode(feature_index, threshold, left_node, right_node)

def predict_sample(node, x):
    if node.is_leaf_node():
        return node.value
    
    if x[node.feature_index] <= node.threshold:
        return predict_sample(node.left, x)
    return predict_sample(node.right, x)

# Random Forest functions
class RandomForest:
    def __init__(self, n_trees, max_depth, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.n_features = X.shape[1] if self.n_features is None else self.n_features
        for _ in range(self.n_trees):
            tree_X, _, tree_y, _ = train_test_split(X, y, test_size=0.8, stratify=y)
            tree = build_decision_tree(tree_X, tree_y, self.n_features, self.max_depth)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([[predict_sample(tree, x) for x in X] for tree in self.trees])
        return np.array([Counter(col).most_common(1)[0][0] for col in predictions.T])


def plot_learning_curve(train_sizes, train_scores, validation_scores):
    plt.figure()
    plt.plot(train_sizes, train_scores, label='Training score')
    plt.plot(train_sizes, validation_scores, label='Validation score')
    plt.xlabel('Training set size')
    plt.ylabel('Accuracy')
    plt.title('Learning curve')
    plt.legend()
    plt.show()

def get_learning_curve_data(rf, X_train, y_train, X_val, y_val, train_sizes):
    train_scores = []
    validation_scores = []

    for size in train_sizes:
        X_train_subset = X_train[:size]
        y_train_subset = y_train[:size]

        rf.fit(X_train_subset, y_train_subset)

        y_train_pred = rf.predict(X_train_subset)
        train_score = accuracy_score(y_train_subset, y_train_pred)
        train_scores.append(train_score)

        y_val_pred = rf.predict(X_val)
        validation_score = accuracy_score(y_val, y_val_pred)
        validation_scores.append(validation_score)

    return train_scores, validation_scores
print(0)
# Train and evaluate the model with different training set sizes
rf = RandomForest(n_trees=7, max_depth=3)
rf.fit(X_train, y_train)
print(0)
train_sizes = (np.linspace(0.1, 1.0, 10) * len(y_train)).astype(int)
print(0)
train_scores, validation_scores = get_learning_curve_data(rf, X_train, y_train, X_val, y_val, train_sizes)
print(0)
# Plot the learning curve
plot_learning_curve(train_sizes, train_scores, validation_scores)
print(0)
# y_pred = rf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)


# In[ ]:




