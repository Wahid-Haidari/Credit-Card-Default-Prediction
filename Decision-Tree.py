import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  
        self.threshold = threshold  
        self.left = left
        self.right = right
        self.value = value  

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        for idx in range(self.n_features):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(value=predicted_class)
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node = Node(feature=idx, threshold=thr)
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree
        while node.left:
            if inputs[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

# Define max depth
max_depth = 3

# Load the credit default dataset as a pandas DataFrame from the provided URL
data_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
df = pd.read_excel(data_path, header=1)

# Drop the 'ID' column from the dataframe
df = df.drop('ID', axis=1)

df = df.drop('SEX', axis=1)
df = df.drop('EDUCATION', axis=1)
df = df.drop('MARRIAGE', axis=1)
df = df.drop('BILL_AMT1', axis=1)




#df = df.drop('PAY_2', axis=1)

# Split data into X and y
X = df.drop(['default payment next month'], axis=1).values
y = df['default payment next month'].values


# Split data into training, validation, and testing sets
train_size = int(0.7 * X.shape[0])
val_size = int(0.15 * X.shape[0])
test_size = X.shape[0] - train_size - val_size

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# Train decision tree on training set
tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)

# Predict on validation set
y_val_pred = tree.predict(X_val)

# Evaluate performance on validation set
val_acc = sum(y_val_pred == y_val) / len(y_val)
print('Validation accuracy:', val_acc)

# Predict on testing set
y_test_pred = tree.predict(X_test)

# Evaluate performance on testing set
test_acc = sum(y_test_pred == y_test) / len(y_test)
print('Testing accuracy:', test_acc)

# Initialize array of training set sizes
train_sizes = [1000, 3000, 5000, 7000, 9000, 11000, 13000]

# Initialize array to store validation accuracies
val_accs = []

# Train and validate model for each training set size
for train_size in train_sizes:
    # Get subset of training data
    X_train_subset = X_train[:train_size]
    y_train_subset = y_train[:train_size]
    
    # Train model
    model = DecisionTree(max_depth=max_depth)
    model.fit(X_train_subset, y_train_subset)
    
    # Validate model on validation set
    y_val_pred = model.predict(X_val)
    val_acc = np.mean(y_val_pred == y_val)
    val_accs.append(val_acc)

# Plot learning curve
plt.plot(train_sizes, val_accs)
plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Validation Accuracy")
plt.show()


# In[ ]:




