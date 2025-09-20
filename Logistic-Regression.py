import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Initialize weights and bias to zeros
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent optimization
        for i in range(self.n_iters):
            # Linear combination of features and weights
            linear_model = np.dot(X, self.weights) + self.bias
            
            # Sigmoid activation function
            y_pred = self._sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        # Linear combination of features and weights
        linear_model = np.dot(X, self.weights) + self.bias
        
        # Sigmoid activation function
        y_pred = self._sigmoid(linear_model)
        
        # Convert probabilities to binary predictions
        y_pred_class = np.where(y_pred > 0.5, 1, 0)
        return y_pred_class
    
    def _sigmoid(self, z):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-z))
    
    
# Load the credit default dataset as a pandas DataFrame from the provided URL
data_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
df = pd.read_excel(data_path, header=1)


# Handle missing values by filling them with appropriate values or removing rows with missing values
df = df.dropna()

# Split the dataset into features (X) and target variable (y)
X = df.drop('default payment next month', axis=1) # Features
y = df['default payment next month'] # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr_model.predict(X_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Calculate accuracy of the logistic regression model
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Generate learning curve data
train_sizes = np.linspace(0.1, 1.0, 10)
train_scores = []
test_scores = []

for train_size in train_sizes:
    # Set the training set size
    train_set_size = int(train_size * len(X_train))
    X_train_subset = X_train[:train_set_size]
    y_train_subset = y_train[:train_set_size]

    # Initialize and fit the logistic regression model
    lr_model = LogisticRegression()
    lr_model.fit(X_train_subset, y_train_subset)

    # Make predictions on the training and test sets
    y_train_pred = lr_model.predict(X_train_subset)
    y_test_pred = lr_model.predict(X_test)

    # Calculate accuracy on the training and test sets
    train_accuracy = accuracy_score(y_train_subset, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    train_scores.append(train_accuracy)
    test_scores.append(test_accuracy)

# Plot the learning curve
plt.plot(train_sizes, train_scores, label='Train')
plt.plot(train_sizes, test_scores, label='Test')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve For Logistic Regression')
plt.legend(loc='best')
plt.show()


# In[ ]:




