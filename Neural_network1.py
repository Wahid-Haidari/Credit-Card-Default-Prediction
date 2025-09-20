#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls', header=1)

X = df.drop('default payment next month', axis=1).values
y = df['default payment next month'].values.reshape(-1, 1)


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

# Define the derivative of the activation function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Shuffle the data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# Split the data
split_idx = int(0.8 * X.shape[0])  # 80% of data for training

X_train = X[:split_idx]
y_train = y[:split_idx]

X_test = X[split_idx:]
y_test = y[split_idx:]

def train_neural_network(X_train, y_train):
    inputLayerSize, hiddenLayerSize, outputLayerSize = X_train.shape[1], 5, 1
    weights_input_to_hidden = np.random.rand(inputLayerSize, hiddenLayerSize)
    weights_hidden_to_output = np.random.rand(hiddenLayerSize, outputLayerSize)

    for _ in range(10000):  # number of iterations
        # Forward propagation
        hidden_layer_input = np.dot(X_train, weights_input_to_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output)
        y_pred = sigmoid(output_layer_input)

        # Backward propagation
        error = y_train - y_pred
        d_y_pred = error * sigmoid_derivative(y_pred)

        error_hidden_layer = d_y_pred.dot(weights_hidden_to_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Updating Weights
        weights_hidden_to_output += hidden_layer_output.T.dot(d_y_pred)
        weights_input_to_hidden += X_train.T.dot(d_hidden_layer)

    return weights_input_to_hidden, weights_hidden_to_output

def cross_validation(X, y, k=5):
    fold_size = X.shape[0] // k
    losses = []

    for i in range(k):
        # Split the data into training and validation sets
        X_val = X[i*fold_size:(i+1)*fold_size]
        y_val = y[i*fold_size:(i+1)*fold_size]

        X_train = np.concatenate([X[:i*fold_size], X[(i+1)*fold_size:]])
        y_train = np.concatenate([y[:i*fold_size], y[(i+1)*fold_size:]])

        # Train the model
        weights_input_to_hidden, weights_hidden_to_output = train_neural_network(X_train, y_train)

        # Compute the validation predictions
        hidden_layer_input = np.dot(X_val, weights_input_to_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output)
        y_pred_val = sigmoid(output_layer_input)

        # Compute the validation loss (mean squared error)
        val_loss = np.mean(np.square(y_val - y_pred_val))
        losses.append(val_loss)

    return np.mean(losses), np.std(losses)

# Use the function
mean_loss, std_loss = cross_validation(X_train, y_train, k=5)
print(f'Cross-validation mean loss: {mean_loss}, standard deviation: {std_loss}')

# Now train the final model on all the training data
weights_input_to_hidden, weights_hidden_to_output = train_neural_network(X_train, y_train)

# Compute the test predictions
hidden_layer_input = np.dot(X_test, weights_input_to_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output)
y_pred_test = sigmoid(output_layer_input)

# Compute the test loss (mean squared error)
test_loss = np.mean(np.square(y_test - y_pred_test))
print(f'Test loss: {test_loss}')


# In[6]:


import pandas as pd
import scipy.stats as stats

data = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls', header=1)
print(data.head())

# Separate the data into two groups: younger people and older people
younger_group = data[data['AGE'] < 30]
older_group = data[data['AGE'] >= 30]

# Calculate the default rate for each group
younger_default_rate = younger_group['default payment next month'].mean()
older_default_rate = older_group['default payment next month'].mean()

# Perform a statistical test 
t_statistic, p_value = stats.ttest_ind(younger_group['default payment next month'], older_group['default payment next month'])

# Print the results
print("Default Rate for Younger People:", younger_default_rate)
print("Default Rate for Older People:", older_default_rate)
print("p-value:", p_value)


# In[ ]:




