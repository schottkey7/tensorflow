import numpy as np


def sigmoid(x):
    """Calculate sigmoid"""
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    import ipdb; ipdb.set_trace()
    
    """Derivative of the sigmoid function"""
    return sigmoid(x) * (1 - sigmoid(x))


learnrate = 0.5
x = np.array([1, 2, 3, 4])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5, 0.3, 0.1])

# Calculate one gradient descent step for each weight
# Note: Some steps have been consilated, so there are
# fewer variable names than in the above sample code

# Calculate the node's linear combination of inputs and weights)
h = np.dot(x, w)

# Calculate output of neural network
nn_output = sigmoid(h)

# Calculate error of neural network
error = y - nn_output

# output gradient (f'(h))
output_grad = sigmoid_prime(h)

# error term (lowercase delta)
error_term = error * output_grad

# Calculate change in weights
del_w = learnrate * error_term * x

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)
