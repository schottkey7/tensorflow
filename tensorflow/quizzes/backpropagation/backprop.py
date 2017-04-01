import numpy as np


def sigmoid(x):
    """Calculate sigmoid"""
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

w_hidden_out = np.array([0.1, -0.3])


# Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
h_layer_out = sigmoid(hidden_layer_input)

output_layer_in = np.dot(h_layer_out, w_hidden_out)
output = sigmoid(output_layer_in)

# Backwards pass
# TODO: Calculate error
error = target - output

# TODO: Calculate error gradient for output layer
del_err_output = error * output * (1 - output)

# TODO: Calculate error gradient for hidden layer
del_err_hidden = np.dot(del_err_output, w_hidden_out) * \
    h_layer_out * (1 - h_layer_out)

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * del_err_output * h_layer_out

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = (learnrate * del_err_hidden) * x[:, None]

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
