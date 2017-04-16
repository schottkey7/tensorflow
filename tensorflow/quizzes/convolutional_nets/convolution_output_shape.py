

def get_output_shape(inpts, fs, s, p, k):
    """Calculates the output shape of a convolution

    :inpts input shape (h, w, d)
    :fs filter shape (h, w, d )
    :s stride (h, w)
    :p padding (h, w)
    :k number of filters
    """
    new_height = (inpts[0] - fs[0] + 2 * p[0]) / s[0] + 1
    new_width = (inpts[1] - fs[1] + 2 * p[1]) / s[1] + 1
    return (new_height, new_width, k)


# This would correspond to the following code:

inp = tf.placeholder(tf.float32, (None, 32, 32, 3))
# (height, width, input_depth, output_depth)
filter_weights = tf.Variable(tf.truncated_normal((8, 8, 3, 20)))
filter_bias = tf.Variable(tf.zeros(20))
strides = [1, 2, 2, 1]  # (batch, height, width, depth)
padding = 'SAME'
conv = tf.nn.conv2d(inp, filter_weights, strides, padding) + filter_bias

#
# In summary TensorFlow uses the following equation for 'SAME' vs 'PADDING'
#
# SAME Padding, the output height and width are computed as:
#
# out_height = ceil(float(in_height) / float(strides1))
#
# out_width = ceil(float(in_width) / float(strides[2]))
#
# VALID Padding, the output height and width are computed as:
#
# out_height = ceil(float(in_height - filter_height + 1) / float(strides1))
#
# out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
