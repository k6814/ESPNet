import tensorflow as tf


def conv_layer(ip, number_of_filters, kernel, stride, layer_name="conv"):
    """
    Convolution layer

    Args:
        ip: Input
        number_of_filters: Number of filters
        kernel: Kernel Size
        stride: Stride for down-sampling
        layer_name:optional parameter to specify layer name
        """
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=ip, use_bias=False, filters=number_of_filters, kernel_size=kernel,
                                   strides=stride, padding="SAME",
                                   kernel_initializer=tf.random_normal_initializer(0, 0.02))
    return network


def dilation_conv_layer(ip, number_of_filters, kernel, stride, dilation_rate, layer_name="conv"):
    """
    Dilation Convolution Layer

    Args:
        ip: Input
        number_of_filters: Number of filters
        kernel: Kernel Size
        stride: Stride for down-sampling
        dilation_rate: dilation rate
        layer_name:optional parameter to specify layer name
    """
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=ip, use_bias=False, filters=number_of_filters, kernel_size=kernel,
                                   strides=stride, padding="SAME", dilation_rate=dilation_rate,
                                   kernel_initializer=tf.random_normal_initializer(0, 0.02))
    return network


def prelu(x, scope=None):
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        alpha = tf.get_variable("prelu", shape=x.get_shape()[-1],
                                dtype=x.dtype, initializer=tf.constant_initializer(0.1))
    return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)


def BN_PRelu(out):
    """
    Does Batch Normalization followed by PReLU.
    """
    batch_conv = tf.contrib.layers.batch_norm(out)
    prelu_batch_norm = prelu(batch_conv)
    return prelu_batch_norm


def conv_one_cross_one(ip, number_of_classes, layer_name="FINAL"):
    """
    Performs 1X1 concolution to project high-dimensional feature maps onto a low-dimensional space.
    """
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=ip, use_bias=False, filters=number_of_classes, kernel_size=[1, 1],
                                   strides=1, padding="SAME", kernel_initializer=tf.random_normal_initializer(0, 0.02))
    return network


def esp(ip, n_out):
    """
    ESP module based  on  principle of:-
    Reduce -> Split -> Transform -> Merge

    Args:
        ip: Input
        n_out: number of output channels
    """
    number_of_branches = 5
    n = int(n_out / number_of_branches)
    n1 = n_out - (number_of_branches - 1) * n

    # Reduce
    output1 = conv_layer(ip, number_of_filters=n, kernel=[3, 3], stride=1)

    # Split and Transform
    dilated_conv1 = dilation_conv_layer(output1, number_of_filters=n1, kernel=[3, 3], stride=1, dilation_rate=(1, 1))
    dilated_conv2 = dilation_conv_layer(output1, number_of_filters=n, kernel=[3, 3], stride=1, dilation_rate=(2, 2))
    dilated_conv4 = dilation_conv_layer(output1, number_of_filters=n, kernel=[3, 3], stride=1, dilation_rate=(4, 4))
    dilated_conv8 = dilation_conv_layer(output1, number_of_filters=n, kernel=[3, 3], stride=1, dilation_rate=(8, 8))
    d16 = dilation_conv_layer(output1, number_of_filters=n, kernel=[3, 3], stride=1, dilation_rate=(16, 16))
    add1 = dilated_conv2
    add2 = tf.add(add1, dilated_conv4)
    add3 = tf.add(add2, dilated_conv8)
    add4 = tf.add(add3, d16)

    # Merge
    concat = tf.concat(
        (dilated_conv1, add1, add2, add3, add4),
        axis=-1)
    concat = BN_PRelu(concat)
    return concat


def esp_alpha(ip, n_out):
    """
    ESP-alpha module where alpha controls depth of network.

    Args:
        ip: Input
        n_out: number of output channels
    """
    number_of_branches = 5
    n = int(n_out / number_of_branches)
    n1 = n_out - (number_of_branches - 1) * n
    output1 = conv_layer(ip, number_of_filters=n, kernel=[1, 1], stride=1)
    dilated_conv1 = dilation_conv_layer(output1, number_of_filters=n1, kernel=[3, 3], stride=1, dilation_rate=(1, 1))
    dilated_conv2 = dilation_conv_layer(output1, number_of_filters=n, kernel=[3, 3], stride=1, dilation_rate=(2, 2))
    dilated_conv4 = dilation_conv_layer(output1, number_of_filters=n, kernel=[3, 3], stride=1, dilation_rate=(4, 4))
    dilated_conv8 = dilation_conv_layer(output1, number_of_filters=n, kernel=[3, 3], stride=1, dilation_rate=(8, 8))
    dilated_conv16 = dilation_conv_layer(output1, number_of_filters=n, kernel=[3, 3], stride=1, dilation_rate=(16, 16))
    add1 = dilated_conv2
    add2 = tf.add(add1, dilated_conv4)
    add3 = tf.add(add2, dilated_conv8)
    add4 = tf.add(add3, dilated_conv16)
    concat = tf.concat(
        (dilated_conv1, add1, add2, add3, add4),
        axis=-1)
    concat = BN_PRelu(concat)
    return concat


def espnet(x, model):
    """
    ESPNet model architecture

    Args:
        x: Input
        model: Name of the model architecture which either "espnet_c" or "espnet"
    """
    conv_output = conv_layer(x, number_of_filters=16, kernel=[3, 3], stride=1, layer_name="first_layer_conv1")
    prelu_ = BN_PRelu(conv_output)
    avg_pooling = tf.layers.average_pooling2d(x, 3, 1, padding='same', data_format='channels_last', name=None)
    concat1 = tf.concat((avg_pooling, prelu_), axis=-1, name='concat_avg_prerelu')
    concat1 = BN_PRelu(concat1)
    esp_1 = esp(concat1, 64)
    esp_1 = BN_PRelu(esp_1)
    esp_alpha_1 = esp_1
    alpha1 = 2
    alpha2 = 8
    for i in range(alpha1):
        esp_alpha_1 = esp_alpha(esp_alpha_1, 64)
    concat2 = tf.concat((esp_alpha_1, esp_1, avg_pooling), axis=-1)
    esp_2 = esp(concat2, 128)
    esp_alpha_2 = esp_2
    for i in range(alpha2):
        esp_alpha_2 = esp_alpha(esp_alpha_2, 128)
    concat3 = tf.concat((esp_alpha_2, esp_2), axis=-1)
    pred = conv_one_cross_one(concat3, 2)
    if model == "espnet_c":
        return pred
    else:
        deconv1 = tf.layers.conv2d_transpose(pred, 2, [2, 2], strides=(1, 1), padding='same')
        conv_1 = conv_one_cross_one(concat2, 2)
        concat4 = tf.concat((deconv1, conv_1), axis=-1)
        esp_3 = esp(concat4, 2)
        deconv2 = tf.layers.conv2d_transpose(esp_3, 2, [2, 2], strides=(1, 1), padding='same')
        conv_2 = conv_one_cross_one(concat1, 2)
        concat5 = tf.concat((deconv2, conv_2), axis=-1)
        conv_3 = conv_one_cross_one(concat5, 2)
        deconv3 = tf.layers.conv2d_transpose(conv_3, 2, [2, 2], strides=(1, 1), padding='same')
        return deconv3
