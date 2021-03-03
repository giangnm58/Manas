import ast
from abc import abstractmethod

from autokeras.constant import Constant
from autokeras.nn.graph import Graph
from autokeras.nn.layers import StubAdd, StubDense, StubReLU, get_conv_class, get_dropout_class, \
    get_global_avg_pooling_class, get_pooling_class, get_avg_pooling_class, get_batch_norm_class, StubDropout1d, \
    StubConcatenate, StubSoftmax, StubFlatten


class NetworkGenerator:
    """The base class for generating a network.
    It can be used to generate a CNN or Multi-Layer Perceptron.
    Attributes:
        n_output_node: Number of output nodes in the network.
        input_shape: A tuple to represent the input shape.
    """

    def __init__(self, n_output_node, input_shape):
        """Initialize the instance.
        Sets the parameters `n_output_node` and `input_shape` for the instance.
        Args:
            n_output_node: An integer. Number of output nodes in the network.
            input_shape: A tuple. Input shape of the network.
        """
        self.n_output_node = n_output_node
        self.input_shape = input_shape

    @abstractmethod
    def generate(self, model_len, model_width):
        pass


class CnnGenerator(NetworkGenerator):
    """A class to generate CNN.
    Attributes:
          n_dim: `len(self.input_shape) - 1`
          conv: A class that represents `(n_dim-1)` dimensional convolution.
          dropout: A class that represents `(n_dim-1)` dimensional dropout.
          global_avg_pooling: A class that represents `(n_dim-1)` dimensional Global Average Pooling.
          pooling: A class that represents `(n_dim-1)` dimensional pooling.
          batch_norm: A class that represents `(n_dim-1)` dimensional batch normalization.
    """

    def __init__(self, n_output_node, input_shape):
        """Initialize the instance.
        Args:
            n_output_node: An integer. Number of output nodes in the network.
            input_shape: A tuple. Input shape of the network.
        """
        super(CnnGenerator, self).__init__(n_output_node, input_shape)
        self.n_dim = len(self.input_shape) - 1
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        if len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')
        print(self.n_dim)
        self.conv = get_conv_class(self.n_dim)
        self.dropout = get_dropout_class(self.n_dim)
        self.global_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.pooling = get_pooling_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)

    def generate(self, model_len=None, model_width=None):
        """Generates a CNN.
        Args:
            model_len: An integer. Number of convolutional layers.
            model_width: An integer. Number of filters for the convolutional layers.
        Returns:
            An instance of the class Graph. Represents the neural architecture graph of the generated model.
        """
        if model_len is None:
            model_len = Constant.MODEL_LEN
        if model_width is None:
            model_width = Constant.MODEL_WIDTH
        pooling_len = int(model_len / 4)
        graph = Graph(self.input_shape, False)
        temp_input_channel = self.input_shape[-1]
        output_node_id = 0
        stride = 1
        for i in range(model_len):
            output_node_id = graph.add_layer(StubReLU(), output_node_id)
            output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
            output_node_id = graph.add_layer(self.conv(temp_input_channel,
                                                       model_width,
                                                       kernel_size=3,
                                                       stride=stride), output_node_id)
            # if stride == 1:
            #     stride = 2
            temp_input_channel = model_width
            if pooling_len == 0 or ((i + 1) % pooling_len == 0 and i != model_len - 1):
                output_node_id = graph.add_layer(self.pooling(), output_node_id)

        output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
        output_node_id = graph.add_layer(self.dropout(Constant.CONV_DROPOUT_RATE), output_node_id)
        output_node_id = graph.add_layer(StubDense(graph.node_list[output_node_id].shape[0], model_width),
                                         output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        graph.add_layer(StubDense(model_width, self.n_output_node), output_node_id)
        return graph


class TestCnnGenerator1_fix(NetworkGenerator):

    def __init__(self, n_output_node, input_shape):

        super(TestCnnGenerator1_fix, self).__init__(n_output_node, input_shape)
        self.n_dim = len(self.input_shape) - 1
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        if len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')
        self.conv = get_conv_class(self.n_dim)
        self.dropout = get_dropout_class(self.n_dim)
        self.global_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.pooling = get_pooling_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)

    def generate(self, model_len=None, model_width=None):

        graph = Graph(self.input_shape, False)
        temp_input_channel = self.input_shape[-1]
        output_node_id = 0
        output_node_id = graph.add_layer(self.conv(3, 32, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        # output_node_id = graph.add_layer(self.dropout(0.01), output_node_id)

        output_node_id = graph.add_layer(self.conv(32, 32, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        # output_node_id = graph.add_layer(StubAdd(), (output_node_id, 4))
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        # output_node_id = graph.add_layer(self.dropout(0.01), output_node_id)

        output_node_id = graph.add_layer(self.conv(32, 64, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        # output_node_id = graph.add_layer(self.dropout(0.01), output_node_id)

        output_node_id = graph.add_layer(self.conv(64, 64, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        # output_node_id = graph.add_layer(StubAdd(), (output_node_id, 11))
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        # output_node_id = graph.add_layer(self.dropout(0.01), output_node_id)

        output_node_id = graph.add_layer(self.conv(64, 128, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        # output_node_id = graph.add_layer(self.dropout(0.01), output_node_id)

        output_node_id = graph.add_layer(self.conv(128, 128, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        # output_node_id = graph.add_layer(StubAdd(), (output_node_id, 18))
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        # output_node_id = graph.add_layer(self.dropout(0.01), output_node_id)

        output_node_id = graph.add_layer(self.conv(128, 256, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        # output_node_id = graph.add_layer(self.dropout(0.01), output_node_id)

        output_node_id = graph.add_layer(self.conv(256, 256, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        # output_node_id = graph.add_layer(StubAdd(), (output_node_id, 25))
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        # output_node_id = graph.add_layer(self.dropout(0.01), output_node_id)

        output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
        # output_node_id = graph.add_layer(StubFlatten(), output_node_id)
        # output_node_id = graph.add_layer(self.dropout(0.01), output_node_id)
        # output_node_id = graph.add_layer(StubReLU(), output_node_id)
        graph.add_layer(StubDense(256, 4), output_node_id)
        return graph


class TestCnnGenerator2_fix(NetworkGenerator):

    def __init__(self, n_output_node, input_shape):

        super(TestCnnGenerator2_fix, self).__init__(n_output_node, input_shape)
        self.n_dim = len(self.input_shape) - 1
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        if len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')
        self.conv = get_conv_class(self.n_dim)
        self.dropout = get_dropout_class(self.n_dim)
        self.global_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.pooling = get_pooling_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)

    def generate(self, model_len=None, model_width=None):

        graph = Graph(self.input_shape, False)
        temp_input_channel = self.input_shape[-1]
        output_node_id = 0

        output_node_id = graph.add_layer(self.conv(3, 64, kernel_size=11, stride=4), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        # output_node_id = graph.add_layer(self.dropout(0.01), output_node_id)

        output_node_id = graph.add_layer(self.conv(64, 192, kernel_size=5, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        # output_node_id = graph.add_layer(self.dropout(0.01), output_node_id)

        output_node_id = graph.add_layer(self.conv(192, 384, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        # output_node_id = graph.add_layer(self.dropout(0.01), output_node_id)

        output_node_id = graph.add_layer(self.conv(384, 256, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        # output_node_id = graph.add_layer(self.dropout(0.01), output_node_id)

        output_node_id = graph.add_layer(self.conv(256, 256, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        # output_node_id = graph.add_layer(StubAdd(), (15, output_node_id))
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        # output_node_id = graph.add_layer(self.dropout(0.01), output_node_id)

        output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
        # output_node_id = graph.add_layer(self.dropout(0.01), output_node_id)
        # output_node_id = graph.add_layer(StubSoftmax(), output_node_id)

        graph.add_layer(StubDense(256, 5), output_node_id)
        return graph


class TestCnnGenerator3_fix(NetworkGenerator):

    def __init__(self, n_output_node, input_shape):

        super(TestCnnGenerator3_fix, self).__init__(n_output_node, input_shape)
        self.n_dim = len(self.input_shape) - 1
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        if len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')
        self.conv = get_conv_class(self.n_dim)
        self.dropout = get_dropout_class(self.n_dim)
        self.global_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.pooling = get_pooling_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)

    def generate(self, model_len=None, model_width=None):

        graph = Graph(self.input_shape, False)
        temp_input_channel = self.input_shape[-1]
        output_node_id = 0
        output_node_id = graph.add_layer(self.conv(3, 32, kernel_size=7, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        output_node_id = graph.add_layer(self.dropout(0.15), output_node_id)

        output_node_id = graph.add_layer(self.conv(32, 64, kernel_size=5, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        output_node_id = graph.add_layer(self.dropout(0.15), output_node_id)

        output_node_id = graph.add_layer(self.conv(64, 128, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        output_node_id = graph.add_layer(self.dropout(0.15), output_node_id)

        output_node_id = graph.add_layer(self.conv(128, 128, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        # output_node_id = graph.add_layer(StubAdd(), (15, output_node_id))
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        output_node_id = graph.add_layer(self.dropout(0.15), output_node_id)

        output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(StubDense(128, 1000), output_node_id)
        output_node_id = graph.add_layer(StubSoftmax(), output_node_id)
        graph.add_layer(StubDense(1000, 5), output_node_id)
        return graph


class TestCnnGenerator4_fix(NetworkGenerator):

    def __init__(self, n_output_node, input_shape):

        super(TestCnnGenerator4_fix, self).__init__(n_output_node, input_shape)
        self.n_dim = len(self.input_shape) - 1
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        if len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')
        self.conv = get_conv_class(self.n_dim)
        self.dropout = get_dropout_class(self.n_dim)
        self.global_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.pooling = get_pooling_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)

    def generate(self, model_len=None, model_width=None):

        graph = Graph(self.input_shape, False)
        temp_input_channel = self.input_shape[-1]
        output_node_id = 0
        output_node_id = graph.add_layer(self.conv(3, 16, kernel_size=5, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)

        output_node_id = graph.add_layer(self.conv(16, 32, kernel_size=5, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)

        output_node_id = graph.add_layer(self.conv(32, 64, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)

        output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)

        output_node_id = graph.add_layer(StubDense(64, 14400), output_node_id)
        output_node_id = graph.add_layer(StubDense(14400, 512), output_node_id)
        output_node_id = graph.add_layer(StubDense(512, 128), output_node_id)
        output_node_id = graph.add_layer(self.dropout(0.2), output_node_id)

        graph.add_layer(StubDense(128, 5), output_node_id)
        return graph


class TestCnnGenerator8(NetworkGenerator):

    def __init__(self, n_output_node, input_shape):

        super(TestCnnGenerator8, self).__init__(n_output_node, input_shape)
        self.n_dim = len(self.input_shape) - 1
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        if len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')
        self.conv = get_conv_class(self.n_dim)
        self.dropout = get_dropout_class(self.n_dim)
        self.global_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.pooling = get_pooling_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)

    def generate(self, model_len=None, model_width=None):

        graph = Graph(self.input_shape, False)
        temp_input_channel = self.input_shape[-1]
        output_node_id = 0
        output_node_id = graph.add_layer(self.conv(3, 32, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)

        output_node_id = graph.add_layer(self.conv(32, 32, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.conv(32, 64, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)

        output_node_id = graph.add_layer(self.conv(64, 64, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.conv(64, 128, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)

        graph.add_layer(StubDense(1152, 2), output_node_id)
        return graph


import os
import pickle

from autokeras.utils import pickle_from_file


class Testretrain(NetworkGenerator):
    def __init__(self, n_output_node, input_shape):

        super(Testretrain, self).__init__(n_output_node, input_shape)
        self.n_dim = len(self.input_shape) - 1
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        if len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')
        self.conv = get_conv_class(self.n_dim)
        self.dropout = get_dropout_class(self.n_dim)
        self.global_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.pooling = get_pooling_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)

    def generate(self, model_len=None, model_width=None):

        graph = Graph(self.input_shape, False)
        temp_input_channel = self.input_shape[-1]
        output_node_id = 0
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        output_node_id = graph.add_layer(self.conv(3, 64, kernel_size=3, stride=1), output_node_id)

        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)

        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.conv(64, 64, kernel_size=1, stride=1), output_node_id)
        output_node_id = graph.add_layer(StubAdd(), (6, output_node_id))

        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)

        output_node_id = graph.add_layer(self.conv(64, 128, kernel_size=3, stride=1), output_node_id)

        output_node_id = graph.add_layer(self.pooling(), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
        output_node_id = graph.add_layer(self.conv(128, 64, kernel_size=3, stride=1), output_node_id)
        output_node_id = graph.add_layer(self.pooling(), output_node_id)

        output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)

        output_node_id = graph.add_layer(self.dropout(0.25), output_node_id)
        output_node_id = graph.add_layer(StubDense(64, 64), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        graph.add_layer(StubDense(64, 2), output_node_id)
        return graph


# 3499 2293


class MlpGenerator(NetworkGenerator):
    """A class to generate Multi-Layer Perceptron.
    """

    def __init__(self, n_output_node, input_shape):
        """Initialize the instance.
        Args:
            n_output_node: An integer. Number of output nodes in the network.
            input_shape: A tuple. Input shape of the network. If it is 1D, ensure the value is appended by a comma
                in the tuple.
        """
        super(MlpGenerator, self).__init__(n_output_node, input_shape)
        if len(self.input_shape) > 1:
            raise ValueError('The input dimension is too high.')

    def generate(self, model_len=None, model_width=None):
        """Generates a Multi-Layer Perceptron.
        Args:
            model_len: An integer. Number of hidden layers.
            model_width: An integer or a list of integers of length `model_len`. If it is a list, it represents the
                number of nodes in each hidden layer. If it is an integer, all hidden layers have nodes equal to this
                value.
        Returns:
            An instance of the class Graph. Represents the neural architecture graph of the generated model.
        """
        if model_len is None:
            model_len = Constant.MODEL_LEN
        if model_width is None:
            model_width = Constant.MODEL_WIDTH
        if isinstance(model_width, list) and not len(model_width) == model_len:
            raise ValueError('The length of \'model_width\' does not match \'model_len\'')
        elif isinstance(model_width, int):
            model_width = [model_width] * model_len

        graph = Graph(self.input_shape, False)
        output_node_id = 0
        n_nodes_prev_layer = self.input_shape[0]
        for width in model_width:
            output_node_id = graph.add_layer(StubDense(n_nodes_prev_layer, width), output_node_id)
            output_node_id = graph.add_layer(StubDropout1d(Constant.MLP_DROPOUT_RATE), output_node_id)
            output_node_id = graph.add_layer(StubReLU(), output_node_id)
            n_nodes_prev_layer = width

        graph.add_layer(StubDense(n_nodes_prev_layer, self.n_output_node), output_node_id)
        return graph


class ResNetGenerator(NetworkGenerator):
    def __init__(self, n_output_node, input_shape, layers=[2, 2, 2, 2], bottleneck=False):
        super(ResNetGenerator, self).__init__(n_output_node, input_shape)
        self.layers = layers
        self.in_planes = 64
        self.n_dim = len(self.input_shape) - 1
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        elif len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')
        self.conv = get_conv_class(self.n_dim)
        self.dropout = get_dropout_class(self.n_dim)
        self.global_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.adaptive_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)
        if bottleneck:
            self.make_block = self._make_bottleneck_block
            self.block_expansion = 4
        else:
            self.make_block = self._make_basic_block
            self.block_expansion = 1

    def generate(self, model_len=None, model_width=None):
        if model_width is None:
            model_width = Constant.MODEL_WIDTH
        graph = Graph(self.input_shape, False)
        temp_input_channel = self.input_shape[-1]
        output_node_id = 0
        output_node_id = graph.add_layer(self.conv(temp_input_channel, model_width, kernel_size=3), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(model_width), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        # output_node_id = graph.add_layer(self.pooling(kernel_size=3, stride=2, padding=1), output_node_id)

        output_node_id = self._make_layer(graph, model_width, self.layers[0], output_node_id, 1)
        model_width *= 2
        output_node_id = self._make_layer(graph, model_width, self.layers[1], output_node_id, 2)
        model_width *= 2
        output_node_id = self._make_layer(graph, model_width, self.layers[2], output_node_id, 2)
        model_width *= 2
        output_node_id = self._make_layer(graph, model_width, self.layers[3], output_node_id, 2)

        output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
        graph.add_layer(StubDense(model_width * self.block_expansion, self.n_output_node), output_node_id)
        return graph

    def _make_layer(self, graph, planes, blocks, node_id, stride):
        strides = [stride] + [1] * (blocks - 1)
        out = node_id
        for current_stride in strides:
            out = self.make_block(graph, self.in_planes, planes, out, current_stride)
            self.in_planes = planes * self.block_expansion
        return out

    def _make_basic_block(self, graph, in_planes, planes, node_id, stride=1):
        out = graph.add_layer(self.conv(in_planes, planes, kernel_size=3, stride=stride), node_id)
        out = graph.add_layer(self.batch_norm(planes), out)
        out = graph.add_layer(StubReLU(), out)
        out = graph.add_layer(self.conv(planes, planes, kernel_size=3), out)
        out = graph.add_layer(self.batch_norm(planes), out)

        residual_node_id = node_id

        if stride != 1 or in_planes != self.block_expansion * planes:
            residual_node_id = graph.add_layer(self.conv(in_planes,
                                                         planes * self.block_expansion,
                                                         kernel_size=1,
                                                         stride=stride), residual_node_id)
            residual_node_id = graph.add_layer(self.batch_norm(self.block_expansion * planes), residual_node_id)

        out = graph.add_layer(StubAdd(), (out, residual_node_id))
        out = graph.add_layer(StubReLU(), out)
        return out

    def _make_bottleneck_block(self, graph, in_planes, planes, node_id, stride=1):
        out = graph.add_layer(self.conv(in_planes, planes, kernel_size=1), node_id)
        out = graph.add_layer(self.batch_norm(planes), out)
        out = graph.add_layer(StubReLU(), out)
        out = graph.add_layer(self.conv(planes, planes, kernel_size=3, stride=stride), out)
        out = graph.add_layer(self.batch_norm(planes), out)
        out = graph.add_layer(StubReLU(), out)
        out = graph.add_layer(self.conv(planes, self.block_expansion * planes, kernel_size=1), out)
        out = graph.add_layer(self.batch_norm(self.block_expansion * planes), out)

        residual_node_id = node_id

        if stride != 1 or in_planes != self.block_expansion * planes:
            residual_node_id = graph.add_layer(self.conv(in_planes,
                                                         planes * self.block_expansion,
                                                         kernel_size=1,
                                                         stride=stride), residual_node_id)
            residual_node_id = graph.add_layer(self.batch_norm(self.block_expansion * planes), residual_node_id)

        out = graph.add_layer(StubAdd(), (out, residual_node_id))
        out = graph.add_layer(StubReLU(), out)
        return out


def ResNet18(n_output_node, input_shape):
    return ResNetGenerator(n_output_node, input_shape)


def ResNet34(n_output_node, input_shape):
    return ResNetGenerator(n_output_node, input_shape, [3, 4, 6, 3])


def ResNet50(n_output_node, input_shape):
    return ResNetGenerator(n_output_node, input_shape, [3, 4, 6, 3], bottleneck=True)


def ResNet101(n_output_node, input_shape):
    return ResNetGenerator(n_output_node, input_shape, [3, 4, 23, 3], bottleneck=True)


def ResNet152(n_output_node, input_shape):
    return ResNetGenerator(n_output_node, input_shape, [3, 8, 36, 3], bottleneck=True)


class DenseNetGenerator(NetworkGenerator):
    def __init__(self, n_output_node, input_shape, block_config=[6, 12, 24, 16], growth_rate=32):
        super().__init__(n_output_node, input_shape)
        # DenseNet Constant
        self.num_init_features = 64
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.bn_size = 4
        self.drop_rate = 0
        # Stub layers
        self.n_dim = len(self.input_shape) - 1
        self.conv = get_conv_class(self.n_dim)
        self.dropout = get_dropout_class(self.n_dim)
        self.global_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.adaptive_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.max_pooling = get_pooling_class(self.n_dim)
        self.avg_pooling = get_avg_pooling_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)

    def generate(self, model_len=None, model_width=None):
        if model_len is None:
            model_len = Constant.MODEL_LEN
        if model_width is None:
            model_width = Constant.MODEL_WIDTH
        graph = Graph(self.input_shape, False)
        temp_input_channel = self.input_shape[-1]
        # First convolution
        output_node_id = 0
        output_node_id = graph.add_layer(self.conv(temp_input_channel, model_width, kernel_size=7),
                                         output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(num_features=self.num_init_features), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        db_input_node_id = graph.add_layer(self.max_pooling(kernel_size=3, stride=2, padding=1), output_node_id)
        # Each Denseblock
        num_features = self.num_init_features
        for i, num_layers in enumerate(self.block_config):
            db_input_node_id = self._dense_block(num_layers=num_layers, num_input_features=num_features,
                                                 bn_size=self.bn_size, growth_rate=self.growth_rate,
                                                 drop_rate=self.drop_rate,
                                                 graph=graph, input_node_id=db_input_node_id)
            num_features = num_features + num_layers * self.growth_rate
            if i != len(self.block_config) - 1:
                db_input_node_id = self._transition(num_input_features=num_features,
                                                    num_output_features=num_features // 2,
                                                    graph=graph, input_node_id=db_input_node_id)
                num_features = num_features // 2
        # Final batch norm
        out = graph.add_layer(self.batch_norm(num_features), db_input_node_id)

        out = graph.add_layer(StubReLU(), out)
        out = graph.add_layer(self.adaptive_avg_pooling(), out)
        # Linear layer
        graph.add_layer(StubDense(num_features, self.n_output_node), out)
        return graph

    def _dense_block(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, graph, input_node_id):
        block_input_node = input_node_id
        for i in range(num_layers):
            block_input_node = self._dense_layer(num_input_features + i * growth_rate, growth_rate,
                                                 bn_size, drop_rate,
                                                 graph, block_input_node)
        return block_input_node

    def _dense_layer(self, num_input_features, growth_rate, bn_size, drop_rate, graph, input_node_id):
        out = graph.add_layer(self.batch_norm(num_features=num_input_features), input_node_id)
        out = graph.add_layer(StubReLU(), out)
        out = graph.add_layer(self.conv(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1), out)
        out = graph.add_layer(self.batch_norm(bn_size * growth_rate), out)
        out = graph.add_layer(StubReLU(), out)
        out = graph.add_layer(self.conv(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1), out)
        out = graph.add_layer(self.dropout(rate=drop_rate), out)

        out = graph.add_layer(StubConcatenate(), (input_node_id, out))
        return out

    def _transition(self, num_input_features, num_output_features, graph, input_node_id):
        out = graph.add_layer(self.batch_norm(num_features=num_input_features), input_node_id)
        out = graph.add_layer(StubReLU(), out)
        out = graph.add_layer(self.conv(num_input_features, num_output_features, kernel_size=1, stride=1), out)
        out = graph.add_layer(self.avg_pooling(kernel_size=2, stride=2), out)
        return out


def DenseNet121(n_output_node, input_shape):
    return DenseNetGenerator(n_output_node, input_shape)


def DenseNet169(n_output_node, input_shape):
    return DenseNetGenerator(n_output_node, input_shape, [6, 12, 32, 32])


def DenseNet201(n_output_node, input_shape):
    return DenseNetGenerator(n_output_node, input_shape, [6, 12, 48, 32])


def DenseNet161(n_output_node, input_shape):
    return DenseNetGenerator(n_output_node, input_shape, [6, 12, 36, 24], growth_rate=48)

'''
class mined_model(NetworkGenerator):
    def __init__(self, n_output_node, input_shape, link, trans = None):
        super(mined_model, self).__init__(n_output_node, input_shape)
        self.n_dim = len(self.input_shape) - 1
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        if len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')
        self.conv = get_conv_class(self.n_dim)
        self.dropout = get_dropout_class(self.n_dim)
        self.global_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.pooling = get_pooling_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)
        self.link = link
    def generate(self, model_len=None, model_width=None):
        CNN = ['conv1d', 'conv2d', 'conv3d', 'maxpool1d', 'maxpool2d', 'maxpool3d', 'avgpool1d', 'avgpool2d',
               'avgpool3d', 'prelu', 'relu', 'relu6', 'batchnorm1d', 'batchnorm2d', 'batchnorm3d', 'dropout',
               'dropout2d', 'dropout3d', 'softmax', 'softmax2d', 'relu', 'conv1d', 'conv2d', 'conv3d', 'avg_pool1d',
               'avg_pool2d', 'avg_pool3d', 'max_pool1d', 'max_pool2d', 'max_pool3d', 'batch_norm', 'normalize',
               'linear', 'dropout', 'dropout2d', 'dropout3d', 'Linear', 'elu', 'softmax', 'Module', 'relu6', 'flatten']
        layer_number = 0
        graph = Graph(self.input_shape, False)
        output_node_id = 0
        model = open(self.link, "r", encoding="ISO-8859-1")
        fullmodel = []
        count_linear = 0
        lastest_output = 0
        count_average = 0
        lastest_layer = None
        last_linear = 0
        first_conv = 0
        for layer in model:
            try:
                dictlayer = ast.literal_eval(layer)
                fullmodel.append(dictlayer)
                if dictlayer['func'].lower() in CNN:
                    #print(dictlayer['func'].lower())
                    layer_number += 1
                    if dictlayer['func'].lower() == 'linear':
                        last_linear = layer_number
            except SyntaxError:
                pass
        #print(last_linear,'xxxxxxxxxxx')
        for i in range(len(fullmodel)):
            first_conv += 1
            # print(fullmodel[i]['func'])
            if 'conv' in fullmodel[i]['func'].lower():
                input = list(fullmodel[i].values())[1]
                if lastest_output != input:
                    input = lastest_output
                out = list(fullmodel[i].values())[2]
                lastest_output = out
                kernel = list(fullmodel[i].values())[3][0]
                if 'strides' in fullmodel[i]:
                    if len(fullmodel[i]['strides']) == 0:
                        stride = 1
                    else:
                        stride = fullmodel[i]['strides'][0]
                else:
                    stride = 1
                if lastest_layer == 'conv':
                    output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                                                     output_node_id)
                    output_node_id = graph.add_layer(StubReLU(), output_node_id)
                if lastest_layer == 'batchnorm':
                    output_node_id = graph.add_layer(StubReLU(), output_node_id)
                if i == 0:
                    output_node_id = graph.add_layer(
                        self.conv(self.input_shape[-1], out, kernel_size=kernel, stride=stride),
                        output_node_id)
                    lastest_layer = 'conv'
                else:
                    output_node_id = graph.add_layer(self.conv(input, out, kernel_size=kernel, stride=stride),
                                                     output_node_id)
                    lastest_layer = 'conv'
            if 'dropout' in fullmodel[i]['func'].lower():
                #if count_average == 0:
                    #count_average += 1
                    #output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
                p_value = list(fullmodel[i].values())[1]
                if i == layer_number - 1:
                    graph.add_layer(self.dropout(p_value), output_node_id)
                    lastest_layer = 'dropout'
                else:
                    output_node_id = graph.add_layer(self.dropout(p_value), output_node_id)
                    lastest_layer = 'dropout'
            if 'batchnorm' in fullmodel[i]['func'].lower():
                if count_linear == 0:
                    output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                                                     output_node_id)
                    lastest_layer = 'batchnorm'
            if 'elu' in fullmodel[i]['func'].lower():
                if lastest_layer == 'conv':
                    output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                                                     output_node_id)
                if i == layer_number-1:
                    print('elu')
                    graph.add_layer(StubReLU(), output_node_id)
                    lastest_layer = 'elu'
                else:
                    output_node_id = graph.add_layer(StubReLU(), output_node_id)
                    lastest_layer = 'elu'
            if 'softmax' in fullmodel[i]['func'].lower():
                if lastest_layer == 'conv':
                    output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                                                     output_node_id)
                if i == layer_number-1:
                    graph.add_layer(StubSoftmax(), output_node_id)
                    lastest_layer = 'softmax'
                else:
                    output_node_id = graph.add_layer(StubSoftmax(), output_node_id)
                    lastest_layer = 'softmax'
            if 'maxpool' in fullmodel[i]['func'].lower():
                if lastest_layer == 'conv':
                    output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                                                     output_node_id)
                    output_node_id = graph.add_layer(StubReLU(), output_node_id)
                if lastest_layer == 'batchnorm':
                    output_node_id = graph.add_layer(StubReLU(), output_node_id)
                output_node_id = graph.add_layer(self.pooling(), output_node_id)
            if 'avgpool' in fullmodel[i]['func'].lower():
                if lastest_layer == 'conv':
                    output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                                                     output_node_id)
                    output_node_id = graph.add_layer(StubReLU(), output_node_id)
                if lastest_layer == 'batchnorm':
                    output_node_id = graph.add_layer(StubReLU(), output_node_id)
                count_average += 1
                output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
                lastest_layer = 'avgpool'
            if 'flatten' in fullmodel[i]['func'].lower():
                if lastest_layer == 'conv':
                    output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                                                     output_node_id)
                    output_node_id = graph.add_layer(StubReLU(), output_node_id)
                if lastest_layer == 'batchnorm':
                    output_node_id = graph.add_layer(StubReLU(), output_node_id)
                count_average += 1
                #output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
                output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
                lastest_layer = 'flatten'
            if 'linear' in fullmodel[i]['func'].lower():
                if count_average == 0:
                    if lastest_layer == 'conv':
                        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                                                         output_node_id)
                        output_node_id = graph.add_layer(StubReLU(), output_node_id)
                    if lastest_layer == 'batchnorm':
                        output_node_id = graph.add_layer(StubReLU(), output_node_id)
                    count_average += 1
                    output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
                count_linear += 1
                input = list(fullmodel[i].values())[1]
                if lastest_output != input:
                    input = lastest_output
                out = list(fullmodel[i].values())[2]
                lastest_output = out
                if i == layer_number-1:
                    print(i,'linear')
                    graph.add_layer(StubDense(input, self.n_output_node), output_node_id)
                    lastest_layer = 'linear'
                else:
                    if i == last_linear-1:
                        output_node_id = graph.add_layer(StubDense(input, self.n_output_node), output_node_id)
                        lastest_layer = 'linear'
                    else:
                        output_node_id = graph.add_layer(StubDense(input, out), output_node_id)
                        lastest_layer = 'linear'
        if count_linear == 0:
            if lastest_layer == 'conv':
                output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                                                 output_node_id)
                output_node_id = graph.add_layer(StubReLU(), output_node_id)
            if count_average == 0:
                count_average += 1
                output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
            graph.add_layer(StubDense(lastest_output, self.n_output_node), output_node_id)
        return graph
'''

class mined_model(NetworkGenerator):

    def __init__(self, n_output_node, input_shape, link, trans = 0):

        super(mined_model, self).__init__(n_output_node, input_shape)
        self.n_dim = len(self.input_shape) - 1
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        if len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')
        self.conv = get_conv_class(self.n_dim)
        self.dropout = get_dropout_class(self.n_dim)
        self.global_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.pooling = get_pooling_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)
        self.link = link
        self.trans = trans

    def generate(self, model_len=None, model_width=None):
        print(self.link,'aaaaaaaaaa')
        CNN = ['conv1d', 'conv2d', 'conv3d', 'maxpool1d', 'maxpool2d', 'maxpool3d', 'avgpool1d', 'avgpool2d',
               'avgpool3d', 'prelu', 'relu', 'relu6', 'batchnorm1d', 'batchnorm2d', 'batchnorm3d', 'dropout',
               'dropout2d', 'dropout3d', 'softmax', 'softmax2d', 'relu', 'conv1d', 'conv2d', 'conv3d', 'avg_pool1d',
               'avg_pool2d', 'avg_pool3d', 'max_pool1d', 'max_pool2d', 'max_pool3d', 'batch_norm', 'normalize',
               'linear', 'dropout', 'dropout2d', 'dropout3d', 'Linear', 'elu', 'tanh', 'softmax', 'sigmoid', 'Module', 'relu6', 'flatten']
        layer_number = 0
        graph = Graph(self.input_shape, False)
        output_node_id = 0
        model = open(self.link, "r", encoding="ISO-8859-1")
        fullmodel = []
        count_linear = 0
        lastest_output = 0
        count_average = 0
        lastest_layer = None
        last_linear = 0
        first_conv = 0

        for layer in model:
            try:
                dictlayer = ast.literal_eval(layer)
                fullmodel.append(dictlayer)
                if dictlayer['func'].lower() in CNN:
                    #print(dictlayer['func'].lower())
                    layer_number += 1
                    if dictlayer['func'].lower() == 'linear':
                        last_linear = layer_number
            except (SyntaxError, TypeError):
                pass
        #print(last_linear,'xxxxxxxxxxx')
        for i in range(len(fullmodel)):
            first_conv += 1
            # print(fullmodel[i]['func'])
            try:
                if 'conv' in fullmodel[i]['func'].lower():
                    input = list(fullmodel[i].values())[1]
                    padding = None
                    if lastest_output != input:
                        input = lastest_output
                    out = list(fullmodel[i].values())[2]
                    lastest_output = out

                    kernel = list(fullmodel[i].values())[3][0]
                    if 'strides' in fullmodel[i]:
                        if len(fullmodel[i]['strides']) == 0:
                            stride = 1
                        else:
                            stride = fullmodel[i]['strides'][0]
                    else:
                        stride = 1
                    ''' 
                    if 'padding' in fullmodel[i]:
                        padding = fullmodel[i]['padding']
                    '''
                    if lastest_layer == 'conv':
                        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                                                         output_node_id)
                        output_node_id = graph.add_layer(StubReLU(), output_node_id)

                    if lastest_layer == 'batchnorm':
                        output_node_id = graph.add_layer(StubReLU(), output_node_id)

                    if i == 0:
                        output_node_id = graph.add_layer(
                            self.conv(self.input_shape[-1], out, kernel_size=kernel, stride=stride),
                            output_node_id)
                        lastest_layer = 'conv'

                    else:
                        if self.trans <= 2 and lastest_layer != 'dropout' and self.trans != 0:
                            output_node_id = graph.add_layer(self.dropout(0.25), output_node_id)

                        output_node_id = graph.add_layer(self.conv(input, out, kernel_size=kernel, stride=stride),
                                                         output_node_id)
                        lastest_layer = 'conv'

                if 'dropout' in fullmodel[i]['func'].lower():
                    #if count_average == 0:
                        #count_average += 1
                        #output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
                    p_value = list(fullmodel[i].values())[1]
                    if self.trans <= 2 and count_linear == 0 and self.trans != 0:
                        p_value = 0.25

                    if self.trans >= 2 and count_linear != 0 and self.trans != 0:
                        p_value = 0.5

                    if i == layer_number - 1:
                        graph.add_layer(self.dropout(p_value), output_node_id)
                        lastest_layer = 'dropout'
                    else:
                        output_node_id = graph.add_layer(self.dropout(p_value), output_node_id)
                        lastest_layer = 'dropout'

                # if 'batchnorm' in fullmodel[i]['func'].lower():
                #     if count_linear == 0:
                #         output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                #                                          output_node_id)
                #         lastest_layer = 'batchnorm'

                if ('elu' in fullmodel[i]['func'].lower()) or ('tanh' in fullmodel[i]['func'].lower()):
                    if lastest_layer == 'conv':
                        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                                                         output_node_id)
                    if i == layer_number-1:
                        print('elu')
                        graph.add_layer(StubReLU(), output_node_id)
                        lastest_layer = 'elu'
                    else:
                        output_node_id = graph.add_layer(StubReLU(), output_node_id)
                        lastest_layer = 'elu'

                # if ('softmax' in fullmodel[i]['func'].lower()) or ('sigmoid' in fullmodel[i]['func'].lower()):
                #     if lastest_layer == 'conv':
                #         output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                #                                          output_node_id)
                #     if i == layer_number-1:
                #         graph.add_layer(StubSoftmax(), output_node_id)
                #         lastest_layer = 'softmax'
                #     else:
                #         output_node_id = graph.add_layer(StubSoftmax(), output_node_id)
                #         lastest_layer = 'softmax'

                if 'maxpool' in fullmodel[i]['func'].lower():
                    '''
                    pool_size = None
                    pool_strides = None
                    if 'arg1' in fullmodel[i]:
                        pool_size = fullmodel[i]['arg1'][0]
                    if 'pool_size' in fullmodel[i]:
                        pool_size = fullmodel[i]['pool_size'][0]
                    if 'arg2' in fullmodel[i]:
                        pool_strides = fullmodel[i]['arg2'][0]
                    if 'strides' in fullmodel[i]:
                        pool_strides = fullmodel[i]['strides'][0]
                    '''
                    if lastest_layer == 'conv':
                        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                                                         output_node_id)
                        output_node_id = graph.add_layer(StubReLU(), output_node_id)
                    if lastest_layer == 'batchnorm':
                        output_node_id = graph.add_layer(StubReLU(), output_node_id)
                    output_node_id = graph.add_layer(self.pooling(),
                                                     output_node_id)

                if 'avgpool' in fullmodel[i]['func'].lower():
                    if lastest_layer == 'conv':
                        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                                                         output_node_id)
                        output_node_id = graph.add_layer(StubReLU(), output_node_id)

                    if lastest_layer == 'batchnorm':
                        output_node_id = graph.add_layer(StubReLU(), output_node_id)

                    if self.trans <= 2 and lastest_layer != 'dropout' and self.trans != 0:
                        output_node_id = graph.add_layer(self.dropout(0.25), output_node_id)

                    count_average += 1
                    output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
                    lastest_layer = 'avgpool'

                if 'flatten' in fullmodel[i]['func'].lower():
                    if lastest_layer == 'conv':
                        output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                                                         output_node_id)
                        output_node_id = graph.add_layer(StubReLU(), output_node_id)
                    if lastest_layer == 'batchnorm':
                        output_node_id = graph.add_layer(StubReLU(), output_node_id)

                    if self.trans <= 2 and lastest_layer != 'dropout' and self.trans != 0:
                        output_node_id = graph.add_layer(self.dropout(0.25), output_node_id)

                    count_average += 1
                    #output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
                    output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)

                    lastest_layer = 'flatten'

                if 'linear' in fullmodel[i]['func'].lower():
                    if count_average == 0:
                        if lastest_layer == 'conv':
                            output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                                                             output_node_id)
                            output_node_id = graph.add_layer(StubReLU(), output_node_id)
                        if lastest_layer == 'batchnorm':
                            output_node_id = graph.add_layer(StubReLU(), output_node_id)
                        count_average += 1
                        output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
                    count_linear += 1

                    if self.trans >= 2 and count_linear != 0 and lastest_layer != "dropout" and self.trans != 0 and count_linear>1:
                        output_node_id = graph.add_layer(self.dropout(0.5), output_node_id)

                    input = list(fullmodel[i].values())[1]
                    if lastest_output != input:
                        input = lastest_output
                    out = list(fullmodel[i].values())[2]
                    lastest_output = out

                    if i == layer_number-1:
                        print(i,'linear')
                        print('kkkkkkkkkkkkkkkkk')
                        graph.add_layer(StubDense(input, self.n_output_node), output_node_id)
                        lastest_layer = 'linear'
                    else:
                        if i == last_linear-1:
                            output_node_id = graph.add_layer(StubDense(input, self.n_output_node), output_node_id)
                            lastest_layer = 'linear'
                        else:
                            output_node_id = graph.add_layer(StubDense(input, out), output_node_id)
                            lastest_layer = 'linear'
            except TypeError:
                pass
        if count_linear == 0:
            if lastest_layer == 'conv':
                output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]),
                                                 output_node_id)
                output_node_id = graph.add_layer(StubReLU(), output_node_id)
            if count_average == 0:
                count_average += 1
                output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)

            graph.add_layer(StubDense(lastest_output, self.n_output_node), output_node_id)

        return graph

