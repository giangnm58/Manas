class Constant:
    torch_optimizer = ['Adadelta', 'Adagrad', 'Adam', 'SparseAdam', 'Adamax', 'ASGD', 'LBFGS', 'RMSprop', 'Rprop', 'SGD']
    unsupported_kerasapis = [
        #Core Layers
        'Reshape', 'Permute', 'RepeatVector', 'Lambda',
        'ActivityRegularization', 'Masking', 'SpatialDropout1D', 'SpatialDropout2D', 'SpatialDropout3D',
        # Convolutional Layers
        'Conv1D', 'Convolution1D', 'SeparableConv1D', 'SeparableConv2D', 'DepthwiseConv2D', 'Conv2DTranspose',
        'Conv3D', 'Convolution3D', 'Conv3DTranspose', 'Cropping1D', 'Cropping2D',
        'Cropping3D', 'UpSampling1D', 'UpSampling2D', 'UpSampling3D', 'ZeroPadding1D', 'ZeroPadding2D',
        'ZeroPadding3D',
        # Pooling Layers
        'MaxPooling1D', 'MaxPooling3D', 'AveragePooling1D', 'AveragePooling3D', 'GlobalMaxPooling1D',
        'GlobalAveragePooling1D', 'GlobalMaxPooling3D', 'GlobalAveragePooling3D',
        # Locally-connected Layers
        'LocallyConnected1D', 'LocallyConnected2D',
        # Recurrent Layers
        'RNN', 'SimpleRNN', 'GRU', 'LSTM', 'ConvLSTM2D', 'ConvLSTM2DCell', 'SimpleRNNCell', 'GRUCell', 'LSTMCell',
        'CuDNNGRU', 'CuDNNLSTM',
        # Embedding Layers
        'Embedding',
        # Merge Layers
        'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum', 'Dot', 'subtract',
        'multiply', 'average', 'maximum', 'minimum', 'concatenate', 'dot',
        # Advanced Activations Layers
        #'LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU', 'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear',
        # Noise layers
        'GaussianNoise', 'GaussianDropout', 'AlphaDropout',
        # Layer wrappers
        'TimeDistributed', 'Bidirectional']
    kerasapis = [
        # Core Layers
        'Dense', 'Activation', 'Dropout', 'Flatten', 'Input', 'Reshape', 'Permute', 'RepeatVector', 'Lambda',
        'ActivityRegularization', 'Masking', 'SpatialDropout1D', 'SpatialDropout2D', 'SpatialDropout3D',
        # Convolutional Layers
        'Conv1D', 'Convolution1D', 'SeparableConv1D', 'SeparableConv2D', 'DepthwiseConv2D', 'Conv2DTranspose',
        'Conv2D', 'Convolution2D', 'Conv3D', 'Convolution3D', 'Conv3DTranspose', 'Cropping1D', 'Cropping2D',
        'Cropping3D', 'UpSampling1D', 'UpSampling2D', 'UpSampling3D', 'ZeroPadding1D', 'ZeroPadding2D',
        'ZeroPadding3D',
        # Pooling Layers
        'MaxPooling1D', 'MaxPooling2D', 'MaxPooling3D', 'AveragePooling1D', 'AveragePooling2D',
        'AveragePooling3D', 'GlobalMaxPooling1D', 'GlobalAveragePooling1D', 'GlobalMaxPooling2D',
        'GlobalAveragePooling2D', 'GlobalMaxPooling3D', 'GlobalAveragePooling3D',
        # Locally-connected Layers
        'LocallyConnected1D', 'LocallyConnected2D',
        # Recurrent Layers
        'RNN', 'SimpleRNN', 'GRU', 'LSTM', 'ConvLSTM2D', 'ConvLSTM2DCell', 'SimpleRNNCell', 'GRUCell',
        'LSTMCell', 'CuDNNGRU', 'CuDNNLSTM',
        # Embedding Layers
        'Embedding',
        # Merge Layers
        'Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum', 'Concatenate', 'Dot', 'subtract', 'multiply',
        'average', 'maximum', 'minimum', 'concatenate', 'dot',
        # 'add',
        # Advanced Activations Layers
        'LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU', 'Softmax', 'ReLU', 'tanh', 'sigmoid', 'hard_sigmoid',
        'exponential', 'linear',
        # Normalization Layers
        'BatchNormalization',
        # Noise layers
        'GaussianNoise', 'GaussianDropout', 'AlphaDropout',
        # Layer wrappers
        'TimeDistributed', 'Bidirectional',
        'compile', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    requirement_kerasapi = ['Conv2D', 'Convolution2D']
    preprocessed_keywords = [  # 'Sequential'
        ': Conv1d(', ': Conv2d(', ': Conv3d(', ': MaxPool1d(', ': MaxPool2d(', ': MaxPool3d(', ': AvgPool1d(',
        ': AvgPool2d(',
        ': AvgPool3d(', ': PReLU(', ': ReLU(', ': ReLU6(', ': BatchNorm1d(', ': BatchNorm2d(', ': BatchNorm3d(',
        ': Dropout(', ': Dropout2d(',
        ': Dropout3d(', ': Softmax(', ': Softmax2d(', ': relu(', ': conv1d(', ': conv2d(', ': conv3d(', ': avg_pool1d(',
        ': avg_pool2d(',
        ': avg_pool3d(', ': max_pool1d(',
        ': max_pool2d(', ': max_pool3d(', ': batch_norm(', ': normalize(', ': linear(', ': dropout(', ': dropout2d(',
        ': dropout3d(', ': Linear(', ': elu(', ': softmax(',
        ': Module(', ': relu6(']
    model3_10 = "S:\manas_data\\new\MANAS_NIPS2020\manas\\autokeras\mined_model\\3_10"
    model3_100 = "/manas/autokeras/mined_model/3_100"
    model3_1000 = "/manas/autokeras/mined_model/3_1000"
    model1_10 = "S:\manas_data\\new\MANAS_NIPS2020\manas\\autokeras\mined_model\\1_10"
    model1_100 = "/manas/autokeras/mined_model/1_100"
    model1_1000 = "/manas/autokeras/mined_model/1_1000"