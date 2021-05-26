"""

    Keras implementation of the EDRAM network of Ablavatski et al. (2017)

        * emission_weights
        * tedram_cell       |
        * tedram_model      |> tEDRAM (with separate batch normalization per time step)
        * edram_cell      |
        * edram_model     |> EDRAM
        * STN_model

"""

from typing import List, Tuple, Any
from numpy import ndarray, array, zeros, linspace, asarray, sqrt
from keras.layers import (Input,
                          LSTM,
                          Dense,
                          Activation,
                          Flatten,
                          Reshape,
                          Conv2D,
                          LocallyConnected2D,
                          MaxPooling2D,
                          BatchNormalization,
                          Dropout,
                          concatenate,
                          multiply,
                          add,
                          average,
                          maximum)


from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.initializers import RandomUniform, RandomNormal, Zeros

from models.spatial_transformer.models.layers import BilinearInterpolation
from models.weighted_losses import weighted_categorical_crossentropy, weighted_mean_squared_error


# number of emotions per class in ANet training file (in thousands)
n: ndarray = asarray([78, 144, 29, 16, 8, 5, 28])
w: float = sum(n)/(n*7)
w2: ndarray = sqrt(w)

# weights for weighted loss functions
emotion_weights: List[ndarray] = []
emotion_weights.append(array(w2))
emotion_weights.append(array(w))
emotion_weights.append(array([0.82, 0.73, 1.14, 1.56, 2.48, 3.52, 1.17]))

emotion_dimension_weights: ndarray = array([1.17, 0.83])

localisation_weights: List[ndarray] = []
localisation_weights.append(array([1.00, 0.25, 1.00, 0.25, 1.00, 1.00]))
localisation_weights.append(array([1.00, 0.00, 0.00, 0.00, 1.00, 0.00]))


def emission_weights(output_size: int, zoom_bias: int) -> List[ndarray]:
    """
        Initialization weights for the Spatial Transformer
        (all weights and biases zero except zoom biases which should be set to 1)

        Parameters:

            * output_size: number of rows
            * zoom_bias: ~

        Returns:

            * weight matrix with zeros and biases [zoom_bias,0,0,0,zoom_bias,0]

    """
    b: ndarray = zeros((2, 3), dtype='float32')
    b[0, 0] = zoom_bias
    b[1, 1] = zoom_bias

    W: ndarray = zeros((output_size, 6), dtype='float32')
    weights: List[ndarray] = [W, b.flatten()]

    return weights


def tedram_cell(input_shape=(120,160, 1), batch_size=192, glimpse_size=(26,26),
                 n_filters=128, filter_size=(3,3), n_features=1024,
                 RNN_size_1=512, RNN_size_2=512, n_classes=7,
                 bn=True, dropout=0, clip_value=1, layers=None,
                 output_localisation=True, output_emotion_dims=False,
                 step='0', unique_emission=False, unique_glimpse=False,
                 emission_bias=1) -> Model:
    """
        One timestep of the EDRAM network with temporally separated batch normalization

        Parameters:

            * input_shape: input image dimensions
            * glimpse_size: dimensions of the extracted image patch (the glimpse)
            * n_filters: determines the number of filters in the glimpse CNN
            * filter_size: dimensions of the glimpse CNN kernel
            * n_features: learned features of the glimpse CNN (fc dimension)
            * RNN_size: number of cells in the LSTMs
            * n_classes: number of classes in the output
            * bn: whether to use batch normalization
            * dropout: dropout percentage
            * network: all layers that should be reused
            * output_localisation: whether to output the localisation matrix
            * output_emotion_dims: whether to output valence and arousal for emotion stimuli
            * step: name suffix for the edram cell
            * unique_emission: whether to use a temporally separated emission layer
            * unique_glimpse: whether to use a temporally separated first layer for the glimpse CNN
            * emission_bias: presets the zoom bias of the emission network
            * clip_value: max value of the zoom factor in the spatial transformer

        Returns:

            * itself

    """
    # activate dropout
    do: bool = True if dropout>0 else False

    # unpack layers
    (conv_1, conv_1_bias, conv_2, conv_2_bias, max_pooling_1,
     conv_3, conv_3_bias, conv_4, conv_4_bias, max_pooling_2, conv_5,
     conv_5_bias, conv_6, conv_6_bias, flatten, glimpse_what, glimpse_where,
     reshape_to_sequence, LSTM_classify, LSTM_localize, reshape_from_sequence,
     cla_fc_1, cla_fc_2, cla_fc_3, dim_fc_1, dim_fc_2, dim_fc_3, em) = layers


    #######################
    ###  Define Inputs  ###
    #######################

    # input image and localization matrix
    input_image: Input = Input(shape=input_shape, dtype='float32', name='input_image')
    input_matrix: Input = Input(shape=(6,), dtype='float32', name='input_matrix')

    # hidden states of the LSTMs
    hidden_state_1: Input = Input(shape=(RNN_size_1,),  dtype='float32', name='hidden_state_1')
    cell_state_1: Input = Input(shape=(RNN_size_1,),  dtype='float32', name='cell_state_1')
    hidden_state_2: Input = Input(shape=(RNN_size_2,),  dtype='float32', name='hidden_state_2')
    cell_state_2: Input = Input(shape=(RNN_size_2,),  dtype='float32', name='cell_state_2')

    # bias matrices
    if glimpse_size==(26,26):
        bias_26: Input = Input(shape=(26,26,1),  dtype='float32', name='bias_26')
        bias_24: Input = Input(shape=(24,24,1),  dtype='float32', name='bias_24')
        bias_12: Input = Input(shape=(12,12,1),  dtype='float32', name='bias_12')
        bias_6: Input = Input(shape=(6,6,1),  dtype='float32', name='bias_6')
        bias_4: Input = Input(shape=(4,4,1),  dtype='float32', name='bias_4')
    else:
        bias_26: Input = Input(shape=(16,16,1),  dtype='float32', name='bias_26')
        bias_24: Input = Input(shape=(16,16,1),  dtype='float32', name='bias_24')
        bias_12: Input = Input(shape=(8,8,1),  dtype='float32', name='bias_12')
        bias_6: Input = Input(shape=(4,4,1),  dtype='float32', name='bias_6')
        bias_4: Input = Input(shape=(4,4,1),  dtype='float32', name='bias_4')

    inputs: List[Input] =[input_image, input_matrix, hidden_state_1, cell_state_1, hidden_state_2, cell_state_2, bias_26, bias_24, bias_12, bias_6, bias_4]


    ########################
    ###  Connect Layers  ###
    ########################

    ## Glimpse Network
    # spatial transformer, performs affine transformation of input image to a 26x26 patch
    x: BilinearInterpolation = BilinearInterpolation(glimpse_size, clip_value)([input_image, input_matrix])
    bn_axis: int = 3

    if unique_glimpse:
        x: Conv2D = Conv2D(int(n_filters/2), (5,5), padding='same', activation='relu', use_bias=False, name='glimpse_conv1')(x)
        b: LocallyConnected2D = LocallyConnected2D(int(n_filters/2), (1,1), padding='valid', use_bias=False, name='glimpse_conv1_bias')(bias_26)
    else:
        x = conv_1(x)
        b = conv_1_bias(bias_26)
    x = add([x, b], name='glimpse_conv1_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv1_bn')(x)
    x = conv_2(x)
    b = conv_2_bias(bias_24)
    x = add([x, b], name='glimpse_conv2_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv2_bn')(x)
    x = max_pooling_1(x)
    x = conv_3(x)
    b = conv_3_bias(bias_12)
    x = add([x, b], name='glimpse_conv3_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv3_bn')(x)
    x = conv_4(x)
    b = conv_4_bias(bias_12)
    x = add([x, b], name='glimpse_conv4_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv4_bn')(x)
    x = max_pooling_2(x)
    x = conv_5(x)
    b = conv_5_bias(bias_6)
    x = add([x, b], name='glimpse_conv5_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv5_bn')(x)
    x = conv_6(x)
    b = conv_6_bias(bias_4)
    x = add([x, b], name='glimpse_conv6_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv6_bn')(x)
    x = flatten(x)
    x_what = glimpse_what(x)
    if bn: x_what = BatchNormalization(name='glimpse_dense_bn')(x_what)
    x_where = glimpse_where(input_matrix)
    if bn: x_where = BatchNormalization(name='glimpse_localisation_bn')(x_where)
    x = multiply([x_where, x_what], name='glimpse_output')

    ## RNNs
    x = reshape_to_sequence(x)
    rnn1, h1, c1 = LSTM_classify(x, initial_state=[hidden_state_1, cell_state_1]) #type: ndarray, ndarray, ndarray
    if output_localisation:
        rnn2, h2, c2 = LSTM_localize(rnn1, initial_state=[hidden_state_2, cell_state_2]) #type: ndarray, ndarray, ndarray
    rnn1: ndarray = reshape_from_sequence(rnn1);

    # apply dropout
    if do:
        rnn1 = Dropout(dropout, name='classification_dropout')(rnn1)
        if output_localisation:
            rnn2 = Dropout(dropout, name='localisation_dropout')(rnn2)


    ## Classification Network
    x = cla_fc_1(rnn1)
    if bn: x = BatchNormalization(name='classification_fc1_bn')(x)
    x = cla_fc_2(x)
    if bn: x = BatchNormalization(name='classification_fc2_bn')(x)
    classification = cla_fc_3(x)

    ## Affective Dimensions Network - outputs valence and arousal
    if output_emotion_dims:
        x = dim_fc_1(rnn1)
        if bn: x = BatchNormalization(name='dimension_fc1_bn')(x)
        x = dim_fc_2(x)
        if bn: x = BatchNormalization(name='dimension_fc2_bn')(x)
        dimension = dim_fc_3(x)

    ## Emission Network - outputs the flat localization matrix
    if output_localisation:
        if unique_emission:
            localisation = Dense(6, activation='tanh', weights=emission_weights(RNN_size_2, emission_bias), name='emission')(rnn2)
        else:
            localisation = em(rnn2)


    ########################
    ###  Define Outputs  ###
    ########################

    if output_emotion_dims:
        if output_localisation:
            outputs=[classification, localisation, h1, c1, h2, c2, dimension]
        else:
            # this will lead to an error
            outputs=[classification, h1, c1, dimension]
    else:
        if output_localisation:
            outputs=[classification, localisation, h1, c1, h2, c2]
        else:
            outputs=[classification, h1, c1]

    return Model(inputs, outputs, name='edram_cell_'+str(step))


def tedram_model(input_shape=(10,120,160,1), batch_size=192, learning_rate=0.0001, steps=3,
                  glimpse_size=(26,26), coarse_size=(12,12), hidden_init=0,
                  n_filters=128, filter_sizes=(3,5), n_features=1024,
                  RNN_size_1=512, RNN_size_2=512, n_classes=7, output_mode=0,
                  use_init_matrix=True, output_emotion_dims=False,
                  emission_bias=1, clip_value=1, unique_emission=False, unique_glimpse=False,
                  bn=True, dropout=0, use_weighted_loss=False, localisation_cost_factor=1):
    """
        EDRAM network with temporally separated batch normalization - takes an image and iteratively extracts image
        patches (glimpses) to produce a classification

        Parameters:

            * input_shape: input image dimensions
            * learning_rate: ~
            * steps: number of iterations over the input (or size of the window for sequence processing)
            * glimpse_size: dimensions of the extracted image patch (the glimpse)
            * coarse_size: dimensions of the downscaled image for the initialization of the network
            * hidden_init: the value of the initial hidden state (leads to ValueError --> unused)
            * n_filters: determines the number of filters in the glimpse CNN
            * filter_sizes: dimensions of the glimpse and initialization and CNN kernels
            * n_features: learned features of the glimpse CNN (fc dimension)
            * RNN_size: number of cells in the LSTMs
            * n_classes: number of classes in the output
            * output_mode: 0 only the outputs of the last step are evaluated for the loss or
                           1 outputs of all time steps are concatenated for evaluation
            * use_init_matrix: whether to use the initialization matrix for the first step
            * output_emotion_dims: whether to output valence and arousal for emotion stimuli
            * emission_bias: presets the zoom bias of the emission network
            * clip_value: >0  max value of the zoom factor in the spatial transformer or
                          0   smaller clip values for every step [1.,.85,.70,.55,.40,.25]
            * unique_emission: whether to use a unique emission layer
            * unique_glimpse: whether to use a unique first layer for the glimpse CNN
            * bn: whether to use batch normalization
            * dropout: dropout percentage
            * use_weighted_loss: whether to use weighted versions of the losses, especially class weights
            * localisation_cost_factor: weighting of the localisation cost

        Returns:

            * itself

    """
    filter_size1, filter_size2 = list(zip(filter_sizes, filter_sizes))

    # activate dropout
    do = True if dropout>0 else False


    #######################
    ###  Define Inputs  ###
    #######################

    # input image and localization matrix
    input_image = Input(shape=input_shape, dtype='float32', name='input_image') # 10 x 120 x 160
    input_matrix = Input(shape=(6,), dtype='float32', name='input_matrix')

    # initial hidden states of the LSTMs
    init_h1 = Input(shape=(RNN_size_1,),  dtype='float32', name='initial_hidden_state_1')
    init_c1 = Input(shape=(RNN_size_1,),  dtype='float32', name='initial_cell_state_1')
    # init_h2 is generated by initialization network
    init_c2 = Input(shape=(RNN_size_2,),  dtype='float32', name='initial_cell_state_2')

    # bias matrices
    if glimpse_size==(26,26):
        b26 = Input(shape=(26,26,1),  dtype='float32', name='b26')
        b24 = Input(shape=(24,24,1),  dtype='float32', name='b24')
        b12 = Input(shape=(12,12,1),  dtype='float32', name='b12')
        b8 = Input(shape=(8,8,1),  dtype='float32', name='b8')
        b6 = Input(shape=(6,6,1),  dtype='float32', name='b6')
        b4 = Input(shape=(4,4,1),  dtype='float32', name='b4')
    else: # glimpse size == (16,16)
        b26 = Input(shape=(16,16,1),  dtype='float32', name='b26')
        b24 = Input(shape=(16,16,1),  dtype='float32', name='b24')
        b12 = Input(shape=(8,8,1),  dtype='float32', name='b12')
        b8 = Input(shape=(8,8,1),  dtype='float32', name='b8')
        b6 = Input(shape=(6,6,1),  dtype='float32', name='b6')
        b4 = Input(shape=(4,4,1),  dtype='float32', name='b4')

    inputs=[input_image, input_matrix, init_h1, init_c1, init_c2, b26, b24, b12, b8, b6, b4]


    #################################
    ###  Network Building Blocks  ###
    #################################

    if glimpse_size==(26,26):
        glimpse_padding = 'valid'
    else:
        glimpse_padding = 'same'

    ### ------------------------------------------- layers for the EDRAM core cell
    ## Glimpse Network (26x26 --> 192x4x4 --> 1024)
    # 64 filters, 3x3 Convolution, zero padding --> 26x26
    if unique_glimpse==False:
        conv_1 = Conv2D(int(n_filters/2), filter_size1, padding='same', activation='relu', use_bias=False, name='glimpse_conv1')
        conv_1_bias = LocallyConnected2D(int(n_filters/2), (1,1), padding='valid', use_bias=False, name='glimpse_conv1_bias')
    else:
        conv_1 = None
        conv_1_bias = None
    # 64 filters, 3x3 Convolution, no padding --> 24x24
    conv_2: Conv2D = Conv2D(int(n_filters/2), filter_size1, padding=glimpse_padding, activation='relu', use_bias=False, name='glimpse_conv2')
    conv_2_bias: LocallyConnected2D = LocallyConnected2D(int(n_filters/2), (1,1), padding='valid', use_bias=False, name='glimpse_conv2_bias')
    # max pooling, 24x24 --> 12x12
    max_pooling_1: MaxPooling2D = MaxPooling2D(pool_size=(2,2), name='glimpse_max_pooling1')
    # 128 filters, 3x3 Convolution, padding to preserve dimensionality of the tensor
    conv_3: Conv2D = Conv2D(n_filters, filter_size1, padding='same', activation='relu', use_bias=False, name='glimpse_conv3')
    conv_3_bias: LocallyConnected2D = LocallyConnected2D(n_filters, (1,1), padding='valid', use_bias=False, name='glimpse_conv3_bias')
    # 128 filters, 3x3 Convolution, padding to preserve dimensionality of the tensor
    conv_4: Conv2D = Conv2D(n_filters, filter_size1, padding='same', activation='relu', use_bias=False, name='glimpse_conv4')
    conv_4_bias: LocallyConnected2D = LocallyConnected2D(n_filters, (1,1), padding='valid', use_bias=False, name='glimpse_conv4_bias')
    # max pooling, 12x12 --> 6x6
    max_pooling_2 = MaxPooling2D(pool_size=(2,2), name='glimpse_max_pooling2')
    # 160 filters, 3x3 Convolution, padding to preserve dimensionality of the tensor
    conv_5 = Conv2D(160, filter_size1, padding='same', activation='relu', use_bias=False, name='glimpse_conv5')
    conv_5_bias = LocallyConnected2D(160, (1,1), padding='valid', use_bias=False, name='glimpse_conv5_bias')
    # 192 filters, 3x3 Convolution, no padding --> 4x4
    conv_6 = Conv2D(192, filter_size1, padding=glimpse_padding, activation='relu', use_bias=False, name='glimpse_conv6')
    conv_6_bias = LocallyConnected2D(192, (1,1), padding='valid', use_bias=False, name='glimpse_conv6_bias')
    # 4*4*192 = 3072
    flatten = Flatten(name='glimpse_flatten')

    # fully connected, output_dim=1024
    glimpse_what = Dense(n_features, activation='relu', name = 'glimpse_what')

    # fully connected, output_dim=1024
    glimpse_where = Dense(n_features, activation='relu', name = 'glimpse_where')
    # --> glimpse_what and where are multiplied to create Glimpse Network output


    ## LSTMs
    # reshape to 1-step sequence for LSTM
    reshape_to_sequence = Reshape((1, n_features), name = 'to_sequence')
    # LSTMs
    LSTM_classify = LSTM(RNN_size_1, return_state=True, return_sequences=True, name="LSTM_classify")
    LSTM_localize = LSTM(RNN_size_2, return_state=True, name="LSTM_localize")
    # classification network - outputs classification probabilities
    reshape_from_sequence = Reshape((RNN_size_1,), name='from_sequence')


    ## Classification Network
    # fully connected, output_dim=1024
    cla_fc_1 = Dense(n_features, activation='relu', name='classification_fc1')
    # fully connected, output_dim=1024
    cla_fc_2 = Dense(n_features, activation='relu', name='classification_fc2')
    # fully connected, output_dim=7, softmax activation
    cla_fc_3 = Dense(n_classes, activation='softmax', name='classification_fc3')


    ## Affective Dimensions Network - outputs valence and arousal
    if output_emotion_dims:
        # fully connected, output_dim=1024
        dim_fc_1 = Dense(n_features, activation='relu', name='dimension_fc1')
        # fully connected, output_dim=1024
        dim_fc_2 = Dense(n_features, activation='relu', name='dimension_fc2')
        # fully connected, output_dim=7, softmax activation
        dim_fc_3 = Dense(2, activation='tanh', name='dimension_fc3')
    else:
        dim_fc_1, dim_fc_2, dim_fc_3 = None, None, None


    ## Emission Network
    if unique_emission==False:
        em = Dense(6, activation='tanh', weights=emission_weights(RNN_size_2, emission_bias), name='emission')
    else:
        em = None

    ## pack layers
    layers = (conv_1, conv_1_bias, conv_2, conv_2_bias, max_pooling_1,
              conv_3, conv_3_bias, conv_4, conv_4_bias, max_pooling_2, conv_5, conv_5_bias,
              conv_6, conv_6_bias, flatten, glimpse_what, glimpse_where,
              reshape_to_sequence, LSTM_classify, LSTM_localize, reshape_from_sequence,
              cla_fc_1, cla_fc_2, cla_fc_3, dim_fc_1, dim_fc_2, dim_fc_3, em)


    ### the EDRAM Core Cells
    output_localisation: bool = False if output_mode==0 and steps==1 else True
    # constant or decreassing clip values for the zoom factor of the glimpse STN
    clip_value: ndarray = linspace(clip_value, clip_value, steps) if clip_value>0 else linspace(1, 0.25, steps)
    # constant or decreassing bias for the zoom factor of the emission network
    emission_bias: ndarray = linspace(emission_bias, emission_bias, steps) if emission_bias>0 else linspace(1, 0.30, steps)
    edram_cell: List[Model] = []
    # STEPS should be set to 10
    for i in range(0, steps):

        print('CALL TEDRAM CELL:', input_shape[0:])
        edram_cell.append(tedram_cell(input_shape[0:], batch_size, glimpse_size, n_filters, filter_size1,
                                       n_features, RNN_size_1, RNN_size_2, n_classes,
                                       bn, dropout, clip_value[i], layers,
                                       output_localisation, output_emotion_dims,
                                       i, unique_emission, unique_glimpse, emission_bias[i]))


    ### Initialization Network
    ### Context Network
    ## downscale the input with STN, TODO: Do this in the preprocessing (but then coarse_size is fixed!)
    input_image_coarse: BilinearInterpolation = BilinearInterpolation(coarse_size, 1)([input_image, input_matrix])

    ## 3 convolutions on 1x12x12 input: 5x5, 16 filters --> 3x3, 16 filters --> 3x3, 32 filters
    bn_axis: int = 3
    x = Conv2D(int(n_filters/8), filter_size2, padding='valid', use_bias=False, name='init_conv1')(input_image_coarse)
    b = LocallyConnected2D(int(n_filters/8), (1,1), padding='valid', use_bias=False, name='init_conv1_bias')(b8)
    x = add([x, b], name='init_conv1_add')

    if do: x = Dropout(dropout, name='init_conv1_dropout')(x)
    if bn: x = BatchNormalization(axis=bn_axis, name='init_conv1_bn')(x)

    nf = n_filters/8 if RNN_size_2==512 else n_filters/4 # increasing glimpse NN size if RNN_size2 is 1024
    x = Conv2D(int(nf), filter_size1, padding='valid', use_bias=False, name='init_conv2')(x)
    b = LocallyConnected2D(int(nf), (1,1), padding='valid', use_bias=False, name='init_conv2_bias')(b6)
    x = add([x, b], name='init_conv2_add')

    if do: x = Dropout(dropout, name='init_conv2_dropout')(x)
    if bn: x = BatchNormalization(axis=bn_axis, name='init_conv2_bn')(x)

    nf = n_filters/4 if RNN_size_2==512 else n_filters/2 # increasing glimpse NN size if RNN_size2 is 1024
    x = Conv2D(int(nf), filter_size1, padding='valid', use_bias=False, name='init_conv3')(x)
    b = LocallyConnected2D(int(nf), (1,1), padding='valid', use_bias=False, name='init_conv3_bias')(b4)
    x = add([x, b], name='init_conv3_add')

    if do: x = Dropout(dropout, name='init_conv3_dropout')(x)
    if bn: x = BatchNormalization(axis=bn_axis, name='init_conv3_bn')(x)
    x = Flatten(name='init_flatten')(x)

    ## Initialization of localization LSTM
    if unique_emission==False:
        init_matrix = em(x)
    else:
        init_matrix = Dense(6, activation='tanh', weights=emission_weights(RNN_size_2, emission_bias[0]), name='emission')(x)

    init_h2 = Reshape((RNN_size_2,), name = 'initial_hidden_state_2')(x)

    #############################
    ###  Assemble everything  ###
    #############################

    # step zero (initialization)
    step = [[None, init_matrix if output_mode==1 else input_matrix]]
    # step 1, INPUT_IMAGE (None, 10, 120,160,1) -> input_image[0][0] -> (120,160,1)
    step.append(edram_cell[0]([input_image[:, 0], init_matrix if use_init_matrix else input_matrix, init_h1, init_c1, init_h2, init_c2, b26, b24, b12, b6 if glimpse_size==(26,26) else b4, b4]))
    # "recurrently" apply edram network
    for i in range(1, steps):
        print('INPUT IMAGE:', input_shape[:,i].shape);
        step.append(edram_cell[i]([input_image[:, i], step[i][1], step[i][2], step[i][3], step[i][4], step[i][5], b26, b24, b12, b6 if glimpse_size==(26,26) else b4, b4]))


    ########################
    ###  Define Outputs  ###
    ########################

    if output_mode==0:
        # only use outputs of last time step
        classifications = Reshape((n_classes,), name='classifications')(step[steps][0])
        if output_emotion_dims:
            dimensions = Reshape((2,), name='dimensions')(step[steps][6])
        localisations = Reshape((6,), name='localisations')(step[steps-1][1])
    else:
        # concatenate outputs of different timesteps
        if steps==1:
            classifications = Reshape((n_classes,), name='classifications')(step[1][0])
            if output_emotion_dims:
                dimensions = Reshape((2,), name='dimensions')(step[1][6])
            localisations = Reshape((2, 6), name='localisations')(concatenate([step[0][1], step[1][1]]))
        else:
            classifications = Reshape((steps, n_classes), name='classifications')(concatenate([step[i][0] for i in range(1, steps+1)]))
            if output_emotion_dims:
                dimensions = Reshape((steps,2), name='dimensions')(concatenate([step[i][6] for i in range(1, steps+1)]))
            localisations = Reshape((steps+1, 6), name='localisations')(concatenate([step[i][1] for i in range(0, steps+1)]))

    if output_emotion_dims:
        outputs=[classifications, dimensions, localisations]
    else:
        outputs=[classifications, localisations]
    # build the model
    model = Model(inputs, outputs, name='tedram_model')


    ############################
    ###  Training Framework  ###
    ############################

    # optimization algorithm
    optimizer = Adam(lr=learning_rate, clipnorm=10.)

    # define losses
    if use_weighted_loss:
        # weighted losses
        if n_classes==7:
            # only use emotion weights for emotion classification
            classification_loss = weighted_categorical_crossentropy(emotion_weights[use_weighted_loss])
        else:
            classification_loss = 'categorical_crossentropy'
        dimension_loss = weighted_mean_squared_error(emotion_dimension_weights)
        localisation_loss = weighted_mean_squared_error(localisation_weights[1 if n_classes==7 else 0])
    else:
        # standard losses
        classification_loss = 'categorical_crossentropy'
        dimension_loss = 'mean_squared_error'
        localisation_loss = 'mean_squared_error'


    ###########################
    ###  Compile the Model  ###
    ###########################

    if output_emotion_dims:
        model.compile(loss={'classifications': classification_loss, 'dimensions': dimension_loss,'localisations': localisation_loss},
                      loss_weights={'classifications': 1, 'dimensions': 1, 'localisations': localisation_cost_factor},
                      metrics={'classifications': 'categorical_accuracy'}, optimizer=optimizer)
    else:
        model.compile(loss={'classifications': classification_loss, 'localisations': localisation_loss},
                      loss_weights={'classifications': 1, 'localisations': localisation_cost_factor},
                      metrics={'classifications': 'categorical_accuracy'}, optimizer=optimizer)

    return model


def edram_cell(input_shape=(100,100,1), batch_size=192, glimpse_size=(26,26),
                 n_filters=128, filter_size=(3,3), n_features=1024,
                 RNN_size_1=512, RNN_size_2=512, n_classes=7,
                 bn=True, dropout=0, clip_value=1, emission_network=None,
                 output_localisation=True, output_emotion_dims=False,
                 headless=False):
    """
        One timestep of the EDRAM network

        Parameters:

            * input_shape: input image dimensions
            * glimpse_size: dimensions of the extracted image patch (the glimpse)
            * n_filters: determines the number of filters in the glimpse CNN
            * filter_size: dimensions of the glimpse CNN kernel
            * n_features: learned features of the glimpse CNN (fc dimension)
            * RNN_size: number of cells in the LSTMs
            * n_classes: number of classes in the output
            * bn: whether to use batch normalization
            * dropout: dropout percentage
            * clip_value: max value of the zoom factor in the spatial transformer
            * emissions_network: a fully connected layer which can be used outside of the cell
            * output_localisation: whether to output the localisation matrix
            * output_emotion_dims: whether to output valence and arousal for emotion stimuli
            * headless: uses the RNN1 hidden states for a final fully connected layer for predictions

        Returns:

            * itself

    """
    # define inputs

    # input image and localization matrix
    input_image = Input(shape=input_shape, dtype='float32', name='input_image')
    input_matrix = Input(shape=(6,), dtype='float32', name='input_matrix')

    # hidden states of the LSTMs
    hidden_state_1 = Input(shape=(RNN_size_1,),  dtype='float32', name='hidden_state_1')
    cell_state_1 = Input(shape=(RNN_size_1,),  dtype='float32', name='cell_state_1')
    hidden_state_2 = Input(shape=(RNN_size_2,),  dtype='float32', name='hidden_state_2')
    cell_state_2 = Input(shape=(RNN_size_2,),  dtype='float32', name='cell_state_2')

    # bias matrices
    if glimpse_size==(26,26):
        bias_26 = Input(shape=(26,26,1),  dtype='float32', name='bias_26')
        bias_24 = Input(shape=(24,24,1),  dtype='float32', name='bias_24')
        bias_12 = Input(shape=(12,12,1),  dtype='float32', name='bias_12')
        bias_6 = Input(shape=(6,6,1),  dtype='float32', name='bias_6')
        bias_4 = Input(shape=(4,4,1),  dtype='float32', name='bias_4')
    else:
        bias_26 = Input(shape=(16,16,1),  dtype='float32', name='bias_26')
        bias_24 = Input(shape=(16,16,1),  dtype='float32', name='bias_24')
        bias_12 = Input(shape=(8,8,1),  dtype='float32', name='bias_12')
        bias_6 = Input(shape=(4,4,1),  dtype='float32', name='bias_6')
        bias_4 = Input(shape=(4,4,1),  dtype='float32', name='bias_4')

    inputs=[input_image, input_matrix, hidden_state_1, cell_state_1, hidden_state_2, cell_state_2, bias_26, bias_24, bias_12, bias_6, bias_4]


    # spatial transformer, performs affine transformation of input image to a 26x26 patch
    x = BilinearInterpolation(glimpse_size, clip_value)([input_image, input_matrix])

    # glimpse network (26x26 --> 192x4x4 --> 1024)
    bn_axis = 3  # axis over which batch normalization is performed

    if glimpse_size==(26,26):
        glimpse_padding = 'valid'
    else:
        glimpse_padding = 'same'

    x = Conv2D(int(n_filters/2), filter_size, padding='same', activation='relu', use_bias=False, name='glimpse_conv1')(x)
    b = LocallyConnected2D(int(n_filters/2), (1,1), padding='valid', use_bias=False, name='glimpse_conv1_bias')(bias_26)
    x = add([x, b], name='glimpse_conv1_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv1_bn')(x)
    # 64 filters, 3x3 Convolution, no padding --> 24x24
    x = Conv2D(int(n_filters/2), filter_size, padding=glimpse_padding, activation='relu', use_bias=False, name='glimpse_conv2')(x)
    b = LocallyConnected2D(int(n_filters/2), (1,1), padding='valid', use_bias=False, name='glimpse_conv2_bias')(bias_24)
    x = add([x, b], name='glimpse_conv2_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv2_bn')(x)
    # max pooling, 24x24 --> 12x12
    x = MaxPooling2D(pool_size=(2,2), name='glimpse_max_pooling1')(x)
    # 128 filters, 3x3 Convolution, padding to preserve dimensionality of the tensor
    x = Conv2D(n_filters, filter_size, padding='same', activation='relu', use_bias=False, name='glimpse_conv3')(x)
    b = LocallyConnected2D(n_filters, (1,1), padding='valid', use_bias=False, name='glimpse_conv3_bias')(bias_12)
    x = add([x, b], name='glimpse_conv3_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv3_bn')(x)
    # 128 filters, 3x3 Convolution, padding to preserve dimensionality of the tensor
    x = Conv2D(n_filters, filter_size, padding='same', activation='relu', use_bias=False, name='glimpse_conv4')(x)
    b = LocallyConnected2D(n_filters, (1,1), padding='valid', use_bias=False, name='glimpse_conv4_bias')(bias_12)
    x = add([x, b], name='glimpse_conv4_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv4_bn')(x)
    # max pooling, 12x12 --> 6x6
    x = MaxPooling2D(pool_size=(2,2), name='glimpse_max_pooling2')(x)
    # 160 filters, 3x3 Convolution, padding to preserve dimensionality of the tensor
    x = Conv2D(160, filter_size, padding='same', activation='relu', use_bias=False, name='glimpse_conv5')(x)
    b = LocallyConnected2D(160, (1,1), padding='valid', use_bias=False, name='glimpse_conv5_bias')(bias_6)
    x = add([x, b], name='glimpse_conv5_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv5_bn')(x)
    # 192 filters, 3x3 Convolution, no padding --> 4x4
    x = Conv2D(192, filter_size, padding=glimpse_padding, activation='relu', use_bias=False, name='glimpse_conv6')(x)
    b = LocallyConnected2D(192, (1,1), padding='valid', use_bias=False, name='glimpse_conv6_bias')(bias_4)
    x = add([x, b], name='glimpse_conv6_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='glimpse_conv6_bn')(x)
    # 4*4*192 = 3072
    x = Flatten(name='glimpse_flatten')(x)
    # fully connected, output_dim=1024
    x_what = Dense(n_features, activation='relu', name = 'glimpse_dense')(x)
    if bn: x_what = BatchNormalization(name='glimpse_dense_bn')(x_what)



    # casting transformation matrix A (l) to a 1024-dimensional vector for multiplication with feature vector from the CNN

    # fully connected, output_dim=1024
    x_where = Dense(n_features, activation='relu', name = 'glimpse_localisation')(input_matrix)
    if bn: x_where = BatchNormalization(name='glimpse_localisation_bn')(x_where)

    # combining "what" and "where"
    x = multiply([x_where, x_what], name='glimpse_output')
    # reshape to 1-step sequence for RNN1
    x = Reshape((1, n_features), name = 'to_sequence')(x)


    # RNNs
    # batch_input_shape = (batch_size,) + (1, n_features), ...

    # "encoder" lstm, 512 hidden units, returns sequence of length 1
    rnn1, h1, c1 = LSTM(RNN_size_1, return_state=True, return_sequences=True, name="RNN_classify")(x, initial_state=[hidden_state_1, cell_state_1])

    # "decoder" lstm, 512 hidden units
    if output_localisation:
        rnn2, h2, c2 = LSTM(RNN_size_2, return_state=True, name="RNN_localize")(rnn1, initial_state=[hidden_state_2, cell_state_2])


    # classification network - outputs classification probabilities
    rnn1 = Reshape((RNN_size_1,), name='from_sequence')(rnn1)
    if headless==False:
        # fully connected, output_dim=1024
        x = Dense(n_features, activation='relu', name='classification_fc1')(rnn1)
        if bn: x = BatchNormalization(name='classification_fc1_bn')(x)
        # fully connected, output_dim=1024
        x = Dense(n_features, activation='relu', name='classification_fc2')(x)
        if bn: x = BatchNormalization(name='classification_fc2_bn')(x)
        # fully connected, output_dim=7, softmax activation
        classification = Dense(n_classes, activation='softmax', name='classification_fc3')(x)

        # affective dimensions network - outputs valence and arousal
        if output_emotion_dims:
            # fully connected, output_dim=1024
            x = Dense(n_features, activation='relu', name='dimension_fc1')(rnn1)
            if bn: x = BatchNormalization(name='dimension_fc1_bn')(x)
            # fully connected, output_dim=1024
            x = Dense(n_features, activation='relu', name='dimension_fc2')(x)
            if bn: x = BatchNormalization(name='dimension_fc2_bn')(x)
            # fully connected, output_dim=7, tanh activation
            dimension = Dense(2, activation='tanh', name='dimension_fc3')(x)

    # emission network - outputs the flat localization matrix
    if output_localisation:
        # fully connected, output_dim=6, zero weights and biases for the labels
        localisation = emission_network(rnn2)

    # define outputs
    outputs = []
    if output_localisation:
        outputs.append(localisation)
    outputs.append(h1)
    outputs.append(c1)
    if output_localisation:
        outputs.append(h2)
    if output_localisation:
        outputs.append(c2)
    if headless==False:
        outputs.append(classification)
        if output_emotion_dims:
            outputs.append(dimension)

    return Model(inputs, outputs, name='edram_cell')


def edram_model(input_shape=(100,100,1), batch_size=192, learning_rate=0.0001, steps=6,
                  glimpse_size=(26,26), coarse_size=(12,12), hidden_init=0,
                  n_filters=128, filter_sizes=(3,5), n_features=1024,
                  RNN_size_1=512, RNN_size_2=512, n_classes=7, output_mode=0,
                  use_init_matrix=True, output_emotion_dims=False, headless=False,
                  emission_bias=1, clip_value=1, init_abv=False, bn=True, dropout=0,
                  use_weighted_loss=False, localisation_cost_factor=1):
    """
        EDRAM network, takes an image and iteratively extracts image patches (glimpses)
        to make a classification

        Parameters:

            * input_shape: input image dimensions
            * learning_rate: ~
            * steps: number of iterations over the input (or size of the window for sequence processing)
            * glimpse_size: dimensions of the extracted image patch (the glimpse)
            * coarse_size: dimensions of the downscaled image for the initialization of the network
            * hidden_init: the value of the initial hidden state (leads to ValueError --> unused)
            * n_filters: determines the number of filters in the glimpse CNN
            * filter_sizes: dimensions of the glimpse and initialization and CNN kernels
            * n_features: learned features of the glimpse CNN (fc dimension)
            * RNN_size: number of cells in the LSTMs
            * n_classes: number of classes in the output
            * output_mode: 0 only the outputs of the last step are evaluated for the loss or
                           1 outputs of all time steps are concatenated for evaluation
                           2 special mode for predictions without emission network in the edram model
            * use_init_matrix: whether to use the initialization matrix for the first step
            * output_emotion_dims: whether to output valence and arousal for emotion stimuli
            * headless: uses the RNN1 hidden states for a final fully connected layer for predictions
            * emission_bias: initial zoom bias of the emission network
            * clip_value: max value of the zoom factor in the spatial transformer
            * init_abv: use normal initializers as in Ablavatsky et al. (2017)
            * bn: whether to use batch normalization
            * dropout: dropout percentage
            * use_weighted_loss: whether to use weighted versions of the losses, especially class weights
            * localisation_cost_factor: weighting of the localisation cost

        Returns:

            * itself

    """
    filter_size1, filter_size2 = list(zip(filter_sizes, filter_sizes))
    do = True if dropout>0 else False

    # define inputs

    # input image and localization matrix
    input_image = Input(shape=input_shape, dtype='float32', name='input_image')
    input_matrix = Input(shape=(6,), dtype='float32', name='input_matrix')

    # initial hidden states of the LSTMs
    # hidden_init = K.variable([hidden_init for i in range(0, RNN_size)]) Input(tensor = hidden_init, ...)
    init_h1 = Input(shape=(RNN_size_1,),  dtype='float32', name='initial_hidden_state_1')
    init_c1 = Input(shape=(RNN_size_1,),  dtype='float32', name='initial_cell_state_1')
    # init_h2 is generated by initialization network
    init_c2 = Input(shape=(RNN_size_2,),  dtype='float32', name='initial_cell_state_2')

    # bias matrices
    b26 = Input(shape=(26,26,1),  dtype='float32', name='b26')
    b24 = Input(shape=(24,24,1),  dtype='float32', name='b24')
    b12 = Input(shape=(12,12,1),  dtype='float32', name='b12')
    b8 = Input(shape=(8,8,1),  dtype='float32', name='b8')
    b6 = Input(shape=(6,6,1),  dtype='float32', name='b6')
    b4 = Input(shape=(4,4,1),  dtype='float32', name='b4')

    inputs=[input_image, input_matrix, init_h1, init_c1, init_c2, b26, b24, b12, b8, b6, b4]


    # define network building blocks

    if init_abv:
        dense_init = RandomNormal(0, 0.001*init_abv)
        # emission network (gets passed to the EDRAM cell for shared weights with the initialization network)
        emission_network = Dense(6, activation='tanh', kernel_initializer=dense_init, bias_initializer=array([1,0,0,0,1,0]), name='emission')
    else:
        # emission network (gets passed to the EDRAM cell for shared weights with the initialization network)
        emission_network = Dense(6, activation='tanh', weights=emission_weights(RNN_size_2, emission_bias), name='emission')

    # the EDRAM core cell
    output_localisation = False if output_mode==0 and steps==1 else True
    edram = edram_cell(input_shape, batch_size, glimpse_size, n_filters,
                       filter_size1, n_features, RNN_size_1, RNN_size_2,
                       n_classes, bn, dropout, clip_value, emission_network,
                       output_localisation, output_emotion_dims, headless)

    # initialization network
    # downscale the input, TODO: Do this in the preprocessing (but then coarse_size is fixed!)
    input_image_coarse = BilinearInterpolation(coarse_size, clip_value)([input_image, input_matrix])
    # 3 convolutions on 1x12x12 input: 5x5, 16 filters, 3x3, 16 filters, 3x3, 32 filters
    bn_axis = 3
    x = Conv2D(int(n_filters/8), filter_size2, padding='valid', use_bias=False, name='init_conv1')(input_image_coarse)
    b = LocallyConnected2D(int(n_filters/8), (1,1), padding='valid', use_bias=False, name='init_conv1_bias')(b8)
    x = add([x, b], name='init_conv1_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='init_conv1_bn')(x)
    x = Conv2D(int(n_filters/8), filter_size1, padding='valid', use_bias=False, name='init_conv2')(x)
    b = LocallyConnected2D(int(n_filters/8), (1,1), padding='valid', use_bias=False, name='init_conv2_bias')(b6)
    x = add([x, b], name='init_conv2_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='init_conv2_bn')(x)
    x = Conv2D(int(n_filters/4), filter_size1, padding='valid', use_bias=False, name='init_conv3')(x)
    b = LocallyConnected2D(int(n_filters/4), (1,1), padding='valid', use_bias=False, name='init_conv3_bias')(b4)
    x = add([x, b], name='init_conv3_add')
    if bn: x = BatchNormalization(axis=bn_axis, name='init_conv3_bn')(x)
    x = Flatten(name='init_flatten')(x)

    init_h2 = Reshape((RNN_size_2,), name = 'initial_hidden_state_2')(x)
    if use_init_matrix:
        init_matrix = emission_network(x)

    # assemble everything

    # step zero (initialization)
    step = [[init_matrix if output_mode==1 and use_init_matrix else input_matrix]]
    # step 1
    step.append(edram([input_image, init_matrix if use_init_matrix else input_matrix, init_h1, init_c1, init_h2, init_c2, b26, b24, b12, b6, b4]))
    # "recurrently" apply edram network
    for i in range(1, steps):
        step.append(edram([input_image, step[i][0], step[i][1], step[i][2], step[i][3], step[i][4], b26, b24, b12, b6, b4]))
    # use a dense classifier on all rnn1 outputs
    if headless:
        rnn1_avg = average([step[i][1] for i in range(1, steps+1)], name='temporal_average')
        rnn1_max = maximum([step[i][1] for i in range(1, steps+1)], name='temporal_maximum')
        rnn1 = concatenate([rnn1_avg, rnn1_max])

        # apply dropout
        if do: rnn1 = Dropout(dropout, name='classifications_dropout')(rnn1)

        # fully connected, output_dim=1024
        x = Dense(n_features, activation='relu', name='classifications_fc1')(rnn1)
        if do: x = Dropout(dropout, name='classifications_fc1_dropout')(x)
        if bn: x = BatchNormalization(name='classifications_fc1_bn')(x)
        # fully connected, output_dim=1024
        x = Dense(n_features, activation='relu', name='classifications_fc2')(x)
        if do: x = Dropout(dropout, name='classifications_fc2_dropout')(x)
        if bn: x = BatchNormalization(name='classifications_fc2_bn')(x)
        # fully connected, output_dim=7, softmax activation
        classifications = Dense(n_classes, activation='softmax', name='classifications')(x)

        # affective dimensions network - outputs valence and arousal
        if output_emotion_dims:
            # fully connected, output_dim=1024
            x = Dense(n_features, activation='relu', name='dimensions_fc1')(rnn1)
            if do: x = Dropout(dropout, name='dimensions_fc1_dropout')(x)
            if bn: x = BatchNormalization(name='dimensions_fc1_bn')(x)
            # fully connected, output_dim=1024
            x = Dense(n_features, activation='relu', name='dimensions_fc2')(x)
            if do: x = Dropout(dropout, name='dimensions_fc2_dropout')(x)
            if bn: x = BatchNormalization(name='dimensions_fc2_bn')(x)
            # fully connected, output_dim=7, tanh activation
            dimensions = Dense(2, activation='tanh', name='dimensions')(x)


    # define outputs
    if output_mode==0:
        # only use outputs of last time step
        if headless:
            classifications = Reshape((n_classes,), name='classifications')(step[steps][5])
            if output_emotion_dims:
                dimensions = Reshape((2,), name='dimensions')(step[steps][6])
            localisations = Reshape((6,), name='localisations')(step[steps-1][0])
        else:
            localisations = Reshape((6,), name='localisations')(step[steps-1][0])
    else:
        # concatenate outputs of different timesteps
        if steps==1:
            classifications = Reshape((n_classes,), name='classifications')(step[1][5])
            if output_emotion_dims:
                dimensions = Reshape((2,), name='dimensions')(step[1][6])
            localisations = Reshape((2, 6), name='localisations')(concatenate([step[0][0], step[1][0]]))
        else:
            if headless:
                localisations = Reshape((steps+use_init_matrix, 6), name='localisations')(concatenate([step[i][0] for i in range(1-use_init_matrix, steps+1)]))
            else:
                classifications = Reshape((steps, n_classes), name='classifications')(concatenate([step[i][5] for i in range(1, steps+1)]))
                if output_emotion_dims:
                    dimensions = Reshape((steps,2), name='dimensions')(concatenate([step[i][6] for i in range(1, steps+1)]))
                localisations = Reshape((steps+use_init_matrix, 6), name='localisations')(concatenate([step[i][0] for i in range(1-use_init_matrix, steps+1)]))

    if output_emotion_dims:
        outputs=[classifications, dimensions, localisations]
    else:
        outputs=[classifications, localisations]
    # build the model
    model = Model(inputs, outputs, name='edram_model')


    # define training framework

    # optimization algorithm
    optimizer = Adam(lr=learning_rate, clipnorm=10.)

    # define losses
    if use_weighted_loss:
        # weighted losses
        if n_classes==7:
            # only use emotion weights for emotion classification
            classification_loss = weighted_categorical_crossentropy(emotion_weights)
        else:
            classification_loss = 'categorical_crossentropy'
        dimension_loss = weighted_mean_squared_error(emotion_dimension_weights)
        localisation_loss = weighted_mean_squared_error(localisation_weights)
    else:
        # standard losses
        classification_loss = 'categorical_crossentropy'
        dimension_loss = 'mean_squared_error'
        localisation_loss = 'mean_squared_error'

    # compile the model
    if output_emotion_dims:
        model.compile(loss={'classifications': classification_loss, 'dimensions': dimension_loss,'localisations': localisation_loss},
                      loss_weights={'classifications': 1, 'dimensions': 1, 'localisations': localisation_cost_factor},
                      metrics={'classifications': 'categorical_accuracy'}, optimizer=optimizer)
    else:
        model.compile(loss={'classifications': classification_loss, 'localisations': localisation_loss},
                      loss_weights={'classifications': 1, 'localisations': localisation_cost_factor},
                      metrics={'classifications': 'categorical_accuracy'}, optimizer=optimizer)

    return model


# %%
def STN_model(input_shape=(100, 100, 1), glimpse_size=(26, 26), n_classes=10,
        learning_rate=0.0001, output_mode=0, use_weighted_loss=False):
    """

        A simple Spatial Transformer CNN

    """

    input_image = Input(shape=input_shape, name='input_image')

    # emission network
    locnet = MaxPooling2D(pool_size=(2, 2))(input_image)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = MaxPooling2D(pool_size=(2, 2))(locnet)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = Flatten()(locnet)
    locnet = Dense(50)(locnet)
    locnet = Activation('relu')(locnet)
    weights = emission_weights(50)
    localisations = Dense(6, weights=weights, name='localisations')(locnet)

    # glimpse CNN
    x = BilinearInterpolation(glimpse_size)([input_image, localisations])

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    classifications = Dense(n_classes, activation='softmax', name='classifications')(x)

    # define training framework
    sgd = Adam(lr=learning_rate, clipnorm=10.)
    if use_weighted_loss:
        classification_loss = weighted_categorical_crossentropy(emotion_weights)
        localisation_loss = weighted_mean_squared_error(localisation_weights)
    else:
        classification_loss = 'categorical_crossentropy'
        localisation_loss = 'mean_squared_error'
    if output_mode==0:
        # only use classification
        model = Model(inputs=input_image, outputs=classifications)
        model.compile(loss={'classifications': classification_loss},
                      metrics={'classifications': 'categorical_accuracy'},
                      optimizer=sgd)
    else:
        # + localisations
        model = Model(inputs=input_image, outputs=[classifications, localisations])
        model.compile(loss={'classifications': classification_loss, 'localisations': localisation_loss},
                      metrics={'classifications': 'categorical_accuracy'},
                      optimizer=sgd)

    return model


def big_STN_model(input_shape=(100, 100, 1), glimpse_size=(26, 26), n_classes=10,
        learning_rate=0.0001, output_mode=0, use_weighted_loss=False):
    """

        A simple Spatial Transformer CNN

    """

    input_image = Input(shape=input_shape, name='input_image')

    # emission network
    """
    locnet = Conv2D(24, (5, 5), activation='relu')(input_image)
    locnet = MaxPooling2D(pool_size=(2, 2))(locnet)
    locnet = Conv2D(48, (5, 5), activation='relu')(locnet)
    locnet = MaxPooling2D(pool_size=(2, 2))(locnet)
    locnet = Flatten()(locnet)
    locnet = Dense(64, activation='relu')(locnet)
    weights = emission_weights(64)
    """

    locnet = MaxPooling2D(pool_size=(2, 2))(input_image)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = MaxPooling2D(pool_size=(2, 2))(locnet)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = Flatten()(locnet)
    locnet = Dense(50, activation='relu')(locnet)
    weights = emission_weights(50)
    localisations = Dense(6, weights=weights, name='localisations')(locnet)

    # glimpse CNN
    x = BilinearInterpolation(glimpse_size)([input_image, localisations])

    n_filters = 64
    filter_size = (3,3)

    x = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(n_filters, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    classifications = Dense(n_classes, activation='softmax', name='classifications')(x)

    # define training framework
    sgd = Adam(lr=learning_rate, clipnorm=10.)
    if use_weighted_loss:
        classification_loss = weighted_categorical_crossentropy(emotion_weights)
        localisation_loss = weighted_mean_squared_error(localisation_weights)
    else:
        classification_loss = 'categorical_crossentropy'
        localisation_loss = 'mean_squared_error'
    if output_mode==0:
        # only use classification
        model = Model(inputs=input_image, outputs=classifications)
        model.compile(loss={'classifications': classification_loss},
                      metrics={'classifications': 'categorical_accuracy'},
                      optimizer=sgd)
    else:
        # + localisations
        model = Model(inputs=input_image, outputs=[classifications, localisations])
        model.compile(loss={'classifications': classification_loss, 'localisations': localisation_loss},
                      metrics={'classifications': 'categorical_accuracy'},
                      optimizer=sgd)

    return model
