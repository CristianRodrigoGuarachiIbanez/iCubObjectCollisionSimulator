from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, MaxPooling2D, Flatten, Conv2D, Dense
from tensorflow.keras.optimizers import Adam, SGD

from .utils import get_initial_weights
from .layers import BilinearInterpolation


def STN(input_shape=(100, 100, 1), sampling_size=(26, 26), n_classes=10,
        learning_rate=0.0001, output_mode=0):

    input_image = Input(shape=input_shape, name='input_image')
    locnet = MaxPooling2D(pool_size=(2, 2))(input_image)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = MaxPooling2D(pool_size=(2, 2))(locnet)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = Flatten()(locnet)
    locnet = Dense(50)(locnet)
    locnet = Activation('relu')(locnet)
    weights = get_initial_weights(50)
    localisations = Dense(6, weights=weights)(locnet)

    x = BilinearInterpolation(sampling_size)([input_image, localisations])
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

    sgd = Adam(lr=learning_rate, clipnorm=10.)

    if output_mode==0:
        model = Model(inputs=input_image, outputs=classifications)
        model.compile(loss={'classifications': 'categorical_crossentropy'},
                      metrics={'classifications': 'categorical_accuracy'},
                      optimizer=sgd)
    else:
        model = Model(inputs=input_image, outputs=[classifications, localisations])
        model.compile(loss={'classifications': 'categorical_crossentropy', 'localisations': 'mean_squared_error'},
                      metrics={'classifications': 'categorical_accuracy'},
                      optimizer=sgd)

    return model
