config = {

    # size of the input faces
    'input_shape': (100, 100, 1),
    'input_shape_200': (200, 200, 1),
    'input_shape_400': (400, 400, 1),

    # Paths to the datasets
    'path_Anet': "/scratch/facs_data/AffectNet/AffectNet_train_data_keras.hdf5",

    'path_Anet_400': "/scratch/facs_data/AffectNet/AffectNet_training_data_EDRAM_400.h5",
    'path_Anet_200': "/scratch/facs_data/AffectNet/AffectNet_training_data_EDRAM_200.h5",

    'path_mnist_cluttered': "/scratch/forch/tEDRAM/datasets/mnist_cluttered_keras.hdf5",


}

datasets = [config['path_mnist_cluttered'],
            config['path_Anet'],
            config['path_Anet_400'],
            config['path_Anet_200']]