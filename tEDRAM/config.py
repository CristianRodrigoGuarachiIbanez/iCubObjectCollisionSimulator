config = {

    # size of the input images
    'input_shape_binocular': (10, 120,160),
    'input_shape_scene': (10, 126, 160),
    #'input_shape': (10, 126, 160),

    # Paths to the datasets
    'path_scene_dataset': "/scratch/gucr/training_data/training_images.h5",
    'path_scene_labels': "/scratch/gucr/training_data/data_left_side",
    'path_binocular_labels': "/scratch/gucr/training_data/data_right_side",

    'path_Anet': "/scratch/facs_data/AffectNet/AffectNet_train_data_keras.hdf5",
    'path_Anet_400': "/scratch/facs_data/AffectNet/AffectNet_training_data_EDRAM_400.h5",
    'path_Anet_200': "/scratch/facs_data/AffectNet/AffectNet_training_data_EDRAM_200.h5",
    'path_mnist_cluttered': "/scratch/forch/tEDRAM/datasets/mnist_cluttered_keras.hdf5",

}

datasets = [config['path_dataset'],
            config['path_scene_labels'],
            config['path_binocular_labels'],

            config['path_mnist_cluttered'],
            config['path_Anet'],
            config['path_Anet_400'],
            config['path_Anet_200']]