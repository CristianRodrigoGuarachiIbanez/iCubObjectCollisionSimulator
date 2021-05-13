import os
import random
import numpy as np

from h5py import File
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau

from config import config, datasets
from batch_generator import batch_generator
from models.model_keras import edram_model, tedram_model, STN_model, big_STN_model


def train(list_params, gpu_id, dataset_id, model_id, load_path, save_path,
         batch_size, learning_rate, n_epochs, augment_input, rotation, n_steps,
         glimpse_size, coarse_size, conv_sizes, n_filters, fc_dim, enc_dim, dec_dim,
         n_classes, output_mode, use_init_matrix, output_emotion_dims, headless,
         emission_bias, clip_value, unique_emission, unique_glimpse, init_abv,
         scale_inputs, normalize_inputs, use_batch_norm, dropout,
         use_weighted_loss, localisation_cost_factor, zoom_factor):

    glimpse_size = (glimpse_size, glimpse_size)
    coarse_size = (coarse_size, coarse_size)

    # train on ANet or MNIST (0)?
    if dataset_id > 0:
        n_classes = 7
    # train on high-res input?
    if dataset_id<2:
        input_shape = config['input_shape']
    elif dataset_id==2:
        input_shape = config['input_shape_400']
    else:
        input_shape = config['input_shape_200']
    # create output directory
    save_path = './output/'+save_path+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # select a GPU
    print("\n[Info] Using GPU", gpu_id)
    if gpu_id == -1:
        print('[Error] You need to select a gpu. (e.g. python train.py --gpu=2)\n')
        exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # create the EDRAM model
    print("[Info] Creating the model...")

    if (load_path != '.'):
        model_path = load_path + 'model.h5'
        print ("[Info] Loading the model from:", model_path,"\n")
        try:
            model = load_model(model_path)
        except:
            model = edram_model(input_shape, batch_size, learning_rate, n_steps,
                                glimpse_size, coarse_size, hidden_init=0,
                                n_filters=128, filter_sizes=(3,5), n_features=fc_dim,
                                RNN_size_1=enc_dim, RNN_size_2=dec_dim, n_classes=n_classes,
                                output_mode=output_mode, use_init_matrix=use_init_matrix,
                                output_emotion_dims=output_emotion_dims, headless=headless,
                                emission_bias=emission_bias, clip_value=clip_value, init_abv=init_abv,
                                bn=use_batch_norm, dropout=dropout, use_weighted_loss=use_weighted_loss,
                                localisation_cost_factor=localisation_cost_factor)
            model.load_weights(load_path+'model_weights.h5')
    else:
        if model_id==1:
            model = edram_model(input_shape, batch_size, learning_rate, n_steps,
                                glimpse_size, coarse_size, hidden_init=0,
                                n_filters=128, filter_sizes=(3,5), n_features=fc_dim,
                                RNN_size_1=enc_dim, RNN_size_2=dec_dim, n_classes=n_classes,
                                output_mode=output_mode, use_init_matrix=use_init_matrix,
                                output_emotion_dims=output_emotion_dims, headless=headless,
                                emission_bias=emission_bias, clip_value=clip_value,
                                bn=use_batch_norm, dropout=dropout, use_weighted_loss=use_weighted_loss,
                                localisation_cost_factor=localisation_cost_factor)
        elif model_id==2:
            model = tedram_model(input_shape, batch_size, learning_rate, n_steps,
                                glimpse_size, coarse_size, hidden_init=0,
                                n_filters=128, filter_sizes=(3,5), n_features=fc_dim,
                                RNN_size_1=enc_dim, RNN_size_2=dec_dim, n_classes=n_classes,
                                output_mode=output_mode, use_init_matrix=use_init_matrix,
                                output_emotion_dims=output_emotion_dims,
                                emission_bias=emission_bias, clip_value=clip_value,
                                unique_emission=unique_emission, unique_glimpse=unique_glimpse,
                                bn=use_batch_norm, dropout=dropout,
                                use_weighted_loss=use_weighted_loss,
                                localisation_cost_factor=localisation_cost_factor)
        elif model_id==3:
            model = STN_model(config['input_shape'], glimpse_size, n_classes,
                              learning_rate, use_weighted_loss=use_weighted_loss)
        elif model_id==4:
            model = big_STN_model(config['input_shape'], glimpse_size, n_classes,
                              learning_rate, use_weighted_loss=use_weighted_loss)
        else:
            print('[Error] Only models 1 through 4 are available!\n')
            exit()
    # model summary
    if list_params=='all' and model_id==1:
        model.get_layer('edram_cell').summary()
    elif list_params!='none':
        model.summary()

    # --------- load the data
    data_path: str = datasets[dataset_id]
    print("\n[Info] Opening", data_path)
    data: File = None;
    try:
        data = File(data_path, 'r')
    except Exception as e:
        print("[Error]", e)
        exit()
    # --------- split into train and test set
    n_train, n_test, t = 0, 0, 0 #type: int, int, int
    if dataset_id==0:
        n_train = 60000
        n_test = 10000
    elif dataset_id==1:
        n_train = 282595
        n_test = data['features'].shape[0] - n_train
    else:
        t = 100000
        train = data['Train']
        while train[t] == b'train':
            t = t+1
        n_train = t-1
        n_test = data['Y_lab'].shape[0] - n_train

    if dataset_id<2:
        train_images = data['features'][:n_train]
        train_labels = data['labels'][:n_train]
        train_locations = data['locations'][:n_train]
        if output_emotion_dims:
            train_dims1 = data['dimensions'][:n_train]
            train_dims2 = None
        else:
            train_dims1 = None
            train_dims2 = None
        test_images = data['features'][n_train:]
        test_labels = data['labels'][n_train:]
        test_locations = data['locations'][n_train:]
        if output_emotion_dims:
            test_dims1 = data['dimensions'][n_train:]
            test_dims2 = None
        else:
            test_dims1 = None
            test_dims2 = None
    else:
        n = n_train//1
        train_images = data['X'][:n]
        train_labels = data['Y_lab'][:n]
        train_locations = None
        if output_emotion_dims:
            train_dims1 = data['Y_val'][:n]
            train_dims2 = data['Y_ars'][:n]
            train_dims1 = np.reshape(train_dims1, (train_dims1.shape[0],1))
            train_dims2 = np.reshape(train_dims2, (train_dims2.shape[0],1))
        else:
            train_dims1 = None
            train_dims2 = None
        test_images = data['X'][n_train:]
        test_labels = data['Y_lab'][n_train:]
        test_locations = None
        if output_emotion_dims:
            test_dims1 = data['Y_val'][n_train:]
            test_dims2 = data['Y_ars'][n_train:]
            test_dims1 = np.reshape(test_dims1, (test_dims1.shape[0],1))
            test_dims2 = np.reshape(test_dims2, (test_dims2.shape[0],1))
        else:
            test_dims1 = None
            test_dims2 = None

    # normalize input data
    if normalize_inputs:
        indices = list(range(n_train))
        random.shuffle(indices)
        samples = train_images[sorted(indices[:1000]), ...]/scale_inputs

        train_mean = np.mean(samples, axis=0)
        train_sd = np.std(samples, axis=0).clip(min=0.00001)

        indices = list(range(n_test))
        random.shuffle(indices)
        samples = test_images[sorted(indices[:1000]), ...]/scale_inputs

        test_mean = np.mean(samples, axis=0)
        test_sd = np.std(samples, axis=0).clip(min=0.00001)
    else:
        train_mean = 0
        train_sd = 1
        test_mean = 0
        test_sd = 1

    print("[Info] Dataset Size\n")
    print(" ", n_train, "training examples")
    print(" ", n_test, "test examples")

    print("\n[Info] Data Dimensions\n")
    print("  Images:   ", train_images.shape[1], "x", train_images.shape[2], "x", train_images.shape[3])
    print("  Labels:   ", train_labels.shape[1])
    if train_locations is not None:
        print("  Locations:", train_locations.shape[1],"\n")
    else:
        print("  Locations:", 6,"\n")

    # create callbacks
    history = History()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.333, patience=1, min_lr=0.00001, verbose=0)
    checkpoint = ModelCheckpoint(save_path+'checkpoint_weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)

    # create data generator for data augmentation
    datagen = None
    if augment_input:
        datagen = ImageDataGenerator(rotation_range=int(20*rotation),
                                 width_shift_range=(0.05+0.10 if dataset_id==1 else 0)*rotation,
                                 height_shift_range=(0.05+0.10 if dataset_id==1 else 0)*rotation,
                                 zoom_range=(0.10+0.10 if dataset_id==1 else 0)*rotation,
                                 shear_range=(0.20-0.10 if dataset_id==1 else 0)*rotation,
                                 horizontal_flip=True if dataset_id==1 else False,
                                 fill_mode='nearest')

    # train the model
    try:
        hist = model.fit_generator(
            batch_generator(n_train, batch_size, (enc_dim, dec_dim), n_steps,
                            train_images, train_labels, train_dims1, train_dims2, train_locations,
                            datagen, scale_inputs, normalize_inputs, train_mean, train_sd,
                            output_mode, use_init_matrix, headless, model_id, glimpse_size,
                            zoom_factor),
            steps_per_epoch=int(n_train/batch_size),
            epochs=n_epochs,
            verbose=1,
            callbacks=[history, reduce_lr, checkpoint],
            validation_data=batch_generator(n_test, batch_size, (enc_dim, dec_dim), n_steps,
                                            test_images, test_labels, test_dims1, test_dims2, test_locations,
                                            None, scale_inputs, normalize_inputs, test_mean, test_sd,
                                            output_mode, use_init_matrix, headless, model_id,
                                            glimpse_size, zoom_factor),
            validation_steps=int(n_test/batch_size)
        )
    except KeyboardInterrupt:
        pass

    # save the history
    if not os.path.exists(save_path+'/history'):
        os.makedirs(save_path+'/history')
    np.save(save_path+'/history/history', history.history)
    for key in history.history.keys():
        np.save(save_path+'/history/'+str(key), history.history[key])

    # save the model
    print('\n[Info] Saving the model...')
    model.save(save_path+'/model.h5')
    model.save_weights(save_path+'/model_weights.h5')