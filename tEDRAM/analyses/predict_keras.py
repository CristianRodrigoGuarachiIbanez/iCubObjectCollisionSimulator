from __future__ import print_function
import os
import h5py
import random
import numpy as np
from argparse import ArgumentParser
from fuel.converters.base import fill_hdf5_file

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from models.model_keras import edram_model, tedram_model, STN_model
from config import config

# default training parameters
_batch_size = 192
_model_id = 2
_n_steps = 6

# paths to the datasets
datasets = [config['path_mnist_cluttered'],
            config['path_Anet'],
            config['path_Anet_400'],
            config['path_Anet_200']]

emotion_labels = ["Neutral","Happy","Sad","Surprise","Fear","Disgust","Anger"]


def main(list_params, gpu_id, dataset_id, model_id, use_checkpoint_weights,
         load_path, batch_size, n_steps, glimpse_size, coarse_size, conv_sizes,
         n_filters, fc_dim, enc_dim, dec_dim, n_classes, clip_value, unique_emission,
         unique_glimpse, output_mode, use_init_matrix, output_emotion_dims, headless,
         scale_inputs, normalize_inputs, use_batch_norm, dropout, weighting,
         iterations, show_steps, zoom_factor):

    # mode = 0 if output_init_matrix==0 and mode==0 else 1
    if dataset_id>0:
        n_classes = 7
    if dataset_id<2:
        input_shape = config['input_shape']
    elif dataset_id==2:
        input_shape = config['input_shape_400']
    else:
        input_shape = config['input_shape_200']

    glimpse_size = (glimpse_size, glimpse_size)
    coarse_size = (coarse_size, coarse_size)

    # select a GPU
    print("[Info] Using GPU", gpu_id)
    if gpu_id == -1:
        print('[Error] You need to select a gpu. (e.g. python train.py --gpu=7)\n')
        exit()
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # create the model
    print ("\n[Info] Loading the model from:", load_path + ("model_weights.h5" if use_checkpoint_weights==0 else "checkpoint_weights.h5"))
    print()
    if model_id==1:
        model = edram_model(input_shape, learning_rate=1, steps=n_steps,
                            glimpse_size=glimpse_size, coarse_size=coarse_size,
                            hidden_init=0, n_filters=128, filter_sizes=(3,5),
                            n_features=fc_dim, RNN_size_1=enc_dim, RNN_size_2=dec_dim,
                            n_classes=n_classes, output_mode=output_mode,
                            use_init_matrix=use_init_matrix, clip_value=clip_value,
                            output_emotion_dims=output_emotion_dims, headless=headless,
                            bn=use_batch_norm, dropout=dropout,
                            use_weighted_loss=False, localisation_cost_factor=1)
    elif model_id==2:
        model = tedram_model(input_shape, learning_rate=1, steps=n_steps,
                            glimpse_size=glimpse_size, coarse_size=coarse_size,
                            hidden_init=0, n_filters=128, filter_sizes=(3,5),
                            n_features=fc_dim, RNN_size_1=enc_dim, RNN_size_2=dec_dim,
                            n_classes=n_classes, output_mode=output_mode,
                            use_init_matrix=use_init_matrix, clip_value=clip_value,
                            output_emotion_dims=output_emotion_dims,
                            unique_emission=unique_emission, unique_glimpse=unique_glimpse,
                            bn=use_batch_norm, dropout=dropout, use_weighted_loss=False,
                            localisation_cost_factor=1)
    elif model_id==3:
        model = STN_model(learning_rate=1, n_classes=n_classes,
                          use_weighted_loss=False, output_mode=1)
    else:
        print('[Error] Only model 1 and 2 is available!\n')
        exit()
    # load weights
    if use_checkpoint_weights:
        model.load_weights(load_path+'checkpoint_weights.h5')
    else:
        model.load_weights(load_path+'model_weights.h5')

    # load the data
    data_path = datasets[dataset_id]
    print("\n[Info] Opening", data_path)

    try:
        data = h5py.File(data_path, 'r')

    except Exception:
        print("[Error]", data_path, "does not exist.\n")
        exit()

    if dataset_id==0:
        n_train = 60000
        n_test = 10000
    elif dataset_id==1:
        n_train = data['features'].shape[0] - 3482
        n_test = data['features'].shape[0] - n_train
    else:
        n_train = data['X'].shape[0] - 3462
        n_test = data['X'].shape[0] - n_train

    if dataset_id<2:

        features = data['features'][n_train:]
        labels = data['labels'][n_train:]
        locations = data['locations'][n_train:]
        if output_emotion_dims:
            dims1 = data['dimensions'][n_train:]
            dims2 = None
        else:
            dims1 = None
            dims2 = None
    else:

        features = data['X'][n_train:]
        labels = data['Y_lab'][n_train:]
        locations = None
        if output_emotion_dims:
            dims1 = data['Y_val'][n_train:]
            dims2 = data['Y_ars'][n_train:]
            dims1 = np.reshape(dims1, (dims1.shape[0],1))
            dims2 = np.reshape(dims2, (dims2.shape[0],1))
        else:
            dims1 = None
            dims2 = None

    # normalize input data
    if normalize_inputs:
        indices = list(range(n_test))
        random.shuffle(indices)
        samples = features[sorted(indices[:1000]), ...]/scale_inputs

        mean = np.mean(samples, axis=0)
        sd = np.std(samples, axis=0).clip(min=0.00001)
    else:
        mean = 0
        sd = 1

    print("[Info] Dataset Size\n")
    print(" using", iterations,"*", batch_size, "out of", n_test, "test examples")

    print("\n[Info] Data Dimensions\n")
    print("  Images:   ", features.shape[1], "x", features.shape[2], "x", features.shape[3])
    print("  Labels:   ", labels.shape[1])
    if locations is not None:
        print("  Locations:", locations.shape[1],"\n")
    else:
        print("  Locations:", 6,"\n")


    predicted_labels, predicted_dimensions, predicted_locations = [], [], []

    # get sample data
    indices = list(range(n_test))
    random.shuffle(indices)
    samples = sorted(indices[0:batch_size*iterations])

    # prepare the minibatch

    # input image
    if scale_inputs!=1 and scale_inputs!=0:
        I = np.array(features[samples, ...], dtype='float32')/scale_inputs
    if normalize_inputs:
        I = (np.array(features[samples, ...], dtype='float32')-mean)/sd
    else:
        I = np.array(features[samples, ...], dtype='float32')
    # transformation matrix with zoom paramters set to 1
    A = np.zeros((batch_size*iterations, 6), dtype='float32')
    A[:, (0,4)] = 1
    # initial RNN states
    S1 = np.zeros((batch_size*iterations, enc_dim), dtype='float32')
    S2 = np.zeros((batch_size*iterations, dec_dim), dtype='float32')
    # biases
    if glimpse_size==(26,26):
        B1 = np.ones((batch_size*iterations, 26, 26, 1), dtype='float32')
        B2 = np.ones((batch_size*iterations, 24, 24, 1), dtype='float32')
        B3 = np.ones((batch_size*iterations, 12, 12, 1), dtype='float32')
        B4 = np.ones((batch_size*iterations, 8, 8, 1), dtype='float32')
        B5 = np.ones((batch_size*iterations, 6, 6, 1), dtype='float32')
        B6 = np.ones((batch_size*iterations, 4, 4, 1), dtype='float32')
    else:
        B1 = np.ones((batch_size*iterations, 16, 16, 1), dtype='float32')
        B2 = np.ones((batch_size*iterations, 16, 16, 1), dtype='float32')
        B3 = np.ones((batch_size*iterations, 8, 8, 1), dtype='float32')
        B4 = np.ones((batch_size*iterations, 8, 8, 1), dtype='float32')
        B5 = np.ones((batch_size*iterations, 6, 6, 1), dtype='float32')
        B6 = np.ones((batch_size*iterations, 4, 4, 1), dtype='float32')
    # concatenation of target outputs for every step
    Y_cla = np.array(labels[samples, ...], dtype='float32')
    if zoom_factor==1:
        Y_loc = np.array(locations[samples, ...], dtype='float32')
    else:
        Y_loc = np.zeros((batch_size*iterations,6), dtype='float32')
        Y_loc[:,(0,4)] = zoom_factor
    if dims1 is not None:
        if dims2 is None:
            Y_dim = np.array(dims1[samples, ...], dtype='float32')
        else:
            Y_dim = np.array(np.hstack([dims1[samples, ...], dims2[samples, ...]]), dtype='float32')

    if model_id==1 or model_id==2:
        inputs = {'input_image': I, 'input_matrix': A,
                  'initial_hidden_state_1': S1, 'initial_cell_state_1': S1,
                  'initial_cell_state_2': S2,
                  'b26': B1, 'b24': B2, 'b12': B3, 'b8': B4, 'b6': B5, 'b4': B6}
        if dims1 is not None:
            outputs = {'classifications': Y_cla, 'dimensions': Y_dim, 'localisations': Y_loc}
        else:
            outputs = {'classifications': Y_cla, 'localisations': Y_loc}
    elif model_id==3:
        inputs = {'input_image': I}
        outputs = {'classifications': Y_cla}

    if dims1 is not None:
        predicted_labels, predicted_dimensions, predicted_locations = model.predict(inputs, batch_size=batch_size, verbose=1)
    else:
        predicted_labels, predicted_locations = model.predict(inputs, batch_size=batch_size, verbose=1)

    batch_size = batch_size*iterations

    # reshape
    if model_id==1 or model_id==2:
        if output_mode:
            predicted_locations = np.vstack([predicted_locations[:,i,:] for i in range(0, n_steps+use_init_matrix)])
        if n_steps>1 and headless==False:
            predicted_labels = np.vstack([predicted_labels[:,i,:] for i in range(0, n_steps)])
            if dims1 is not None:
                predicted_dimensions = np.vstack([predicted_dimensions[:,i,:] for i in range(0, n_steps)])

    # save smaple data and predictions
    h5file = h5py.File(load_path+'predictions.h5', mode='w')

    if dims1 is not None:
        data = (
                ('true','features', np.array(features[samples, ...], dtype='float32')),
                ('normalized','features', np.array(I, dtype='float32')),
                ('true', 'locations', np.array(Y_loc, dtype='float32')),
                ('predicted', 'locations', np.array(predicted_locations, dtype='float32')),
                ('true', 'dimension', np.array(Y_dim, dtype='float32')),
                ('predcited', 'dimensions', np.array(predicted_dimensions, dtype='float32')),
                ('true', 'labels', np.array(Y_cla, dtype='float32')),
                ('predcited', 'labels', np.array(predicted_labels, dtype='float32')),
        )
    else:
        data = (
                ('true','features', np.array(features[samples, ...], dtype='float32')),
                ('normalized','features', np.array(I, dtype='float32')),
                ('true', 'locations', np.array(Y_loc, dtype='float32')),
                ('predicted', 'locations', np.array(predicted_locations, dtype='float32')),
                ('true', 'labels', np.array(Y_cla, dtype='float32')),
                ('predcited', 'labels', np.array(predicted_labels, dtype='float32')),
        )
    fill_hdf5_file(h5file, data)

    h5file.flush()
    h5file.close()

    print("\n[INFO] Saved data to", load_path+'predictions.h5',"\n")

    # some statistics
    hist = np.zeros(n_classes, dtype='int')
    acc = np.zeros((1 if headless else n_steps, n_classes), dtype='int')
    acc_avg = np.zeros((1 if headless else n_steps, n_classes), dtype='int')
    pos = np.zeros((n_steps+use_init_matrix, n_classes, 2), dtype='float')
    zoom = np.zeros((n_steps+use_init_matrix, n_classes, 2), dtype='float')
    mse_pos = np.zeros((n_steps+use_init_matrix, n_classes), dtype='float')
    mse_zoom = np.zeros((n_steps+use_init_matrix, n_classes), dtype='float')
    val_ars = np.zeros((n_steps, n_classes, 2), dtype='float')
    mse_val = np.zeros((n_steps, n_classes), dtype='float')
    mse_ars = np.zeros((n_steps, n_classes), dtype='float')

    if weighting:
        # average predictions per step in inverted order
        for j in range(0, n_steps):
            k = 0
            predicted_labels_avg = predicted_labels[(n_steps-1)*batch_size:(n_steps)*batch_size,:]
            for k in range(1,j+1):
                predicted_labels_avg += predicted_labels[(n_steps-1-k)*batch_size:(n_steps-k)*batch_size,:]
            predicted_labels_avg /= k+1
            for i in range(0, batch_size):
                # count correct classifications per class
                if np.argmax(Y_cla[i,:])==np.argmax(predicted_labels_avg[i,:]):
                    acc_avg[j,:] = acc_avg[j,:] + Y_cla[i,:]

    for i in range(0, batch_size):
        # count class occurences
        hist = hist + Y_cla[i,:]
        # count correct classifications per class
        for j in range(0, 1 if headless else n_steps):
            if np.argmax(Y_cla[i,:])==np.argmax(predicted_labels[i+j*batch_size,:]):
                acc[j,:] = acc[j,:] + Y_cla[i,:]

    # compute accuracy
    acc_mean = np.zeros(1 if headless else n_steps)
    for j in range(0, 1 if headless else n_steps):
        acc_mean[j] = np.dot(hist/batch_size, acc[j,:]/hist)
    hist[hist==0] = 0.00000001
    acc = np.asarray(acc*100/(hist), dtype='int')/100
    acc_avg = np.asarray(acc_avg*100/(hist), dtype='int')/100
    hist[hist<1] = 0
    # compute bb info per class and mse
    for j in range(0, n_steps+use_init_matrix):
        for i in range(0, n_classes):
            pos[j,i,:] = np.mean(predicted_locations[j*batch_size:(j+1)*batch_size,:][Y_cla[:,i]==1,:][:,(2,5)], axis=0)
            zoom[j,i,:] = np.mean(predicted_locations[j*batch_size:(j+1)*batch_size,:][Y_cla[:,i]==1,:][:,(0,4)], axis=0)
            mse_pos[j,i] = np.mean(np.square(Y_loc[Y_cla[:,i]==1,:][:,(2,5)] - predicted_locations[j*batch_size:(j+1)*batch_size,:][Y_cla[:,i]==1,:][:,(2,5)]))
            mse_zoom[j,i] = np.mean(np.square(Y_loc[Y_cla[:,i]==1,:][:,(0,4)] - predicted_locations[j*batch_size:(j+1)*batch_size,:][Y_cla[:,i]==1,:][:,(0,4)]))
    # compute mean dimensional ratings and mse
    if dims1 is not None:
        for j in range(0, n_steps):
            for i in range(0, n_classes):
                val_ars[j,i,:] = np.mean(predicted_dimensions[j*batch_size:(j+1)*batch_size,:][Y_cla[:,i]==1,:], axis=0)
                mse_val[j,i] = np.mean(np.square(Y_dim[Y_cla[:,i]==1,:][:,0] - predicted_dimensions[j*batch_size:(j+1)*batch_size,:][Y_cla[:,i]==1,:][:,0]))
                mse_ars[j,i] = np.mean(np.square(Y_dim[Y_cla[:,i]==1,:][:,1] - predicted_dimensions[j*batch_size:(j+1)*batch_size,:][Y_cla[:,i]==1,:][:,1]))

    print("Sample Class Distribution:", hist,"\n")
    print("Accuracy per Class:")
    for j in range(0, 1 if headless else n_steps):
        print("  Step "+str(j+1)+":                 ", acc[j,:], "= %.3f" % acc_mean[j])
    if weighting:
        print("\nWeighted Accuracy per Class:")
        for j in range(0, n_steps):
            print("  Step "+str(n_steps-j)+" to "+str(n_steps)+":             ", acc_avg[n_steps-1-j,:], "= %.3f" % np.dot(hist/batch_size, acc_avg[n_steps-1-j,:]))
    if dims1 is not None:
        print("\nValence Error per Class:")
        for j in range(n_steps-show_steps, n_steps):
            print("  Step "+str(j+1)+":                 ", np.asarray(mse_val[j,:]*100, dtype='int')/100, "= %.3f" % np.dot(hist/batch_size, mse_val[j,:]))
        print("\nAverage Valence per Class:")
        for j in range(n_steps-show_steps, n_steps):
            print("  Step "+str(j+1)+":                 ", np.asarray(val_ars[j,:,0]*100, dtype='int')/100, "= %.3f" % np.dot(hist/batch_size, val_ars[j,:,0]))
        print("\nArousal Error per Class:")
        for j in range(n_steps-show_steps, n_steps):
            print("  Step "+str(j+1)+":                 ", np.asarray(mse_ars[j,:]*100, dtype='int')/100, "= %.3f" % np.dot(hist/batch_size, mse_ars[j,:]))
        print("\nAverage Arousal per Class:")
        for j in range(n_steps-show_steps, n_steps):
            print("  Step "+str(j+1)+":                 ", np.asarray(val_ars[j,:,1]*100, dtype='int')/100, "= %.3f" % np.dot(hist/batch_size, val_ars[j,:,1]))
    print("\nAverage Position per Class:")
    for j in range(0, n_steps+use_init_matrix):
        print("  Step "+str(j)+":                 ", np.asarray(pos[j,:,0]*100, dtype='int')/100, "= %.3f" % np.dot(hist/batch_size, pos[j,:,0]))
        print("                          ", np.asarray(pos[j,:,1]*100, dtype='int')/100, "= %.3f" % np.dot(hist/batch_size, pos[j,:,1]))
    print("\nPosition Variance: %.3f" % np.var(np.asarray(pos)), "  Position SD: %.3f" % np.std(np.asarray(pos)))
    if False:
        print("\nLocation Error per Class:")
        for j in range(0, n_steps+use_init_matrix):
            print("  Step "+str(j)+":                 ", np.asarray(mse_pos[j,:]*100, dtype='int')/100, "= %.3f" % np.dot(hist/batch_size, mse_pos[j,:]))
        print("\nZoom Error per Class:")
        for j in range(0, n_steps+use_init_matrix):
            print("  Step "+str(j)+":                 ", np.asarray(mse_zoom[j,:]*100, dtype='int')/100, "= %.3f" % np.dot(hist/batch_size, mse_zoom[j,:]))
    print("\nAverage Zoom per Class:")
    for j in range(0, n_steps+use_init_matrix):
        print("  Step "+str(j)+":                 ", np.asarray(zoom[j,:,0]*100, dtype='int')/100, "= %.3f" % np.dot(hist/batch_size, zoom[j,:,0]))
        print("                          ", np.asarray(zoom[j,:,1]*100, dtype='int')/100, "= %.3f" % np.dot(hist/batch_size, zoom[j,:,1]))
    print("")

    exit()

if __name__ == "__main__":

    # argument list
    parser = ArgumentParser(description="Generate predictions based on trained EDRAM network")
    parser.add_argument("--l", "--list_params", type=str, nargs='?', default='none', const='none',
                        dest='list_params', help="Show a parameter list")
    # high-level options
    parser.add_argument("--gpu", type=int, nargs='?', default=-1, const=7,
                        dest='gpu_id', help="Specifies the GPU.")
    parser.add_argument("--data", type=int, default=1,
                        dest='dataset_id', help="ID of the test data set or path to the dataset.")
    parser.add_argument("--model", type=int, default=_model_id,
                        dest='model_id', help="Selects model type.")
    parser.add_argument("--checkpoint", "--checkpoint_weights", type=int, default=1,
                        dest='use_checkpoint_weights', help="Whether to load checkpoint weights.")
    parser.add_argument("--path","--load_path", type=str, default='.',
                        dest='load_path', help="Path for loading the model weights.")
    parser.add_argument("--bs", "--batch_size", type=int, default=_batch_size,
                        dest="batch_size", help="Size of each mini-batch")
    parser.add_argument("--iter", "--iterations", type=int, default=1,
                        dest="iterations", help="Number of mini-batches to process.")
    parser.add_argument("--show", "--show_steps", type=int, default=_n_steps,
                        dest="show_steps", help="Steps to show.")
    # model structure
    parser.add_argument("--steps", type=int, default=_n_steps,
                        dest="n_steps", help="Step size for digit recognition.")
    parser.add_argument("--glimpse", "--glimpse_size", "-a", type=int, default=26,
                        dest='glimpse_size', help="Window size of attention mechanism.")
    parser.add_argument("--coarse_size", type=int, default=12,
                        dest='coarse_size', help="Size of the rescaled input image for initialization of the network.")
    parser.add_argument("--conv_sizes", type=int, nargs='+', default=[5, 3],
                        dest="conv_sizes", help="List of sizes of convolution filters.")
    parser.add_argument("--conv_filters", type=int, default=128,
                        dest="n_filters", help="Number of filters in convolution.")
    parser.add_argument("--fc_dim", type=int, default=1024,
                        dest="fc_dim", help="Fully connected dimension.")
    parser.add_argument("--enc_dim", type=int, default=512,
                        dest="enc_dim", help="Encoder RNN state dimension.")
    parser.add_argument("--dec_dim", type=int, default=512,
                        dest="dec_dim", help="Decoder  RNN state dimension.")
    parser.add_argument("--classes", type=int, default=10,
                        dest="n_classes", help="Number of classes for recognition.")
    parser.add_argument("--clip", "--clip_value", type=float, default=1.0,
                        dest="clip_value", help="Clips Zoom Value in Spatial Transformer.")
    parser.add_argument("--unique", "--unique_emission", type=int, default=0,
                        dest="unique_emission", help="Inserts unique emission layer")
    parser.add_argument("--unique_glimpse", type=int, default=0,
                        dest="unique_glimpse", help="Inserts unique first glimpse layer")
    # output options
    parser.add_argument("--mode", "--output_mode", type=int, default=1,
                          dest="output_mode", help="Output last step or all steps.")
    parser.add_argument("--use_init", "--use_init_matrix", type=int, default=1,
                        dest="use_init_matrix", help="Whether to use the init matrix as output.")
    parser.add_argument("--dims", "--output_dims", "--output_emotion_dims", type=int, default=1,
                        dest="output_emotion_dims", help="Whether to output valence and arousal.")
    parser.add_argument("--headless", type=int, default=0,
                          dest="headless", help="Whether to use a dense classifier on all timesteps in parallel.")
    # normalisation of inputs and model layers
    parser.add_argument("--scale", "--scale_inputs", type=float, default=255,
                        dest="scale_inputs", help="Scaling Factor for Input Images.")
    parser.add_argument("--normalize", "--normalize_inputs", type=int, default=0,
                        dest="normalize_inputs", help="Whether to normalize the input images.")
    parser.add_argument("--bn", "--use_batch_norm", type=int, default=1,
                        dest="use_batch_norm", help="Whether to use batch normalization.")
    parser.add_argument("--do", "--dropout", type=float, default=0,
                        dest="dropout", help="Whether to use dropout (dropout precentage).")
    # pertaining to the accuracy computation
    parser.add_argument("--weighting", type=int, nargs='?', default=0, const=1,
                        dest='weighting', help="Weighting applied for accuracy from average model predictions.")
    parser.add_argument("--zoom", "--zoom_factor", type=float, default=1,
                        dest='zoom_factor', help="Targte Zoom Factor.")

    args = parser.parse_args()

    def list_args(**args):
        for i, arg in enumerate(args.items()):
            if i <= 2:
                if arg[1]=='none' and i == 0:
                    break
                elif i == 0:
                    print("\n[Info] Training Parameters\n")
            else:
                print(' '*(24-len(arg[0])), arg[0], "=", arg[1])

    list_args(**vars(args))
    main(**vars(args))

