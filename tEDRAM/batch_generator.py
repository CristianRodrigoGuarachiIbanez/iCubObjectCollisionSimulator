"""
    Batch Generator for Training
"""

import random
from typing import Tuple, List, Generator, Dict
from sequenceGenerator import SequenceGenerator
from numpy import ndarray, array, asarray, zeros, ones, hstack, reshape

def batch_generator(dataset_size: int, batch_size: int, init_state_size: Tuple, n_steps: int, features: ndarray,
                    labels: ndarray, dims1, dims2, locations, augment, scale, normalize, mean, std,
                    mode, mode2, mode3, model_id, glimpse_size, zoom):

    state_size_1, state_size_2 = init_state_size

    #indices = list(range(dataset_size))
    #random.shuffle(indices)
    indices: SequenceGenerator = SequenceGenerator(dataset_size//10, 10);
    inputs: Dict[str, ndarray] = None;
    outputs: Dict[str, ndarray] = None;
    start, end = 0, 0 #type: int, int
    # iterate over the minibatches
    i: int = 0
    while True:

        # select the sample indices
        start = batch_size*i
        end = batch_size*(i+1)
        #samples = sorted(indices[start:end])

        # prepare the minibatch

        # input image
        I: ndarray = None;

        if scale!=1 and scale!=0:
            I = indices.samples(features, start, end)/scale;
            #I = array(features[samples, ...], dtype='float32')/scale
        if normalize:
            I = (indices.samples(features, start, end)-mean)/std;
            #I = (array(features[samples, ...], dtype='float32')-mean)/std
        else:
            I = indices.samples(features, start, end)
            #I = array(features[samples, ...], dtype='float32')

        # transformation matrix with zoom paramters set to 1
        A = zeros((batch_size, 6), dtype='float32')
        A[:, (0,4)] = 1
        # initial RNN states
        S1:zeros = zeros((batch_size, state_size_1), dtype='float32')
        S2:zeros = zeros((batch_size, state_size_2), dtype='float32')
        # biases
        B1, B2, B3, B4, B5, B6 = None, None, None, None, None, None #type: ndarray, ndarray, ndarray, ndarray, ndarray, ndarray;
        if glimpse_size==(26,26):
            B1 = ones((batch_size, 26, 26, 1), dtype='float32')
            B2 = ones((batch_size, 24, 24, 1), dtype='float32')
            B3 = ones((batch_size, 12, 12, 1), dtype='float32')
            B4 = ones((batch_size, 8, 8, 1), dtype='float32')
            B5 = ones((batch_size, 6, 6, 1), dtype='float32')
            B6 = ones((batch_size, 4, 4, 1), dtype='float32')
        else:
            B1 = ones((batch_size, 16, 16, 1), dtype='float32')
            B2 = ones((batch_size, 16, 16, 1), dtype='float32')
            B3 = ones((batch_size, 8, 8, 1), dtype='float32')
            B4 = ones((batch_size, 8, 8, 1), dtype='float32')
            B5 = ones((batch_size, 6, 6, 1), dtype='float32')
            B6 = ones((batch_size, 4, 4, 1), dtype='float32')
        # target outputs
        Y_dim, Y_loc, Y_cla = None, None, None; #type: ndarray, ndarray, ndarray
        #Y_cla = array(labels[samples, ...], dtype='float32')
        Y_cla = indices.samples(labels, start, end)

        if dims1 is not None:
            if dims2 is None:
                #Y_dim = array(dims1[samples, ...], dtype='float32')
                #Y_dim = indices.samples(dims1, start, end)
                pass
            else:
                #Y_dim = array(hstack([dims1[samples, ...], dims2[samples, ...]]), dtype='float32')
                #Y_dim = array(hstack(indices.samples(dims1, start, end), indices.samples(dims2, start, end)), dtype='float32')
                pass
        if zoom==1:
            #Y_loc = array(locations[samples, ...], dtype='float32')
            print('LOCATION GENERATOR:',locations)
            Y_loc = indices.samples(locations, start, end)
        else:
            Y_loc = zeros((batch_size,6), dtype='float32')
            Y_loc[:,(0,4)] = zoom
            print("Y LOCATION BATCH GENERATOR:",Y_loc)

        # when using all outputs for training
        if (mode):
            Y_loc = reshape(Y_loc, (batch_size,1,6))
            Y_loc = hstack([Y_loc for i in range(0, n_steps+mode2)])
            if n_steps>1 and not mode3:
                Y_cla = reshape(Y_cla, (batch_size,1,Y_cla.shape[1]))
                Y_cla = hstack([Y_cla for i in range(0, n_steps)])
                if dims1 is not None:
                    Y_dim = reshape(Y_dim, (batch_size,1,2))
                    Y_dim = hstack([Y_dim for i in range(0, n_steps)])

        i +=1

        if (batch_size*(i+1) > len(indices.getSequenceList())):
            i = 0

        if augment is not None:
            for I in augment.flow(I, batch_size=batch_size, shuffle=False):
                break

        if(model_id==1 or model_id==2):
            inputs = {'input_image': I, 'input_matrix': A,
                      'initial_hidden_state_1': S1, 'initial_cell_state_1': S1,
                      'initial_cell_state_2': S2,
                      'b26': B1, 'b24': B2, 'b12': B3, 'b8': B4, 'b6': B5, 'b4': B6};
            if dims1 is not None:
                outputs = {'classifications': Y_cla, 'dimensions': Y_dim, 'localisations': Y_loc};
            else:
                outputs = {'classifications': Y_cla, 'localisations': Y_loc};
        elif model_id==3 or model_id==4:
            inputs = {'input_image': I}
            outputs = {'classifications': Y_cla}

        yield inputs, outputs;